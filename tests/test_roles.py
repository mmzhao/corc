"""Tests for the agent role system.

Covers: YAML parsing, schema validation, role composition (extends),
role loading from built-in and project directories, constraint derivation,
system prompt generation, and CLI commands.
"""

import textwrap
from pathlib import Path

import pytest
import yaml

from corc.roles import (
    RoleConfig,
    RoleLoader,
    ValidationResult,
    compose_roles,
    constraints_from_role,
    get_system_prompt_for_role,
    parse_role_yaml,
    validate_role_data,
)
from corc.dispatch import Constraints


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_role(path: Path, data: dict):
    """Write a role YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _minimal_role(name: str = "test-role", **overrides) -> dict:
    """Return a minimal valid role dict."""
    role = {
        "name": name,
        "description": f"Test role: {name}",
        "extends": None,
        "system_prompt": "You are a test agent.",
        "knowledge_write_access": "findings_only",
        "allowed_tools": ["Read", "Grep", "Glob"],
        "cost_limits": {
            "max_budget_per_invocation_usd": 2.0,
            "max_turns_per_invocation": 30,
        },
    }
    role.update(overrides)
    return role


@pytest.fixture
def roles_dir(tmp_path):
    """Create a project with .corc/roles/ directory."""
    rd = tmp_path / ".corc" / "roles"
    rd.mkdir(parents=True)
    return rd


@pytest.fixture
def loader(tmp_path, roles_dir):
    """RoleLoader pointed at a temp project."""
    return RoleLoader(tmp_path)


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


class TestParseRoleYaml:
    def test_parse_valid_yaml(self, roles_dir):
        data = _minimal_role()
        path = roles_dir / "test-role.yaml"
        _write_role(path, data)

        parsed = parse_role_yaml(path)
        assert parsed["name"] == "test-role"
        assert parsed["allowed_tools"] == ["Read", "Grep", "Glob"]

    def test_parse_invalid_yaml(self, roles_dir):
        path = roles_dir / "bad.yaml"
        path.write_text(": : : not valid yaml [[[")
        with pytest.raises(Exception):
            parse_role_yaml(path)

    def test_parse_non_mapping(self, roles_dir):
        path = roles_dir / "list.yaml"
        path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            parse_role_yaml(path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateRoleData:
    def test_valid_role(self):
        data = _minimal_role()
        result = validate_role_data(data)
        assert result.valid
        assert result.errors == []

    def test_missing_required_field(self):
        data = _minimal_role()
        del data["system_prompt"]
        result = validate_role_data(data)
        assert not result.valid
        assert any("system_prompt" in e for e in result.errors)

    def test_missing_multiple_fields(self):
        result = validate_role_data({"name": "x"})
        assert not result.valid
        assert len(result.errors) >= 4  # missing description, system_prompt, allowed_tools, etc.

    def test_invalid_knowledge_write_access(self):
        data = _minimal_role(knowledge_write_access="admin")
        result = validate_role_data(data)
        assert not result.valid
        assert any("knowledge_write_access" in e for e in result.errors)

    def test_valid_knowledge_write_access_values(self):
        for kwa in ("none", "findings_only", "full"):
            data = _minimal_role(knowledge_write_access=kwa)
            result = validate_role_data(data)
            assert result.valid, f"Failed for {kwa}: {result.errors}"

    def test_invalid_cost_limits_missing(self):
        data = _minimal_role()
        data["cost_limits"] = {}
        result = validate_role_data(data)
        assert not result.valid
        assert any("max_budget_per_invocation_usd" in e for e in result.errors)

    def test_invalid_cost_limits_negative(self):
        data = _minimal_role()
        data["cost_limits"]["max_budget_per_invocation_usd"] = -1.0
        result = validate_role_data(data)
        assert not result.valid
        assert any("positive" in e for e in result.errors)

    def test_invalid_cost_limits_type(self):
        data = _minimal_role()
        data["cost_limits"]["max_budget_per_invocation_usd"] = "three"
        result = validate_role_data(data)
        assert not result.valid
        assert any("number" in e for e in result.errors)

    def test_allowed_tools_not_list(self):
        data = _minimal_role(allowed_tools="Read,Grep")
        result = validate_role_data(data)
        assert not result.valid
        assert any("list" in e for e in result.errors)

    def test_unknown_tool_warning(self):
        data = _minimal_role(allowed_tools=["Read", "UnknownTool"])
        result = validate_role_data(data)
        assert result.valid  # warnings don't block validity
        assert any("UnknownTool" in w for w in result.warnings)

    def test_tool_restriction_syntax_no_warning(self):
        """Tool restrictions like 'Bash(git*)' should not warn."""
        data = _minimal_role(allowed_tools=["Read", "Bash(git*, gh pr*)"])
        result = validate_role_data(data)
        assert result.valid
        assert len(result.warnings) == 0

    def test_empty_system_prompt(self):
        data = _minimal_role(system_prompt="   ")
        result = validate_role_data(data)
        assert not result.valid
        assert any("system_prompt" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Role composition
# ---------------------------------------------------------------------------


class TestRoleComposition:
    def test_child_overrides_parent(self):
        parent = RoleConfig(
            name="parent",
            description="Parent role",
            extends=None,
            system_prompt="Parent prompt.",
            knowledge_write_access="findings_only",
            allowed_tools=["Read", "Grep"],
            cost_limits={"max_budget_per_invocation_usd": 2.0, "max_turns_per_invocation": 30},
        )
        child_data = {
            "name": "child",
            "description": "Child role",
            "extends": "parent",
            "allowed_tools": ["Read", "Grep", "Edit", "Write"],
            "system_prompt": "Child prompt.",
            "cost_limits": {"max_budget_per_invocation_usd": 5.0},
        }
        composed = compose_roles(child_data, parent)
        assert composed["name"] == "child"
        assert composed["allowed_tools"] == ["Read", "Grep", "Edit", "Write"]
        assert composed["system_prompt"] == "Child prompt."
        assert composed["cost_limits"]["max_budget_per_invocation_usd"] == 5.0
        assert composed["cost_limits"]["max_turns_per_invocation"] == 30  # inherited

    def test_child_inherits_missing_fields(self):
        parent = RoleConfig(
            name="parent",
            description="Parent role",
            extends=None,
            system_prompt="Parent prompt.",
            knowledge_write_access="findings_only",
            allowed_tools=["Read", "Grep"],
            cost_limits={"max_budget_per_invocation_usd": 2.0, "max_turns_per_invocation": 30},
        )
        child_data = {
            "name": "child",
            "extends": "parent",
        }
        composed = compose_roles(child_data, parent)
        assert composed["system_prompt"] == "Parent prompt."
        assert composed["knowledge_write_access"] == "findings_only"
        assert composed["allowed_tools"] == ["Read", "Grep"]

    def test_system_prompt_append(self):
        parent = RoleConfig(
            name="parent",
            description="Parent role",
            extends=None,
            system_prompt="Base instructions.",
            knowledge_write_access="findings_only",
            allowed_tools=["Read"],
            cost_limits={"max_budget_per_invocation_usd": 2.0, "max_turns_per_invocation": 30},
        )
        child_data = {
            "name": "child",
            "extends": "parent",
            "system_prompt": "+Extra instructions.",
        }
        composed = compose_roles(child_data, parent)
        assert "Base instructions." in composed["system_prompt"]
        assert "Extra instructions." in composed["system_prompt"]
        assert composed["system_prompt"].startswith("Base instructions.")


# ---------------------------------------------------------------------------
# Role loader
# ---------------------------------------------------------------------------


class TestRoleLoader:
    def test_load_from_project_dir(self, loader, roles_dir):
        data = _minimal_role("custom-role")
        _write_role(roles_dir / "custom-role.yaml", data)

        role = loader.load("custom-role")
        assert role.name == "custom-role"
        assert role.allowed_tools == ["Read", "Grep", "Glob"]

    def test_load_builtin_roles(self):
        """All 5 built-in roles should load successfully."""
        loader = RoleLoader(None)  # no project dir, built-in only
        for name in ("scout", "implementer", "reviewer", "adversarial-reviewer", "planner"):
            role = loader.load(name)
            assert role.name == name
            assert len(role.system_prompt) > 0
            assert len(role.allowed_tools) > 0
            assert role.max_budget_usd > 0
            assert role.max_turns > 0

    def test_project_overrides_builtin(self, loader, roles_dir):
        """Project-local role should override built-in with same name."""
        custom = _minimal_role("implementer", description="Custom implementer")
        _write_role(roles_dir / "implementer.yaml", custom)

        role = loader.load("implementer")
        assert role.description == "Custom implementer"

    def test_load_nonexistent_role(self, loader):
        with pytest.raises(ValueError, match="not found"):
            loader.load("nonexistent-role")

    def test_load_with_extends(self, loader, roles_dir):
        parent = _minimal_role("base", allowed_tools=["Read", "Grep"])
        child = {
            "name": "extended",
            "description": "Extended role",
            "extends": "base",
            "system_prompt": "+Additional instructions.",
            "allowed_tools": ["Read", "Grep", "Edit", "Write"],
            "knowledge_write_access": "findings_only",
            "cost_limits": {"max_budget_per_invocation_usd": 5.0, "max_turns_per_invocation": 50},
        }
        _write_role(roles_dir / "base.yaml", parent)
        _write_role(roles_dir / "extended.yaml", child)

        role = loader.load("extended")
        assert role.name == "extended"
        assert "Additional instructions." in role.system_prompt
        assert "You are a test agent." in role.system_prompt  # inherited
        assert role.max_budget_usd == 5.0
        assert "Edit" in role.allowed_tools

    def test_circular_extends_detected(self, loader, roles_dir):
        a = _minimal_role("role-a")
        a["extends"] = "role-b"
        b = _minimal_role("role-b")
        b["extends"] = "role-a"
        _write_role(roles_dir / "role-a.yaml", a)
        _write_role(roles_dir / "role-b.yaml", b)

        with pytest.raises(ValueError, match="Circular"):
            loader.load("role-a")

    def test_caching(self, loader, roles_dir):
        data = _minimal_role("cached-role")
        _write_role(roles_dir / "cached-role.yaml", data)

        role1 = loader.load("cached-role")
        role2 = loader.load("cached-role")
        assert role1 is role2  # same object from cache

    def test_list_roles(self, loader, roles_dir):
        data = _minimal_role("my-role")
        _write_role(roles_dir / "my-role.yaml", data)

        roles = loader.list_roles()
        names = {r["name"] for r in roles}
        # Should include project role and built-in roles
        assert "my-role" in names
        assert "implementer" in names  # built-in

    def test_list_roles_includes_source(self, loader, roles_dir):
        data = _minimal_role("project-role")
        _write_role(roles_dir / "project-role.yaml", data)

        roles = loader.list_roles()
        project_roles = [r for r in roles if r["name"] == "project-role"]
        assert len(project_roles) == 1
        assert project_roles[0]["source"] == "project"

        builtin_roles = [r for r in roles if r["name"] == "scout"]
        assert len(builtin_roles) == 1
        assert builtin_roles[0]["source"] == "built-in"

    def test_validate_valid_role(self, loader, roles_dir):
        data = _minimal_role("valid-role")
        _write_role(roles_dir / "valid-role.yaml", data)

        result = loader.validate("valid-role")
        assert result.valid

    def test_validate_invalid_role(self, loader, roles_dir):
        data = {"name": "bad-role"}  # missing fields
        _write_role(roles_dir / "bad-role.yaml", data)

        result = loader.validate("bad-role")
        assert not result.valid
        assert len(result.errors) > 0

    def test_validate_nonexistent(self, loader):
        result = loader.validate("ghost-role")
        assert not result.valid
        assert any("not found" in e for e in result.errors)

    def test_validate_with_extends(self, loader, roles_dir):
        parent = _minimal_role("base-valid")
        child = {
            "name": "child-valid",
            "description": "Child with parent",
            "extends": "base-valid",
            "system_prompt": "+More.",
            "allowed_tools": ["Read"],
            "knowledge_write_access": "findings_only",
            "cost_limits": {"max_budget_per_invocation_usd": 1.0, "max_turns_per_invocation": 10},
        }
        _write_role(roles_dir / "base-valid.yaml", parent)
        _write_role(roles_dir / "child-valid.yaml", child)

        result = loader.validate("child-valid")
        assert result.valid

    def test_clear_cache(self, loader, roles_dir):
        data = _minimal_role("cacheable")
        _write_role(roles_dir / "cacheable.yaml", data)

        role1 = loader.load("cacheable")
        loader.clear_cache()
        role2 = loader.load("cacheable")
        assert role1 is not role2  # different objects after cache clear


# ---------------------------------------------------------------------------
# Constraint derivation
# ---------------------------------------------------------------------------


class TestConstraintsFromRole:
    def test_converts_to_constraints(self):
        role = RoleConfig(
            name="test",
            description="Test",
            extends=None,
            system_prompt="Test.",
            knowledge_write_access="findings_only",
            allowed_tools=["Read", "Grep"],
            cost_limits={"max_budget_per_invocation_usd": 5.0, "max_turns_per_invocation": 25},
        )
        c = constraints_from_role(role)
        assert isinstance(c, Constraints)
        assert c.allowed_tools == ["Read", "Grep"]
        assert c.max_budget_usd == 5.0
        assert c.max_turns == 25

    def test_scout_role_constraints(self):
        """Scout should be read-only (no Edit/Write/Bash)."""
        loader = RoleLoader(None)
        scout = loader.load("scout")
        c = constraints_from_role(scout)
        assert "Edit" not in c.allowed_tools
        assert "Write" not in c.allowed_tools
        assert "Read" in c.allowed_tools

    def test_implementer_role_constraints(self):
        """Implementer should have full write access."""
        loader = RoleLoader(None)
        impl = loader.load("implementer")
        c = constraints_from_role(impl)
        assert "Edit" in c.allowed_tools
        assert "Write" in c.allowed_tools
        assert "Bash" in c.allowed_tools

    def test_planner_role_constraints(self):
        """Planner should be read-only."""
        loader = RoleLoader(None)
        planner = loader.load("planner")
        c = constraints_from_role(planner)
        assert "Read" in c.allowed_tools
        assert "Grep" in c.allowed_tools
        assert "Edit" not in c.allowed_tools
        assert "Write" not in c.allowed_tools
        assert "Bash" not in c.allowed_tools


# ---------------------------------------------------------------------------
# System prompt generation
# ---------------------------------------------------------------------------


class TestSystemPromptGeneration:
    def test_includes_role_and_task(self):
        role = RoleConfig(
            name="implementer",
            description="Code gen",
            extends=None,
            system_prompt="You write code.",
            knowledge_write_access="findings_only",
            allowed_tools=["Read"],
            cost_limits={"max_budget_per_invocation_usd": 3.0, "max_turns_per_invocation": 50},
        )
        task = {"name": "build-feature", "done_when": "tests pass"}
        context = "File: main.py\n..."

        prompt = get_system_prompt_for_role(role, task, context)
        assert "ROLE: implementer" in prompt
        assert "You write code." in prompt
        assert "TASK: build-feature" in prompt
        assert "File: main.py" in prompt


# ---------------------------------------------------------------------------
# Built-in role integrity
# ---------------------------------------------------------------------------


class TestBuiltinRoles:
    """Verify all 5 built-in roles are well-formed and meet spec requirements."""

    BUILTIN_NAMES = ["scout", "implementer", "reviewer", "adversarial-reviewer", "planner"]

    def test_all_builtin_roles_exist(self):
        loader = RoleLoader(None)
        roles = loader.list_roles()
        names = {r["name"] for r in roles}
        for name in self.BUILTIN_NAMES:
            assert name in names, f"Built-in role '{name}' not found"

    def test_all_builtin_roles_validate(self):
        loader = RoleLoader(None)
        for name in self.BUILTIN_NAMES:
            result = loader.validate(name)
            assert result.valid, f"Role '{name}' validation failed: {result.errors}"

    def test_all_builtin_roles_have_findings_only(self):
        """Per spec, all built-in roles have findings_only knowledge write access."""
        loader = RoleLoader(None)
        for name in self.BUILTIN_NAMES:
            role = loader.load(name)
            assert role.knowledge_write_access == "findings_only", \
                f"Role '{name}' has {role.knowledge_write_access}, expected findings_only"

    def test_scout_is_readonly(self):
        loader = RoleLoader(None)
        role = loader.load("scout")
        for tool in ("Edit", "Write", "Bash"):
            assert tool not in role.allowed_tools, f"Scout should not have {tool}"

    def test_implementer_has_write_tools(self):
        loader = RoleLoader(None)
        role = loader.load("implementer")
        for tool in ("Read", "Edit", "Write", "Bash", "Grep", "Glob"):
            assert tool in role.allowed_tools, f"Implementer missing {tool}"

    def test_reviewer_has_restricted_bash(self):
        loader = RoleLoader(None)
        role = loader.load("reviewer")
        bash_tools = [t for t in role.allowed_tools if "Bash" in t]
        assert len(bash_tools) == 1
        assert "git" in bash_tools[0]

    def test_adversarial_reviewer_context_reset(self):
        loader = RoleLoader(None)
        role = loader.load("adversarial-reviewer")
        assert "CONTEXT RESET" in role.system_prompt
        assert "adversarial" in role.description.lower()

    def test_planner_is_readonly(self):
        loader = RoleLoader(None)
        role = loader.load("planner")
        for tool in ("Edit", "Write", "Bash"):
            assert tool not in role.allowed_tools, f"Planner should not have {tool}"


# ---------------------------------------------------------------------------
# RoleConfig properties
# ---------------------------------------------------------------------------


class TestRoleConfigProperties:
    def test_max_budget_usd(self):
        rc = RoleConfig(
            name="t", description="t", extends=None, system_prompt="t",
            knowledge_write_access="none", allowed_tools=[],
            cost_limits={"max_budget_per_invocation_usd": 7.5, "max_turns_per_invocation": 10},
        )
        assert rc.max_budget_usd == 7.5

    def test_max_turns(self):
        rc = RoleConfig(
            name="t", description="t", extends=None, system_prompt="t",
            knowledge_write_access="none", allowed_tools=[],
            cost_limits={"max_budget_per_invocation_usd": 1.0, "max_turns_per_invocation": 42},
        )
        assert rc.max_turns == 42

    def test_defaults_for_missing_cost_limits(self):
        rc = RoleConfig(
            name="t", description="t", extends=None, system_prompt="t",
            knowledge_write_access="none", allowed_tools=[],
            cost_limits={},
        )
        assert rc.max_budget_usd == 3.0  # default
        assert rc.max_turns == 50  # default
