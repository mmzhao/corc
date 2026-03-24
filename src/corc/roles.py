"""Agent role system — loads, validates, and composes role configs from YAML.

Roles define system prompts, tool restrictions, cost limits, and access levels.
They are stored as YAML files in .corc/roles/ and the built-in defaults ship
with the package in src/corc/_builtin_roles/.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Schema & validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "name",
    "description",
    "system_prompt",
    "allowed_tools",
    "cost_limits",
    "knowledge_write_access",
}
VALID_KNOWLEDGE_WRITE_ACCESS = {"none", "findings_only", "full"}
VALID_TOOL_NAMES = {
    "Read",
    "Edit",
    "Write",
    "Bash",
    "Grep",
    "Glob",
    "WebSearch",
    "WebFetch",
    "Agent",
}

COST_LIMITS_SCHEMA = {
    "max_budget_per_invocation_usd": (int | float, True),
    "max_turns_per_invocation": (int, True),
}


@dataclass
class RoleConfig:
    """Parsed and validated role configuration."""

    name: str
    description: str
    extends: str | None
    system_prompt: str
    knowledge_write_access: str
    allowed_tools: list[str]
    cost_limits: dict[str, Any]
    source_path: Path | None = None

    @property
    def max_budget_usd(self) -> float:
        return float(self.cost_limits.get("max_budget_per_invocation_usd", 3.0))

    @property
    def max_turns(self) -> int:
        return int(self.cost_limits.get("max_turns_per_invocation", 50))


@dataclass
class ValidationResult:
    """Result of role validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_role_yaml(path: Path) -> dict:
    """Parse a role YAML file and return raw dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Role file {path} does not contain a YAML mapping")
    return data


def validate_role_data(data: dict) -> ValidationResult:
    """Validate raw role data against the schema."""
    errors = []
    warnings = []

    # Check required fields
    for field_name in REQUIRED_FIELDS:
        if field_name not in data:
            errors.append(f"Missing required field: {field_name}")

    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # Validate name
    name = data["name"]
    if not isinstance(name, str) or not name.strip():
        errors.append("'name' must be a non-empty string")

    # Validate description
    if not isinstance(data["description"], str):
        errors.append("'description' must be a string")

    # Validate extends
    extends = data.get("extends")
    if extends is not None and not isinstance(extends, str):
        errors.append("'extends' must be a string or null")

    # Validate system_prompt
    if not isinstance(data["system_prompt"], str) or not data["system_prompt"].strip():
        errors.append("'system_prompt' must be a non-empty string")

    # Validate knowledge_write_access
    kwa = data.get("knowledge_write_access", "")
    if kwa not in VALID_KNOWLEDGE_WRITE_ACCESS:
        errors.append(
            f"'knowledge_write_access' must be one of {VALID_KNOWLEDGE_WRITE_ACCESS}, got '{kwa}'"
        )

    # Validate allowed_tools
    tools = data.get("allowed_tools", [])
    if not isinstance(tools, list):
        errors.append("'allowed_tools' must be a list")
    else:
        for tool in tools:
            if not isinstance(tool, str):
                errors.append(f"Tool name must be a string, got {type(tool).__name__}")
            elif "(" not in tool and tool not in VALID_TOOL_NAMES:
                # Allow tool restrictions like "Bash(git*, gh pr*)"
                warnings.append(f"Unknown tool name: '{tool}'")

    # Validate cost_limits
    cl = data.get("cost_limits", {})
    if not isinstance(cl, dict):
        errors.append("'cost_limits' must be a mapping")
    else:
        for key, (expected_type, required) in COST_LIMITS_SCHEMA.items():
            if key not in cl:
                if required:
                    errors.append(f"Missing required cost_limits field: {key}")
            else:
                val = cl[key]
                if not isinstance(val, (int, float)):
                    errors.append(
                        f"cost_limits.{key} must be a number, got {type(val).__name__}"
                    )
                elif val <= 0:
                    errors.append(f"cost_limits.{key} must be positive, got {val}")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def _data_to_role_config(data: dict, source_path: Path | None = None) -> RoleConfig:
    """Convert validated raw data to a RoleConfig."""
    return RoleConfig(
        name=data["name"],
        description=data["description"],
        extends=data.get("extends"),
        system_prompt=data["system_prompt"],
        knowledge_write_access=data["knowledge_write_access"],
        allowed_tools=data["allowed_tools"],
        cost_limits=data["cost_limits"],
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# Role composition (extends)
# ---------------------------------------------------------------------------


def compose_roles(child_data: dict, parent: RoleConfig) -> dict:
    """Compose a child role with its parent.

    The child inherits all parent fields, then overrides with its own values.
    Lists (allowed_tools) are replaced, not merged. system_prompt is appended
    to parent's if child's starts with '+'.
    """
    composed = {
        "name": child_data.get("name", parent.name),
        "description": child_data.get("description", parent.description),
        "extends": child_data.get("extends"),
        "knowledge_write_access": child_data.get(
            "knowledge_write_access", parent.knowledge_write_access
        ),
        "allowed_tools": child_data.get("allowed_tools", list(parent.allowed_tools)),
        "cost_limits": {**parent.cost_limits, **child_data.get("cost_limits", {})},
    }

    # System prompt: append if starts with '+', else replace
    child_prompt = child_data.get("system_prompt", "")
    if child_prompt.startswith("+"):
        composed["system_prompt"] = (
            parent.system_prompt.rstrip() + "\n\n" + child_prompt[1:].lstrip()
        )
    elif child_prompt:
        composed["system_prompt"] = child_prompt
    else:
        composed["system_prompt"] = parent.system_prompt

    return composed


# ---------------------------------------------------------------------------
# Role loader
# ---------------------------------------------------------------------------

_BUILTIN_ROLES_DIR = Path(__file__).parent / "_builtin_roles"


class RoleLoader:
    """Loads roles from project .corc/roles/ with fallback to built-in roles."""

    def __init__(self, project_root: Path | None = None):
        self._project_roles_dir = (
            (project_root / ".corc" / "roles") if project_root else None
        )
        self._builtin_dir = _BUILTIN_ROLES_DIR
        self._cache: dict[str, RoleConfig] = {}

    def _search_paths(self) -> list[Path]:
        """Return role directories, project-local first (takes precedence)."""
        paths = []
        if self._project_roles_dir and self._project_roles_dir.is_dir():
            paths.append(self._project_roles_dir)
        if self._builtin_dir.is_dir():
            paths.append(self._builtin_dir)
        return paths

    def _find_role_file(self, name: str) -> Path | None:
        """Find the YAML file for a role by name (project overrides built-in)."""
        for dir_path in self._search_paths():
            candidate = dir_path / f"{name}.yaml"
            if candidate.exists():
                return candidate
        return None

    def load(self, name: str) -> RoleConfig:
        """Load a role by name, resolving extends chain.

        Raises ValueError if role not found or validation fails.
        """
        if name in self._cache:
            return self._cache[name]

        role = self._load_recursive(name, seen=set())
        self._cache[name] = role
        return role

    def _load_recursive(self, name: str, seen: set[str]) -> RoleConfig:
        """Load a role, recursively resolving extends."""
        if name in seen:
            raise ValueError(
                f"Circular role inheritance detected: {name} -> {' -> '.join(seen)}"
            )
        seen.add(name)

        path = self._find_role_file(name)
        if path is None:
            raise ValueError(f"Role '{name}' not found in any roles directory")

        data = parse_role_yaml(path)

        extends = data.get("extends")
        if extends:
            parent = self._load_recursive(extends, seen)
            data = compose_roles(data, parent)

        # Validate the (possibly composed) data
        result = validate_role_data(data)
        if not result.valid:
            raise ValueError(
                f"Role '{name}' validation failed: {'; '.join(result.errors)}"
            )

        return _data_to_role_config(data, source_path=path)

    def list_roles(self) -> list[dict[str, str]]:
        """List all available roles with name, description, and source."""
        roles = {}
        # Built-in first (can be overridden by project)
        for dir_path in reversed(self._search_paths()):
            source = "built-in" if dir_path == self._builtin_dir else "project"
            for yaml_file in sorted(dir_path.glob("*.yaml")):
                name = yaml_file.stem
                try:
                    data = parse_role_yaml(yaml_file)
                    roles[name] = {
                        "name": name,
                        "description": data.get("description", ""),
                        "source": source,
                        "path": str(yaml_file),
                    }
                except Exception:
                    roles[name] = {
                        "name": name,
                        "description": "<error loading>",
                        "source": source,
                        "path": str(yaml_file),
                    }
        return sorted(roles.values(), key=lambda r: r["name"])

    def validate(self, name: str) -> ValidationResult:
        """Validate a role by name without caching."""
        path = self._find_role_file(name)
        if path is None:
            return ValidationResult(valid=False, errors=[f"Role '{name}' not found"])

        try:
            data = parse_role_yaml(path)
        except Exception as e:
            return ValidationResult(valid=False, errors=[f"YAML parse error: {e}"])

        # If it extends another role, resolve the chain first
        extends = data.get("extends")
        if extends:
            parent_path = self._find_role_file(extends)
            if parent_path is None:
                return ValidationResult(
                    valid=False, errors=[f"Parent role '{extends}' not found"]
                )
            try:
                parent = self.load(extends)
                data = compose_roles(data, parent)
            except ValueError as e:
                return ValidationResult(valid=False, errors=[str(e)])

        return validate_role_data(data)

    def clear_cache(self):
        """Clear the role cache."""
        self._cache.clear()


# ---------------------------------------------------------------------------
# Convenience: get constraints from a role
# ---------------------------------------------------------------------------


def constraints_from_role(role: RoleConfig) -> "Constraints":
    """Convert a RoleConfig into a dispatch Constraints object."""
    from corc.dispatch import Constraints

    return Constraints(
        allowed_tools=list(role.allowed_tools),
        max_budget_usd=role.max_budget_usd,
        max_turns=role.max_turns,
    )


def get_system_prompt_for_role(role: RoleConfig, task: dict, context: str) -> str:
    """Build a full system prompt from role config and task context."""
    return (
        f'<role name="{role.name}">\n'
        f"{role.system_prompt}\n\n"
        f'<task name="{task["name"]}">\n\n'
        f"{context}"
    )
