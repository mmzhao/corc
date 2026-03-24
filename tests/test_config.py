"""Tests for centralized configuration system.

Verifies:
- Loading defaults when no config file exists
- Loading overrides from .corc/config.yaml
- Deep merge of nested dicts (overrides only touched keys, preserves rest)
- Dot-notation get() access for nested keys
- set() modifies values and creates intermediate dicts
- save() writes only user overrides (diff from defaults)
- _parse_value handles booleans, ints, floats, null, lists, strings
- All modules read their defaults from DEFAULTS dict
- Config CLI: corc config show / corc config set
"""

import json
from pathlib import Path

import pytest
import yaml

from corc.config import (
    CorcConfig,
    DEFAULTS,
    _deep_merge,
    _diff_from_defaults,
    _parse_value,
    load_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project directory with .corc/."""
    corc_dir = tmp_path / ".corc"
    corc_dir.mkdir()
    # Write a .git marker so get_project_root would find this
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def tmp_project_with_config(tmp_project):
    """Project with a .corc/config.yaml containing some overrides."""
    config_path = tmp_project / ".corc" / "config.yaml"
    config_data = {
        "dispatch": {
            "agent_timeout_s": 3600,
            "max_budget_usd": 5.0,
        },
        "retry": {
            "default_retries": 4,
        },
        "audit": {
            "rotate_after_days": 30,
        },
    }
    with open(config_path, "w") as f:
        yaml.safe_dump(config_data, f)
    return tmp_project


# ---------------------------------------------------------------------------
# CorcConfig: defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    """Config returns correct defaults when no file exists."""

    def test_defaults_no_file(self, tmp_project):
        """load_config returns defaults when no config.yaml exists."""
        (tmp_project / ".corc" / "config.yaml").unlink(missing_ok=True)
        cfg = load_config(tmp_project)
        assert cfg.get("dispatch.agent_timeout_s") == 3600
        assert cfg.get("retry.default_retries") == 2
        assert cfg.get("retry.reduced_retries") == 1
        assert cfg.get("retry.increased_retries") == 3
        assert cfg.get("retry.min_samples") == 5
        assert cfg.get("retry.high_success_threshold") == 0.90
        assert cfg.get("retry.low_success_threshold") == 0.50
        assert cfg.get("daemon.poll_interval") == 5.0
        assert cfg.get("daemon.parallel") == 1
        assert cfg.get("audit.backup_interval") == "daily"
        assert cfg.get("audit.rotate_after_days") == 90
        assert cfg.get("alerts.cost.enabled") is True
        assert cfg.get("alerts.cost.daily_limit_usd") == 50.0
        assert cfg.get("patterns.low_score_threshold") == 5.0
        assert cfg.get("patterns.high_score_threshold") == 9.0
        assert cfg.get("patterns.flag_threshold") == 7.0
        assert cfg.get("patterns.min_sample_size") == 3
        assert cfg.get("patterns.trust_min_sample") == 20
        assert cfg.get("curation.blacklist_threshold") == 3
        assert cfg.get("knowledge.target_tokens") == 500
        assert cfg.get("webhooks.timeout") == 10.0

    def test_defaults_empty_file(self, tmp_project):
        """load_config returns defaults when config.yaml is empty."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text("")
        cfg = load_config(tmp_project)
        assert cfg.get("dispatch.agent_timeout_s") == 3600

    def test_defaults_via_constructor(self):
        """CorcConfig() with no data returns pure defaults."""
        cfg = CorcConfig()
        assert cfg.get("dispatch.agent_timeout_s") == 3600
        assert cfg.get("retry.default_retries") == 2

    def test_system_works_without_config_file(self, tmp_path):
        """System works even when no .corc directory exists."""
        cfg = load_config(tmp_path)
        # Should still return defaults
        assert cfg.get("dispatch.agent_timeout_s") == 3600
        assert cfg.get("retry.default_retries") == 2


# ---------------------------------------------------------------------------
# CorcConfig: overrides
# ---------------------------------------------------------------------------


class TestOverrides:
    """Config correctly merges overrides from file."""

    def test_file_overrides_applied(self, tmp_project_with_config):
        """Values from config.yaml override defaults."""
        cfg = load_config(tmp_project_with_config)
        assert cfg.get("dispatch.agent_timeout_s") == 3600
        assert cfg.get("dispatch.max_budget_usd") == 5.0
        assert cfg.get("retry.default_retries") == 4
        assert cfg.get("audit.rotate_after_days") == 30

    def test_non_overridden_values_preserved(self, tmp_project_with_config):
        """Values not in config.yaml still have defaults."""
        cfg = load_config(tmp_project_with_config)
        # These weren't in the override file
        assert cfg.get("dispatch.provider") == "claude-code"
        assert cfg.get("retry.reduced_retries") == 1
        assert cfg.get("audit.backup_interval") == "daily"
        assert cfg.get("daemon.poll_interval") == 5.0

    def test_deep_merge_preserves_sibling_keys(self):
        """Deep merge preserves sibling keys in nested dicts."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"x": 10}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 10, "y": 2}, "b": 3}

    def test_deep_merge_adds_new_keys(self):
        """Deep merge adds new keys that aren't in base."""
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_deep_merge_does_not_mutate(self):
        """Deep merge doesn't modify input dicts."""
        base = {"a": {"x": 1}}
        override = {"a": {"x": 10}}
        base_copy = {"a": {"x": 1}}
        _deep_merge(base, override)
        assert base == base_copy


# ---------------------------------------------------------------------------
# CorcConfig: get / set
# ---------------------------------------------------------------------------


class TestGetSet:
    """Dot-notation get/set operations."""

    def test_get_dotted_key(self):
        cfg = CorcConfig()
        assert cfg.get("dispatch.agent_timeout_s") == 3600

    def test_get_top_level_section(self):
        cfg = CorcConfig()
        dispatch = cfg.get("dispatch")
        assert isinstance(dispatch, dict)
        assert "agent_timeout_s" in dispatch

    def test_get_nonexistent_key(self):
        cfg = CorcConfig()
        assert cfg.get("nonexistent.key") is None

    def test_get_nonexistent_with_default(self):
        cfg = CorcConfig()
        assert cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_get_deep_nested(self):
        cfg = CorcConfig()
        assert cfg.get("alerts.cost.daily_limit_usd") == 50.0

    def test_set_existing_key(self):
        cfg = CorcConfig()
        cfg.set("dispatch.agent_timeout_s", 7200)
        assert cfg.get("dispatch.agent_timeout_s") == 7200

    def test_set_new_key(self):
        cfg = CorcConfig()
        cfg.set("custom.new_setting", "hello")
        assert cfg.get("custom.new_setting") == "hello"

    def test_set_deep_nested(self):
        cfg = CorcConfig()
        cfg.set("a.b.c.d", 42)
        assert cfg.get("a.b.c.d") == 42


# ---------------------------------------------------------------------------
# CorcConfig: save
# ---------------------------------------------------------------------------


class TestSave:
    """Config save writes only overrides (diff from defaults)."""

    def test_save_creates_file(self, tmp_project):
        cfg = CorcConfig(config_path=tmp_project / ".corc" / "config.yaml")
        cfg.set("dispatch.agent_timeout_s", 9999)
        saved_path = cfg.save()
        assert saved_path.exists()

    def test_save_only_overrides(self, tmp_project):
        """Save only writes keys that differ from defaults."""
        config_path = tmp_project / ".corc" / "config.yaml"
        cfg = CorcConfig(config_path=config_path)
        cfg.set("dispatch.agent_timeout_s", 9999)
        cfg.save()

        with open(config_path) as f:
            saved = yaml.safe_load(f)

        # Only the override should be in the file
        assert saved == {"dispatch": {"agent_timeout_s": 9999}}

    def test_save_empty_when_all_defaults(self, tmp_project):
        """Save writes empty/minimal file when nothing differs from defaults."""
        config_path = tmp_project / ".corc" / "config.yaml"
        cfg = CorcConfig(config_path=config_path)
        cfg.save()

        with open(config_path) as f:
            saved = yaml.safe_load(f)

        # No overrides — file should be empty or None
        assert saved is None or saved == {}

    def test_save_raises_without_path(self):
        cfg = CorcConfig()
        with pytest.raises(ValueError, match="No config path"):
            cfg.save()

    def test_save_roundtrip(self, tmp_project):
        """Set a value, save, reload — value persists."""
        config_path = tmp_project / ".corc" / "config.yaml"
        cfg = CorcConfig(config_path=config_path)
        cfg.set("retry.default_retries", 7)
        cfg.save()

        cfg2 = load_config(tmp_project)
        assert cfg2.get("retry.default_retries") == 7
        # Other defaults still intact
        assert cfg2.get("retry.reduced_retries") == 1


# ---------------------------------------------------------------------------
# _diff_from_defaults
# ---------------------------------------------------------------------------


class TestDiffFromDefaults:
    def test_identical_returns_empty(self):
        assert _diff_from_defaults(DEFAULTS, DEFAULTS) == {}

    def test_one_change(self):
        import copy

        data = copy.deepcopy(DEFAULTS)
        data["dispatch"]["agent_timeout_s"] = 9999
        diff = _diff_from_defaults(data, DEFAULTS)
        assert diff == {"dispatch": {"agent_timeout_s": 9999}}

    def test_new_top_level_key(self):
        import copy

        data = copy.deepcopy(DEFAULTS)
        data["custom"] = {"foo": "bar"}
        diff = _diff_from_defaults(data, DEFAULTS)
        assert diff == {"custom": {"foo": "bar"}}


# ---------------------------------------------------------------------------
# _parse_value
# ---------------------------------------------------------------------------


class TestParseValue:
    def test_boolean_true(self):
        assert _parse_value("true") is True
        assert _parse_value("True") is True
        assert _parse_value("yes") is True

    def test_boolean_false(self):
        assert _parse_value("false") is False
        assert _parse_value("False") is False
        assert _parse_value("no") is False

    def test_null(self):
        assert _parse_value("null") is None
        assert _parse_value("none") is None
        assert _parse_value("~") is None

    def test_integer(self):
        assert _parse_value("42") == 42
        assert _parse_value("0") == 0
        assert _parse_value("-1") == -1

    def test_float(self):
        assert _parse_value("3.14") == 3.14
        assert _parse_value("0.5") == 0.5

    def test_json_list(self):
        result = _parse_value('["Read", "Write"]')
        assert result == ["Read", "Write"]

    def test_plain_string(self):
        assert _parse_value("hello") == "hello"
        assert _parse_value("claude-code") == "claude-code"

    def test_integer_not_float(self):
        """Integer values should remain int, not float."""
        assert isinstance(_parse_value("42"), int)


# ---------------------------------------------------------------------------
# as_dict
# ---------------------------------------------------------------------------


class TestAsDict:
    def test_as_dict_returns_copy(self):
        cfg = CorcConfig()
        d = cfg.as_dict()
        d["dispatch"]["agent_timeout_s"] = 99999
        # Original should be unchanged
        assert cfg.get("dispatch.agent_timeout_s") == 3600

    def test_as_dict_contains_all_sections(self):
        cfg = CorcConfig()
        d = cfg.as_dict()
        for section in [
            "dispatch",
            "daemon",
            "retry",
            "alerts",
            "audit",
            "notifications",
            "patterns",
            "curation",
            "knowledge",
            "webhooks",
        ]:
            assert section in d, f"Missing section: {section}"


# ---------------------------------------------------------------------------
# Module integration: verify modules read from DEFAULTS
# ---------------------------------------------------------------------------


class TestModuleIntegration:
    """Verify that existing modules source their constants from DEFAULTS."""

    def test_adaptive_retry_constants(self):
        from corc.adaptive_retry import (
            DEFAULT_RETRIES,
            REDUCED_RETRIES,
            INCREASED_RETRIES,
            MIN_SAMPLES,
            HIGH_SUCCESS_THRESHOLD,
            LOW_SUCCESS_THRESHOLD,
        )

        assert DEFAULT_RETRIES == DEFAULTS["retry"]["default_retries"]
        assert REDUCED_RETRIES == DEFAULTS["retry"]["reduced_retries"]
        assert INCREASED_RETRIES == DEFAULTS["retry"]["increased_retries"]
        assert MIN_SAMPLES == DEFAULTS["retry"]["min_samples"]
        assert HIGH_SUCCESS_THRESHOLD == DEFAULTS["retry"]["high_success_threshold"]
        assert LOW_SUCCESS_THRESHOLD == DEFAULTS["retry"]["low_success_threshold"]

    def test_patterns_constants(self):
        from corc.patterns import (
            LOW_SCORE_THRESHOLD,
            HIGH_SCORE_THRESHOLD,
            FLAG_THRESHOLD,
            MIN_SAMPLE_SIZE,
            TRUST_MIN_SAMPLE,
        )

        assert LOW_SCORE_THRESHOLD == DEFAULTS["patterns"]["low_score_threshold"]
        assert HIGH_SCORE_THRESHOLD == DEFAULTS["patterns"]["high_score_threshold"]
        assert FLAG_THRESHOLD == DEFAULTS["patterns"]["flag_threshold"]
        assert MIN_SAMPLE_SIZE == DEFAULTS["patterns"]["min_sample_size"]
        assert TRUST_MIN_SAMPLE == DEFAULTS["patterns"]["trust_min_sample"]

    def test_curate_blacklist_threshold(self):
        from corc.curate import BLACKLIST_THRESHOLD

        assert BLACKLIST_THRESHOLD == DEFAULTS["curation"]["blacklist_threshold"]

    def test_knowledge_target_tokens(self):
        from corc.knowledge import TARGET_TOKENS

        assert TARGET_TOKENS == DEFAULTS["knowledge"]["target_tokens"]

    def test_dispatch_constraints_defaults(self):
        from corc.dispatch import Constraints

        c = Constraints()
        assert c.allowed_tools == DEFAULTS["dispatch"]["default_allowed_tools"]

    def test_backup_default_config(self):
        from corc.backup import DEFAULT_BACKUP_CONFIG

        assert DEFAULT_BACKUP_CONFIG == DEFAULTS["audit"]

    def test_backup_load_audit_config_defaults(self, tmp_project):
        """load_audit_config returns centralized defaults."""
        from corc.backup import load_audit_config

        config = load_audit_config(tmp_project / ".corc")
        assert config["backup_interval"] == "daily"
        assert config["rotate_after_days"] == 90

    def test_backup_load_audit_config_overrides(self, tmp_project):
        """load_audit_config picks up overrides from config file."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text(yaml.safe_dump({"audit": {"rotate_after_days": 30}}))
        from corc.backup import load_audit_config

        config = load_audit_config(tmp_project / ".corc")
        assert config["rotate_after_days"] == 30
        assert config["backup_interval"] == "daily"  # default preserved

    def test_analyze_load_alert_config_defaults(self, tmp_project):
        """load_alert_config returns centralized defaults."""
        from corc.analyze import load_alert_config

        config = load_alert_config(tmp_project / ".corc")
        assert config.daily_limit_usd == 50.0
        assert config.project_limit_usd == 200.0
        assert config.task_limit_usd == 10.0
        assert config.enabled is True

    def test_analyze_load_alert_config_overrides(self, tmp_project):
        """load_alert_config picks up overrides from config file."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text(
            yaml.safe_dump({"alerts": {"cost": {"daily_limit_usd": 100.0}}})
        )
        from corc.analyze import load_alert_config

        config = load_alert_config(tmp_project / ".corc")
        assert config.daily_limit_usd == 100.0
        assert config.project_limit_usd == 200.0  # default preserved

    def test_notification_config_defaults(self, tmp_project):
        """load_notification_config returns centralized defaults."""
        from corc.notifications import load_notification_config

        config = load_notification_config(tmp_project / ".corc")
        assert config.channels["terminal"]["enabled"] is True
        assert config.channels["slack"]["enabled"] is False
        assert "escalation" in config.triggers
        assert "terminal" in config.triggers["escalation"]


# ---------------------------------------------------------------------------
# CLI: corc config show / set
# ---------------------------------------------------------------------------


class TestConfigCLI:
    """Test the CLI commands for config show/set."""

    def test_config_show(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "show"])
        assert result.exit_code == 0
        # Should contain known section names
        assert "dispatch" in result.output
        assert "retry" in result.output
        assert "daemon" in result.output

    def test_config_show_key(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["config", "show", "--key", "dispatch.agent_timeout_s"]
        )
        assert result.exit_code == 0
        assert "3600" in result.output

    def test_config_show_section(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "show", "--key", "dispatch"])
        assert result.exit_code == 0
        assert "agent_timeout_s" in result.output

    def test_config_set(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["config", "set", "dispatch.agent_timeout_s", "3600"]
        )
        assert result.exit_code == 0
        assert "3600" in result.output

        # Verify it was persisted
        cfg = load_config(tmp_project)
        assert cfg.get("dispatch.agent_timeout_s") == 3600

    def test_config_set_boolean(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "set", "alerts.cost.enabled", "false"])
        assert result.exit_code == 0

        cfg = load_config(tmp_project)
        assert cfg.get("alerts.cost.enabled") is False

    def test_config_set_float(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "set", "daemon.poll_interval", "10.5"])
        assert result.exit_code == 0

        cfg = load_config(tmp_project)
        assert cfg.get("daemon.poll_interval") == 10.5

    def test_config_show_nonexistent_key(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "show", "--key", "does.not.exist"])
        assert result.exit_code != 0
        assert "not found" in result.output


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_corrupted_yaml(self, tmp_project):
        """Corrupted YAML falls back to defaults."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text("this: is: not: valid: [yaml")
        cfg = load_config(tmp_project)
        # Should still get defaults
        assert cfg.get("dispatch.agent_timeout_s") == 3600

    def test_non_dict_yaml(self, tmp_project):
        """Non-dict YAML (e.g. a string) falls back to defaults."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text('"just a string"')
        cfg = load_config(tmp_project)
        assert cfg.get("dispatch.agent_timeout_s") == 3600

    def test_notifications_section_defaults(self):
        """Notifications section has full SPEC defaults."""
        cfg = CorcConfig()
        channels = cfg.get("notifications.channels")
        assert "terminal" in channels
        assert "slack" in channels
        assert "discord" in channels
        assert "telegram" in channels
        assert channels["terminal"]["enabled"] is True
        assert channels["slack"]["enabled"] is False

        triggers = cfg.get("notifications.triggers")
        assert "escalation" in triggers
        assert "task_complete" in triggers
        assert "task_failure" in triggers
        assert "cost_threshold" in triggers
        assert "pause" in triggers
        assert "daily_summary" in triggers

    def test_config_path_property(self, tmp_project):
        cfg = load_config(tmp_project)
        assert cfg.config_path == tmp_project / ".corc" / "config.yaml"

    def test_config_path_none(self):
        cfg = CorcConfig()
        assert cfg.config_path is None
