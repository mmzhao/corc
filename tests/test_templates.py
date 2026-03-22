"""Tests for document templates."""

import re
import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from corc.templates import get_template, render_template, list_types, VALID_TYPES, get_templates_dir
from corc.cli import cli


# --- Unit tests for templates module ---


EXPECTED_TYPES = ("decision", "task-outcome", "architecture", "repo-context", "research")


def test_valid_types_match_expected():
    assert set(VALID_TYPES) == set(EXPECTED_TYPES)
    assert list_types() == list(EXPECTED_TYPES)


@pytest.fixture
def templates_root(tmp_path):
    """Copy real templates into a tmp_path-based project structure."""
    src = Path(__file__).parent.parent / "knowledge" / "_templates"
    dst = tmp_path / "knowledge" / "_templates"
    shutil.copytree(src, dst)
    return tmp_path


class TestGetTemplate:
    """Tests for get_template()."""

    @pytest.mark.parametrize("doc_type", EXPECTED_TYPES)
    def test_template_exists_and_loads(self, doc_type, templates_root):
        content = get_template(doc_type, project_root=templates_root)
        assert content, f"Template for {doc_type} should not be empty"
        assert "---" in content, f"Template for {doc_type} should contain YAML frontmatter"

    @pytest.mark.parametrize("doc_type", EXPECTED_TYPES)
    def test_template_has_valid_yaml_frontmatter(self, doc_type, templates_root):
        content = get_template(doc_type, project_root=templates_root)
        # Extract frontmatter between --- markers
        parts = content.split("---", 2)
        assert len(parts) >= 3, f"Template for {doc_type} should have YAML frontmatter delimiters"
        frontmatter = yaml.safe_load(parts[1])
        assert isinstance(frontmatter, dict)

    @pytest.mark.parametrize("doc_type", EXPECTED_TYPES)
    def test_template_frontmatter_has_required_fields(self, doc_type, templates_root):
        content = get_template(doc_type, project_root=templates_root)
        parts = content.split("---", 2)
        frontmatter = yaml.safe_load(parts[1])

        required_fields = ["id", "type", "tags", "created", "updated", "source", "status"]
        for field in required_fields:
            assert field in frontmatter, (
                f"Template {doc_type} missing required field: {field}"
            )

    @pytest.mark.parametrize("doc_type", EXPECTED_TYPES)
    def test_template_type_matches(self, doc_type, templates_root):
        content = get_template(doc_type, project_root=templates_root)
        parts = content.split("---", 2)
        frontmatter = yaml.safe_load(parts[1])
        assert frontmatter["type"] == doc_type

    @pytest.mark.parametrize("doc_type", EXPECTED_TYPES)
    def test_template_has_placeholder_variables(self, doc_type, templates_root):
        content = get_template(doc_type, project_root=templates_root)
        assert "${id}" in content, f"Template {doc_type} should have ${{id}} placeholder"
        assert "${title}" in content, f"Template {doc_type} should have ${{title}} placeholder"
        assert "${created}" in content, f"Template {doc_type} should have ${{created}} placeholder"
        assert "${updated}" in content, f"Template {doc_type} should have ${{updated}} placeholder"

    @pytest.mark.parametrize("doc_type", EXPECTED_TYPES)
    def test_template_has_markdown_heading(self, doc_type, templates_root):
        content = get_template(doc_type, project_root=templates_root)
        assert "# ${title}" in content, f"Template {doc_type} should have a title heading"

    def test_invalid_type_raises(self, templates_root):
        with pytest.raises(ValueError, match="Unknown template type"):
            get_template("invalid-type", project_root=templates_root)

    def test_missing_file_raises(self, tmp_path):
        # Point to a directory with no templates
        (tmp_path / "knowledge" / "_templates").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            get_template("decision", project_root=tmp_path)


class TestRenderTemplate:
    """Tests for render_template()."""

    @pytest.mark.parametrize("doc_type", EXPECTED_TYPES)
    def test_render_replaces_placeholders(self, doc_type, templates_root):
        rendered = render_template(
            doc_type,
            title="My Test Document",
            project="test-project",
            doc_id="test-uuid-1234",
            project_root=templates_root,
        )
        assert "${id}" not in rendered
        assert "${title}" not in rendered
        assert "${project}" not in rendered
        assert "${created}" not in rendered
        assert "${updated}" not in rendered

        assert "test-uuid-1234" in rendered
        assert "My Test Document" in rendered
        assert "test-project" in rendered

    @pytest.mark.parametrize("doc_type", EXPECTED_TYPES)
    def test_render_produces_valid_yaml(self, doc_type, templates_root):
        rendered = render_template(
            doc_type,
            title="Test Doc",
            project="myproject",
            doc_id="abc-123",
            project_root=templates_root,
        )
        parts = rendered.split("---", 2)
        frontmatter = yaml.safe_load(parts[1])
        assert frontmatter["id"] == "abc-123"
        assert frontmatter["type"] == doc_type
        assert frontmatter["status"] == "active"
        assert frontmatter["source"] == "human"

    def test_render_generates_uuid_when_not_provided(self, templates_root):
        rendered = render_template(
            "decision",
            title="Auto ID Test",
            project_root=templates_root,
        )
        parts = rendered.split("---", 2)
        frontmatter = yaml.safe_load(parts[1])
        # UUID v4 format
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            frontmatter["id"],
        )

    def test_render_generates_iso_timestamps(self, templates_root):
        rendered = render_template(
            "research",
            title="Timestamp Test",
            project_root=templates_root,
        )
        parts = rendered.split("---", 2)
        frontmatter = yaml.safe_load(parts[1])
        # YAML safe_load auto-parses ISO timestamps to datetime objects
        from datetime import datetime
        assert isinstance(frontmatter["created"], datetime)
        assert isinstance(frontmatter["updated"], datetime)
        # Also verify the raw string format
        iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
        assert re.search(f"created: {iso_pattern}", rendered)
        assert re.search(f"updated: {iso_pattern}", rendered)


class TestCLITemplateCommand:
    """Tests for `corc template TYPE` CLI command."""

    @pytest.mark.parametrize("doc_type", EXPECTED_TYPES)
    def test_template_outputs_content(self, doc_type):
        runner = CliRunner()
        result = runner.invoke(cli, ["template", doc_type])
        assert result.exit_code == 0, f"Failed for {doc_type}: {result.output}"
        assert "---" in result.output
        assert f"type: {doc_type}" in result.output

    @pytest.mark.parametrize("doc_type", EXPECTED_TYPES)
    def test_template_output_has_placeholders(self, doc_type):
        runner = CliRunner()
        result = runner.invoke(cli, ["template", doc_type])
        assert result.exit_code == 0
        assert "${id}" in result.output
        assert "${title}" in result.output

    def test_template_invalid_type_exits_with_error(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["template", "invalid-type"])
        assert result.exit_code != 0
        assert "Unknown template type" in result.output

    def test_template_render_flag(self):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["template", "decision", "--render", "--title", "My Decision", "--project", "corc"]
        )
        assert result.exit_code == 0
        assert "My Decision" in result.output
        assert "corc" in result.output
        # Placeholders should be resolved
        assert "${id}" not in result.output
        assert "${title}" not in result.output

    def test_template_render_without_options(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["template", "architecture", "--render"])
        assert result.exit_code == 0
        assert "Untitled" in result.output
        assert "${id}" not in result.output


class TestTemplateFiles:
    """Verify the actual template files in the repo."""

    @pytest.mark.parametrize("doc_type", EXPECTED_TYPES)
    def test_template_file_exists(self, doc_type):
        template_path = (
            Path(__file__).parent.parent / "knowledge" / "_templates" / f"{doc_type}.md"
        )
        assert template_path.exists(), f"Template file missing: {template_path}"

    @pytest.mark.parametrize("doc_type", EXPECTED_TYPES)
    def test_template_file_not_empty(self, doc_type):
        template_path = (
            Path(__file__).parent.parent / "knowledge" / "_templates" / f"{doc_type}.md"
        )
        content = template_path.read_text()
        assert len(content) > 50, f"Template {doc_type} is suspiciously short"

    def test_no_extra_template_files(self):
        """Ensure only expected template types exist."""
        templates_dir = Path(__file__).parent.parent / "knowledge" / "_templates"
        actual_files = {f.stem for f in templates_dir.glob("*.md")}
        expected_files = set(EXPECTED_TYPES)
        assert actual_files == expected_files, (
            f"Unexpected template files: {actual_files - expected_files}"
        )
