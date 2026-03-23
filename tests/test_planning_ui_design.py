"""Tests for the planning UI design document.

Validates that the design document at knowledge/architecture/planning-ui-design.md
is complete, well-structured, and consistent with the existing codebase.
"""

import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
DESIGN_DOC = ROOT / "knowledge" / "architecture" / "planning-ui-design.md"


# ------------------------------------------------------------------
# Fixture: load and parse the design doc
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def design_content():
    """Raw content of the design document."""
    assert DESIGN_DOC.exists(), f"Design doc not found at {DESIGN_DOC}"
    return DESIGN_DOC.read_text()


@pytest.fixture(scope="module")
def frontmatter(design_content):
    """Parsed YAML frontmatter from the design document."""
    match = re.match(r"^---\n(.*?)\n---", design_content, re.DOTALL)
    assert match, "Design doc must have YAML frontmatter"
    return yaml.safe_load(match.group(1))


@pytest.fixture(scope="module")
def headings(design_content):
    """All markdown headings extracted from the design document."""
    return re.findall(r"^(#{1,4})\s+(.+)$", design_content, re.MULTILINE)


# ------------------------------------------------------------------
# Frontmatter validation
# ------------------------------------------------------------------


class TestFrontmatter:
    """Validate YAML frontmatter follows knowledge store conventions."""

    def test_has_id(self, frontmatter):
        assert "id" in frontmatter
        assert frontmatter["id"] == "planning-ui-design"

    def test_has_type(self, frontmatter):
        assert frontmatter.get("type") == "architecture"

    def test_has_project(self, frontmatter):
        assert frontmatter.get("project") == "corc"

    def test_has_tags(self, frontmatter):
        tags = frontmatter.get("tags", [])
        assert isinstance(tags, list)
        assert len(tags) > 0
        # Should contain relevant tags
        assert "planning" in tags or "browser-ui" in tags or "gui" in tags

    def test_has_required_fields(self, frontmatter):
        required = {"id", "type", "project", "tags", "created", "status"}
        missing = required - set(frontmatter.keys())
        assert not missing, f"Frontmatter missing required fields: {missing}"

    def test_status_is_active(self, frontmatter):
        assert frontmatter.get("status") == "active"

    def test_source_field(self, frontmatter):
        assert frontmatter.get("source") in ("human", "agent")


# ------------------------------------------------------------------
# Required sections per the task definition
# ------------------------------------------------------------------


class TestRequiredSections:
    """Verify the design doc covers all areas specified in the task's 'done when'."""

    def test_has_planning_flow_section(self, design_content):
        """Must describe planning flow in browser GUI."""
        assert re.search(
            r"planning flow|stage 1|stage 2|stage 3|spec.*development|task.*decomposition|review.*approve",
            design_content,
            re.IGNORECASE,
        ), "Design doc must describe the planning flow in browser GUI"

    def test_has_query_api_endpoints(self, design_content):
        """Must describe query API endpoints needed."""
        assert re.search(
            r"query api|api endpoint|/api/", design_content, re.IGNORECASE
        ), "Design doc must describe query API endpoints"

    def test_has_draft_review_approval(self, design_content):
        """Must describe how draft tasks are reviewed and approved."""
        assert re.search(
            r"draft.*review|review.*approv|draft.*approv|approval.*flow",
            design_content,
            re.IGNORECASE,
        ), "Design doc must describe draft task review and approval"

    def test_has_spec_task_dag_linking(self, design_content):
        """Must describe how spec and task DAG are linked."""
        assert re.search(
            r"spec.*task.*link|spec.*dag.*link|bidirectional|spec_section|spec_id|task_ids",
            design_content,
            re.IGNORECASE,
        ), "Design doc must describe spec-task DAG linking"

    def test_has_text_wireframes(self, design_content):
        """Must include text wireframes of key screens."""
        # Wireframes use box-drawing characters or ASCII art
        wireframe_indicators = [
            r"┌.*┐",  # box drawing
            r"Screen \d",  # labeled screens
            r"wireframe",  # explicit mention
        ]
        found = any(
            re.search(p, design_content, re.IGNORECASE) for p in wireframe_indicators
        )
        assert found, "Design doc must include text wireframes"


# ------------------------------------------------------------------
# Document structure quality
# ------------------------------------------------------------------


class TestDocumentStructure:
    """Validate overall document structure and organization."""

    def test_has_problem_section(self, design_content):
        assert re.search(r"^##\s+Problem", design_content, re.MULTILINE)

    def test_has_requirements_section(self, design_content):
        assert re.search(r"^##\s+Requirements", design_content, re.MULTILINE)

    def test_has_non_requirements_section(self, design_content):
        assert re.search(r"^##\s+Non-Requirements", design_content, re.MULTILINE)

    def test_has_design_section(self, design_content):
        # Either "Design" or a section about architecture/flow
        assert re.search(r"^##\s+(Design|Architecture)", design_content, re.MULTILINE)

    def test_has_rationale_section(self, design_content):
        assert re.search(r"^##\s+Rationale", design_content, re.MULTILINE)

    def test_minimum_content_length(self, design_content):
        """Design doc should be substantive (>2000 chars of actual content)."""
        # Strip frontmatter
        body = re.sub(r"^---\n.*?\n---\n", "", design_content, flags=re.DOTALL)
        assert len(body) > 2000, "Design doc body is too short to be substantive"

    def test_multiple_wireframes(self, design_content):
        """Should have multiple screen wireframes (at least 3)."""
        screen_count = len(re.findall(r"Screen \d", design_content))
        assert screen_count >= 3, (
            f"Expected >=3 wireframe screens, found {screen_count}"
        )


# ------------------------------------------------------------------
# API endpoint consistency with existing codebase
# ------------------------------------------------------------------


class TestAPIConsistency:
    """Validate that proposed API endpoints align with existing QueryAPI methods."""

    def test_references_existing_query_methods(self, design_content):
        """Should reference existing QueryAPI methods."""
        existing_methods = [
            "get_active_plan_tasks",
            "get_ready_tasks",
            "get_blocked_tasks_with_reasons",
            "get_running_tasks_with_agents",
            "get_recent_events",
            "get_task_stream_events",
        ]
        referenced = [m for m in existing_methods if m in design_content]
        assert len(referenced) >= 3, (
            f"Design should reference existing QueryAPI methods. Found: {referenced}"
        )

    def test_references_draft_status(self, design_content):
        """Should reference the existing 'draft' task status."""
        assert "draft" in design_content.lower()

    def test_references_task_approved_mutation(self, design_content):
        """Should reference the existing task_approved mutation type."""
        assert "task_approved" in design_content

    def test_references_mutation_log(self, design_content):
        """Should build on existing MutationLog for writes."""
        assert re.search(r"mutation.*log|MutationLog", design_content), (
            "Design should reference MutationLog for write operations"
        )

    def test_references_knowledge_store(self, design_content):
        """Should reference the KnowledgeStore for spec storage."""
        assert re.search(
            r"knowledge.*store|KnowledgeStore|knowledge/", design_content
        ), "Design should reference KnowledgeStore"

    def test_no_new_data_storage(self, design_content):
        """Design should reuse existing data layers, not introduce new ones."""
        assert re.search(
            r"no new data storage|reuse.*existing|existing data layers",
            design_content,
            re.IGNORECASE,
        ), "Design should explicitly state it reuses existing data layers"


# ------------------------------------------------------------------
# Planning flow stages
# ------------------------------------------------------------------


class TestPlanningStages:
    """Validate that all three planning stages are described."""

    def test_stage_1_spec_development(self, design_content):
        assert re.search(
            r"stage 1|spec.*development|spec.*editor",
            design_content,
            re.IGNORECASE,
        )

    def test_stage_2_task_decomposition(self, design_content):
        assert re.search(
            r"stage 2|task.*decomposition|task.*dag",
            design_content,
            re.IGNORECASE,
        )

    def test_stage_3_review_approve(self, design_content):
        assert re.search(
            r"stage 3|review.*approv|approve.*commit",
            design_content,
            re.IGNORECASE,
        )

    def test_describes_transitions(self, design_content):
        """Should describe how the user moves between stages."""
        assert re.search(
            r"transition|advance|decompose|next.*stage|→",
            design_content,
            re.IGNORECASE,
        )


# ------------------------------------------------------------------
# Wireframe quality
# ------------------------------------------------------------------


class TestWireframes:
    """Validate wireframes are meaningful and cover key screens."""

    def test_spec_editor_wireframe(self, design_content):
        """Should have a wireframe for the spec editor screen."""
        assert re.search(r"spec.*editor|editor.*spec", design_content, re.IGNORECASE)

    def test_dag_view_wireframe(self, design_content):
        """Should have a wireframe showing the DAG view."""
        assert re.search(
            r"task.*dag|dag.*canvas|dag.*view", design_content, re.IGNORECASE
        )

    def test_review_screen_wireframe(self, design_content):
        """Should have a wireframe for the review/approval screen."""
        assert re.search(
            r"review.*approve|approve.*commit|review.*screen",
            design_content,
            re.IGNORECASE,
        )

    def test_wireframes_use_box_drawing(self, design_content):
        """Wireframes should use box-drawing characters for structure."""
        box_chars = ["┌", "┐", "└", "┘", "│", "─", "├", "┤"]
        found = sum(1 for c in box_chars if c in design_content)
        assert found >= 4, "Wireframes should use box-drawing characters"


# ------------------------------------------------------------------
# Spec-DAG linking specifics
# ------------------------------------------------------------------


class TestSpecDagLinking:
    """Validate the spec-DAG linking design is complete."""

    def test_spec_to_task_direction(self, design_content):
        """Should describe spec → task linking direction."""
        assert re.search(
            r"spec.*→.*task|spec.*frontmatter.*task_ids|task_ids",
            design_content,
            re.IGNORECASE,
        )

    def test_task_to_spec_direction(self, design_content):
        """Should describe task → spec linking direction."""
        assert re.search(
            r"task.*→.*spec|spec_id|spec_section",
            design_content,
            re.IGNORECASE,
        )

    def test_coverage_analysis(self, design_content):
        """Should describe how unlinked spec sections are detected."""
        assert re.search(
            r"coverage|unlinked|uncovered|unimplemented",
            design_content,
            re.IGNORECASE,
        )


# ------------------------------------------------------------------
# Write endpoint coverage
# ------------------------------------------------------------------


class TestWriteEndpoints:
    """Validate that write/mutation endpoints are defined."""

    def test_has_create_draft_endpoint(self, design_content):
        assert re.search(
            r"POST.*/draft/task|create.*draft", design_content, re.IGNORECASE
        )

    def test_has_edit_draft_endpoint(self, design_content):
        assert re.search(r"PUT.*/draft/task|edit.*draft", design_content, re.IGNORECASE)

    def test_has_approve_endpoint(self, design_content):
        assert re.search(
            r"POST.*/plan/approve|approve.*endpoint", design_content, re.IGNORECASE
        )

    def test_has_reject_endpoint(self, design_content):
        assert re.search(
            r"POST.*/plan/reject|reject.*endpoint|discard.*plan",
            design_content,
            re.IGNORECASE,
        )

    def test_has_dag_validation_endpoint(self, design_content):
        """Should have an endpoint to validate DAG acyclicity."""
        assert re.search(
            r"validate.*dag|acyclic|cycle.*check|validate-dag",
            design_content,
            re.IGNORECASE,
        )
