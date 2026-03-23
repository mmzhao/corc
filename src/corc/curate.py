"""Knowledge curation — tiered write access quality gate.

Low-level agents report structured findings as part of their task output.
Only the orchestrator or operator curates findings into the knowledge store.
This prevents knowledge pollution from ephemeral agents making permanent writes.

Usage:
    engine = CurationEngine(work_state, mutation_log, audit_log, knowledge_store)
    findings = engine.get_findings(task_id)
    engine.approve_finding(task_id, index, finding)
    engine.reject_finding(task_id, index, finding, reason="Low quality")
    suggestions = engine.get_blacklist_suggestions()
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from corc.audit import AuditLog
from corc.config import DEFAULTS
from corc.knowledge import KnowledgeStore
from corc.mutations import MutationLog
from corc.state import WorkState

BLACKLIST_THRESHOLD = DEFAULTS["curation"]["blacklist_threshold"]


@dataclass
class Finding:
    """A structured finding from an agent."""

    index: int
    content: str
    finding_type: str = "general"
    source_task_id: str = ""

    @classmethod
    def from_raw(cls, index: int, raw, task_id: str = "") -> "Finding":
        """Create a Finding from either a string or a dict."""
        if isinstance(raw, dict):
            return cls(
                index=index,
                content=raw.get("content", str(raw)),
                finding_type=raw.get("type", "general"),
                source_task_id=task_id,
            )
        return cls(
            index=index,
            content=str(raw),
            finding_type="general",
            source_task_id=task_id,
        )


@dataclass
class CurationResult:
    """Summary of a curation session."""

    task_id: str
    approved: int = 0
    rejected: int = 0
    skipped: int = 0
    blacklist_suggestions: list[str] = field(default_factory=list)


class CurationEngine:
    """Engine for curating agent findings into the knowledge store.

    Provides approve/reject workflow with rejection tracking and
    automatic blacklist suggestions when patterns emerge.
    """

    def __init__(
        self,
        work_state: WorkState,
        mutation_log: MutationLog,
        audit_log: AuditLog,
        knowledge_store: KnowledgeStore,
    ):
        self.ws = work_state
        self.ml = mutation_log
        self.al = audit_log
        self.ks = knowledge_store

    def get_findings(self, task_id: str) -> list[Finding]:
        """Get findings from a completed task.

        Returns a list of Finding objects parsed from the task's findings field.
        Raises ValueError if the task doesn't exist or isn't completed.
        """
        self.ws.refresh()
        task = self.ws.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        if task["status"] != "completed":
            raise ValueError(
                f"Task {task_id} is '{task['status']}', not 'completed'. "
                f"Only completed tasks can be curated."
            )

        raw_findings = task.get("findings", [])
        if not raw_findings:
            return []

        return [
            Finding.from_raw(i, f, task_id=task_id) for i, f in enumerate(raw_findings)
        ]

    def approve_finding(
        self,
        task_id: str,
        finding: Finding,
        doc_type: str = "note",
        project: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Approve a finding and write it to the knowledge store.

        Creates a markdown document in the knowledge store with
        source=orchestrator, linking back to the originating task.

        Returns the document ID in the knowledge store.
        """
        # Build markdown content with frontmatter
        tag_list = tags or []
        if finding.finding_type != "general":
            tag_list.append(finding.finding_type)
        tag_list.append("curated")

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        frontmatter_parts = [
            f"type: {doc_type}",
            f"source: orchestrator",
            f"finding_type: {finding.finding_type}",
            f"source_task: {task_id}",
            f"created: {now}",
        ]
        if project:
            frontmatter_parts.append(f"project: {project}")
        if tag_list:
            tags_yaml = ", ".join(tag_list)
            frontmatter_parts.append(f"tags: [{tags_yaml}]")

        frontmatter = "\n".join(frontmatter_parts)
        content = f"---\n{frontmatter}\n---\n\n# {finding.content[:80]}\n\n{finding.content}\n"

        # Write to knowledge store
        doc_id = self.ks.add(
            content=content, doc_type=doc_type, project=project, tags=tag_list
        )

        # Log approval to mutation log
        self.ml.append(
            "finding_approved",
            {
                "finding_index": finding.index,
                "finding_content": finding.content,
                "finding_type": finding.finding_type,
                "doc_id": doc_id,
            },
            reason=f"Finding approved and written to knowledge store as {doc_id}",
            task_id=task_id,
        )

        # Audit log
        self.al.log(
            "finding_approved",
            task_id=task_id,
            finding_index=finding.index,
            doc_id=doc_id,
            finding_type=finding.finding_type,
        )

        return doc_id

    def reject_finding(
        self,
        task_id: str,
        finding: Finding,
        reason: str,
    ) -> None:
        """Reject a finding, logging it to the mutation log with a reason.

        The rejection is tracked by finding_type for blacklist suggestion
        purposes.
        """
        self.ml.append(
            "finding_rejected",
            {
                "finding_index": finding.index,
                "finding_content": finding.content,
                "finding_type": finding.finding_type,
                "rejection_reason": reason,
            },
            reason=f"Finding rejected: {reason}",
            task_id=task_id,
        )

        self.al.log(
            "finding_rejected",
            task_id=task_id,
            finding_index=finding.index,
            finding_type=finding.finding_type,
            reason=reason,
        )

    def get_rejection_counts(self) -> dict[str, int]:
        """Get rejection counts grouped by finding type.

        Scans the mutation log for all finding_rejected entries
        and tallies them by finding_type.
        """
        counts: dict[str, int] = {}
        for entry in self.ml.read_all():
            if entry["type"] == "finding_rejected":
                ftype = entry["data"].get("finding_type", "general")
                counts[ftype] = counts.get(ftype, 0) + 1
        return counts

    def get_blacklist_suggestions(self) -> list[dict]:
        """Return finding types that have been rejected 3+ times.

        Each suggestion includes the type, count, and recent rejection reasons.
        """
        # Gather all rejections by type
        rejections_by_type: dict[str, list[dict]] = {}
        for entry in self.ml.read_all():
            if entry["type"] == "finding_rejected":
                data = entry["data"]
                ftype = data.get("finding_type", "general")
                rejections_by_type.setdefault(ftype, []).append(
                    {
                        "task_id": entry.get("task_id", ""),
                        "reason": data.get("rejection_reason", ""),
                        "content": data.get("finding_content", ""),
                        "ts": entry.get("ts", ""),
                    }
                )

        suggestions = []
        for ftype, rejections in rejections_by_type.items():
            if len(rejections) >= BLACKLIST_THRESHOLD:
                # Collect unique reasons
                reasons = list({r["reason"] for r in rejections if r["reason"]})
                suggestions.append(
                    {
                        "finding_type": ftype,
                        "rejection_count": len(rejections),
                        "recent_reasons": reasons[-5:],  # Last 5 unique reasons
                        "suggestion": (
                            f"Finding type '{ftype}' has been rejected {len(rejections)} times. "
                            f"Consider adding to blacklist to auto-reject future findings of this type."
                        ),
                    }
                )

        return suggestions
