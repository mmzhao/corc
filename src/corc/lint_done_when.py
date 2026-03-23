"""Done-when quality linter.

Validates that done_when criteria are machine-testable, not subjective.
Checks for measurable assertions, references to tests/schemas/files,
and rejects subjective adjectives.

Integrated into ``corc task create`` as a warning; ``--strict`` makes it an error.

Per SPEC.md "Done When" Quality Linting:
  - Accepted: "All unit tests pass", "Output matches JSON schema X"
  - Rejected: "Implementation works correctly", "Code is clean", "Good performance"
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Subjective patterns to reject.
#
# Words that indicate non-measurable, opinion-based criteria.
# Each entry: (compiled regex, human-readable label).
# ---------------------------------------------------------------------------
SUBJECTIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bgood\b", re.IGNORECASE), "good"),
    (re.compile(r"\bclean\b", re.IGNORECASE), "clean"),
    (re.compile(r"\bcorrect(ly)?\b", re.IGNORECASE), "correct"),
    (re.compile(r"\bproper(ly)?\b", re.IGNORECASE), "proper"),
    (re.compile(r"\bnice(ly)?\b", re.IGNORECASE), "nice"),
    (re.compile(r"\bwell\b", re.IGNORECASE), "well"),
    (re.compile(r"\bappropriate(ly)?\b", re.IGNORECASE), "appropriate"),
    (re.compile(r"\breasonab(le|ly)\b", re.IGNORECASE), "reasonable"),
    (re.compile(r"\belegant(ly)?\b", re.IGNORECASE), "elegant"),
    (re.compile(r"\bbeautiful(ly)?\b", re.IGNORECASE), "beautiful"),
    (re.compile(r"\breadable\b", re.IGNORECASE), "readable"),
    (re.compile(r"\bmaintainable\b", re.IGNORECASE), "maintainable"),
    (re.compile(r"\brobust\b", re.IGNORECASE), "robust"),
    (re.compile(r"\boptimal(ly)?\b", re.IGNORECASE), "optimal"),
    (re.compile(r"\bsuitable\b", re.IGNORECASE), "suitable"),
    (re.compile(r"\bacceptab(le|ly)\b", re.IGNORECASE), "acceptable"),
    (re.compile(r"\bintuitive(ly)?\b", re.IGNORECASE), "intuitive"),
    (re.compile(r"\bsimple\b", re.IGNORECASE), "simple"),
    (re.compile(r"\bperformant\b", re.IGNORECASE), "performant"),
]

# ---------------------------------------------------------------------------
# Testable patterns to accept.
#
# Words / tokens that indicate measurable, machine-verifiable criteria.
# ---------------------------------------------------------------------------
TESTABLE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\btests?\b", re.IGNORECASE),
    re.compile(r"\bpass(es|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bexists?\b", re.IGNORECASE),
    re.compile(r"\bmatch(es|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bcontains?\b", re.IGNORECASE),
    re.compile(r"\bschema\b", re.IGNORECASE),
    re.compile(r"\boutputs?\b", re.IGNORECASE),
    re.compile(r"\breturn(s|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bfail(s|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bfiles?\b", re.IGNORECASE),
    re.compile(r"\bvalidat(e|es|ed|ion|ing)\b", re.IGNORECASE),
    re.compile(r"\bcreated?\b", re.IGNORECASE),
    re.compile(r"\b\d+\b"),  # numbers indicate measurable thresholds
    re.compile(r"\berrors?\b", re.IGNORECASE),
    re.compile(r"\brunnabl[ey]\b", re.IGNORECASE),
    re.compile(r"\bcompil(e|es|ed|ing)\b", re.IGNORECASE),
    re.compile(r"\bpytest\b", re.IGNORECASE),
    re.compile(r"\bexits?\b", re.IGNORECASE),
    re.compile(r"\bstatus\b", re.IGNORECASE),
    re.compile(r"\blint(s|ed|ing|er)?\b", re.IGNORECASE),
    re.compile(r"\bcoverage\b", re.IGNORECASE),
    re.compile(r"\bcommit(s|ted|ting)?\b", re.IGNORECASE),
    re.compile(r"\bmerge[ds]?\b", re.IGNORECASE),
    re.compile(r"\bresponse\b", re.IGNORECASE),
    re.compile(r"\blog(s|ged|ging)?\b", re.IGNORECASE),
    re.compile(r"\breview(s|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bpost(s|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bfound\b", re.IGNORECASE),
    re.compile(r"\bissues?\b", re.IGNORECASE),
    re.compile(r"\brun(s|ning)?\b", re.IGNORECASE),
    re.compile(r"\bproduce[ds]?\b", re.IGNORECASE),
    re.compile(r"\breject(s|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\baccepts?\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Valid task types and type-specific required patterns.
#
# Each task type can define required patterns that must appear in done_when.
# If none match, a warning is emitted.
# ---------------------------------------------------------------------------
VALID_TASK_TYPES = ("implementation", "investigation", "bugfix")

TYPE_SPECIFIC_PATTERNS: dict[str, list[tuple[re.Pattern, str]]] = {
    "investigation": [
        (re.compile(r"\broot\s+cause\b", re.IGNORECASE), "root cause"),
        (re.compile(r"\bdocumented\b", re.IGNORECASE), "documented"),
    ],
    "bugfix": [
        (re.compile(r"\bregression\s+test\b", re.IGNORECASE), "regression test"),
        (re.compile(r"\breproduced\b", re.IGNORECASE), "reproduced"),
    ],
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class LintResult:
    """Result of linting a done_when string."""

    criteria: str
    warnings: list[str] = field(default_factory=list)
    subjective_words: list[str] = field(default_factory=list)
    has_testable_pattern: bool = False

    @property
    def passed(self) -> bool:
        """True when no warnings were raised."""
        return len(self.warnings) == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_compound_well(text: str, match: re.Match) -> bool:
    """Return True if 'well' is part of a compound adjective like 'well-tested'."""
    end = match.end()
    return end < len(text) and text[end] == "-"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lint_done_when(criteria: str, task_type: str = "implementation") -> LintResult:
    """Lint a *done_when* criteria string.

    Checks for:
    1. Subjective adjectives (good, clean, correct, proper, nice, well, ...).
    2. Presence of at least one testable pattern (test, pass, exist, ...).
    3. Type-specific required patterns (investigation: root cause/documented;
       bugfix: regression test/reproduced).

    Parameters
    ----------
    criteria:
        The done_when string to lint.
    task_type:
        One of ``"implementation"`` (default), ``"investigation"``, ``"bugfix"``.
        Controls which type-specific rules are applied.

    Returns a :class:`LintResult` whose :attr:`~LintResult.passed` property
    is ``False`` when any warning was emitted.
    """
    result = LintResult(criteria=criteria)

    if not criteria or not criteria.strip():
        result.warnings.append("done_when criteria is empty")
        return result

    # --- subjective words ---------------------------------------------------
    for pattern, label in SUBJECTIVE_PATTERNS:
        for match in pattern.finditer(criteria):
            # "well-tested", "well-defined" are fine
            if label == "well" and _is_compound_well(criteria, match):
                continue
            result.subjective_words.append(label)
            result.warnings.append(
                f"Subjective word '{match.group()}' found — use measurable criteria instead"
            )

    # --- testable patterns ---------------------------------------------------
    for pattern in TESTABLE_PATTERNS:
        if pattern.search(criteria):
            result.has_testable_pattern = True
            break

    if not result.has_testable_pattern:
        result.warnings.append(
            "No testable pattern found — include references to tests, files, "
            "schemas, or measurable outcomes "
            "(e.g. 'tests pass', 'file exists', 'output matches schema')"
        )

    # --- type-specific patterns ---------------------------------------------
    type_patterns = TYPE_SPECIFIC_PATTERNS.get(task_type)
    if type_patterns:
        matched_any = False
        labels = []
        for pattern, label in type_patterns:
            if pattern.search(criteria):
                matched_any = True
                break
            labels.append(label)
        if not matched_any:
            options = " or ".join(f"'{l}'" for l in labels)
            result.warnings.append(
                f"Task type '{task_type}' requires done_when to mention {options}"
            )

    return result
