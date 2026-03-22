"""Root conftest.py — ensures the main project's src/ is always first in sys.path.

Defense-in-depth against worktree Python path conflicts:
If a worktree's src/ accidentally ends up in sys.path (e.g., via a stale
editable install .pth file), this conftest ensures the main project's src/
takes precedence when running tests from the project root.
"""

import sys
from pathlib import Path

# Compute the main project's src/ directory relative to this conftest.py
_PROJECT_ROOT = Path(__file__).resolve().parent
_MAIN_SRC = str(_PROJECT_ROOT / "src")

# Ensure our src/ is first, ahead of any worktree src/ that might be
# on sys.path via a stale .pth file.
if _MAIN_SRC not in sys.path:
    sys.path.insert(0, _MAIN_SRC)
elif sys.path[0] != _MAIN_SRC:
    sys.path.remove(_MAIN_SRC)
    sys.path.insert(0, _MAIN_SRC)
