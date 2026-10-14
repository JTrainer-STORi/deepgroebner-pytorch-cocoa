from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
