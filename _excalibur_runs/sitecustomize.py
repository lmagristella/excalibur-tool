"""Auto-configure PYTHONPATH for scripts run from _excalibur_runs.

Python automatically imports `sitecustomize` (if available on sys.path) during
startup. Since this folder is on sys.path when running a script from here,
we can reliably add the repository root so `import excalibur` works on any OS.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
