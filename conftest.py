"""
conftest.py — project-root pytest configuration.

Adds the repo root to sys.path so that `import api`, `import ranking`, etc.
work correctly when running tests without `pip install -e .`.
This is the single correct place for this bootstrap; individual test files
must NOT contain sys.path.insert().
"""
import sys
from pathlib import Path

# Insert only if not already present (idempotent)
_root = str(Path(__file__).parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
