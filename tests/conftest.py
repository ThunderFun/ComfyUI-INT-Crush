"""Pytest conftest — registers the hyphenated package name in sys.modules.

Python's import system rejects hyphens in module names, so we inject
a namespace package manually so that relative imports inside
convlinear.py / _quant_utils.py / etc. resolve correctly.

pytest imports conftest.py before collecting any test module, guaranteeing
this shim is registered before the ``from _intcrush import load`` calls
in each test file execute.
"""

import sys
import types
from pathlib import Path

# Register "ComfyUI-INT-Crush" as a valid package in sys.modules.
pkg_dir = Path(__file__).resolve().parent.parent
pkg_name = "ComfyUI-INT-Crush"

if pkg_name not in sys.modules:
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(pkg_dir)]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg
# else: already registered (e.g. pytest re-import within the same process)
