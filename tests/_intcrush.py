"""Shared import helpers for INT-Crush test modules.

Centralizes the conftest-registered hyphenated-package import so that
each test file doesn't need to repeat the ``"ComfyUI-INT-Crush."`` prefix.
The package shim in ``conftest.py`` must run before any test module is
collected (pytest guarantees this ordering).
"""

import importlib

_PACKAGE = "ComfyUI-INT-Crush"


def load(name: str):
    """Import a submodule of the INT-Crush package by short name.

    Example::

        _qu = load("_quant_utils")
        _qu.pack_int4(...)
    """
    return importlib.import_module(f"{_PACKAGE}.{name}")
