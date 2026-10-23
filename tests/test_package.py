"""Verify that pytest can import the installed src-layout package."""

from importlib import import_module


def test_package_imports() -> None:
    import_module("pln_fx")
