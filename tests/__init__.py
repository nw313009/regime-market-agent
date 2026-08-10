"""Test package marker.

Present so pytest resolves the repo root as the import base: that makes ``import src.*`` work
the same way the notebooks do (spec C-a), and lets test modules import the shared payload
fixtures from ``tests.conftest`` as one module instance rather than two.
"""
