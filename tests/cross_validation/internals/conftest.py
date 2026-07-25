"""Every test under ``cross_validation/internals/`` is white-box diagnostic
tooling that compares ripple's private per-mode functions against LAL
internals -- it never gates CI. Marking happens here so the test files
underneath don't each need a decorator.
"""

from pathlib import Path

import pytest

_ROOT = str(Path(__file__).parent)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if str(item.fspath).startswith(_ROOT):
            item.add_marker(pytest.mark.internals)
