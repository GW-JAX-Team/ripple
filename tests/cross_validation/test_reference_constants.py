"""ripplegw's physical constants match every available reference backend.

Parametrized over every backend in ``tests/cross_validation/reference/`` whose
dependency is importable -- a backend that is not installed contributes zero
test cases rather than failing, so adding a new backend needs no edits here.
This is deliberately left unmarked (not ``accuracy``): it compares numeric
literals, not waveform output, so it is cheap enough to run on every PR
wherever the backend happens to be installed.
"""

import pytest

import ripplegw.constants as const
from tests.cross_validation.reference import REFERENCE_BACKENDS, get_backend

CONST_NAMES = [
    "MSUN",
    "MRSUN",
    "MTSUN",
    "G",
    "C",
    "PI",
    "TWO_PI",
    "MPC",
    "AU",
    "EULERGAMMA",
]

AVAILABLE_BACKENDS = [
    name for name, cls in REFERENCE_BACKENDS.items() if cls.available()
]


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("const_name", CONST_NAMES)
def test_constant_matches_reference(backend_name, const_name):
    backend = get_backend(backend_name)
    assert getattr(const, const_name) == backend.constants()[const_name]
