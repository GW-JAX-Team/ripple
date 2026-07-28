"""Compare ripple's physical constants with every available reference backend.

Unavailable backends contribute no cases. The test is unmarked because it
compares constants rather than waveform output.
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
