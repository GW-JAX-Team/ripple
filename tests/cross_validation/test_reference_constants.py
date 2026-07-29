"""Compare ripple's physical constants with every available reference backend.

Unavailable backends contribute no cases. The test is unmarked because it
compares constants rather than waveform output.

``CONST_NAMES`` is discovered from ``ripplegw.constants`` rather than
hand-listed, so a newly added constant is automatically covered instead of
silently slipping through an out-of-date list.
"""

import pytest

import ripplegw.constants as const
from tests.cross_validation.reference import REFERENCE_BACKENDS, get_backend

CONST_NAMES = sorted(
    name
    for name, value in vars(const).items()
    if name.isupper() and isinstance(value, (int, float))
)

AVAILABLE_BACKENDS = [
    name for name, cls in REFERENCE_BACKENDS.items() if cls.available()
]


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_backend_defines_every_constant(backend_name):
    """Each backend's ``constants()`` must cover exactly ``CONST_NAMES``.

    Guards both directions: a constant added to ``ripplegw.constants``
    without a matching reference value, and a stale backend entry left
    behind after a ripple constant is renamed or removed.
    """
    backend = get_backend(backend_name)
    backend_names = set(backend.constants())
    missing = set(CONST_NAMES) - backend_names
    extra = backend_names - set(CONST_NAMES)
    assert not missing, (
        f"{backend_name} backend has no reference value for: {sorted(missing)}"
    )
    assert not extra, (
        f"{backend_name} backend defines constants ripple doesn't have: {sorted(extra)}"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("const_name", CONST_NAMES)
def test_constant_matches_reference(backend_name, const_name):
    backend = get_backend(backend_name)
    assert getattr(const, const_name) == backend.constants()[const_name]


def test_derived_constants_are_self_consistent():
    """MSUN, MRSUN, and MTSUN are all derived from the same GM_sun, alongside
    C and G (see LALConstants.h). Ripple hard-codes each as an independently
    transcribed literal rather than computing it, so equality against LAL
    alone wouldn't catch a transcription slip that happened to also match a
    stale/incorrect LAL value -- this checks ripple's own values are
    mutually consistent regardless of what LAL says.
    """
    assert const.TWO_PI == 2 * const.PI
    assert const.MTSUN == const.MRSUN / const.C
    assert const.MSUN == const.MRSUN * const.C**2 / const.G
