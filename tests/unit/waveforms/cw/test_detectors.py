"""Unit tests for CW ground-based detector lookups."""

import pytest

from ripplegw.waveforms.cw.detectors import DETECTORS, get_detector


def test_get_detector_known():
    """Known detectors resolve and carry a 3-vector location."""
    for name in ("H1", "L1", "V1"):
        det = get_detector(name)
        assert det.name == name
        assert len(det.location) == 3
    # H1 x-coordinate from LALDetectors.h
    assert get_detector("H1").location[0] == pytest.approx(-2.16141492636e06)


def test_get_detector_unknown():
    """An unknown detector name raises KeyError."""
    with pytest.raises(KeyError):
        get_detector("ZZ")
    assert set(DETECTORS) == {"H1", "L1", "V1"}
