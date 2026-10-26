"""Parametrization marks shared by the integration tests."""

import pytest

import ripplegw

# Eager (un-jitted) calls on these models dispatch about 285,000 operations one
# at a time: 6-9 s per forward call and 25-30 s per gradient, and the XLA cache
# cannot speed that up. The jitted forward pass runs the same lines on every build.
HIGHER_MODE_WAVEFORMS = {
    name
    for name in ripplegw.list_waveforms()
    if ripplegw.get_waveform_metadata(name).get("highest_mode", 2) > 2
}


def slow_for(names, slow_names):
    """``names`` as pytest parameters, with those in ``slow_names`` marked ``slow``."""
    return [
        pytest.param(name, marks=pytest.mark.slow) if name in slow_names else name
        for name in names
    ]
