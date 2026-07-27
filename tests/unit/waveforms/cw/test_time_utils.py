"""Unit tests for CW GPS time-system utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from ripplegw.waveforms.cw.time_utils import leap_seconds_since_2000


@pytest.mark.parametrize(
    "gps, expected",
    [
        (630720013, 0),  # 2000-01-01: TAI-UTC = 32 -> 0 since 2000
        (1000000000, 2),  # 2011-ish: TAI-UTC = 34 -> 2
        (1126259462, 4),  # GW150914 (2015): TAI-UTC = 36 -> 4
        (1369000000, 5),  # post-2017: TAI-UTC = 37 -> 5
    ],
)
def test_leap_seconds_since_2000(gps, expected):
    """Leap-second offsets match known TAI-UTC values minus 32."""
    val = int(leap_seconds_since_2000(jnp.asarray([float(gps)]))[0])
    assert val == expected


def test_leap_seconds_vectorized():
    """The function is vectorized over input times."""
    gps = jnp.asarray([630720013.0, 1000000000.0, 1126259462.0])
    out = np.asarray(leap_seconds_since_2000(gps))
    np.testing.assert_array_equal(out, [0, 2, 4])
