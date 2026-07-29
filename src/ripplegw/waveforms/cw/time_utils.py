"""GPS time-system utilities for continuous-wave barycentering.

These reproduce the relevant pieces of LAL's date/barycentering code:

* the leap-second table used by ``XLALGPSLeapSeconds`` (lal/lib/date), and
* the Greenwich Mean / Apparent Sidereal Time computation embedded in
  ``XLALBarycenterEarth`` (lalpulsar/lib/LALBarycenter.c).

The leap-second count enters the conversion from the GPS/TAI clock to the UT1
clock that drives Earth's rotation angle; getting it exactly right matters at
the ~µs level in the diurnal (``erot``) part of the Roemer delay.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from ripplegw.constants import PI

# GPS second at 2000-01-01 00:00:00 UTC (= J2000-ish reference used by LAL).
GPS_EPOCH_2000 = 630720013

# Leap-second table: (GPS second at which it takes effect, TAI-UTC seconds).
# This is the table behind lal's XLALLeapSeconds (lal/lib/date/XLALLeapSeconds.h).
# XLALGPSLeapSeconds returns TAI-UTC - 19 (= GPS-UTC); LALBarycenter then uses
#   leapsSince2000 = XLALGPSLeapSeconds(gps) - 13 = (TAI-UTC) - 32.
_LEAP_TABLE_GPS = (
    -43200,
    46828800,
    78364801,
    109900802,
    173059203,
    252028804,
    315187205,
    346723206,
    393984007,
    425520008,
    457056009,
    504489610,
    551750411,
    599184012,
    820108813,
    914803214,
    1025136015,
    1119744016,
    1167264017,
)
_LEAP_TABLE_TAI_UTC = (
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
)

# TAI-UTC at 2000-01-01, used to offset to "leaps since 2000".
_TAI_UTC_2000 = 32


def leap_seconds_since_2000(gps_seconds: Float[Array, " n"]) -> Float[Array, " n"]:
    """Number of leap seconds added to UTC since 2000-01-01, at GPS times.

    Matches LAL's ``leapsSince2000 = XLALGPSLeapSeconds(gps) - 13``.

    Args:
        gps_seconds (Float[Array, " n"]): Integer GPS seconds (as float).

    Returns:
        Float[Array, " n"]: ``(TAI-UTC)(gps) - 32`` for each input time.
    """
    gps_seconds = jnp.asarray(gps_seconds)
    edges = jnp.asarray(_LEAP_TABLE_GPS, dtype=gps_seconds.dtype)
    tai_utc = jnp.asarray(_LEAP_TABLE_TAI_UTC, dtype=gps_seconds.dtype)
    # Count how many leap-second boundaries are at or before each time. The
    # table value at index (count-1) is the active TAI-UTC.
    idx = jnp.sum(gps_seconds[..., None] >= edges[None, ...], axis=-1) - 1
    idx = jnp.clip(idx, 0, len(_LEAP_TABLE_TAI_UTC) - 1)
    return tai_utc[idx] - _TAI_UTC_2000


def gmst_gast_rad(
    gps_seconds: Float[Array, " n"],
    gps_frac: Float[Array, " n"],
    delpsi: Float[Array, " n"],
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Greenwich Mean and Apparent Sidereal Time, in radians.

    Direct port of the rotational-state block of ``XLALBarycenterEarth``
    (LALBarycenter.c lines ~193-298). The returned angles are *not* wrapped to
    ``[0, 2π)`` (LAL leaves them unwrapped; only their sin/cos are used).

    Args:
        gps_seconds (Float[Array, " n"]): Integer GPS seconds (as float).
        gps_frac (Float[Array, " n"]): Fractional GPS seconds in ``[0, 1)``.
        delpsi (Float[Array, " n"]): Nutation in longitude ``Δψ`` (radians),
            used for the equation of the equinoxes (GAST = GMST + Δψ cos ε₀).

    Returns:
        tuple: ``(gmstRad, gastRad)``.
    """
    eps0 = 0.40909280422232891  # obliquity of the ecliptic, OBLQ in LAL

    leaps = leap_seconds_since_2000(gps_seconds)

    tu_int = gps_seconds - GPS_EPOCH_2000
    ut1_sec_since_2000 = tu_int - leaps

    tu_jc = (ut1_sec_since_2000 + gps_frac - 43200.0) / (8.64e4 * 36525.0)
    full_ut1_days = jnp.floor(ut1_sec_since_2000 / 8.64e4)
    tu0_jc = (full_ut1_days - 0.5) / 36525.0
    dtu = tu_jc - tu0_jc

    gmst0 = 24110.54841 + tu0_jc * (
        8640184.812866 + tu0_jc * (0.093104 - tu0_jc * 6.2e-6)
    )
    gmst = gmst0 + dtu * (
        8.64e4 * 36525.0
        + 8640184.812866
        + 0.093104 * (tu_jc + tu0_jc)
        - 6.2e-6 * (tu_jc * tu_jc + tu_jc * tu0_jc + tu0_jc * tu0_jc)
    )
    gmst_rad = gmst * PI / 43200.0
    gast_rad = gmst_rad + delpsi * jnp.cos(eps0)
    return gmst_rad, gast_rad


def nutation_delpsi(gps_seconds: Float[Array, " n"], gps_frac: Float[Array, " n"]):
    """Nutation in longitude ``Δψ`` (radians), two dominant terms.

    Port of the ``earth->delpsi`` computation in ``XLALBarycenterEarth``.
    Only ``Δψ`` is needed (for GAST); ``Δε`` is not used by the exact CW model.

    Args:
        gps_seconds (Float[Array, " n"]): Integer GPS seconds (as float).
        gps_frac (Float[Array, " n"]): Fractional GPS seconds.

    Returns:
        Float[Array, " n"]: ``Δψ`` in radians.
    """
    deg = PI / 180.0
    days_since_j2000 = (gps_seconds - GPS_EPOCH_2000 - 43200.0 + gps_frac) / 8.64e4
    return (-0.0048 * deg) * jnp.sin((125.0 - 0.05295 * days_since_j2000) * deg) - (
        4.0e-4 * deg
    ) * jnp.sin((200.9 + 1.97129 * days_since_j2000) * deg)
