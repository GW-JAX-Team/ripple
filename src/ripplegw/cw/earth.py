"""Full Earth state for high-accuracy barycentering.

Port of ``XLALBarycenterEarth`` (lalpulsar/lib/LALBarycenter.c) for the
``TIMECORRECTION_ORIGINAL`` time system: Earth-centre position/velocity in the
SSB frame, sidereal time, lunisolar precession and nutation angles, the
Einstein delay (TDB-TDT, via the TEMPO sin-series), and the Earth-Sun vector
needed for the Shapiro delay.

These feed :func:`ripplegw.cw.barycenter.emission_delay`, which assembles the
*full* barycentering delay (Roemer + Earth-rotation + Einstein - Shapiro) used
by ``XLALGeneratePulsarSignal`` — as opposed to the geometric-only delay used
by the exact reference signal.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..constants import PI
from .time_utils import GPS_EPOCH_2000, leap_seconds_since_2000

__all__ = ["EarthState", "earth_state"]

# Obliquity of the ecliptic at J2000 (OBLQ in LALBarycenter.c), radians.
_EPS0 = 0.40909280422232891


class EarthState(NamedTuple):
    """Earth state at a set of GPS times (arrays of shape ``(n,)`` / ``(n,3)``).

    Mirrors the fields of LAL's ``EarthState`` that the barycentering delay
    needs.
    """

    pos_now: Float[Array, "n 3"]  # Earth-centre position in SSB (light-seconds)
    vel_now: Float[Array, "n 3"]  # Earth-centre velocity (v/c)
    gmst_rad: Float[Array, " n"]  # Greenwich mean sidereal time (rad, unwrapped)
    gast_rad: Float[Array, " n"]  # Greenwich apparent sidereal time (rad)
    tze_a: Float[Array, " n"]  # precession angle ζ_A (rad)
    z_a: Float[Array, " n"]  # precession angle z_A (rad)
    theta_a: Float[Array, " n"]  # precession angle θ_A (rad)
    delpsi: Float[Array, " n"]  # nutation in longitude Δψ (rad)
    deleps: Float[Array, " n"]  # nutation in obliquity Δε (rad)
    einstein: Float[Array, " n"]  # Einstein delay TDB-TDT (s)
    se: Float[Array, "n 3"]  # Sun->Earth vector (light-seconds)
    rse: Float[Array, " n"]  # |se| (light-seconds)


def _interp_ephemeris(gps_int, gps_frac, gps0, dt, pos, vel, acc):
    """Taylor-extrapolate pos/vel about the nearest table entry (see earth_pos_now).

    Note: the table index is *clamped* to the table (JAX gather semantics),
    unlike ``XLALBarycenterEarth`` which errors out of range. The caller must
    ensure all sample times lie within the ephemeris span; times outside it
    yield extrapolated, unphysical values rather than an error.
    """
    t0 = gps_int - gps0
    ientry = jnp.floor(t0 / dt + 0.5).astype(jnp.int32)
    ientry = jnp.clip(ientry, 0, pos.shape[0] - 1)
    tdiff = (t0 - dt * ientry) + gps_frac
    tdiff2 = tdiff * tdiff
    p = pos[ientry] + vel[ientry] * tdiff[:, None] + 0.5 * acc[ientry] * tdiff2[:, None]
    v = vel[ientry] + acc[ientry] * tdiff[:, None]
    return p, v


def _einstein_delay(gps_int, gps_frac):
    """Einstein delay TDB-TDT (s), the leading TEMPO sin-series (40 terms)."""
    jedtdt = -7300.5 + (gps_int + 51.184 + gps_frac) / 8.64e4
    jt = jedtdt / 3.6525e5  # Julian millennia
    # (amplitude[µs], angular rate, phase) — tiers 1 and 2 from LALBarycenter.c
    terms = (
        (1656.674564, 6283.075849991, 6.240054195),
        (22.417471, 5753.384884897, 4.296977442),
        (13.839792, 12566.151699983, 6.196904410),
        (4.770086, 529.690965095, 0.444401603),
        (4.676740, 6069.776754553, 4.021195093),
        (2.256707, 213.299095438, 5.543113262),
        (1.694205, -3.523118349, 5.025132748),
        (1.554905, 77713.771467920, 5.198467090),
        (1.276839, 7860.419392439, 5.988822341),
        (1.193379, 5223.693919802, 3.649823730),
        (1.115322, 3930.209696220, 1.422745069),
        (0.794185, 11506.769769794, 2.322313077),
        (0.447061, 26.298319800, 3.615796498),
        (0.435206, -398.149003408, 4.349338347),
        (0.600309, 1577.343542448, 2.678271909),
        (0.496817, 6208.294251424, 5.696701824),
        (0.486306, 5884.926846583, 0.520007179),
        (0.432392, 74.781598567, 2.435898309),
        (0.468597, 6244.942814354, 5.866398759),
        (0.375510, 5507.553238667, 4.103476804),
        (0.243085, -775.522611324, 3.651837925),
        (0.173435, 18849.227549974, 6.153743485),
        (0.230685, 5856.477659115, 4.773852582),
        (0.203747, 12036.460734888, 4.333987818),
        (0.143935, -796.298006816, 5.957517795),
        (0.159080, 10977.078804699, 1.890075226),
        (0.119979, 38.133035638, 4.551585768),
        (0.118971, 5486.777843175, 1.914547226),
        (0.116120, 1059.381930189, 0.873504123),
        (0.137927, 11790.629088659, 1.135934669),
        (0.098358, 2544.314419883, 0.092793886),
        (0.101868, -5573.142801634, 5.984503847),
        (0.080164, 206.185548437, 2.095377709),
        (0.079645, 4694.002954708, 2.949233637),
        (0.062617, 20.775395492, 2.654394814),
        (0.075019, 2942.463423292, 4.980931759),
        (0.064397, 5746.271337896, 1.280308748),
        (0.063814, 5760.498431898, 4.167901731),
        (0.048042, 2146.165416475, 1.495846011),
        (0.048373, 155.420399434, 2.251573730),
    )
    total = jnp.zeros_like(jt)
    for amp, rate, phase in terms:
        total = total + amp * jnp.sin(rate * jt + phase)
    return 1.0e-6 * total


def earth_state(
    gps_int: Float[Array, " n"],
    gps_frac: Float[Array, " n"],
    earth_gps0: float,
    earth_dt: float,
    earth_pos: Float[Array, "m 3"],
    earth_vel: Float[Array, "m 3"],
    earth_acc: Float[Array, "m 3"],
    sun_gps0: float,
    sun_dt: float,
    sun_pos: Float[Array, "k 3"],
    sun_vel: Float[Array, "k 3"],
    sun_acc: Float[Array, "k 3"],
) -> EarthState:
    """Compute the full Earth state at the given GPS times.

    Args:
        gps_int, gps_frac: Integer / fractional GPS seconds.
        earth_gps0, earth_dt, earth_pos, earth_vel, earth_acc: Earth ephemeris.
        sun_gps0, sun_dt, sun_pos, sun_vel, sun_acc: Sun ephemeris (for the
            Earth-Sun vector used in the Shapiro delay).

    Returns:
        EarthState: All quantities needed by :func:`emission_delay`.
    """
    pos_now, vel_now = _interp_ephemeris(
        gps_int, gps_frac, earth_gps0, earth_dt, earth_pos, earth_vel, earth_acc
    )
    sun_pos_now, sun_vel_now = _interp_ephemeris(
        gps_int, gps_frac, sun_gps0, sun_dt, sun_pos, sun_vel, sun_acc
    )

    # ----- Earth rotation: GMST/GAST (TEMPO formulation) -----
    leaps = leap_seconds_since_2000(gps_int)
    tu_int = gps_int - GPS_EPOCH_2000
    ut1_sec = tu_int - leaps
    tu_jc = (ut1_sec + gps_frac - 43200.0) / (8.64e4 * 36525.0)
    full_ut1_days = jnp.floor(ut1_sec / 8.64e4)
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

    # ----- lunisolar precession (Eqs. 3.212 Explan. Supp.) -----
    arc = PI / 6.48e5  # arcsec -> rad
    tze_a = tu_jc * (2306.2181 + (0.30188 + 0.017998 * tu_jc) * tu_jc) * arc
    z_a = tu_jc * (2306.2181 + (1.09468 + 0.018203 * tu_jc) * tu_jc) * arc
    theta_a = tu_jc * (2004.3109 - (0.42665 + 0.041833 * tu_jc) * tu_jc) * arc

    # ----- nutation (two dominant terms) -----
    deg = PI / 180.0
    days_j2000 = (tu_int - 43200.0 + gps_frac) / 8.64e4
    delpsi = (-0.0048 * deg) * jnp.sin((125.0 - 0.05295 * days_j2000) * deg) - (
        4.0e-4 * deg
    ) * jnp.sin((200.9 + 1.97129 * days_j2000) * deg)
    deleps = (0.0026 * deg) * jnp.cos((125.0 - 0.05295 * days_j2000) * deg) + (
        2.0e-4 * deg
    ) * jnp.cos((200.9 + 1.97129 * days_j2000) * deg)
    gast_rad = gmst_rad + delpsi * jnp.cos(_EPS0)

    einstein = _einstein_delay(gps_int, gps_frac)

    se = pos_now - sun_pos_now
    rse = jnp.sqrt(jnp.sum(se * se, axis=-1))

    return EarthState(
        pos_now=pos_now,
        vel_now=vel_now,
        gmst_rad=gmst_rad,
        gast_rad=gast_rad,
        tze_a=tze_a,
        z_a=z_a,
        theta_a=theta_a,
        delpsi=delpsi,
        deleps=deleps,
        einstein=einstein,
        se=se,
        rse=rse,
    )
