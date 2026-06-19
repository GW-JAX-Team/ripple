"""JAX barycentering: detector position in the SSB frame and Roemer delay.

This reproduces, in differentiable JAX, the geometric quantity used by
``XLALSimulateExactPulsarSignal``: the detector position vector in the
solar-system-barycentre (SSB) frame, ``rDetector(t)``, and the light-travel
delay to the source

    dT(t) = n . rDetector(t)

where ``n`` is the unit vector pointing from the SSB to the source. The exact
CW model uses precisely this geometric delay (it deliberately neglects the
Einstein and Shapiro relativistic timing terms), so ``rDetector`` here matches
LAL's ``emit->rDetector`` (LALBarycenter.c lines ~1097-1116): the Earth-centre
position from the JPL ephemeris plus the diurnal rotation term evaluated with
GAST, *without* applying precession/nutation to the rotation term.

Times are represented as an integer GPS second plus a fractional second to
preserve ~ns precision through the table lookup and delay arithmetic.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..constants import AU, C, PI
from .earth import EarthState
from .time_utils import gmst_gast_rad, nutation_delpsi

__all__ = [
    "earth_pos_now",
    "detector_ssb_position",
    "roemer_delay",
    "source_unit_vector",
    "emission_delay",
]

# Obliquity of the ecliptic at J2000 (OBLQ in LALBarycenter.c), radians.
_EPS0 = 0.40909280422232891
# Earth angular velocity (rad/s) and Sun radius (light-seconds), from LAL.
_AU_OVER_C = AU / C
_RSUN_SEC = 2.322


def _detector_geocentric(det_location_m):
    """Return (rd, longitude, latitude, sinLat, cosLat) for a detector site.

    ``rd`` is the geocentric distance in light-seconds; longitude/latitude are
    geocentric (not geodetic), matching LAL's barycentering convention.
    """
    lx, ly, lz = (jnp.asarray(c) / C for c in det_location_m)
    rd = jnp.sqrt(lx * lx + ly * ly + lz * lz)
    longitude = jnp.arctan2(ly, lx)
    latitude = PI / 2.0 - jnp.arccos(lz / rd)
    return rd, longitude, latitude, jnp.sin(latitude), jnp.cos(latitude)


def source_unit_vector(alpha: Float, delta: Float) -> Float[Array, " 3"]:
    """Unit vector pointing from the SSB to the source (J2000 Cartesian).

    Args:
        alpha (Float): Right ascension ``α`` in radians.
        delta (Float): Declination ``δ`` in radians.

    Returns:
        Float[Array, " 3"]: ``(cosδ cosα, cosδ sinα, sinδ)``.
    """
    cos_delta = jnp.cos(delta)
    return jnp.stack(
        [cos_delta * jnp.cos(alpha), cos_delta * jnp.sin(alpha), jnp.sin(delta)]
    )


def earth_pos_now(
    gps_int: Float[Array, " n"],
    gps_frac: Float[Array, " n"],
    eph_gps0: float,
    eph_dt: float,
    eph_pos: Float[Array, "m 3"],
    eph_vel: Float[Array, "m 3"],
    eph_acc: Float[Array, "m 3"],
) -> Float[Array, "n 3"]:
    """Earth-centre position in the SSB frame (light-seconds) at given times.

    Note: unlike ``XLALBarycenterEarth`` (which errors out of range), the table
    index is *clamped* to the table here (JAX gather semantics) — the caller is
    responsible for ensuring all sample times lie within the ephemeris span
    (``Ephemeris.gps0 .. gps_end``); times outside it yield extrapolated,
    unphysical positions rather than an error.

    Reproduces the Taylor extrapolation about the nearest ephemeris entry used
    by ``XLALBarycenterEarth``::

        ientry = floor(t0e / dt + 0.5)
        tdiff  = t0e - dt * ientry + frac
        posNow = pos[ientry] + vel[ientry]*tdiff + 0.5*acc[ientry]*tdiff^2

    Args:
        gps_int (Float[Array, " n"]): Integer GPS seconds (as float).
        gps_frac (Float[Array, " n"]): Fractional GPS seconds in ``[0, 1)``.
        eph_gps0 (float): GPS time of the first ephemeris entry.
        eph_dt (float): Ephemeris table spacing (seconds).
        eph_pos, eph_vel, eph_acc (Float[Array, "m 3"]): Ephemeris tables in
            light-seconds, ``v/c`` and ``1/s`` respectively.

    Returns:
        Float[Array, "n 3"]: Earth-centre position per time, in light-seconds.
    """
    t0e = gps_int - eph_gps0
    ientry = jnp.floor(t0e / eph_dt + 0.5).astype(jnp.int32)
    ientry = jnp.clip(ientry, 0, eph_pos.shape[0] - 1)

    tdiff = (t0e - eph_dt * ientry) + gps_frac
    tdiff2 = tdiff * tdiff

    pos = eph_pos[ientry]  # (n, 3)
    vel = eph_vel[ientry]
    acc = eph_acc[ientry]
    return pos + vel * tdiff[:, None] + 0.5 * acc * tdiff2[:, None]


def detector_ssb_position(
    gps_int: Float[Array, " n"],
    gps_frac: Float[Array, " n"],
    det_location_m: tuple[float, float, float],
    eph_gps0: float,
    eph_dt: float,
    eph_pos: Float[Array, "m 3"],
    eph_vel: Float[Array, "m 3"],
    eph_acc: Float[Array, "m 3"],
) -> Float[Array, "n 3"]:
    """Detector position in the SSB frame (light-seconds), matching LAL.

    ``rDetector = posNow + rd * (cosLat cos(lon+gast), cosLat sin(lon+gast),
    sinLat)`` where ``rd``, ``lat`` (geocentric) and ``lon`` come from the
    detector's geocentric Cartesian location and GAST is the Greenwich
    apparent sidereal time.

    Args:
        gps_int, gps_frac: Integer / fractional GPS seconds.
        det_location_m (tuple): Geocentric Cartesian detector location (metres).
        eph_gps0, eph_dt, eph_pos, eph_vel, eph_acc: Earth ephemeris table.

    Returns:
        Float[Array, "n 3"]: Detector position per time, in light-seconds.
    """
    pos_now = earth_pos_now(
        gps_int, gps_frac, eph_gps0, eph_dt, eph_pos, eph_vel, eph_acc
    )

    rd, longitude, _latitude, sin_lat, cos_lat = _detector_geocentric(det_location_m)

    delpsi = nutation_delpsi(gps_int, gps_frac)
    _, gast = gmst_gast_rad(gps_int, gps_frac, delpsi)

    angle = longitude + gast  # (n,)
    rot = jnp.stack(
        [
            rd * cos_lat * jnp.cos(angle),
            rd * cos_lat * jnp.sin(angle),
            jnp.broadcast_to(rd * sin_lat, angle.shape),
        ],
        axis=-1,
    )  # (n, 3)
    return pos_now + rot


def roemer_delay(
    gps_int: Float[Array, " n"],
    gps_frac: Float[Array, " n"],
    alpha: Float,
    delta: Float,
    det_location_m: tuple[float, float, float],
    eph_gps0: float,
    eph_dt: float,
    eph_pos: Float[Array, "m 3"],
    eph_vel: Float[Array, "m 3"],
    eph_acc: Float[Array, "m 3"],
) -> Float[Array, " n"]:
    """Geometric light-travel delay ``dT(t) = n . rDetector(t)`` in seconds.

    This is the quantity added to the detector time to obtain the (geometric)
    SSB arrival time used in the pulsar phase model.

    Args:
        gps_int, gps_frac: Integer / fractional GPS seconds.
        alpha, delta (Float): Source right ascension / declination (radians).
        det_location_m (tuple): Detector geocentric location (metres).
        eph_gps0, eph_dt, eph_pos, eph_vel, eph_acc: Earth ephemeris table.

    Returns:
        Float[Array, " n"]: Delay ``dT`` per time (seconds).
    """
    r_det = detector_ssb_position(
        gps_int,
        gps_frac,
        det_location_m,
        eph_gps0,
        eph_dt,
        eph_pos,
        eph_vel,
        eph_acc,
    )
    n = source_unit_vector(alpha, delta)
    return r_det @ n


def emission_delay(
    earth: EarthState,
    alpha: Float,
    delta: Float,
    det_location_m: tuple[float, float, float],
) -> Float[Array, " n"]:
    """Full barycentering delay ``emit.deltaT`` (seconds), matching XLALBarycenter.

    Assembles ``roemer + erot + einstein - shapiro`` for the
    ``TIMECORRECTION_ORIGINAL`` time system (observatory term = 0), with the
    Earth-rotation term ``erot`` including lunisolar precession and nutation,
    and a finite source distance ``dInv = 0`` (no finite-distance correction).
    The SSB emission time is ``t_e(t) = t + emit.deltaT(t)``.

    Args:
        earth (EarthState): Output of :func:`ripplegw.cw.earth.earth_state`.
        alpha (Float): Source right ascension ``α`` (radians).
        delta (Float): Source declination ``δ`` (radians).
        det_location_m (tuple): Detector geocentric location (metres).

    Returns:
        Float[Array, " n"]: Full delay per time (seconds).
    """
    sin_a, cos_a = jnp.sin(alpha), jnp.cos(alpha)
    sin_d, cos_d = jnp.sin(delta), jnp.cos(delta)
    s = jnp.stack([cos_d * cos_a, cos_d * sin_a, sin_d])

    # ----- Roemer delay (geometric, Earth centre) -----
    roemer = earth.pos_now @ s

    rd, lon, _lat, sin_lat, cos_lat = _detector_geocentric(det_location_m)
    sin_eps0, cos_eps0 = jnp.sin(_EPS0), jnp.cos(_EPS0)

    # ----- Earth rotation with lunisolar precession (Explan. Supp. 3.212-2) -----
    cosd_sina_za = jnp.sin(alpha + earth.tze_a) * cos_d
    cosd_cosa_za = (
        jnp.cos(alpha + earth.tze_a) * jnp.cos(earth.theta_a) * cos_d
        - jnp.sin(earth.theta_a) * sin_d
    )
    sin_delta_p = (
        jnp.cos(alpha + earth.tze_a) * jnp.sin(earth.theta_a) * cos_d
        + jnp.cos(earth.theta_a) * sin_d
    )
    gast_lon_za = earth.gast_rad + lon - earth.z_a
    ndotd = sin_lat * sin_delta_p + cos_lat * (
        jnp.cos(gast_lon_za) * cosd_cosa_za + jnp.sin(gast_lon_za) * cosd_sina_za
    )
    erot = rd * ndotd

    # ----- forced nutation correction to the rotation term -----
    del_x = -earth.delpsi * (cos_d * sin_a * cos_eps0 + sin_d * sin_eps0)
    del_y = cos_d * cos_a * cos_eps0 * earth.delpsi - sin_d * earth.deleps
    del_z = cos_d * cos_a * sin_eps0 * earth.delpsi + cos_d * sin_a * earth.deleps
    gast_lon = earth.gast_rad + lon
    ndotd_nut = (
        sin_lat * del_z
        + cos_lat * jnp.cos(gast_lon) * del_x
        + cos_lat * jnp.sin(gast_lon) * del_y
    )
    erot = erot + rd * ndotd_nut

    # ----- Shapiro delay (Sun) -----
    se_dot_n = (
        earth.se[:, 2] * sin_d
        + (earth.se[:, 0] * cos_a + earth.se[:, 1] * sin_a) * cos_d
    )
    b = jnp.sqrt(earth.rse * earth.rse - se_dot_n * se_dot_n)
    shapiro_through = 9.852e-6 * jnp.log(
        _AU_OVER_C / (se_dot_n + jnp.sqrt(_RSUN_SEC * _RSUN_SEC + se_dot_n * se_dot_n))
    ) + 19.704e-6 * (1.0 - b / _RSUN_SEC)
    shapiro_usual = 9.852e-6 * jnp.log(_AU_OVER_C / (earth.rse + se_dot_n))
    through_sun = (b < _RSUN_SEC) & (se_dot_n < 0.0)
    shapiro = jnp.where(through_sun, shapiro_through, shapiro_usual)

    return roemer + erot + earth.einstein - shapiro
