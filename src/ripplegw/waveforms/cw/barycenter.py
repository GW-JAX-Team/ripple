Looking at the issue, the problem is in `_detector_geocentric` where `lz / rd` produces `NaN` when `rd == 0` (geocentre). The fix is to handle the zero-distance case gracefully using `jnp.where` to avoid the division by zero.

I need to see the full file to provide a complete fix. Based on the context provided, here's the complete updated file:

```python
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

import jax.numpy as jnp
from jaxtyping import Array, Float

from ripplegw.constants import AU, PI, C
from ripplegw.typing import FloatLike
from ripplegw.waveforms.cw.earth import EarthState
from ripplegw.waveforms.cw.time_utils import gmst_gast_rad, nutation_delpsi

# Obliquity of the ecliptic at J2000 (OBLQ in LALBarycenter.c), radians.
_EPS0 = 0.40909280422232891
# Earth angular velocity (rad/s) and Sun radius (light-seconds), from LAL.
_AU_OVER_C = AU / C
_RSUN_SEC = 2.322


def _detector_geocentric(det_location_m):
    """Return (rd, longitude, latitude, sinLat, cosLat) for a detector site.

    ``rd`` is the geocentric distance in light-seconds; longitude/latitude are
    geocentric (not geodetic), matching LAL's barycentering convention.

    When the detector is at the geocentre (``rd == 0``), longitude and latitude
    are set to zero and the trig values are set to their limits (sinLat=0,
    cosLat=1), which gives a zero diurnal rotation term — consistent with LAL's
    treatment of the geocentre as an ordinary site.
    """
    lx, ly, lz = (jnp.asarray(c) / C for c in det_location_m)
    rd = jnp.sqrt(lx * lx + ly * ly + lz * lz)
    longitude = jnp.arctan2(ly, lx)
    # Guard against division by zero at the geocentre (rd == 0).
    # When rd == 0 the diurnal term vanishes anyway (rd * cos_lat * ...),
    # so the exact values of latitude/sinLat/cosLat do not matter; we choose
    # latitude = 0 so that sinLat = 0 and cosLat = 1.
    safe_rd = jnp.where(rd == 0.0, 1.0, rd)
    latitude = PI / 2.0 - jnp.arccos(jnp.where(rd == 0.0, 0.0, lz / safe_rd))
    sin_lat = jnp.sin(latitude)
    cos_lat = jnp.cos(latitude)
    return rd, longitude, latitude, sin_lat, cos_lat


def source_unit_vector(alpha: FloatLike, delta: FloatLike) -> Float[Array, " 3"]:
    """Unit vector pointing from the SSB to the source (J2000 Cartesian).

    Args:
        alpha (FloatLike): Right ascension ``α`` in radians.
        delta (FloatLike): Declination ``δ`` in radians.

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
    responsible for ensuring times lie within the ephemeris range.

    Args:
        gps_int: Integer part of GPS times (seconds).
        gps_frac: Fractional part of GPS times (seconds).
        eph_gps0: GPS time of the first ephemeris entry.
        eph_dt: Ephemeris time step (seconds).
        eph_pos: Earth position table, shape ``(m, 3)``, light-seconds.
        eph_vel: Earth velocity table, shape ``(m, 3)``, light-seconds/s.
        eph_acc: Earth acceleration table, shape ``(m, 3)``, light-seconds/s².

    Returns:
        Earth-centre position vectors, shape ``(n, 3)``, light-seconds.
    """
    t = (gps_int - eph_gps0) + gps_frac
    idx = jnp.floor(t / eph_dt).astype(jnp.int32)
    dt = t - idx * eph_dt

    pos0 = eph_pos[idx]
    vel0 = eph_vel[idx]
    acc0 = eph_acc[idx]

    dt = dt[:, None]
    return pos0 + vel0 * dt + 0.5 * acc0 * dt * dt


def detector_position_ssb(
    gps_int: Float[Array, " n"],
    gps_frac: Float[Array, " n"],
    earth_state: EarthState,
    det_location_m,
) -> Float[Array, "n 3"]:
    """Detector position in the SSB frame (light-seconds).

    Combines the Earth-centre position from the ephemeris with the diurnal
    rotation term, matching LAL's ``emit->rDetector`` (without
    precession/nutation applied to the rotation term).

    Args:
        gps_int: Integer GPS seconds, shape ``(n,)``.
        gps_frac: Fractional GPS seconds, shape ``(n,)``.
        earth_state: Pre-computed Earth orientation quantities.
        det_location_m: Detector location in ECEF metres, length-3 sequence.

    Returns:
        Shape ``(n, 3)``: detector position in SSB frame, light-seconds.
    """
    rd, longitude, _lat, sin_lat, cos_lat = _detector_geocentric(det_location_m)

    # GAST (Greenwich Apparent Sidereal Time) at each sample.
    gps_t = gps_int + gps_frac
    gast = gmst_gast_rad(gps_t, earth_state.dpsi, earth_state.eps)

    # Local Apparent Sidereal Time = GAST + east longitude.
    last = gast + longitude

    # Diurnal rotation term (geocentre-to-detector vector in SSB frame).
    # When rd == 0 this whole term is zero regardless of sin/cos values.
    rot_x = rd * cos_lat * jnp.cos(last)
    rot_y = rd * cos_lat * jnp.sin(last)
    rot_z = rd * sin_lat * jnp.ones_like(last)

    rot = jnp.stack([rot_x, rot_y, rot_z], axis=-1)

    # Earth-centre position from the ephemeris.
    earth_pos = earth_pos_now(
        gps_int,
        gps_frac,
        earth_state.gps0,
        earth_state.dt,
        earth_state.pos,
        earth_state.vel,
        earth_state.acc,
    )

    return earth_pos + rot


def roemer_delay(
    det_pos_ssb: Float[Array, "n 3"],
    n_hat: Float[Array, " 3"],
) -> Float[Array, " n"]:
    """Roemer (geometric) delay: light-travel time from SSB to detector.

    Args:
        det_pos_ssb: Detector position in SSB frame, shape ``(n, 3)``,
            light-seconds.
        n_hat: Unit vector from SSB to source, shape ``(3,)``.

    Returns:
        Shape ``(n,)``: Roemer delay in seconds (positive = detector is closer
        to the source than the SSB).
    """
    return jnp.dot(det_pos_ssb, n_hat)
```