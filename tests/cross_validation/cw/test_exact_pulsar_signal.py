"""Cross-validation of ``ExactPulsarSignal`` against LALPulsar.

Reproduces the body of ``SimulatePulsarSignal.c`` (lines ~246-284) in Python using LAL's
*own* building blocks -- the detector states (``XLALGetDetectorStates`` -> ``rDetector``)
and antenna-pattern coefficients (``XLALComputeAMCoeffs`` -> ``a, b``) -- to form the
reference detector strain. We then reconstruct that strain from ripple's plus/cross
polarizations combined with LAL's antenna patterns, and require agreement. This
simultaneously checks the barycentered geometric delay, the spindown phase polynomial,
the reference-time handling, the amplitudes, and the ``{p, c}`` <-> (A1...A4)
decomposition. A second test checks the underlying geometric delay
(:func:`ripplegw.waveforms.cw.barycenter.roemer_delay`) against ``XLALBarycenter``
directly, since it is the primitive ``ExactPulsarSignal`` is built on.

Skipped unless both ``lalpulsar`` and an Earth/Sun ephemeris file are available -- point
``RIPPLE_EARTH_EPHEMERIS`` (and optionally ``RIPPLE_SUN_EPHEMERIS``) at a LALPulsar
``earth*``/``sun*`` file to run it.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

lal = pytest.importorskip("lal")
lalpulsar = pytest.importorskip("lalpulsar")

from ripplegw.waveforms.cw.barycenter import roemer_delay
from ripplegw.waveforms.cw.ephemeris import read_ephemeris_file
from ripplegw.waveforms.cw.pulsar_signal import exact_pulsar_polarizations
from tests.cross_validation.cw._lal_helpers import (
    find_ephemeris,
    log10_str,
    overlap_loss,
)

EARTH_FILE, SUN_FILE = find_ephemeris()
pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.skipif(
        EARTH_FILE is None or SUN_FILE is None,
        reason="LALPulsar Earth and Sun ephemeris files required",
    ),
]


def test_exact_pulsar_matches_lal_reference():
    """ripple {p,c} + LAL antenna patterns reproduces the LAL exact strain."""
    sun_file = SUN_FILE if SUN_FILE is not None else EARTH_FILE.replace("earth", "sun")
    eph = read_ephemeris_file(EARTH_FILE)
    edat = lalpulsar.InitBarycenter(EARTH_FILE, sun_file)
    det = lal.CachedDetectors[lal.LALDetectorIndexLHODIFF]
    loc = tuple(det.location)

    start_gps, duration, fs = 1_000_000_000, 3600, 64.0
    alpha, delta = 1.3, -0.5
    f0, f1, f2 = 12.3, -1.1e-9, 2.0e-18
    phi0, psi = 1.1, 0.37
    aplus, across = 1.0, 0.64
    dt = 1.0 / fs
    n_steps = math.ceil(fs * duration)

    # --- LAL detector states + antenna patterns on the sample timestamps ---
    ts = lalpulsar.CreateTimestampVector(n_steps)
    for i in range(n_steps):
        sec = start_gps + int((i * dt) // 1)
        ns = round(((i * dt) % 1) * 1e9)
        if ns == 1_000_000_000:
            sec, ns = sec + 1, 0
        ts.data[i] = lal.LIGOTimeGPS(sec, ns)
    ts.deltaT = dt
    det_states = lalpulsar.GetDetectorStates(ts, det, edat, 0.0)
    skypos = lal.SkyPosition()
    skypos.system = lal.COORDINATESYSTEM_EQUATORIAL
    skypos.longitude, skypos.latitude = alpha, delta
    amcoe = lalpulsar.ComputeAMCoeffs(det_states, skypos)
    a = np.asarray(amcoe.a.data, dtype=float)
    b = np.asarray(amcoe.b.data, dtype=float)
    r_det = np.array([list(det_states.data[i].rDetector) for i in range(n_steps)])
    t_gps = np.array(
        [
            det_states.data[i].tGPS.gpsSeconds
            + 1e-9 * det_states.data[i].tGPS.gpsNanoSeconds
            for i in range(n_steps)
        ]
    )
    n_hat = np.array(
        [
            math.cos(delta) * math.cos(alpha),
            math.cos(delta) * math.sin(alpha),
            math.sin(delta),
        ]
    )

    # --- reference: transcription of SimulatePulsarSignal.c body ---
    fr = det.frDetector
    sin_zeta = math.sin(abs(fr.xArmAzimuthRadians - fr.yArmAzimuthRadians))
    a_p, a_c, two_psi = sin_zeta * aplus, sin_zeta * across, 2 * psi
    cphi, sphi = math.cos(phi0), math.sin(phi0)
    c2, s2 = math.cos(two_psi), math.sin(two_psi)
    big_a1 = a_p * cphi * c2 - a_c * sphi * s2
    big_a2 = a_p * cphi * s2 + a_c * sphi * c2
    big_a3 = -a_p * sphi * c2 - a_c * cphi * s2
    big_a4 = -a_p * sphi * s2 + a_c * cphi * c2
    ref_time = t_gps[0] + n_hat @ r_det[0]
    d_t = r_det @ n_hat
    tau = (t_gps - ref_time) + d_t
    phase = 2 * math.pi * (f0 * tau + 0.5 * f1 * tau**2 + (1 / 6) * f2 * tau**3)
    h_lal = (
        big_a1 * a * np.cos(phase)
        + big_a2 * b * np.cos(phase)
        + big_a3 * a * np.sin(phase)
        + big_a4 * b * np.sin(phase)
    )

    # --- ripple polarizations + LAL antenna patterns ---
    t_rel = jnp.arange(n_steps, dtype=jnp.float64) * dt
    hp, hc = exact_pulsar_polarizations(
        t_rel,
        start_gps,
        alpha,
        delta,
        f0,
        phi0,
        aplus,
        across,
        loc,
        eph.gps0,
        eph.dt,
        jnp.asarray(eph.pos),
        jnp.asarray(eph.vel),
        jnp.asarray(eph.acc),
        fkdot=(f1, f2),
    )
    f_plus = sin_zeta * (a * c2 + b * s2)
    f_cross = sin_zeta * (b * c2 - a * s2)
    h_ripple = f_plus * np.asarray(hp) + f_cross * np.asarray(hc)

    loss = overlap_loss(h_ripple, h_lal)
    print(f"\nexact pulsar: overlap loss = {loss:.2e} (log10 = {log10_str(loss)})")
    # ~log10 -12.5 in practice; the floor is LAL's float64 GPS-time arithmetic.
    assert loss < 1e-10, f"overlap loss {loss:.2e} (log10={log10_str(loss)})"


def test_barycenter_matches_lal_to_microsecond():
    """The geometric delay n.rDetector matches XLALBarycenter to << 1 us."""
    sun_file = SUN_FILE if SUN_FILE is not None else EARTH_FILE.replace("earth", "sun")
    eph = read_ephemeris_file(EARTH_FILE)
    edat = lalpulsar.InitBarycenter(EARTH_FILE, sun_file)
    det = lal.CachedDetectors[lal.LALDetectorIndexLHODIFF]
    alpha, delta = 1.3, -0.5
    n_hat = np.array(
        [
            math.cos(delta) * math.cos(alpha),
            math.cos(delta) * math.sin(alpha),
            math.sin(delta),
        ]
    )

    gps_ints = np.array([1_000_000_000, 1_000_043_200, 1_126_259_462], dtype=float)
    gps_fracs = np.array([0.0, 0.5, 0.25], dtype=float)

    lal_dt = []
    for gi, gf in zip(gps_ints, gps_fracs):
        t = lal.LIGOTimeGPS(int(gi), round(gf * 1e9))
        bi = lalpulsar.BarycenterInput()
        bi.site = det
        for i in range(3):
            bi.site.location[i] = det.location[i] / lal.C_SI
        bi.alpha, bi.delta, bi.dInv, bi.tgps = alpha, delta, 0.0, t
        earth = lalpulsar.EarthState()
        lalpulsar.BarycenterEarth(earth, t, edat)
        emit = lalpulsar.EmissionTime()
        lalpulsar.Barycenter(emit, bi, earth)
        lal_dt.append(float(n_hat @ np.array(list(emit.rDetector))))

    my_dt = np.asarray(
        roemer_delay(
            jnp.asarray(gps_ints),
            jnp.asarray(gps_fracs),
            alpha,
            delta,
            tuple(det.location),
            eph.gps0,
            eph.dt,
            jnp.asarray(eph.pos),
            jnp.asarray(eph.vel),
            jnp.asarray(eph.acc),
        )
    )
    assert np.max(np.abs(my_dt - np.array(lal_dt))) < 1e-9
