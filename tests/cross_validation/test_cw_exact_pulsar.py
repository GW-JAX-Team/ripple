"""Cross-validation of the exact pulsar signal against LALPulsar.

LAL does not SWIG-wrap ``PulsarSignalParams``, so ``XLALSimulateExactPulsarSignal``
cannot be called directly from Python. Instead we reproduce the body of
``SimulatePulsarSignal.c`` (lines ~246-284) in Python using LAL's *own*
building blocks — the detector states (``XLALGetDetectorStates`` → ``rDetector``)
and antenna-pattern coefficients (``XLALComputeAMCoeffs`` → ``a, b``) — to form
the reference detector strain. We then reconstruct that strain from ripple's
plus/cross polarizations combined with LAL's antenna patterns, and require
agreement.

This simultaneously checks: the barycentered geometric delay (vs
``XLALBarycenter``), the spindown phase polynomial, the reference-time
handling, the amplitudes, and the ``{p, c}`` ↔ (A₁…A₄) decomposition.

The test is skipped unless both ``lalpulsar`` and an Earth ephemeris file are
available. Point ``RIPPLE_EARTH_EPHEMERIS`` (and optionally
``RIPPLE_SUN_EPHEMERIS``) at a LALPulsar ``earth*``/``sun*`` file to run it.
"""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

lal = pytest.importorskip("lal")
lalpulsar = pytest.importorskip("lalpulsar")

from ripplegw.cw.barycenter import roemer_delay  # noqa: E402
from ripplegw.cw.ephemeris import read_ephemeris_file  # noqa: E402
from ripplegw.cw.pulsar_signal import (  # noqa: E402
    _binary_source_phase,
    exact_pulsar_polarizations,
    generate_pulsar_polarizations,
)


def _overlap_loss(h1, h2) -> float:
    """1 − normalized overlap between two real time series (white inner product).

    Uses the same numerically stable identity as
    ``tests.utils.compute_overlap_loss`` —
    ``(AB − C²) / (√(AB)·(√(AB) + C))`` with ``A=⟨h1|h1⟩`` etc. — but with a flat
    (white) time-domain inner product, the conventional figure of merit for
    these quasi-monochromatic continuous-wave strain series. Returns the
    mismatch; report ``log10`` of it.
    """
    h1 = np.asarray(h1, dtype=float)
    h2 = np.asarray(h2, dtype=float)
    a = float(h1 @ h1)
    b = float(h2 @ h2)
    c = float(h1 @ h2)
    denom = math.sqrt(a * b)
    return max((a * b - c * c) / (denom * (denom + c)), 0.0)


def _find_ephemeris():
    """Locate an Earth (and optional Sun) ephemeris file, or return (None, None)."""
    earth = os.environ.get("RIPPLE_EARTH_EPHEMERIS")
    sun = os.environ.get("RIPPLE_SUN_EPHEMERIS")
    if earth and os.path.exists(earth):
        return earth, sun
    candidates = [
        "earth00-40-DE405.dat.gz",
        "/usr/share/lalpulsar/earth00-40-DE405.dat.gz",
    ]
    for c in candidates:
        if os.path.exists(c):
            s = c.replace("earth", "sun")
            return c, (s if os.path.exists(s) else None)
    return None, None


EARTH_FILE, SUN_FILE = _find_ephemeris()
pytestmark = pytest.mark.skipif(
    EARTH_FILE is None, reason="no LALPulsar Earth ephemeris file found"
)


def test_exact_pulsar_matches_lal_reference():
    """ripple {p,c} + LAL antenna patterns reproduces the LAL exact strain."""
    assert EARTH_FILE is not None  # guaranteed by pytestmark skip
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
    n_steps = int(math.ceil(fs * duration))

    # --- LAL detector states + antenna patterns on the sample timestamps ---
    ts = lalpulsar.CreateTimestampVector(n_steps)
    for i in range(n_steps):
        sec = start_gps + int((i * dt) // 1)
        ns = int(round(((i * dt) % 1) * 1e9))
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
        t_rel, start_gps, alpha, delta, f0, phi0, aplus, across, loc,
        eph.gps0, eph.dt, jnp.asarray(eph.pos), jnp.asarray(eph.vel),
        jnp.asarray(eph.acc), fkdot=(f1, f2),
    )
    f_plus = sin_zeta * (a * c2 + b * s2)
    f_cross = sin_zeta * (b * c2 - a * s2)
    h_ripple = f_plus * np.asarray(hp) + f_cross * np.asarray(hc)

    loss = _overlap_loss(h_ripple, h_lal)
    print(f"\nexact pulsar: overlap loss = {loss:.2e} (log10 = {math.log10(loss):.2f})")
    # ~log10 -12.5 in practice; the floor is LAL's float64 GPS-time arithmetic.
    assert loss < 1e-10, f"overlap loss {loss:.2e} (log10={math.log10(loss):.2f})"


def test_barycenter_matches_lal_to_microsecond():
    """The geometric delay n·rDetector matches XLALBarycenter to << 1 µs."""
    assert EARTH_FILE is not None  # guaranteed by pytestmark skip
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
        t = lal.LIGOTimeGPS(int(gi), int(round(gf * 1e9)))
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
            jnp.asarray(gps_ints), jnp.asarray(gps_fracs), alpha, delta,
            tuple(det.location), eph.gps0, eph.dt, jnp.asarray(eph.pos),
            jnp.asarray(eph.vel), jnp.asarray(eph.acc),
        )
    )
    assert np.max(np.abs(my_dt - np.array(lal_dt))) < 1e-9


def test_full_pulsar_matches_lal_barycenter_reference():
    """Full (isolated) generator matches a per-sample XLALBarycenter reference.

    The full model uses the complete ``emit.deltaT`` (Roemer + Earth-rotation +
    Einstein - Shapiro). We build the reference strain sample-by-sample from
    ``XLALBarycenter`` (the same routine LAL uses internally) and the standard
    antenna response, and require agreement to ~1e-9 (LAL's high-level
    ``CWSimulator`` only reaches ~1e-3 here due to its internal interpolation).
    """
    assert EARTH_FILE is not None
    sun_file = SUN_FILE if SUN_FILE is not None else EARTH_FILE.replace("earth", "sun")
    eph = read_ephemeris_file(EARTH_FILE)
    seph = read_ephemeris_file(sun_file)
    edat = lalpulsar.InitBarycenter(EARTH_FILE, sun_file)
    det = lal.CachedDetectors[lal.LALDetectorIndexLHODIFF]

    start_gps, fs, n_steps = 1_000_000_000, 16.0, 8192
    alpha, delta, f0, f1, phi0, psi = 1.3, -0.5, 12.3, -1.1e-9, 1.1, 0.37
    aplus, across = 1.0, 0.64
    tref = float(start_gps)
    dt = 1.0 / fs

    bi = lalpulsar.BarycenterInput()
    bi.site = det
    for i in range(3):
        bi.site.location[i] = det.location[i] / lal.C_SI
    bi.alpha, bi.delta, bi.dInv = alpha, delta, 0.0
    f_plus = np.empty(n_steps)
    f_cross = np.empty(n_steps)
    phi_ref = np.empty(n_steps)
    for i in range(n_steps):
        g = start_gps + i * dt
        tg = lal.LIGOTimeGPS(int(g // 1), int(round((g % 1) * 1e9)))
        bi.tgps = tg
        es = lalpulsar.EarthState()
        lalpulsar.BarycenterEarth(es, tg, edat)
        em = lalpulsar.EmissionTime()
        lalpulsar.Barycenter(em, bi, es)
        tau = (tg.gpsSeconds - tref) + tg.gpsNanoSeconds * 1e-9 + em.deltaT
        phi_ref[i] = phi0 + 2 * math.pi * (f0 * tau + 0.5 * f1 * tau * tau)
        gmst = lal.GreenwichMeanSiderealTime(tg)
        fp, fc = lal.ComputeDetAMResponse(det.response, alpha, delta, psi, gmst)
        f_plus[i], f_cross[i] = fp, fc
    h_ref = f_plus * aplus * np.cos(phi_ref) + f_cross * across * np.sin(phi_ref)

    t_rel = jnp.arange(n_steps, dtype=jnp.float64) * dt
    hp, hc = generate_pulsar_polarizations(
        t_rel, start_gps, alpha, delta, f0, phi0, aplus, across,
        tuple(det.location),
        eph.gps0, eph.dt, jnp.asarray(eph.pos), jnp.asarray(eph.vel),
        jnp.asarray(eph.acc),
        seph.gps0, seph.dt, jnp.asarray(seph.pos), jnp.asarray(seph.vel),
        jnp.asarray(seph.acc),
        fkdot=(f1,), ref_time_ssb=tref,
    )
    h_mine = f_plus * np.asarray(hp) + f_cross * np.asarray(hc)
    loss = _overlap_loss(h_mine, h_ref)
    print(f"\nfull pulsar: overlap loss = {loss:.2e} (log10 = {math.log10(loss):.2f})")
    assert loss < 1e-10, f"overlap loss {loss:.2e} (log10={math.log10(loss):.2f})"


def test_binary_source_phase_matches_lal_spinorbit():
    """Binary source phase matches XLALGenerateSpinOrbitCW (tight-Kepler regime).

    LAL solves Kepler only to a phase tolerance ``dxMax = 0.01/(f0·P)``; at high
    f0 (tight tolerance) our machine-precision solve agrees to ~float level.
    """
    f0, phi0, f1 = 1000.0, 0.7, -1.0e-10
    ecc, asini, period, argp = 0.18, 1.44, 6.3 * 3600, 1.05
    epoch_gps, orbit_epoch, spin_epoch = 900_000_000, 900_050_000, 900_040_000
    deltaT, length = 1.0, 4000

    sp = lalpulsar.SpinOrbitCWParamStruc()
    sp.position.system = lal.COORDINATESYSTEM_EQUATORIAL
    sp.position.longitude, sp.position.latitude = 1.0, 0.4
    sp.psi, sp.aPlus, sp.aCross, sp.phi0, sp.f0, sp.omega = 0.0, 1.0, 0.5, phi0, f0, argp
    sp.rPeriNorm = asini * (1.0 - ecc)
    sp.oneMinusEcc = 1.0 - ecc
    sp.angularSpeed = (lal.TWOPI / period) * math.sqrt((1.0 + ecc) / (1.0 - ecc) ** 3)
    sp.orbitEpoch = lal.LIGOTimeGPS(orbit_epoch, 0)
    sp.spinEpoch = lal.LIGOTimeGPS(spin_epoch, 0)
    sp.epoch = lal.LIGOTimeGPS(epoch_gps, 0)
    sp.deltaT, sp.length = deltaT, length
    fvec = lal.CreateREAL8Vector(1)
    fvec.data[0] = f1 / (1.0 * f0)
    sp.f = fvec
    cw = lalpulsar.PulsarCoherentGW()
    lalpulsar.GenerateSpinOrbitCW(cw, sp)
    phi_lal = np.array(cw.phi.data.data, dtype=float)

    i = np.arange(length, dtype=np.float64)
    tau = epoch_gps + i * deltaT
    phi_mine = np.asarray(
        _binary_source_phase(
            jnp.asarray(tau - orbit_epoch), float(orbit_epoch - spin_epoch),
            f0, phi0, (f1,), asini, ecc, period, argp,
        )
    )
    # Compare as a strain shape cos(φ): overlap loss is the conventional metric.
    # At f0=1000 LAL solves Kepler tightly, so this reaches the float floor
    # (~log10 -15); at lower f0 LAL's dxMax = 0.01/(f0·P) tolerance dominates.
    loss = _overlap_loss(np.cos(phi_mine), np.cos(phi_lal))
    print(f"\nbinary phase: overlap loss = {loss:.2e} (log10 = {math.log10(loss):.2f})")
    assert loss < 1e-12, f"overlap loss {loss:.2e} (log10={math.log10(loss):.2f})"
