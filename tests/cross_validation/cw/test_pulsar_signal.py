"""Cross-validation of ``PulsarSignal`` against LALPulsar.

The full (isolated-pulsar) generator uses the complete barycentering delay (Roemer +
Earth-rotation with precession/nutation + Einstein - Shapiro), unlike
``ExactPulsarSignal``'s geometric-only delay (see ``test_exact_pulsar_signal.py``). The
reference strain is built sample-by-sample from ``XLALBarycenter`` (the same routine LAL
uses internally) and the standard antenna response; LAL's high-level ``CWSimulator`` only
reaches ~1e-3 here due to its internal interpolation, so this compares against
``XLALBarycenter`` directly instead.

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

from ripplegw.waveforms.cw.ephemeris import read_ephemeris_file
from ripplegw.waveforms.cw.pulsar_signal import generate_pulsar_polarizations
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


def test_full_pulsar_matches_lal_barycenter_reference():
    """Full (isolated) generator matches a per-sample XLALBarycenter reference."""
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
        tg = lal.LIGOTimeGPS(int(g // 1), round((g % 1) * 1e9))
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
        t_rel,
        start_gps,
        alpha,
        delta,
        f0,
        phi0,
        aplus,
        across,
        tuple(det.location),
        eph.gps0,
        eph.dt,
        jnp.asarray(eph.pos),
        jnp.asarray(eph.vel),
        jnp.asarray(eph.acc),
        seph.gps0,
        seph.dt,
        jnp.asarray(seph.pos),
        jnp.asarray(seph.vel),
        jnp.asarray(seph.acc),
        fkdot=(f1,),
        ref_time_ssb=tref,
    )
    h_mine = f_plus * np.asarray(hp) + f_cross * np.asarray(hc)
    loss = overlap_loss(h_mine, h_ref)
    print(f"\nfull pulsar: overlap loss = {loss:.2e} (log10 = {log10_str(loss)})")
    assert loss < 1e-10, f"overlap loss {loss:.2e} (log10={log10_str(loss)})"
