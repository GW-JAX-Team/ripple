"""Compare ripple's CW polarizations against the *compiled* LALPulsar functions.

``XLALSimulateExactPulsarSignal`` and ``XLALGeneratePulsarSignal`` take a
``PulsarSignalParams`` argument that swiglal does not wrap (it contains
anonymous nested structs), so they cannot be driven from Python. The companion
``harness.c`` calls them from C and dumps the resulting ``REAL4TimeSeries`` to
binary files; this script reconstructs the detector strain from ripple's
``{p, c}`` polarizations (using LAL's own antenna response) and compares.

See ``README.md`` for the build/run recipe. Run with the same lalsuite-enabled
interpreter used to build the ephemerides, e.g.::

    JAX_ENABLE_X64=1 PYTHONPATH=<ripple>/src python compare.py \
        <earth> <sun> out_exact.bin out_gen0.bin out_genhet.bin
"""

import math
import struct
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import lal
import lalpulsar
from ripplegw.cw.detectors import get_detector
from ripplegw.cw.ephemeris import read_ephemeris_file
from ripplegw.cw.pulsar_signal import (
    exact_pulsar_polarizations,
    generate_pulsar_polarizations,
)

# Parameters must match the constants in harness.c.
START_GPS = 1_000_000_000
ALPHA, DELTA = 1.3, -0.5
F0, F1, F2 = 12.3, -1.1e-9, 2.0e-18
PHI0, PSI = 1.1, 0.37
APLUS, ACROSS = 1.0, 0.64


def _overlap_loss(h1, h2) -> float:
    """1 − normalized overlap (white time-domain inner product), stable form."""
    h1 = np.asarray(h1, dtype=float)
    h2 = np.asarray(h2, dtype=float)
    a, b, c = float(h1 @ h1), float(h2 @ h2), float(h1 @ h2)
    denom = math.sqrt(a * b)
    return max((a * b - c * c) / (denom * (denom + c)), 0.0)


def _log10_str(loss: float) -> str:
    """log10(loss) formatted, or 'N/A' for an exactly-zero (clamped) loss."""
    return f"{math.log10(loss):.2f}" if loss > 0.0 else "N/A (0)"


def _load(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("I", f.read(4))
        es, en = struct.unpack("ii", f.read(8))
        dt, f0 = struct.unpack("dd", f.read(16))
        h = np.frombuffer(f.read(4 * n), dtype=np.float32).astype(np.float64)
    return n, es, en, dt, f0, h


def main(earth, sun, out_exact, out_gen0, out_genhet):
    eph = read_ephemeris_file(earth)
    seph = read_ephemeris_file(sun)
    loc = get_detector("H1").location
    edat = lalpulsar.InitBarycenter(earth, sun)
    det = lal.CachedDetectors[lal.LALDetectorIndexLHODIFF]
    eph_e = (
        eph.gps0,
        eph.dt,
        jnp.asarray(eph.pos),
        jnp.asarray(eph.vel),
        jnp.asarray(eph.acc),
    )
    eph_s = (
        seph.gps0,
        seph.dt,
        jnp.asarray(seph.pos),
        jnp.asarray(seph.vel),
        jnp.asarray(seph.acc),
    )

    # ---- EXACT: antenna = sinZeta*(a cos2psi + b sin2psi), etc. ----
    n, _, _, dt, _, h_c = _load(out_exact)
    ts = lalpulsar.CreateTimestampVector(n)
    for i in range(n):
        g = START_GPS + i * dt
        ts.data[i] = lal.LIGOTimeGPS(int(g // 1), int(round((g % 1) * 1e9)))
    ts.deltaT = dt
    ds = lalpulsar.GetDetectorStates(ts, det, edat, 0.0)
    sk = lal.SkyPosition()
    sk.system = lal.COORDINATESYSTEM_EQUATORIAL
    sk.longitude, sk.latitude = ALPHA, DELTA
    am = lalpulsar.ComputeAMCoeffs(ds, sk)
    a, b = np.array(am.a.data, float), np.array(am.b.data, float)
    fr = det.frDetector
    sinzeta = math.sin(abs(fr.xArmAzimuthRadians - fr.yArmAzimuthRadians))
    c2, s2 = math.cos(2 * PSI), math.sin(2 * PSI)
    fp, fc = sinzeta * (a * c2 + b * s2), sinzeta * (b * c2 - a * s2)
    t = jnp.arange(n, dtype=jnp.float64) * dt
    hp, hc = exact_pulsar_polarizations(
        t, START_GPS, ALPHA, DELTA, F0, PHI0, APLUS, ACROSS, loc, *eph_e, fkdot=(F1, F2)
    )
    h_me = fp * np.array(hp) + fc * np.array(hc)
    loss = _overlap_loss(h_me, h_c)
    print(
        f"EXACT       vs compiled XLALSimulateExactPulsarSignal: "
        f"overlap loss = {loss:.2e}  log10 = {_log10_str(loss)}  (n={n})"
    )

    # ---- GENERATE: standard ComputeDetAMResponse antenna ----
    for fh, fname in [(0.0, out_gen0), (12.0, out_genhet)]:
        n, _, _, dt, _, h_c = _load(fname)
        fp = np.empty(n)
        fc = np.empty(n)
        for i in range(n):
            g = START_GPS + i * dt
            gmst = lal.GreenwichMeanSiderealTime(
                lal.LIGOTimeGPS(int(g // 1), int(round((g % 1) * 1e9)))
            )
            fp[i], fc[i] = lal.ComputeDetAMResponse(
                det.response, ALPHA, DELTA, PSI, gmst
            )
        t = jnp.arange(n, dtype=jnp.float64) * dt
        hp, hc = generate_pulsar_polarizations(
            t,
            START_GPS,
            ALPHA,
            DELTA,
            F0,
            PHI0,
            APLUS,
            ACROSS,
            loc,
            *eph_e,
            *eph_s,
            fkdot=(F1, F2),
            f_heterodyne=fh,
        )
        h_me = fp * np.array(hp) + fc * np.array(hc)
        loss = _overlap_loss(h_me, h_c)
        print(
            f"GENERATE fHet={fh:4.1f} vs compiled XLALGeneratePulsarSignal:  "
            f"overlap loss = {loss:.2e}  log10 = {_log10_str(loss)}  (n={n})"
        )


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(
            "usage: python compare.py <earth> <sun> <out_exact> <out_gen0> "
            "<out_genhet>",
            file=sys.stderr,
        )
        sys.exit(2)
    main(*sys.argv[1:6])
