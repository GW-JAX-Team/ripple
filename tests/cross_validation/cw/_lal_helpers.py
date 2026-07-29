"""Helpers shared by CW/LALPulsar cross-validation tests.

Direct-LAL tests build reference strains from SWIG-exposed LALPulsar routines;
the MakeFakeData tests use ``XLALCWMakeFakeData``. This module provides
ephemeris lookup, same-grid time-domain mismatch, detector projection, and
MakeFakeData setup. Individual tests own their dependency skips.
"""

import math
import os

import numpy as np

from tests.helpers.metrics import time_domain_overlap_loss


def find_ephemeris():
    """Return existing Earth and Sun ephemeris paths, or ``(None, None)``."""
    earth = os.environ.get("RIPPLE_EARTH_EPHEMERIS")
    if earth and os.path.exists(earth):
        # Prefer the explicit Sun path; otherwise infer a sibling file.
        sun = os.environ.get("RIPPLE_SUN_EPHEMERIS") or earth.replace("earth", "sun")
        if os.path.exists(sun):
            return earth, sun
    candidates = [
        "earth00-40-DE405.dat.gz",
        "/usr/share/lalpulsar/earth00-40-DE405.dat.gz",
    ]
    for c in candidates:
        s = c.replace("earth", "sun")
        if os.path.exists(c) and os.path.exists(s):
            return c, s
    return None, None


def overlap_loss(h1, h2) -> float:
    """CW alias for the shared same-grid time-domain mismatch."""
    return time_domain_overlap_loss(h1, h2)


def log10_str(loss: float) -> str:
    """``log10(loss)`` formatted, or 'N/A' for an exactly-zero (clamped) loss."""
    return f"{math.log10(loss):.2f}" if loss > 0.0 else "N/A (0)"


def make_fake_data_v5(
    *,
    lal,
    lalpulsar,
    edat,
    det_prefix: str,
    alpha: float,
    delta: float,
    psi: float,
    phi0: float,
    aplus: float,
    across: float,
    f0: float,
    fkdot=(),
    asini: float = 0.0,
    ecc: float = 0.0,
    period: float = 0.0,
    argp: float = 0.0,
    tp_ssb: float = 0.0,
    start_gps: int,
    duration: float,
    fmin: float,
    band: float,
    source_delta_t: float = 0.0,
):
    """Generate noise-free detector strain through LALPulsar's ``CWMakeFakeData``.

    ``asini=0`` selects an isolated source. ``fmin`` heterodynes the returned
    series, so its ``.f0`` must be passed to ripple as ``f_heterodyne``.
    """
    data_params = lalpulsar.CWMFDataParams()
    data_params.fMin = fmin
    data_params.Band = band
    lalpulsar.ParseMultiLALDetector(data_params.multiIFO, [det_prefix])
    lalpulsar.ParseMultiNoiseFloor(data_params.multiNoiseFloor, ["0"], 1)
    data_params.multiTimestamps = lalpulsar.MakeMultiTimestamps(
        start_gps, duration, duration, 0, 1
    )
    data_params.randSeed = 1
    data_params.inputMultiTS = None
    data_params.sourceDeltaT = source_delta_t

    sources = lalpulsar.CreatePulsarParamsVector(1)
    p = sources.data[0]
    p.Amp.psi, p.Amp.phi0, p.Amp.aPlus, p.Amp.aCross = psi, phi0, aplus, across
    p.Doppler.refTime = lal.LIGOTimeGPS(int(start_gps))
    p.Doppler.Alpha, p.Doppler.Delta = alpha, delta
    for i in range(7):
        p.Doppler.fkdot[i] = 0.0
    p.Doppler.fkdot[0] = f0
    for i, fk in enumerate(fkdot, start=1):
        p.Doppler.fkdot[i] = fk
    p.Doppler.asini, p.Doppler.ecc = asini, ecc
    p.Doppler.period, p.Doppler.argp = period, argp
    p.Doppler.tp = lal.LIGOTimeGPS(tp_ssb)
    p.Transient.type = lalpulsar.TRANSIENT_NONE

    _, tseries = lalpulsar.CWMakeFakeData(None, 0, sources, data_params, 0, edat)
    return tseries


def detector_strain_from_am_response(lal, det, alpha, delta, psi, tseries, hp, hc):
    """Project ripple polarizations through LAL's antenna response on ``tseries``'s grid."""
    n = tseries.data.length
    f_plus = np.empty(n)
    f_cross = np.empty(n)
    for i in range(n):
        gmst = lal.GreenwichMeanSiderealTime(
            lal.LIGOTimeGPS(float(tseries.epoch) + i * tseries.deltaT)
        )
        f_plus[i], f_cross[i] = lal.ComputeDetAMResponse(
            det.response, alpha, delta, psi, gmst
        )
    return f_plus * np.asarray(hp) + f_cross * np.asarray(hc)
