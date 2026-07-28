"""Shared helpers for CW-vs-LALPulsar cross-validation (tests/cross_validation/cw/).

LAL does not SWIG-wrap ``PulsarSignalParams``, so ``XLALSimulateExactPulsarSignal``/
``XLALGeneratePulsarSignal`` cannot be called directly from Python. Most tests in this
directory instead reproduce the relevant LAL reference computation in Python from its
SWIG-exposed building blocks (``XLALGetDetectorStates``, ``XLALComputeAMCoeffs``,
``XLALBarycenter``, ``XLALGenerateSpinOrbitCW``) -- this is why CW has its own directory
here instead of a ``ReferenceBackend`` in ``reference/`` like the campaign in
``cross_validation/fd/``. ``test_makefakedata_v5.py`` instead drives the actual
``lalpulsar_Makefakedata_v5`` engine (``XLALCWMakeFakeData``, SWIG-wrapped via the
"modern" ``PulsarParams``/``CWMFDataParams`` structs, unlike the anonymous-nested-struct
``PulsarSignalParams`` those functions wrap internally) -- a genuinely independent code
path, at that pipeline's own (~1e-7, ``REAL4``-limited) precision rather than the
``XLALBarycenter``-based tests' ~1e-9 to ~1e-13. See ``docs/dev/reference_implementations.md``
for the per-model methodology and accuracy numbers.

Not itself a test module (no ``test_`` prefix, not collected by pytest) -- just the
pieces every file in this directory would otherwise duplicate: locating the ephemeris
files, the white (unweighted) time-domain overlap-loss metric these tests use instead
of ``tests.helpers.metrics.overlap_loss`` (which requires a PSD and a frequency axis),
and (for ``test_makefakedata_v5.py``) driving ``CWMakeFakeData`` and combining ripple's
polarizations into detector strain via LAL's own antenna response.
Each test module still does its own ``pytest.importorskip("lal")`` / ``"lalpulsar"`` and
``pytest.mark.skipif`` on the ephemeris files -- a module-level skip in a shared
``conftest.py`` does not degrade gracefully when the import itself is what's missing (it
raises during pytest's initial conftest loading instead of being reported as a skip), so
each module keeps its own skip guard rather than centralizing it there.
"""

import math
import os

import numpy as np


def find_ephemeris():
    """Locate Earth and Sun ephemeris files (both must exist), or ``(None, None)``.

    ``lalpulsar.InitBarycenter`` requires both files, so this only returns a pair
    when both are present (otherwise the caller should skip rather than hard-fail).
    """
    earth = os.environ.get("RIPPLE_EARTH_EPHEMERIS")
    if earth and os.path.exists(earth):
        # Use RIPPLE_SUN_EPHEMERIS if given, else guess the Sun file next to it.
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
    """1 - normalized overlap between two real time series (white inner product).

    Uses the same numerically stable identity as
    ``tests.helpers.metrics.overlap_loss`` --
    ``(AB - C**2) / (sqrt(AB)*(sqrt(AB) + C))`` with ``A=<h1|h1>`` etc. -- but with a flat
    (white) time-domain inner product, the conventional figure of merit for these
    quasi-monochromatic continuous-wave strain series. Returns the mismatch; report
    ``log10`` of it via :func:`log10_str`.
    """
    h1 = np.asarray(h1, dtype=float)
    h2 = np.asarray(h2, dtype=float)
    a = float(h1 @ h1)
    b = float(h2 @ h2)
    c = float(h1 @ h2)
    denom = math.sqrt(a * b)
    return max((a * b - c * c) / (denom * (denom + c)), 0.0)


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
    """Noise-free ``REAL8TimeSeries`` for one pulsar at one site, via LALPulsar's
    ``CWMakeFakeData`` -- the same ``XLALCWMakeFakeData`` engine ``lalpulsar_Makefakedata_v5``
    calls, driven entirely from Python (no CLI, no C).

    ``asini=0`` (the default) gives an isolated pulsar, matching
    ``XLALGeneratePulsarSignal``'s own convention (see ``GeneratePulsarSignal.c``:
    ``if (params->orbit.asini > 0) {...} else {/* isolated pulsar */}``).

    Args:
        lal, lalpulsar: The ``lal``/``lalpulsar`` modules, already imported by the
            caller via ``pytest.importorskip`` (this helper takes no dependency on
            them existing beyond what's needed when it's actually called).
        edat: Output of ``lalpulsar.InitBarycenter``.
        det_prefix (str): Two-character detector prefix, e.g. ``"H1"``.
        alpha, delta (float): Sky position (radians).
        psi (float): Polarization angle (radians) -- needed here (unlike ripple's own
            polarizations-only output) because ``CWMakeFakeData`` returns combined
            detector strain, not ``{"p", "c"}``.
        phi0 (float): Initial phase (radians).
        aplus, across (float): Polarization amplitudes.
        f0 (float): Wave frequency at ``start_gps`` (Hz).
        fkdot (tuple): Spindown terms ``(f1, f2, ...)``.
        asini, ecc, period, argp, tp_ssb: Orbital elements (as in
            ``generate_binary_pulsar_polarizations``); ``asini=0`` means isolated.
        start_gps (int): GPS second of the reference time and of the first output
            sample.
        duration (float): Requested output duration (seconds).
        fmin, band (float): Frequency band passed to ``CWMFDataParams`` -- this also
            determines the output sampling rate, and (see the module docstring in
            ``test_makefakedata_v5.py``) ``CWMakeFakeData`` heterodynes its output by
            ``fmin``, so the returned series' ``.f0`` must be passed to ripple's own
            ``f_heterodyne`` for a fair comparison.
        source_delta_t (float): ``CWMFDataParams.sourceDeltaT`` (0 = LAL's internal
            default, 60s isolated / 5s binary). Empirically has no measurable effect
            on agreement with ripple -- the ~1e-7 floor is ``REAL4`` truncation inside
            ``XLALGeneratePulsarSignal``, not this tabulation interval.

    Returns:
        A LAL ``REAL8TimeSeries``: noise-free detector strain, heterodyned by ``fmin``.
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
    """Combine ripple polarizations into detector strain via LAL's antenna response.

    Uses the same per-sample ``GreenwichMeanSiderealTime`` + ``ComputeDetAMResponse``
    convention as ``test_full_pulsar_signal.py``, evaluated at ``tseries``'s own time
    grid so it lines up sample-for-sample with a :func:`make_fake_data_v5` reference.

    Args:
        lal: The ``lal`` module, already imported by the caller via
            ``pytest.importorskip``.
        det: A ``lal.CachedDetectors`` entry (for ``det.response``).
        alpha, delta, psi (float): Sky position and polarization angle (radians).
        tseries: A LAL ``REAL8TimeSeries`` (only ``.epoch``/``.deltaT``/``.data.length``
            are used -- e.g. the output of :func:`make_fake_data_v5`).
        hp, hc (array): Ripple's plus/cross polarizations, sampled on the same grid.

    Returns:
        numpy.ndarray: Detector strain ``F+ hp + F× hc``.
    """
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
