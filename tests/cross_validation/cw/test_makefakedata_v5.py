"""Cross-validation of ``PulsarSignal``/``BinaryPulsarSignal`` against ``lalpulsar_Makefakedata_v5``.

This is deliberately **not** a replacement for ``test_full_pulsar_signal.py`` /
``test_binary_pulsar_signal.py`` -- it checks something those tests structurally
cannot. Those tests reconstruct LAL's reference computation in Python from the same
low-level building block (``XLALBarycenter``) that ripple's own ``barycenter.py``/
``earth.py`` were ported from, reaching ~1e-9 to ~1e-13: a translation-fidelity check.

This file instead drives ``XLALCWMakeFakeData`` -- the literal engine behind the
``lalpulsar_Makefakedata_v5`` CLI real CW searches use to generate injections/MDCs --
via its SWIG-wrapped "modern" ``PulsarParams``/``CWMFDataParams`` structs (unlike the
anonymous-nested-struct ``PulsarSignalParams`` that ``XLALGeneratePulsarSignal`` takes
directly, which is why that function "cannot be called directly from Python", per
``docs/dev/reference_implementations.md``). ``XLALCWMakeFakeData`` is a thin wrapper
around exactly that function (see ``CWMakeFakeData.c``), so this is a genuinely
independent code path from what the other two files check, at that pipeline's own,
looser precision. (LALPulsar's other Python-native option,
``lalpulsar.simulateCW.CWSimulator``, was already tried for this purpose -- see
``test_full_pulsar_signal.py``'s docstring -- and reaches only ~1e-3 due to its own
internal interpolation; ``CWMakeFakeData`` is a meaningfully tighter independent check.)

**The normalized time-domain mismatch against this pipeline is not a flat floor -- it
scales as roughly f0**2** (fitted exponent ~1.5-1.9 depending on population). The
dominant known LAL approximation is its 400-second barycentric-delay-table half
interval (800-second node spacing), linearly interpolated by
``XLALPulsarSimulateCoherentGW``: a microsecond-scale delay residual becomes a phase
residual proportional to ``f0``. ``_threshold()`` is therefore
frequency-dependent, calibrated from a 500-trial HPC large-scale test
(``test_makefakedata_v5_large_scale.py``, f0 log-uniform 10-2000 Hz) rather than a single
point. The comparison is direct in time domain; it uses no FFT and no time/phase
maximization.

Covers ``PulsarSignal`` and ``BinaryPulsarSignal`` -- both use the full barycentering
delay that ``XLALGeneratePulsarSignal`` implements. **Not** ``ExactPulsarSignal``:
LAL has no toggle to disable the Einstein/Shapiro terms in this pipeline, so a
comparison would just show the expected, deliberate omission, not a meaningful check.

For ``BinaryPulsarSignal`` this is the first automated, in-repo, end-to-end check of
the full waveform (barycentering + orbital modulation + antenna response combined) --
previously this was only checked "manually once, off-repo" against the compiled
``XLALGeneratePulsarSignal`` entry point directly (see
``docs/dev/reference_implementations.md``); ``CWMakeFakeData``'s SWIG-friendly wrapper
structs make that check reproducible from Python alone, in-tree, without a compiled
helper program.

Skipped unless both ``lalpulsar`` and an Earth/Sun ephemeris file are available -- point
``RIPPLE_EARTH_EPHEMERIS`` (and optionally ``RIPPLE_SUN_EPHEMERIS``) at a LALPulsar
``earth*``/``sun*`` file to run it.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

lal = pytest.importorskip("lal")
lalpulsar = pytest.importorskip("lalpulsar")

from ripplegw.waveforms.cw.ephemeris import read_ephemeris_file
from ripplegw.waveforms.cw.pulsar_signal import (
    generate_binary_pulsar_polarizations,
    generate_pulsar_polarizations,
)
from tests.cross_validation.cw._lal_helpers import (
    detector_strain_from_am_response,
    find_ephemeris,
    log10_str,
    make_fake_data_v5,
    overlap_loss,
)
from tests.helpers.metrics import relative_norm_error

EARTH_FILE, SUN_FILE = find_ephemeris()
pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.skipif(
        EARTH_FILE is None or SUN_FILE is None,
        reason="LALPulsar Earth and Sun ephemeris files required",
    ),
]


# Overlap-loss threshold vs. f0 (Hz), calibrated from a 500-trial HPC large-scale test
# (test_makefakedata_v5_large_scale.py, 2026-07-28, n=250 per population, f0 log-uniform
# 10-2000 Hz -- see docs/dev/reference_implementations.md). Not a flat floor: the
# loss scales as roughly f0**2 because LAL linearly interpolates a barycentric-delay
# table with a 400-second half interval (see module docstring), plus an additive floor
# dominating at low f0 (binary's floor is set by LAL's own Kepler-solver tolerance, looser than
# isolated's). Smallest observed margin over any sampled trial: isolated 25x, binary
# 13x -- a fresh large-scale test run may want to re-derive these constants rather than
# assume they still hold with the same margin at a much larger --n-samples.
def _threshold(f0: float, *, is_binary: bool) -> float:
    floor = 2e-3 if is_binary else 1e-4
    coeff = 4e-4 if is_binary else 3e-4
    return floor + coeff * (f0 / 100.0) ** 2


# ``XLALPulsarSimulateCoherentGW`` documents its interpolated antenna response as
# accurate to about 0.1%; this 1% ceiling leaves a factor-of-ten reference margin
# while catching a global amplitude-scale regression that normalized mismatch misses.
_MAX_RELATIVE_NORM_ERROR = 1e-2


_START_GPS = 1_000_000_000
_ALPHA, _DELTA, _PSI = 1.3, -0.5, 0.37
_F0, _F1, _PHI0 = 12.3, -1.1e-9, 1.1
_APLUS, _ACROSS = 1.0, 0.64
_FMIN, _BAND = 1.0, 31.0
_DURATION = 16.0


def test_pulsar_signal_matches_makefakedata_v5():
    """Isolated ``PulsarSignal`` matches ``CWMakeFakeData`` end to end."""
    edat = lalpulsar.InitBarycenter(EARTH_FILE, SUN_FILE)
    det = lal.CachedDetectors[lal.LALDetectorIndexLHODIFF]

    tseries = make_fake_data_v5(
        lal=lal,
        lalpulsar=lalpulsar,
        edat=edat,
        det_prefix=det.frDetector.prefix,
        alpha=_ALPHA,
        delta=_DELTA,
        psi=_PSI,
        phi0=_PHI0,
        aplus=_APLUS,
        across=_ACROSS,
        f0=_F0,
        fkdot=(_F1,),
        start_gps=_START_GPS,
        duration=_DURATION,
        fmin=_FMIN,
        band=_BAND,
    )
    n = tseries.data.length
    t_rel = (float(tseries.epoch) - _START_GPS) + jnp.arange(
        n, dtype=jnp.float64
    ) * tseries.deltaT

    eph = read_ephemeris_file(EARTH_FILE)
    seph = read_ephemeris_file(SUN_FILE)
    hp, hc = generate_pulsar_polarizations(
        t_rel,
        _START_GPS,
        _ALPHA,
        _DELTA,
        _F0,
        _PHI0,
        _APLUS,
        _ACROSS,
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
        fkdot=(_F1,),
        ref_time_ssb=float(_START_GPS),
        f_heterodyne=float(tseries.f0),
    )
    h_mine = detector_strain_from_am_response(
        lal, det, _ALPHA, _DELTA, _PSI, tseries, hp, hc
    )
    h_ref = tseries.data.data

    loss = overlap_loss(h_mine, h_ref)
    norm_error = relative_norm_error(h_mine, h_ref)
    threshold = _threshold(_F0, is_binary=False)
    print(
        f"\nPulsarSignal vs CWMakeFakeData: mismatch = {loss:.2e} "
        f"(log10 = {log10_str(loss)}), relative norm error = {norm_error:.2e}"
    )
    assert loss < threshold, (
        f"time-domain mismatch {loss:.2e} (log10={log10_str(loss)}) >= "
        f"threshold {threshold:.2e}"
    )
    assert norm_error < _MAX_RELATIVE_NORM_ERROR, (
        f"relative norm error {norm_error:.2e} >= {_MAX_RELATIVE_NORM_ERROR:.2e}"
    )


def test_binary_pulsar_signal_matches_makefakedata_v5():
    """``BinaryPulsarSignal`` (orbital modulation + full barycentering) matches
    ``CWMakeFakeData`` end to end -- the first automated, in-repo check of this."""
    edat = lalpulsar.InitBarycenter(EARTH_FILE, SUN_FILE)
    det = lal.CachedDetectors[lal.LALDetectorIndexLHODIFF]

    asini, ecc, period, argp = 1.44, 0.18, 6.3 * 3600.0, 1.05
    tp_ssb = float(_START_GPS) + 1234.0

    tseries = make_fake_data_v5(
        lal=lal,
        lalpulsar=lalpulsar,
        edat=edat,
        det_prefix=det.frDetector.prefix,
        alpha=_ALPHA,
        delta=_DELTA,
        psi=_PSI,
        phi0=_PHI0,
        aplus=_APLUS,
        across=_ACROSS,
        f0=_F0,
        asini=asini,
        ecc=ecc,
        period=period,
        argp=argp,
        tp_ssb=tp_ssb,
        start_gps=_START_GPS,
        duration=_DURATION,
        fmin=_FMIN,
        band=_BAND,
    )
    n = tseries.data.length
    t_rel = (float(tseries.epoch) - _START_GPS) + jnp.arange(
        n, dtype=jnp.float64
    ) * tseries.deltaT

    eph = read_ephemeris_file(EARTH_FILE)
    seph = read_ephemeris_file(SUN_FILE)
    hp, hc = generate_binary_pulsar_polarizations(
        t_rel,
        _START_GPS,
        _ALPHA,
        _DELTA,
        _F0,
        _PHI0,
        _APLUS,
        _ACROSS,
        asini,
        ecc,
        period,
        argp,
        tp_ssb,
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
        fkdot=(),
        ref_time_ssb=float(_START_GPS),
        f_heterodyne=float(tseries.f0),
    )
    h_mine = detector_strain_from_am_response(
        lal, det, _ALPHA, _DELTA, _PSI, tseries, hp, hc
    )
    h_ref = tseries.data.data

    loss = overlap_loss(h_mine, h_ref)
    norm_error = relative_norm_error(h_mine, h_ref)
    threshold = _threshold(_F0, is_binary=True)
    print(
        f"\nBinaryPulsarSignal vs CWMakeFakeData: mismatch = {loss:.2e} "
        f"(log10 = {log10_str(loss)}), relative norm error = {norm_error:.2e}"
    )
    assert loss < threshold, (
        f"time-domain mismatch {loss:.2e} (log10={log10_str(loss)}) >= "
        f"threshold {threshold:.2e}"
    )
    assert norm_error < _MAX_RELATIVE_NORM_ERROR, (
        f"relative norm error {norm_error:.2e} >= {_MAX_RELATIVE_NORM_ERROR:.2e}"
    )
