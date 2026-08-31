"""End-to-end CW cross-validation through LALPulsar's ``CWMakeFakeData``.

The tests generate noise-free detector strain with LAL and compare it with
ripple polarizations projected through LAL's antenna response on identical
timestamps. They cover ``PulsarSignal`` and ``BinaryPulsarSignal``; the latter
also exercises orbital modulation. This is a separate LAL API path from the
direct-building-block checks in this directory.

``ExactPulsarSignal`` is excluded because it deliberately omits timing terms
that ``CWMakeFakeData`` always includes. The tests require ``lalpulsar`` and
Earth/Sun ephemerides.
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
from tests.cross_validation.cw.runner import MAX_RELATIVE_NORM_ERROR
from tests.cross_validation.cw.runner import mismatch_threshold as _threshold
from tests.helpers.metrics import relative_norm_error

EARTH_FILE, SUN_FILE = find_ephemeris()
pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.skipif(
        EARTH_FILE is None or SUN_FILE is None,
        reason="LALPulsar Earth and Sun ephemeris files required",
    ),
]

# Limits shared with the large-scale sweep.
_MAX_RELATIVE_NORM_ERROR = MAX_RELATIVE_NORM_ERROR


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
    """``BinaryPulsarSignal`` matches ``CWMakeFakeData`` end to end."""
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
