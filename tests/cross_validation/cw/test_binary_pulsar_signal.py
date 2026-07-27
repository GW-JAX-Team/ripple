"""Cross-validation of ``BinaryPulsarSignal`` against LALPulsar.

Only the orbital source-phase model (:func:`ripplegw.waveforms.cw.pulsar_signal._binary_source_phase`)
is checked here, against ``XLALGenerateSpinOrbitCW`` in the tight-Kepler regime
(``f0=1000 Hz``, where LAL's own Kepler-solver tolerance is tightest). The full binary
waveform end-to-end (combined with the barycentering delay ``PulsarSignal`` already
validates in ``test_pulsar_signal.py``) is not part of the automated suite -- see
``docs/dev/reference_implementations.md`` for the supplementary manual check that covers it.

Skipped unless ``lalpulsar`` is available. Unlike the other two files in this directory,
this test needs no ephemeris -- ``XLALGenerateSpinOrbitCW`` is a pure orbital-phase
computation with no barycentering involved -- so it only depends on the ``lal``/
``lalpulsar`` import, not on ``RIPPLE_EARTH_EPHEMERIS``/``RIPPLE_SUN_EPHEMERIS``.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

lal = pytest.importorskip("lal")
lalpulsar = pytest.importorskip("lalpulsar")

from ripplegw.waveforms.cw.pulsar_signal import _binary_source_phase
from tests.cross_validation.cw._lal_helpers import log10_str, overlap_loss

pytestmark = pytest.mark.accuracy


def test_binary_source_phase_matches_lal_spinorbit():
    """Binary source phase matches XLALGenerateSpinOrbitCW (tight-Kepler regime).

    LAL solves Kepler only to a phase tolerance ``dxMax = 0.01/(f0*P)``; at high
    f0 (tight tolerance) our machine-precision solve agrees to ~float level.
    """
    f0, phi0, f1 = 1000.0, 0.7, -1.0e-10
    ecc, asini, period, argp = 0.18, 1.44, 6.3 * 3600, 1.05
    epoch_gps, orbit_epoch, spin_epoch = 900_000_000, 900_050_000, 900_040_000
    deltaT, length = 1.0, 4000

    sp = lalpulsar.SpinOrbitCWParamStruc()
    sp.position.system = lal.COORDINATESYSTEM_EQUATORIAL
    sp.position.longitude, sp.position.latitude = 1.0, 0.4
    sp.psi, sp.aPlus, sp.aCross, sp.phi0, sp.f0, sp.omega = (
        0.0,
        1.0,
        0.5,
        phi0,
        f0,
        argp,
    )
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
            jnp.asarray(tau - orbit_epoch),
            float(orbit_epoch - spin_epoch),
            f0,
            phi0,
            (f1,),
            asini,
            ecc,
            period,
            argp,
        )
    )
    # Compare as a strain shape cos(phi): overlap loss is the conventional metric.
    # At f0=1000 LAL solves Kepler tightly, so this reaches the float floor
    # (~log10 -15); at lower f0 LAL's dxMax = 0.01/(f0*P) tolerance dominates.
    loss = overlap_loss(np.cos(phi_mine), np.cos(phi_lal))
    print(f"\nbinary phase: overlap loss = {loss:.2e} (log10 = {log10_str(loss)})")
    assert loss < 1e-12, f"overlap loss {loss:.2e} (log10={log10_str(loss)})"
