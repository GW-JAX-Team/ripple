"""Validate ``BinaryPulsarSignal``'s orbital source phase against LALPulsar.

This test isolates the orbital phase through ``XLALGenerateSpinOrbitCW``. The
end-to-end binary strain is checked separately through ``CWMakeFakeData``. No
ephemerides are needed because this comparison does not barycenter the signal.
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
    """Compare binary source phase with ``XLALGenerateSpinOrbitCW``."""
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
    # Compare phase-induced strain shapes with normalized time-domain mismatch.
    loss = overlap_loss(np.cos(phi_mine), np.cos(phi_lal))
    print(f"\nbinary phase: overlap loss = {loss:.2e} (log10 = {log10_str(loss)})")
    assert loss < 1e-12, f"overlap loss {loss:.2e} (log10={log10_str(loss)})"
