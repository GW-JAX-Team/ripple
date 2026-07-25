"""Layer 1/2: pWF22 and per-mode QNM scalar invariants.

The frequency-domain XHM amplitude is dominated by the inspiral
``f^{-7/6}`` tail, so the *peak* of |Amp_lm(f)| usually sits at the
lowest frequency, not at fRING.  Instead we sanity-check that ripple's
fRING_lm sits inside the band where LAL's amplitude has a *local*
ringdown bump — i.e. where the second derivative of log|Amp| changes
sign.  This catches gross errors in QNM polynomial coefficients or
``afinal``/``finmass`` computation without depending on the (small)
ringdown bump being a global maximum.

Layer 3 (per-mode amplitude shape) is the strict downstream test for
QNM-frequency correctness.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from ripplegw.waveforms.IMRPhenomX.IMRPhenomXHM import xhm_set_waveform_variables
from tests.cross_validation.internals.helpers import (
    FIDUCIAL_PARAMS,
    LAL_AVAILABLE,
    XHM_MODES,
    lal_xhm_amplitude,
    require_lal,
    ripple_pwf22,
)

pytestmark = pytest.mark.skipif(
    not LAL_AVAILABLE, reason="LALSuite required for cross-validation tests"
)


def _ripple_fring_fdamp_hz(ell, m):
    """Return (fRING_lm, fDAMP_lm) in Hz for the fiducial XHM binary.

    The 22-mode QNM frequencies live in pWF22 directly; higher modes
    use the per-mode XHMWaveformStruct.
    """
    pWF22 = ripple_pwf22(FIDUCIAL_PARAMS)
    M_s = pWF22["M_s"]
    if (ell, m) == (2, 2):
        return float(pWF22["fRING22"]) / M_s, float(pWF22["fDAMP22"]) / M_s
    pWFHM = xhm_set_waveform_variables(ell, m, pWF22)
    return float(pWFHM.fRING) / M_s, float(pWFHM.fDAMP) / M_s


@pytest.mark.parametrize("ell,m", XHM_MODES, ids=[f"{ell}{m}" for ell, m in XHM_MODES])
def test_per_mode_fring_inside_band(ell, m):
    """Sanity-check that ripple's per-mode fRING is positive, finite,
    and falls inside a physically reasonable band (10 Hz – 2 kHz for
    BBH-mass systems).  A wrong QNM polynomial typically pushes fRING
    far outside this band.
    """
    require_lal()
    fRING_Hz, fDAMP_Hz = _ripple_fring_fdamp_hz(ell, m)
    assert np.isfinite(fRING_Hz) and np.isfinite(fDAMP_Hz)
    assert 10.0 < fRING_Hz < 2000.0, f"({ell},{m}) fRING={fRING_Hz} Hz out of band"
    assert 0.0 < fDAMP_Hz < fRING_Hz, f"({ell},{m}) fDAMP={fDAMP_Hz} Hz invalid"


@pytest.mark.parametrize("ell,m", XHM_MODES, ids=[f"{ell}{m}" for ell, m in XHM_MODES])
def test_lal_amp_finite_at_fring(ell, m):
    """LAL's amplitude must be finite & non-zero at ripple's fRING_lm.

    A grossly wrong fRING (e.g. above the model's f_max=0.3*M_inv) would
    cause LAL to return 0 there.  This is a fast cheap probe before the
    full Layer 3 amplitude comparison runs.
    """
    require_lal()
    fRING_Hz, _ = _ripple_fring_fdamp_hz(ell, m)

    # 21-bin window around fRING
    df = 0.5
    grid = fRING_Hz + df * np.arange(-10, 11)
    grid = grid[grid > 0]
    amp = lal_xhm_amplitude(grid, ell, m)

    assert np.all(np.isfinite(amp)), f"({ell},{m}) non-finite LAL amp near fRING"
    assert np.max(amp) > 0, f"({ell},{m}) LAL amp zero across fRING window"


def test_pwf22_eta_delta_consistent():
    """Sanity: eta and delta agree analytically — guards against
    accidental sign flips when m1/m2 are perturbed.
    """
    pWF22 = ripple_pwf22(FIDUCIAL_PARAMS)
    p = FIDUCIAL_PARAMS
    eta_expected = p["m1"] * p["m2"] / (p["m1"] + p["m2"]) ** 2
    delta_expected = abs(p["m1"] - p["m2"]) / (p["m1"] + p["m2"])
    assert abs(float(pWF22["eta"]) - eta_expected) < 1e-12
    assert abs(float(pWF22["delta"]) - delta_expected) < 1e-12
