"""Layer 5: per-mode complex hlm vs SimIMRPhenomXHMFrequencySequenceOneMode.

Combines the amp + phase + t0 + phi0 + (-1)^l + amp0 prefactors that
Layers 3 and 4 deliberately split apart.  Tests both phi_ref=0 and
phi_ref=pi/3 to flush out the (mm/2)*phifRef + mm*phi0 reference-phase
accounting that initial_plan.md §3 flagged.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from tests.cross_validation.xhm_helpers import (
    LAL_AVAILABLE,
    FIDUCIAL_PARAMS,
    XHM_MODES,
    lal_xhm_one_mode_freqseq,
    make_freq_grid,
    relative_amp_error,
    ripple_xhm_one_mode_complex,
)

pytestmark = pytest.mark.skipif(
    not LAL_AVAILABLE, reason="LALSuite required for cross-validation tests"
)


# Combined per-mode tolerances (relative on amplitude, absolute on phase
# in radians).  Tighter than per-region because we are at this layer
# expecting all polynomial / fit errors already pinned at Layers 3-4.
_TOL = {
    (2, 2): dict(amp=1e-12, phase=1e-9),
    (2, 1): dict(amp=5e-10, phase=5e-9),
    (3, 3): dict(amp=5e-10, phase=5e-9),
    (3, 2): dict(amp=1e-9,  phase=1e-8),
    (4, 4): dict(amp=5e-10, phase=5e-9),
}


@pytest.mark.parametrize("phi_ref_label,phi_ref", [("phi0", 0.0), ("phi_pi3", np.pi / 3.0)])
@pytest.mark.parametrize("ell,m", XHM_MODES, ids=[f"{l}{m}" for l, m in XHM_MODES])
def test_per_mode_complex_hlm(ell, m, phi_ref_label, phi_ref):
    params = dict(FIDUCIAL_PARAMS)
    params["phi_ref"] = phi_ref

    freqs_hz = make_freq_grid(f_min=20.0, f_max=512.0, df=0.125)

    h_ripple = ripple_xhm_one_mode_complex(freqs_hz, ell, m, params)
    h_lal = lal_xhm_one_mode_freqseq(freqs_hz, ell, m, params)

    # Mask zero-amplitude bins (LAL zeros out beyond f_max_lm and below
    # f_min_lm).  Compare only where LAL produced data.
    nonzero = np.abs(h_lal) > 0
    if nonzero.sum() < 16:
        pytest.skip(f"({ell},{m}): too few non-zero LAL bins")

    h_r = h_ripple[nonzero]
    h_l = h_lal[nonzero]

    # Relative amplitude error
    amp_err = relative_amp_error(np.abs(h_r), np.abs(h_l))
    max_amp_err = float(np.max(amp_err))

    # Absolute phase error after subtracting the f_min value
    # (constant offset is irrelevant for the mismatch).
    dphi = np.unwrap(np.angle(h_r) - np.angle(h_l))
    dphi = dphi - dphi[0]
    max_phase_err = float(np.max(np.abs(dphi)))

    tol = _TOL[(ell, m)]
    assert max_amp_err < tol["amp"], (
        f"({ell},{m}) {phi_ref_label}: max relative amp err = {max_amp_err:.3e} > {tol['amp']:.0e}"
    )
    assert max_phase_err < tol["phase"], (
        f"({ell},{m}) {phi_ref_label}: max abs phase err = {max_phase_err:.3e} rad > {tol['phase']:.0e}"
    )
