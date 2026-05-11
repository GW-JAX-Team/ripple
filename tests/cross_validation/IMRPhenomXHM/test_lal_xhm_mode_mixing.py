"""Layer 6: 32-mode mixing isolation test.

If Layer 5 fails for (3,2) but the other modes pass, the bug is inside
`_compute_32_hlm` (spheroidal coefficients, mu mixing, or the s2s
function).  We exercise the mixing across a small parameter grid where
the mixing amplitude is large.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from tests.cross_validation.IMRPhenomXHM.xhm_helpers import (
    LAL_AVAILABLE,
    FIDUCIAL_PARAMS,
    lal_xhm_one_mode_freqseq,
    make_freq_grid,
    relative_amp_error,
    ripple_xhm_one_mode_complex,
)

pytestmark = pytest.mark.skipif(
    not LAL_AVAILABLE, reason="LALSuite required for cross-validation tests"
)


# Mass ratio + spin combinations chosen to maximise mode-mixing strength:
# - large mass ratio (q >= 4) makes the (3,2) mode a substantial fraction of (2,2)
# - high aligned spin shifts QNM frequencies
_MIXING_CASES = [
    dict(m1=50.0, m2=10.0, chi1z=0.7,  chi2z=0.0),
    dict(m1=50.0, m2=10.0, chi1z=-0.5, chi2z=0.2),
    dict(m1=80.0, m2=20.0, chi1z=0.6,  chi2z=-0.3),
    dict(m1=40.0, m2=15.0, chi1z=0.0,  chi2z=0.5),
]


@pytest.mark.parametrize(
    "case", _MIXING_CASES, ids=[
        f"q{int(c['m1']/c['m2'])}_s{c['chi1z']:+.1f}_{c['chi2z']:+.1f}" for c in _MIXING_CASES
    ]
)
def test_32_mixing_complex_hlm(case):
    params = dict(FIDUCIAL_PARAMS, **case)
    freqs_hz = make_freq_grid(f_min=20.0, f_max=512.0, df=0.125)

    h_ripple = ripple_xhm_one_mode_complex(freqs_hz, 3, 2, params)
    h_lal = lal_xhm_one_mode_freqseq(freqs_hz, 3, 2, params)

    nonzero = np.abs(h_lal) > 0
    if nonzero.sum() < 16:
        pytest.skip("too few non-zero LAL bins for (3,2)")

    h_r = h_ripple[nonzero]
    h_l = h_lal[nonzero]

    max_amp_err = float(np.max(relative_amp_error(np.abs(h_r), np.abs(h_l))))
    dphi = np.unwrap(np.angle(h_r) - np.angle(h_l))
    dphi = dphi - dphi[0]
    max_phase_err = float(np.max(np.abs(dphi)))

    # Slightly looser than Layer 5 (3,2) because mode mixing amplifies
    # any small QNM/coefficient errors.
    assert max_amp_err < 1e-8, (
        f"{case}: max relative amp err = {max_amp_err:.3e}"
    )
    assert max_phase_err < 1e-7, (
        f"{case}: max abs phase err = {max_phase_err:.3e} rad"
    )
