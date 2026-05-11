"""Layer 7: hp/hc spherical-harmonic assembly vs SimIMRPhenomXHM.

By Layer 6 each (l,m) mode has been validated.  This test exercises
the (-1)^l, Y_lm and 1/2*(F_neg ± F_pos) factors in
`gen_IMRPhenomXHM_hphc`.  We sweep iota to flush conventions that
collapse at face-on (iota=0).
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from tests.cross_validation.IMRPhenomXHM.xhm_helpers import (
    LAL_AVAILABLE,
    FIDUCIAL_PARAMS,
    lal_xhm_full,
    make_freq_grid,
    ripple_xhm_hphc,
)

pytestmark = pytest.mark.skipif(
    not LAL_AVAILABLE, reason="LALSuite required for cross-validation tests"
)


_INC_CASES = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0, np.pi]


@pytest.mark.parametrize("inclination", _INC_CASES, ids=[f"inc{i:.2f}" for i in _INC_CASES])
def test_xhm_hphc_strict(inclination):
    params = dict(FIDUCIAL_PARAMS)
    params["inclination"] = inclination

    freqs_hz = make_freq_grid(f_min=20.0, f_max=512.0, df=0.125)

    hp_r, hc_r = ripple_xhm_hphc(freqs_hz, params)
    hp_l, hc_l = lal_xhm_full(freqs_hz, params)

    # Trim to common length.
    n = min(len(hp_r), len(hp_l))
    hp_r, hc_r = hp_r[:n], hc_r[:n]
    hp_l, hc_l = hp_l[:n], hc_l[:n]

    nonzero = (np.abs(hp_l) + np.abs(hc_l)) > 0
    if nonzero.sum() < 16:
        pytest.skip("too few non-zero bins")

    # Use the larger of |hp_l|, |hc_l| as denominator
    scale = max(np.max(np.abs(hp_l)), np.max(np.abs(hc_l)))
    hp_err = np.abs(hp_r[nonzero] - hp_l[nonzero]) / scale
    hc_err = np.abs(hc_r[nonzero] - hc_l[nonzero]) / scale

    max_hp = float(np.max(hp_err))
    max_hc = float(np.max(hc_err))

    assert max_hp < 1e-9, f"inc={inclination:.2f}: max |Δhp|/scale = {max_hp:.3e}"
    assert max_hc < 1e-9, f"inc={inclination:.2f}: max |Δhc|/scale = {max_hc:.3e}"
