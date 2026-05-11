"""Layer 9: XPHM hp/hc strict comparison vs SimIMRPhenomXPHM.

The existing `test_lal_mismatch.py::IMRPhenomXPHM` is the *final*
correctness criterion (phase-maximised mismatch ≤ 1e-6 over 10 random
samples).  This test instead does a *strict* per-bin hp/hc comparison
on a single fiducial precessing binary, to expose the *shape* of any
remaining discrepancy when the mismatch test fails.

We test PrecVersion=222 (raises on MSA-init failure) — same setting
the mismatch test uses.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from tests.cross_validation.IMRPhenomXHM.xhm_helpers import (
    LAL_AVAILABLE,
    FIDUCIAL_PARAMS_PREC,
    lal_xphm_full,
    make_freq_grid,
    ripple_xphm_hphc,
)

pytestmark = pytest.mark.skipif(
    not LAL_AVAILABLE, reason="LALSuite required for cross-validation tests"
)


def _strict_xphm(inclination):
    params = dict(FIDUCIAL_PARAMS_PREC)
    params["inclination"] = inclination

    freqs_hz = make_freq_grid(f_min=20.0, f_max=512.0, df=0.125)

    try:
        hp_l, hc_l = lal_xphm_full(freqs_hz, params)
    except RuntimeError as e:
        pytest.skip(f"LAL XPHM raised (likely MSA init): {e}")

    hp_r, hc_r = ripple_xphm_hphc(freqs_hz, params)

    n = min(len(hp_r), len(hp_l))
    hp_r, hc_r = hp_r[:n], hc_r[:n]
    hp_l, hc_l = hp_l[:n], hc_l[:n]

    nonzero = (np.abs(hp_l) + np.abs(hc_l)) > 0
    if nonzero.sum() < 16:
        pytest.skip("too few non-zero bins")

    scale = max(np.max(np.abs(hp_l)), np.max(np.abs(hc_l)))
    return (
        float(np.max(np.abs(hp_r[nonzero] - hp_l[nonzero])) / scale),
        float(np.max(np.abs(hc_r[nonzero] - hc_l[nonzero])) / scale),
    )


@pytest.mark.parametrize(
    "inclination",
    [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0],
    ids=lambda i: f"inc{i:.2f}",
)
def test_xphm_hphc_strict(inclination):
    max_hp, max_hc = _strict_xphm(inclination)
    # Looser than XHM because precession adds an MSA integration whose
    # numeric path differs slightly between JAX and LAL.  Tighten when
    # XPHM mismatch starts passing 1e-6.
    assert max_hp < 1e-6, f"inc={inclination:.2f}: max |Δhp|/scale = {max_hp:.3e}"
    assert max_hc < 1e-6, f"inc={inclination:.2f}: max |Δhc|/scale = {max_hc:.3e}"
