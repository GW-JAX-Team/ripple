"""Layer 3: per-mode amplitude vs lalsim.SimIMRPhenomXHMAmplitude.

For each mode the test computes the relative amplitude error in three
disjoint frequency bands (inspiral / intermediate / ringdown) so that a
failing region pinpoints which polynomial fit is wrong.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from ripplegw.waveforms.IMRPhenomX.IMRPhenomXHM import (
    _compute_32_hlm,
    xhm_amp_noModeMixing,
    xhm_get_amp_coefficients,
    xhm_set_waveform_variables,
)
from tests.cross_validation.internals.helpers import (
    FIDUCIAL_PARAMS,
    LAL_AVAILABLE,
    NON_22_MODES,
    lal_xhm_amplitude,
    make_freq_grid,
    region_masks,
    relative_amp_error,
    ripple_pwf22,
)

pytestmark = pytest.mark.skipif(
    not LAL_AVAILABLE, reason="LALSuite required for cross-validation tests"
)


# Per-region tolerances on the relative amplitude error.
# The 22 mode dominates the SNR so its tolerance must be tightest;
# subdominant modes are allowed proportionally looser bands.
# Values calibrated so that, when satisfied, the 1e-6 overlap loss budget
# is comfortably met after the noise-weighted integral.
_TOL = {
    # (2,2) goes through XAS path (validated by IMRPhenomXAS tests).
    (2, 1): {"inspiral": 1e-10, "intermediate": 1e-10, "ringdown": 1e-10},
    (3, 3): {"inspiral": 1e-10, "intermediate": 1e-10, "ringdown": 1e-10},
    # (3,2) intermediate/ringdown: measured ~1.7e-8, >2x margin below.
    # Same S2S-phase-sampling sensitivity documented for this mode in
    # docs/dev/lal_agreement.md (IMRPhenomXHM section) -- not a regression.
    (3, 2): {"inspiral": 1e-9, "intermediate": 5e-8, "ringdown": 5e-8},
    (4, 4): {"inspiral": 1e-10, "intermediate": 1e-10, "ringdown": 1e-10},
}


def _ripple_amp_only(freqs_hz, ell, m, params=FIDUCIAL_PARAMS):
    """Per-mode |Amp_lm(f)| from ripple, distance-normalised the same
    way as LAL's SimIMRPhenomXHMAmplitude.

    For (3,2) we extract the amplitude from the magnitude of
    `_compute_32_hlm` (which entangles the mixing).  For all other
    modes the path is `xhm_amp_noModeMixing`.
    """
    pWF22 = ripple_pwf22(params)
    M_s = pWF22["M_s"]
    freqs_geom = jnp.array(freqs_hz) * M_s
    pWFHM = xhm_set_waveform_variables(int(ell), int(m), pWF22)

    if pWFHM.MixingOn:
        # 32 with mixing: amplitude is the magnitude of the complex hlm
        # before the (-1)^l and amp0 prefactors.
        from ripplegw.waveforms.IMRPhenomX.IMRPhenomXHM import IMRPhenomX_TimeShift_22

        t0 = IMRPhenomX_TimeShift_22(pWF22)
        # phifRef does not affect the magnitude.
        hlm = _compute_32_hlm(freqs_geom, pWFHM, pWF22, t0, 0.0, 0.0)
        amp_geom = np.abs(np.asarray(hlm))
    else:
        pAmp = xhm_get_amp_coefficients(pWFHM, pWF22)
        amp_geom = np.asarray(xhm_amp_noModeMixing(freqs_geom, pAmp, pWFHM))

    # Convert geometric amplitude to LAL's distance-and-mass units.
    # LAL's SimIMRPhenomXHMAmplitude returns the *physical* amplitude in
    # SI: Amp_lm = amp0 * Amp_geom_lm with amp0 = M^2 * MRSUN * MTSUN / d.
    from ripplegw.constants import MPC, MRSUN, MTSUN

    Mtot = params["m1"] + params["m2"]
    amp0 = Mtot * MRSUN * Mtot * MTSUN / (params["distance_mpc"] * MPC)
    return amp_geom * amp0


@pytest.mark.parametrize(
    "ell,m", NON_22_MODES, ids=[f"{ell}{m}" for ell, m in NON_22_MODES]
)
def test_per_mode_amplitude_per_region(ell, m):
    """Bisect amplitude error across inspiral/intermediate/ringdown."""
    freqs_hz = make_freq_grid(f_min=20.0, f_max=512.0, df=0.125)
    amp_ripple = _ripple_amp_only(freqs_hz, ell, m)
    amp_lal = lal_xhm_amplitude(freqs_hz, ell, m)

    rel = relative_amp_error(amp_ripple, amp_lal)
    insp, inter, ring = region_masks(freqs_hz, ell, m)
    tol = _TOL[(ell, m)]

    failures = []
    for name, mask, t in [
        ("inspiral", insp, tol["inspiral"]),
        ("intermediate", inter, tol["intermediate"]),
        ("ringdown", ring, tol["ringdown"]),
    ]:
        if not np.any(mask):
            continue
        max_err = float(np.max(rel[mask]))
        if max_err > t:
            failures.append(f"({ell},{m}) {name}: {max_err:.3e} > {t:.0e}")

    assert not failures, "\n".join(failures)
