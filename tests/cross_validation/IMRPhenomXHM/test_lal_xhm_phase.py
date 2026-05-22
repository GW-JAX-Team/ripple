"""Layer 4: per-mode phase vs lalsim.SimIMRPhenomXHMPhase.

Both ripple and LAL define the per-mode phase up to a global additive
constant; therefore we compare the phase *difference relative to the
lowest-frequency bin*, not the absolute value.

A linear-in-Mf mismatch (slope error) shows up as a non-zero residual
after subtracting the value at f_min — this catches the
DeltaT/t0 bug suspected in `initial_plan.md` §3.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tests.cross_validation.IMRPhenomXHM.xhm_helpers import (
    LAL_AVAILABLE,
    FIDUCIAL_PARAMS,
    NON_22_MODES,
    lal_xhm_phase,
    make_freq_grid,
    region_masks,
    ripple_pwf22,
    unwrap_after_subtract,
)

from ripplegw.waveforms.IMRPhenomXHM import (
    IMRPhenomX_TimeShift_22,
    IMRPhenomXAS_Phase,
    xhm_set_waveform_variables,
    xhm_get_phase_coefficients,
    xhm_phase_noModeMixing,
    _compute_32_hlm,
)

pytestmark = pytest.mark.skipif(
    not LAL_AVAILABLE, reason="LALSuite required for cross-validation tests"
)


# Per-region absolute phase tolerances (radians).  See debug_plan_xhm.md
# for the calibration.
_TOL_RAD = {
    # (2,2) goes through XAS path (validated by IMRPhenomXAS tests).
    (2, 1): dict(inspiral=5e-9, intermediate=5e-9, ringdown=5e-9),
    (3, 3): dict(inspiral=5e-9, intermediate=5e-9, ringdown=5e-9),
    (3, 2): dict(inspiral=1e-8, intermediate=1e-8, ringdown=1e-8),
    (4, 4): dict(inspiral=5e-9, intermediate=5e-9, ringdown=5e-9),
}


def _ripple_phase_only(freqs_hz, ell, m, params=FIDUCIAL_PARAMS):
    """Per-mode phase from ripple WITHOUT the (mm/2)*phifRef + mm*phi0
    term — that lives in `XLALSimIMRPhenomXHMGethlmModes` and is
    re-introduced at Layer 5.

    This lets Layer 4 isolate the polynomial phase from the reference-
    phase / time-shift assembly that Layer 5 tests.
    """
    pWF22 = ripple_pwf22(params)
    M_s = pWF22["M_s"]
    freqs_geom = jnp.array(freqs_hz) * M_s
    pWFHM = xhm_set_waveform_variables(int(ell), int(m), pWF22)
    t0 = IMRPhenomX_TimeShift_22(pWF22)

    if ell == 2 and m == 2:
        # 22 mode: ripple uses XAS phase + t0.
        phase_22 = np.asarray(
            IMRPhenomXAS_Phase(
                jnp.array(freqs_hz), pWF22["theta"], pWF22["phase_coeffs"]
            )
        )
        phase = phase_22 + float(t0) * np.asarray(freqs_geom)
        return phase

    if pWFHM.MixingOn:
        # Use angle of the complex hlm with phi_ref=0, phifRef set to 0
        # so we subtract a constant only when comparing to LAL.
        hlm = _compute_32_hlm(freqs_geom, pWFHM, pWF22, t0, 0.0, 0.0)
        return np.unwrap(np.angle(np.asarray(hlm)))

    pPhase = xhm_get_phase_coefficients(pWFHM, pWF22, t0)
    phase = np.asarray(xhm_phase_noModeMixing(freqs_geom, pPhase, pWFHM, pWF22, t0))
    return phase


@pytest.mark.parametrize(
    "ell,m", NON_22_MODES, ids=[f"{l}{m}" for l, m in NON_22_MODES]
)
def test_per_mode_phase_per_region(ell, m):
    freqs_hz = make_freq_grid(f_min=20.0, f_max=512.0, df=0.125)

    phase_ripple = _ripple_phase_only(freqs_hz, ell, m)
    phase_lal = lal_xhm_phase(freqs_hz, ell, m)

    # Subtract the value at f_min — both phases are defined up to a
    # global constant.  After subtraction, any *linear* residual
    # indicates a t0/timeshift bug; any *nonlinear* residual indicates a
    # polynomial fit bug.
    diff = (phase_ripple - phase_ripple[0]) - (phase_lal - phase_lal[0])
    diff = unwrap_after_subtract(diff)

    insp, inter, ring = region_masks(freqs_hz, ell, m)
    tol = _TOL_RAD[(ell, m)]

    failures = []
    for name, mask, t in [
        ("inspiral", insp, tol["inspiral"]),
        ("intermediate", inter, tol["intermediate"]),
        ("ringdown", ring, tol["ringdown"]),
    ]:
        if not np.any(mask):
            continue
        max_err = float(np.max(np.abs(diff[mask])))
        if max_err > t:
            failures.append(
                f"({ell},{m}) {name}: max|Δφ - linear_offset| = {max_err:.3e} rad > {t:.0e}"
            )

    assert not failures, "\n".join(failures)
