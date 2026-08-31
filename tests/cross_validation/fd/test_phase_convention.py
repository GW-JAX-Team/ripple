"""Check absolute phase for non-precessing frequency-domain waveforms.

Overlap loss is invariant under a constant phase offset. This test therefore
measures the plus and cross inner-product phases at ``tc = phi_c = 0``.
Precessing models require a separate phase-convention analysis.
"""

import jax.numpy as jnp
import pytest

import ripplegw
from ripplegw.conversions import ms_to_Mc_eta
from tests.cross_validation.runner import (
    default_grid,
    get_tolerance,
    load_psd,
    load_tolerances,
)
from tests.helpers.metrics import get_nyquist_mask, inner_product_phase
from tests.helpers.params import canonical_params, regime

_TOLERANCES = load_tolerances()

# This fiducial phase setup covers only non-precessing FD models.
NON_PRECESSING = ripplegw.list_waveforms(domain="FD", is_precessing=False)


def _fiducial_params(wf) -> dict:
    """Return canonical parameters with zero coalescence time and phase."""
    if regime(wf) == "bns":
        m1, m2 = 1.4, 1.2
        s1_z = s2_z = 0.0  # zero spin/deformability: BBH limit, for a clean phase check
    else:
        m1, m2 = 40.0, 25.0
        s1_z, s2_z = 0.3, -0.2
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return canonical_params(
        wf,
        M_c=float(Mc),
        eta=float(eta),
        s1_z=s1_z,
        s2_z=s2_z,
        lambda_1=0.0,
        lambda_2=0.0,
        d_L=400.0,
        phase_c=0.0,
        iota=jnp.pi / 3,
    )


@pytest.mark.accuracy
@pytest.mark.parametrize("waveform_name", NON_PRECESSING)
def test_reference_phase_convention(
    waveform_name, reference, segment_duration_override
):
    if not reference.supports(waveform_name):
        pytest.skip(f"{reference.name} backend does not support {waveform_name!r}")

    threshold = get_tolerance(
        _TOLERANCES, reference.name, waveform_name, "phase_offset"
    )
    wf = ripplegw.waveform(waveform_name, f_ref=5.0)
    grid = default_grid(wf, T_override=segment_duration_override)
    params = _fiducial_params(wf)

    ref = reference.generate(waveform_name, params, grid)
    rip = wf(grid.axis, params)

    nyquist = get_nyquist_mask(grid.axis)
    psd_freqs, psd_vals = load_psd()
    psd = jnp.interp(grid.axis, jnp.asarray(psd_freqs), jnp.asarray(psd_vals))
    hp_r = jnp.asarray(rip["p"]) * nyquist
    hc_r = jnp.asarray(rip["c"]) * nyquist
    hp_ref = jnp.asarray(ref["p"]) * nyquist
    hc_ref = jnp.asarray(ref["c"]) * nyquist

    phase_p = float(inner_product_phase(hp_r, hp_ref, psd, grid.axis))
    phase_c = float(inner_product_phase(hc_r, hc_ref, psd, grid.axis))
    max_offset = max(abs(phase_p), abs(phase_c))

    assert max_offset < threshold, (
        f"{waveform_name}: max phase offset {max_offset:.4f} rad exceeds threshold "
        f"{threshold:.0e} rad (p={phase_p:.4f}, c={phase_c:.4f}) against "
        f"reference {reference.name!r} -- possible global phase convention mismatch"
    )
