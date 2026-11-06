from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from ripplegw.constants import PI
from ripplegw.typing import FloatLike

from .IMRPhenomXHM import (
    IMRPhenomX_TimeShift_22,
    IMRPhenomXAS_Phase,
    _compute_32_hlm,
    xhm_get_phase_coefficients,
    xhm_phase_noModeMixing,
    xhm_set_waveform_variables,
)

# This contains the code to generate the amplitude and phase for individual modes
# You will need to generate the 22-mode parameter dict first with build_pWF22 from IMRPhenomXHM.py as this is common between all HOMs.
# These modes are not yet multiplied by the spin-weighted spherical harmonics, so you will need to do that separately in order to get the final waveform.


def extract_22_mode_params(pWF22: dict[str, Any], phi0: float | FloatLike) -> Array:
    """
    TODO: write this out comprehensively but it basically does what the first few lines do in XLALSimIMRPhenomXHMGethlmModes in IMRPhenomXHM.py
    """

    M_s = pWF22["M_s"]
    theta = pWF22["theta"]
    phase_coeffs = pWF22["phase_coeffs"]
    Mf_ref = pWF22["Mf_ref"]
    chip = pWF22.get("chip", 0.0)
    # Step 1: time shift for 22 mode
    t0 = IMRPhenomX_TimeShift_22(pWF22)

    # Step 2: Reference phase of the 22-mode at Mf_ref.
    # phifRef (for the (2,2) mode): sets phase(Mf_ref) = 2*phi0.
    # phiref22 (for higher modes): LAL's pWFHM->phiref22 convention.
    phi_22_at_ref = IMRPhenomXAS_Phase(Mf_ref / M_s, theta, phase_coeffs, chip)
    # phifRef sets the reference phase for the (2,2) mode (and equals LAL's phiref22).
    # LAL: phiref22 = -phi_22(Mf_ref)/eta - timeshift*Mf_ref - phaseshift + 2*phi0 + PI/4
    # (phaseshift=0 in LAL; phi_22_at_ref already includes 1/eta normalization in ripple)
    phifRef = -phi_22_at_ref - t0 * Mf_ref + 2.0 * phi0 + PI / 4.0

    return jnp.array([M_s, theta, phase_coeffs, chip, t0, phifRef])


def get_single_mode_phase(
    freqs_geom: Float[Array, " n_freq"],
    pWF22: dict[str, Any],
    phi0: float | FloatLike,
    ell_mm_pair: tuple[int, int],
):
    ell, m = ell_mm_pair

    # 3 branches: 22-mode, 32-mode with mode mixing, all others
    def phase_22(freqs_geom, pWF22, ell, m):
        M_s, theta, phase_coeffs, chip, t0, phifRef = extract_22_mode_params(
            pWF22, phi0
        )

        phase_22 = IMRPhenomXAS_Phase(freqs_geom / M_s, theta, phase_coeffs, chip)
        phase_total = t0 * freqs_geom + phase_22 + phifRef
        return phase_total

    def phase_32(freqs_geom, pWF22, ell, m):
        _, _, _, _, t0, phifRef = extract_22_mode_params(pWF22, phi0)
        pWFHM = xhm_set_waveform_variables(ell, m, pWF22)

        phase_32 = jnp.unwrap(
            jnp.angle(  # FIXME This is not the best way to do this as this function also computes the amplitude. We should split these at some point
                _compute_32_hlm(freqs_geom, pWFHM, pWF22, t0, phifRef, phi0)
            )
        )

        phase_total = phase_32 + m / 2.0 * phifRef + m * phi0
        return phase_total

    def phase_other(freqs_geom, pWF22, ell, m):
        _, _, _, _, t0, phifRef = extract_22_mode_params(pWF22, phi0)
        pWFHM = xhm_set_waveform_variables(ell, m, pWF22)

        pPhase = xhm_get_phase_coefficients(pWFHM, pWF22, t0)
        phase_lm = (
            xhm_phase_noModeMixing(freqs_geom, pPhase, pWFHM, pWF22, t0) - m * phi0
        )  # Phase of XLALSimIMRPhenomXHMEvaluateOnehlmMode

        phase_total = phase_lm + (m / 2.0) * phifRef + m * phi0
        return phase_total

    mode_phase = jax.lax.cond(
        (ell, m) == (2, 2),
        phase_22,
        lambda freqs_geom, pWF22, ell, m: jax.lax.cond(
            (ell, m) == (3, 2), phase_32, phase_other, freqs_geom, pWF22, ell, m
        ),
        freqs_geom,
        pWF22,
        ell,
        m,
    )
    return mode_phase


def get_single_mode_amplitude(
    freqs_geom: Float[Array, " n_freq"],
    pWF22: dict[str, Any],
    phi0: float | FloatLike,
    ell_mm_pair: tuple[int, int],
):
    pass


def get_single_mode():
    pass
