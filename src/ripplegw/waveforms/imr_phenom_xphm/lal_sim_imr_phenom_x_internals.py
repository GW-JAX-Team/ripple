"""Helper functions for IMRPhenomXPHM waveform model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_inspiral_waveform_flags import (
    xlal_sim_inspiral_mode_array_is_mode_active,
)


@checkify.checkify
def imr_phenom_x_initialize_powers(number: float | jnp.ndarray) -> dict:
    """Initialize various powers of the input number.

    Args:
        number: Input number (float or JAX array).

    Returns:
        Dictionary mapping power names to their computed values.
    """
    # Ensure number is a JAX array for consistent operations
    number = jnp.asarray(number)

    # Sanity check
    checkify.check(jnp.all(number >= 0), "Error: number must be non-negative.")

    # Compute sixth root and its reciprocal
    sixth = jnp.power(number, 1.0 / 6.0)
    m_sixth = 1.0 / sixth

    # Build the powers dictionary
    powers = {
        "one_sixth": sixth,
        "m_one_sixth": m_sixth,
        "m_one": 1.0 / number,
        "itself": number,
        "one_third": sixth * sixth,
        "m_one_third": 1.0 / (sixth * sixth),
        "two_thirds": (sixth * sixth) ** 2,
        "m_two_thirds": 1.0 / ((sixth * sixth) ** 2),
        "four_thirds": ((sixth * sixth) ** 2) ** 2,
        "m_four_thirds": 1.0 / (((sixth * sixth) ** 2) ** 2),
        "five_thirds": (((sixth * sixth) ** 2) ** 2) * (sixth * sixth),
        "m_five_thirds": 1.0 / ((((sixth * sixth) ** 2) ** 2) * (sixth * sixth)),
        "seven_thirds": (((sixth * sixth) ** 2) ** 2) * number,
        "m_seven_thirds": 1.0 / ((((sixth * sixth) ** 2) ** 2) * number),
        "eight_thirds": ((((sixth * sixth) ** 2) ** 2) * number) * (sixth * sixth),
        "m_eight_thirds": 1.0 / (((((sixth * sixth) ** 2) ** 2) * number) * (sixth * sixth)),
        "ten_thirds": ((((sixth * sixth) ** 2) ** 2) * number) * number,
        "m_ten_thirds": 1.0 / (((((sixth * sixth) ** 2) ** 2) * number) * number),
        "two": number * number,
        "three": number * number * number,
        "four": number * number * number * number,
        "five": number * number * number * number * number,
        "m_two": 1.0 / (number * number),
        "m_three": 1.0 / (number * number * number),
        "m_four": 1.0 / (number * number * number * number),
        "m_five": 1.0 / (number * number * number * number * number),
        "seven_sixths": sixth * number,
        "m_seven_sixths": m_sixth / number,
        "log": jnp.log(number),
        "sqrt": sixth * sixth * sixth,  # Equivalent to sqrt(number)
    }

    return powers


def _generate_valid_modes(max_l: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute valid (ell, emm) pairs statically.

    Args:
        max_l: Maximum ell value.

    Returns:
        Tuple of JAX arrays (ell_array, emm_array) containing valid modes.
    """
    ell_list = []
    emm_list = []
    for ell in range(2, max_l + 1):
        for emm in range(0, ell + 1):
            ell_list.append(ell)
            emm_list.append(emm)
    return jnp.array(ell_list, dtype=int), jnp.array(emm_list, dtype=int)


@checkify.checkify
def _check_input_mode_array_impl(
    mode_array: jnp.ndarray, ell_valid: jnp.ndarray, emm_valid: jnp.ndarray, max_l: int
) -> bool:
    """Internal implementation of mode array checking.

    Args:
        mode_array: JAX boolean array indicating active modes.
        ell_valid: JAX array of valid ell values to check.
        emm_valid: JAX array of valid emm values to check.
        max_l: Maximum ell value for bounds checking.

    Returns:
        True if all active modes are allowed, raises error otherwise.
    """
    # Allowed modes: (l, |m|) pairs
    allowed_l = jnp.array([2, 2, 3, 3, 4])
    allowed_m = jnp.array([2, 1, 3, 2, 4])

    # Check if each valid mode is active (positive or negative m)
    active_pos = jax.vmap(
        lambda ell, emm: xlal_sim_inspiral_mode_array_is_mode_active(modes=mode_array, ell=ell, m=emm, max_l=max_l)
    )(ell_valid, emm_valid)
    active_neg = jax.vmap(
        lambda ell, emm: xlal_sim_inspiral_mode_array_is_mode_active(modes=mode_array, ell=ell, m=-emm, max_l=max_l)
    )(ell_valid, emm_valid)
    active = active_pos | active_neg

    # For active modes, check if they are allowed
    def is_allowed(ell, emm):
        # Check if (ell, emm) is in allowed
        matches = (allowed_l == ell) & (allowed_m == emm)
        return jnp.any(matches)

    allowed_mask = jax.vmap(is_allowed)(ell_valid, emm_valid)

    # Invalid if active but not allowed
    invalid = active & ~allowed_mask

    # Check that no invalid modes are active
    checkify.check(
        ~jnp.any(invalid),
        "Invalid modes activated in mode_array. Only the modes (2,2), (2,1), (3,3), (3,2), and (4,4) are supported.",
    )

    # Always check a trivial condition to ensure err is properly initialized
    checkify.check(True, "")

    return True


def check_input_mode_array(lal_params: dict, max_l: int = 8) -> tuple[checkify.Error, bool]:
    """Check if mode_array in lal_params contains only allowed modes.

    Args:
        lal_params: Dictionary that may contain a 'mode_array' key.
        max_l: Maximum angular momentum number to check (must be concrete/static).

    Returns:
        Tuple of (error, result) from checkify.
    """
    # Precompute valid modes statically before any JAX operations
    ell_valid, emm_valid = _generate_valid_modes(max_l)

    # Check if mode_array is present
    has_mode_array = "mode_array" in lal_params

    def check_with_mode_array():
        """Check modes when mode_array is present."""
        mode_array = lal_params["mode_array"]
        return _check_input_mode_array_impl(mode_array, ell_valid, emm_valid, max_l)

    def check_without_mode_array():
        """Skip check when mode_array is not present."""
        return checkify.checkify(lambda: True)()

    # Use Python if since this is before JIT transforms the whole function
    # (check_input_mode_array itself is not transformed, only called by JIT)
    if has_mode_array:
        return check_with_mode_array()
    return check_without_mode_array()
