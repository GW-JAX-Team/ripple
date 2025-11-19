"""
Module for handling waveform flags for the IMRPhenomXPHM model.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.experimental import checkify


def xlal_sim_inspiral_create_mode_array(max_l: int = 8) -> jnp.ndarray:
    """
    Create a mode array initialized to empty (all modes inactive).

    Returns:
        JAX boolean array where index i corresponds to mode bit position.
    """
    max_modes = max_l**2 + 2 * max_l + 1
    return jnp.zeros(max_modes, dtype=bool)


def xlal_sim_inspiral_mode_array_activate_mode(modes: jnp.ndarray, ell: int, m: int, max_l: int = 8) -> jnp.ndarray:
    """
    Activate a specific mode (ell, m) in the mode array.

    Args:
        modes: JAX boolean array for mode states.
        ell: Spherical harmonic ell.
        m: Spherical harmonic m.
        max_l: Maximum ell value for bounds checking.

    Returns:
        New JAX array with the mode activated.
    """
    # Sanity checks
    checkify.check(ell <= max_l, f"Invalid value of ell={ell} must not be greater than {max_l}")
    checkify.check(abs(m) <= ell, f"Invalid value of m={m} for ell={ell}")

    # Calculate index
    i = ell * ell + ell + m

    # Check bounds
    checkify.check(i < modes.size, f"Index {i} out of bounds for modes array of size {modes.size}")

    # Activate mode
    return modes.at[i].set(True)


def xlal_sim_inspiral_mode_array_deactivate_mode(modes: jnp.ndarray, ell: int, m: int, max_l: int = 8) -> jnp.ndarray:
    """
    Deactivate a specific mode (ell, m) in the mode array.

    Args:
        modes: JAX boolean array for mode states.
        ell: Spherical harmonic ell.
        m: Spherical harmonic m.
        max_l: Maximum ell value for bounds checking.

    Returns:
        New JAX array with the mode deactivated.
    """
    # Sanity checks
    checkify.check(ell <= max_l, f"Invalid value of ell={ell} must not be greater than {max_l}")
    checkify.check(abs(m) <= ell, f"Invalid value of m={m} for ell={ell}")

    # Calculate index
    i = ell * ell + ell + m

    # Check bounds
    checkify.check(i < modes.size, f"Index {i} out of bounds for modes array of size {modes.size}")

    # Deactivate mode
    return modes.at[i].set(False)


def xlal_sim_inspiral_mode_array_is_mode_active(modes: jnp.ndarray, ell: int, m: int, max_l: int = 8) -> bool:
    """
    Check if a specific mode (ell, m) is active in the mode array.

    Args:
        modes: JAX boolean array for mode states.
        ell: Spherical harmonic ell.
        m: Spherical harmonic m.
        max_l: Maximum ell value for bounds checking.

    Returns:
        True if the mode is active, False otherwise.
    """
    # Sanity checks
    checkify.check(ell <= max_l, f"Invalid value of ell={ell} must not be greater than {max_l}")
    checkify.check(abs(m) <= ell, f"Invalid value of m={m} for ell={ell}")

    # Calculate index
    i = ell * ell + ell + m

    # Check bounds
    checkify.check(i < modes.size, f"Index {i} out of bounds for modes array of size {modes.size}")

    # Check if active
    return modes[i]
