"""
General benchmarking utilities for ripple.

This module provides common utilities used across different benchmarking scripts,
including git information retrieval, device detection, and parameter generation.
"""

import subprocess

import jax
import jax.numpy as jnp
import numpy as np


def get_git_hash():
    """Get the current git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_device_name():
    """Get the actual device name (CPU or GPU model like H100)."""
    devices = jax.devices()
    if len(devices) == 0:
        return "cpu"

    device = devices[0]
    if device.platform == "cpu":
        return "cpu"
    else:
        # For GPU, get the device kind (e.g., "NVIDIA H100")
        device_kind = device.device_kind
        # Try to extract just the GPU model name
        if "H100" in device_kind:
            return "H100"
        elif "A100" in device_kind:
            return "A100"
        elif "V100" in device_kind:
            return "V100"
        elif "T4" in device_kind:
            return "T4"
        else:
            # Fallback to the full device kind or just "gpu"
            return device_kind if device_kind else "gpu"


def generate_bbh_parameters(n_waveforms, seed=42):
    """Generate binary black hole parameters using numpy random sampling.

    Returns parameters for BBH systems including both aligned and precessing spins.
    """
    rng = np.random.default_rng(seed)

    # Mass parameters (5-100 solar masses)
    mass_1 = rng.uniform(10, 100, n_waveforms)
    mass_2 = rng.uniform(0.5, 1.0, n_waveforms) * mass_1  # mass_2 < mass_1

    # Spin magnitudes (0 to 0.99)
    a_1 = rng.uniform(0, 0.99, n_waveforms)
    a_2 = rng.uniform(0, 0.99, n_waveforms)

    # Precessing spin components - sample randomly then rescale to correct magnitude
    # For spin 1
    spin_1x_raw = rng.uniform(-1, 1, n_waveforms)
    spin_1y_raw = rng.uniform(-1, 1, n_waveforms)
    spin_1z_raw = rng.uniform(-1, 1, n_waveforms)
    spin_1_mag = np.sqrt(spin_1x_raw**2 + spin_1y_raw**2 + spin_1z_raw**2)
    spin_1x = a_1 * spin_1x_raw / spin_1_mag
    spin_1y = a_1 * spin_1y_raw / spin_1_mag
    spin_1z = a_1 * spin_1z_raw / spin_1_mag

    # For spin 2
    spin_2x_raw = rng.uniform(-1, 1, n_waveforms)
    spin_2y_raw = rng.uniform(-1, 1, n_waveforms)
    spin_2z_raw = rng.uniform(-1, 1, n_waveforms)
    spin_2_mag = np.sqrt(spin_2x_raw**2 + spin_2y_raw**2 + spin_2z_raw**2)
    spin_2x = a_2 * spin_2x_raw / spin_2_mag
    spin_2y = a_2 * spin_2y_raw / spin_2_mag
    spin_2z = a_2 * spin_2z_raw / spin_2_mag

    # Distance (100-2000 Mpc)
    luminosity_distance = rng.uniform(100, 2000, n_waveforms)

    # Angles
    theta_jn = rng.uniform(0, np.pi, n_waveforms)  # inclination
    phase = rng.uniform(0, 2 * np.pi, n_waveforms)

    # Time of coalescence
    geocent_time = rng.uniform(0, 1, n_waveforms)

    # Convert to JAX arrays
    params = {
        "mass_1": jnp.array(mass_1),
        "mass_2": jnp.array(mass_2),
        "a_1": jnp.array(a_1),
        "a_2": jnp.array(a_2),
        "spin_1x": jnp.array(spin_1x),
        "spin_1y": jnp.array(spin_1y),
        "spin_1z": jnp.array(spin_1z),
        "spin_2x": jnp.array(spin_2x),
        "spin_2y": jnp.array(spin_2y),
        "spin_2z": jnp.array(spin_2z),
        "luminosity_distance": jnp.array(luminosity_distance),
        "theta_jn": jnp.array(theta_jn),
        "phase": jnp.array(phase),
        "geocent_time": jnp.array(geocent_time),
    }

    return params


def generate_bns_parameters(n_waveforms, seed=42):
    """Generate binary neutron star parameters using numpy random sampling.

    Includes tidal deformability parameters.

    # TODO: Implement precessing spins. For now, this is limited to aligned spins only.
    """
    rng = np.random.default_rng(seed)

    # Mass parameters (1-3 solar masses for neutron stars)
    mass_1 = rng.uniform(1.2, 3.0, n_waveforms)
    mass_2 = rng.uniform(0.5, 1.0, n_waveforms) * mass_1  # mass_2 < mass_1

    # Aligned spin magnitudes (neutron stars typically have low spins)
    a_1 = rng.uniform(-0.4, 0.4, n_waveforms)
    a_2 = rng.uniform(-0.4, 0.4, n_waveforms)

    # Tidal deformability parameters (0-5000)
    lambda_1 = rng.uniform(0, 5000, n_waveforms)
    lambda_2 = rng.uniform(0, 5000, n_waveforms)

    # Distance (100-2000 Mpc)
    luminosity_distance = rng.uniform(100, 2000, n_waveforms)

    # Angles
    theta_jn = rng.uniform(0, np.pi, n_waveforms)  # inclination
    phase = rng.uniform(0, 2 * np.pi, n_waveforms)

    # Time of coalescence
    geocent_time = rng.uniform(0, 1, n_waveforms)

    # Convert to JAX arrays
    params = {
        "mass_1": jnp.array(mass_1),
        "mass_2": jnp.array(mass_2),
        "a_1": jnp.array(a_1),
        "a_2": jnp.array(a_2),
        "lambda_1": jnp.array(lambda_1),
        "lambda_2": jnp.array(lambda_2),
        "luminosity_distance": jnp.array(luminosity_distance),
        "theta_jn": jnp.array(theta_jn),
        "phase": jnp.array(phase),
        "geocent_time": jnp.array(geocent_time),
    }

    return params
