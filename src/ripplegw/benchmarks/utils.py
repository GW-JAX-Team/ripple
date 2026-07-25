"""
General benchmarking utilities for ripple.

This module provides common utilities used across different benchmarking scripts,
including git information retrieval, device detection, and parameter generation.
"""

import subprocess

import jax
import jax.numpy as jnp


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
    """Generate binary black hole parameters using JAX random sampling.

    Returns parameters for BBH systems including both aligned and precessing spins.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 14)

    def uniform(key, low, high):
        return jax.random.uniform(key, shape=(n_waveforms,), minval=low, maxval=high)

    # Mass parameters (5-100 solar masses)
    mass_1 = uniform(keys[0], 10, 100)
    mass_2 = uniform(keys[1], 0.5, 1.0) * mass_1  # mass_2 < mass_1

    # Spin magnitudes (0 to 0.99)
    a_1 = uniform(keys[2], 0, 0.99)
    a_2 = uniform(keys[3], 0, 0.99)

    # Precessing spin components - sample randomly then rescale to correct magnitude
    # For spin 1
    spin_1x_raw = uniform(keys[4], -1, 1)
    spin_1y_raw = uniform(keys[5], -1, 1)
    spin_1z_raw = uniform(keys[6], -1, 1)
    spin_1_mag = jnp.sqrt(spin_1x_raw**2 + spin_1y_raw**2 + spin_1z_raw**2)
    spin_1x = a_1 * spin_1x_raw / spin_1_mag
    spin_1y = a_1 * spin_1y_raw / spin_1_mag
    spin_1z = a_1 * spin_1z_raw / spin_1_mag

    # For spin 2
    spin_2x_raw = uniform(keys[7], -1, 1)
    spin_2y_raw = uniform(keys[8], -1, 1)
    spin_2z_raw = uniform(keys[9], -1, 1)
    spin_2_mag = jnp.sqrt(spin_2x_raw**2 + spin_2y_raw**2 + spin_2z_raw**2)
    spin_2x = a_2 * spin_2x_raw / spin_2_mag
    spin_2y = a_2 * spin_2y_raw / spin_2_mag
    spin_2z = a_2 * spin_2z_raw / spin_2_mag

    # Distance (100-2000 Mpc)
    luminosity_distance = uniform(keys[10], 100, 2000)

    # Angles
    theta_jn = uniform(keys[11], 0, jnp.pi)  # inclination
    phase = uniform(keys[12], 0, 2 * jnp.pi)

    # Time of coalescence
    geocent_time = uniform(keys[13], 0, 1)

    return {
        "mass_1": mass_1,
        "mass_2": mass_2,
        "a_1": a_1,
        "a_2": a_2,
        "spin_1x": spin_1x,
        "spin_1y": spin_1y,
        "spin_1z": spin_1z,
        "spin_2x": spin_2x,
        "spin_2y": spin_2y,
        "spin_2z": spin_2z,
        "luminosity_distance": luminosity_distance,
        "theta_jn": theta_jn,
        "phase": phase,
        "geocent_time": geocent_time,
    }


def generate_bns_parameters(n_waveforms, seed=42):
    """Generate binary neutron star parameters using JAX random sampling.

    Includes tidal deformability parameters.

    # TODO: Implement precessing spins. For now, this is limited to aligned spins only.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 10)

    def uniform(key, low, high):
        return jax.random.uniform(key, shape=(n_waveforms,), minval=low, maxval=high)

    # Mass parameters (1-3 solar masses for neutron stars)
    mass_1 = uniform(keys[0], 1.2, 3.0)
    mass_2 = uniform(keys[1], 0.5, 1.0) * mass_1  # mass_2 < mass_1

    # Aligned spin magnitudes (neutron stars typically have low spins)
    a_1 = uniform(keys[2], -0.4, 0.4)
    a_2 = uniform(keys[3], -0.4, 0.4)

    # Tidal deformability parameters (0-5000)
    lambda_1 = uniform(keys[4], 0, 5000)
    lambda_2 = uniform(keys[5], 0, 5000)

    # Distance (100-2000 Mpc)
    luminosity_distance = uniform(keys[6], 100, 2000)

    # Angles
    theta_jn = uniform(keys[7], 0, jnp.pi)  # inclination
    phase = uniform(keys[8], 0, 2 * jnp.pi)

    # Time of coalescence
    geocent_time = uniform(keys[9], 0, 1)

    return {
        "mass_1": mass_1,
        "mass_2": mass_2,
        "a_1": a_1,
        "a_2": a_2,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "luminosity_distance": luminosity_distance,
        "theta_jn": theta_jn,
        "phase": phase,
        "geocent_time": geocent_time,
    }


def generate_burst_parameters(n_waveforms, seed=42):
    """Generate sine-Gaussian burst parameters using JAX random sampling.

    Keys match ``SineGaussian.parameter_names`` directly -- no mass/spin
    conversion is needed, unlike the BBH/BNS generators above.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)

    def uniform(key, low, high):
        return jax.random.uniform(key, shape=(n_waveforms,), minval=low, maxval=high)

    return {
        "Q": uniform(keys[0], 2.0, 20.0),
        "f_0": uniform(keys[1], 50.0, 1000.0),
        "hrss": uniform(keys[2], 1e-23, 1e-20),
        "phase": uniform(keys[3], 0.0, 2 * jnp.pi),
        "e": uniform(keys[4], 0.0, 1.0),
    }
