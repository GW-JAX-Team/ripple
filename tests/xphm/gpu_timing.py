"""
GPU test script for IMRPhenomXPHM waveform generation.
"""

import jax
import jax.numpy as jnp
import bilby
import numpy as np
import time
from ripplegw.waveforms import IMRPhenomXPHM

jax.config.update("jax_enable_x64", False)

# Check available devices
print("JAX devices:", jax.devices())
print("Default backend:", jax.default_backend())
for d in jax.devices():
    print(f"Device: {d.device_kind}, Platform: {d.platform}")


def setup_injection_parameters(N_WAVEFORMS, reference_frequency):
    # Waveform parameters
    population = bilby.gw.prior.BBHPriorDict()
    population["chirp_mass"] = bilby.core.prior.Uniform(10, 100)
    population["mass_ratio"] = bilby.core.prior.Uniform(0.25, 1)
    population.pop("mass_1")
    population.pop("mass_2")

    _injection_parameters = bilby.gw.conversion.generate_component_masses(
        population.sample(N_WAVEFORMS)
    )
    _injection_parameters["reference_frequency"] = (
        np.ones(N_WAVEFORMS) * reference_frequency
    )

    _injection_parameters = bilby.gw.conversion.generate_component_spins(
        _injection_parameters
    )

    injection_parameters = {
        key: jnp.array(value) for key, value in _injection_parameters.items()
    }

    return injection_parameters


def main():
    N_injections = int(10)
    seed = 3232
    np.random.seed(seed)
    bilby.core.utils.random.seed(seed)
    # Frequency settings
    minimum_frequency = 20.0
    maximum_frequency = 512.0
    duration = 4.0
    reference_frequency = 50.0
    frequency_array = jnp.arange(minimum_frequency, maximum_frequency, 1.0 / duration)

    # Waveform parameters batch
    batch_injection_parameters = setup_injection_parameters(
        N_injections, reference_frequency
    )

    # Batch function
    generate_xphm_batched = jax.vmap(
        IMRPhenomXPHM.generate_xphm,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
    )

    # Warm-up run (JIT compilation)
    print("\n--- Warm-up run (JIT compilation) ---")
    start = time.time()
    hp, hc = generate_xphm_batched(
        batch_injection_parameters["mass_1"],
        batch_injection_parameters["mass_2"],
        batch_injection_parameters["spin_1x"],
        batch_injection_parameters["spin_1y"],
        batch_injection_parameters["spin_1z"],
        batch_injection_parameters["spin_2x"],
        batch_injection_parameters["spin_2y"],
        batch_injection_parameters["spin_2z"],
        batch_injection_parameters["luminosity_distance"],
        batch_injection_parameters["theta_jn"],
        batch_injection_parameters["phase"],
        frequency_array,
        reference_frequency,
    )

    hp.block_until_ready()
    hc.block_until_ready()
    warmup_time = time.time() - start
    print(f"Warm-up time (includes JIT): {warmup_time:.3f} s")

    # Timed run
    print("\n--- Timed run ---")
    start = time.time()
    hp, hc = generate_xphm_batched(
        batch_injection_parameters["mass_1"],
        batch_injection_parameters["mass_2"],
        batch_injection_parameters["spin_1x"],
        batch_injection_parameters["spin_1y"],
        batch_injection_parameters["spin_1z"],
        batch_injection_parameters["spin_2x"],
        batch_injection_parameters["spin_2y"],
        batch_injection_parameters["spin_2z"],
        batch_injection_parameters["luminosity_distance"],
        batch_injection_parameters["theta_jn"],
        batch_injection_parameters["phase"],
        frequency_array,
        reference_frequency,
    )

    hp.block_until_ready()
    hc.block_until_ready()
    exec_time = time.time() - start
    print(f"Execution time for {N_injections} waveforms: {exec_time:.6f} s")
    print(f"Time per waveform: {exec_time / N_injections * 1000:.3f} ms")


if __name__ == "__main__":
    main()
