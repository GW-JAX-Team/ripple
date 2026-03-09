"""
GPU test script for IMRPhenomXPHM waveform generation.
"""

import jax
import jax.numpy as jnp
import bilby
import numpy as np
import time
from ripplegw.waveforms import IMRPhenomXPHM

jax.config.update("jax_enable_x64", True)

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
        key: jnp.array(value, dtype=jnp.float64)
        for key, value in _injection_parameters.items()
    }

    return injection_parameters


def main():
    N_injections = int(1e4)
    batch_size = 2000
    N_batches = N_injections // batch_size
    seed = 3232
    np.random.seed(seed)
    bilby.core.utils.random.seed(seed)
    # Frequency settings
    minimum_frequency = 20.0
    maximum_frequency = 1024.0
    duration = 8.0
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
        batch_injection_parameters["mass_1"][0:batch_size],
        batch_injection_parameters["mass_2"][0:batch_size],
        batch_injection_parameters["spin_1x"][0:batch_size],
        batch_injection_parameters["spin_1y"][0:batch_size],
        batch_injection_parameters["spin_1z"][0:batch_size],
        batch_injection_parameters["spin_2x"][0:batch_size],
        batch_injection_parameters["spin_2y"][0:batch_size],
        batch_injection_parameters["spin_2z"][0:batch_size],
        batch_injection_parameters["luminosity_distance"][0:batch_size],
        batch_injection_parameters["theta_jn"][0:batch_size],
        batch_injection_parameters["phase"][0:batch_size],
        frequency_array,
        reference_frequency,
    )

    hp.block_until_ready()
    hc.block_until_ready()
    warmup_time = time.time() - start
    print(f"Warm-up time (includes JIT): {warmup_time:.3f} s")

    # Timed run
    print("\n--- Timed run ---")
    t0 = time.time()
    for ii in range(N_batches):
        begin = ii * batch_size
        end = (ii + 1) * batch_size
        hp, hc = generate_xphm_batched(
            batch_injection_parameters["mass_1"][begin:end],
            batch_injection_parameters["mass_2"][begin:end],
            batch_injection_parameters["spin_1x"][begin:end],
            batch_injection_parameters["spin_1y"][begin:end],
            batch_injection_parameters["spin_1z"][begin:end],
            batch_injection_parameters["spin_2x"][begin:end],
            batch_injection_parameters["spin_2y"][begin:end],
            batch_injection_parameters["spin_2z"][begin:end],
            batch_injection_parameters["luminosity_distance"][begin:end],
            batch_injection_parameters["theta_jn"][begin:end],
            batch_injection_parameters["phase"][begin:end],
            frequency_array,
            reference_frequency,
        )

    hp.block_until_ready()
    hc.block_until_ready()
    exec_time = time.time() - t0
    print(f"Execution time for {N_injections} waveforms: {exec_time:.6f} s")
    print(f"Time per waveform: {exec_time / N_injections * 1000:.3f} ms")


if __name__ == "__main__":
    main()
