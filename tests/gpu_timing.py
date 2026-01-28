"""
GPU test script for IMRPhenomXPHM waveform generation.
"""
import jax
import jax.numpy as jnp
import numpy as np
import time
jax.config.update("jax_enable_x64", False)
import bilby


# Check available devices
print("JAX devices:", jax.devices())
print("Default backend:", jax.default_backend())
for d in jax.devices():
    print(f"Device: {d.device_kind}, Platform: {d.platform}")

from ripplegw.waveforms import IMRPhenomXPHM
from ripplegw.constants import MSUN

N_WAVEFORMS = int(2e3)


# Waveform parameters
np.random.seed(42)
population = bilby.gw.prior.BBHPriorDict()
population['chirp_mass'] = bilby.core.prior.Uniform(10, 100)
population['mass_ratio'] = bilby.core.prior.Uniform(0.25, 1)
population.pop('mass_1')
population.pop('mass_2')

_injection_parameters = bilby.gw.conversion.generate_component_masses(population.sample(N_WAVEFORMS))
_injection_parameters['reference_frequency'] = np.ones(N_WAVEFORMS)*50

_injection_parameters = bilby.gw.conversion.generate_component_spins(_injection_parameters)

injection_parameters =  {key: jnp.array(value) for key, value in _injection_parameters.items()}

del _injection_parameters

print(injection_parameters.keys())



# Frequency settings
minimum_frequency = 20.0
maximum_frequency = 512.0
duration = 4.0
reference_frequency = 50.0


generate_xphm_batched = jax.vmap(
    IMRPhenomXPHM.generate_xphm,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None))

# Warm-up run (JIT compilation)
print("\n--- Warm-up run (JIT compilation) ---")
start = time.time()
hp, hc = generate_xphm_batched(
    injection_parameters['mass_1'], injection_parameters['mass_2'],
    injection_parameters['spin_1x'], injection_parameters['spin_1y'], injection_parameters['spin_1z'],
    injection_parameters['spin_2x'], injection_parameters['spin_2y'], injection_parameters['spin_2z'],
    injection_parameters['luminosity_distance'], injection_parameters['theta_jn'], injection_parameters['phase'],
    duration,
    minimum_frequency, maximum_frequency, reference_frequency)

hp.block_until_ready()
hc.block_until_ready()
warmup_time = time.time() - start
print(f"Warm-up time (includes JIT): {warmup_time:.3f} s")


# Timed run
print("\n--- Timed run ---")
start = time.time()
hp, hc = generate_xphm_batched(
    injection_parameters['mass_1'], injection_parameters['mass_2'],
    injection_parameters['spin_1x'], injection_parameters['spin_1y'], injection_parameters['spin_1z'],
    injection_parameters['spin_2x'], injection_parameters['spin_2y'], injection_parameters['spin_2z'],
    injection_parameters['luminosity_distance'], injection_parameters['theta_jn'], injection_parameters['phase'],
    duration, minimum_frequency, maximum_frequency, reference_frequency)

hp.block_until_ready()
hc.block_until_ready()
exec_time = time.time() - start
print(f"Execution time for {N_WAVEFORMS} waveforms: {exec_time:.6f} s")
print(f"Time per waveform: {exec_time/N_WAVEFORMS*1000:.3f} ms")

# Output info