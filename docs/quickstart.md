# Getting Started

## Installation

The simplest way to install ripple is through pip:

```bash
pip install ripplegw
```

This will install the latest stable release and its dependencies.
ripple is built on [JAX](https://github.com/google/jax).
By default, this installs the CPU version of JAX. If you have a GPU, install the CUDA-enabled version:

```bash
pip install ripplegw[cuda]
```

For local development:

```bash
git clone https://github.com/GW-JAX-Team/ripple.git
cd ripple
pip install -e .
```

!!! note "Float precision"
    By default, ripple uses float32 precision for improved performance. If you require float64 precision, add the following at the start of your script:

    ```python
    from jax import config
    config.update("jax_enable_x64", True)
    ```

    See [JAX - The Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) for other common JAX gotchas.

## Supported Waveforms

All waveforms have been extensively tested and match `lalsuite` implementations to machine precision across the full parameter space.

| Waveform | Type | Notes |
|----------|------|-------|
| **IMRPhenomXAS** | Aligned spin | |
| **IMRPhenomD** | Aligned spin | |
| **IMRPhenomPv2** | Precessing spin | Finalizing sampling validation |
| **IMRPhenomXPHM** | Precessing spin, higher modes | |
| **TaylorF2** | Tidal effects | |
| **IMRPhenomD_NRTidalv2** | Tidal | Verified for low spin ($\chi_1, \chi_2 < 0.05$) |
| **IMRPhenomXAS_NRTidalv3** | Tidal | |

## Basic Example

Generating waveforms with ripple is straightforward. Below is an example using the IMRPhenomXAS model to compute the $h_+$ and $h_\times$ polarizations.

```python
import jax
import jax.numpy as jnp
from ripplegw import IMRPhenomXAS
from ripplegw.conversions import ms_to_Mc_eta

# Define source parameters
m1_msun = 20.0           # Primary mass (solar masses)
m2_msun = 19.0           # Secondary mass (solar masses)
chi1 = 0.5               # Primary dimensionless spin
chi2 = -0.5              # Secondary dimensionless spin
tc = 0.0                 # Time of coalescence (seconds)
phic = 0.0               # Phase at coalescence (radians)
dist_mpc = 440           # Luminosity distance (Mpc)
inclination = 0.0        # Inclination angle (radians)

# Convert to chirp mass and symmetric mass ratio
Mc, eta = ms_to_Mc_eta(jnp.array([m1_msun, m2_msun]))

# Construct parameter array
theta_ripple = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination])

# Generate frequency grid
f_l = 24                 # Lower frequency bound (Hz)
f_u = 512                # Upper frequency bound (Hz)
del_f = 0.01             # Frequency resolution (Hz)
fs = jnp.arange(f_l, f_u, del_f)
f_ref = f_l              # Reference frequency

# Generate the waveform
hp, hc = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(fs, theta_ripple, f_ref)
```

!!! tip "JIT compilation"
    For better performance, JIT-compile your waveform function:

    ```python
    @jax.jit
    def waveform(theta):
        return IMRPhenomXAS.gen_IMRPhenomXAS_hphc(fs, theta, f_ref)
    ```

## What's Next?

- **[Tutorials](tutorials/Generating_waveforms.ipynb)** — Detailed waveform generation and validation notebooks
- **[API Reference](api/)** — Full API documentation
