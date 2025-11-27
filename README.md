# Ripple :ocean:

**A small `jax` package for differentiable and fast gravitational wave data analysis**

<a href="https://ripplegw.readthedocs.io/">
<img src="https://badgen.net/badge/Read/the doc/blue" alt="doc"/>
</a>
<a href="https://github.com/GW-JAX-Team/ripple/blob/main/LICENSE">
<img src="https://badgen.net/badge/License/MIT/blue" alt="license"/>
</a>
<a href='https://coveralls.io/github/GW-JAX-Team/ripple?branch=main'>
<img src='https://badgen.net/coveralls/c/github/GW-JAX-Team/ripple/main' alt='Coverage Status' />
</a> 

Ripple is a JAX-based package for differentiable and hardware-accelerated gravitational wave data analysis. It is maintained by the GW-JAX-Team organization and was originally developed by Thomas Edwards and Adam Coogan, with significant contributions from Kaze Wong and the community.

See the accompanying paper, [Edwards et al. (2024)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.064028), for more details. For questions or comments, please open an issue on the [GitHub repository](https://github.com/GW-JAX-Team/ripple).

# Installation

The simplest way to install Ripple is through pip:

```
pip install ripplegw
```

This will install the latest stable release and its dependencies.
Ripple is built on [JAX](https://github.com/google/jax).
By default, this installs the CPU version of JAX from [PyPI](https://pypi.org).
If you have a GPU and want to leverage hardware acceleration, install the CUDA-enabled version:

```
pip install ripplegw[cuda]
```

If you want to install the latest version of Ripple, you can clone this repo and install it locally:

```
git clone https://github.com/GW-JAX-Team/ripple.git
cd ripple
pip install -e .
```

**Note:** By default, Ripple uses float32 precision for improved performance. If you require float64 precision, add the following at the start of your script:

```python
from jax import config
config.update("jax_enable_x64", True)
```

See https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html for other common `jax` gotchas.

# Supported Waveforms

All waveforms have been extensively tested and match `lalsuite` implementations to machine precision across the full parameter space.

- **IMRPhenomXAS** (aligned spin)
- **IMRPhenomD** (aligned spin)
- **IMRPhenomPv2** (finalizing sampling validation)
- **TaylorF2** with tidal effects
- **IMRPhenomD_NRTidalv2** (verified for low spin: $\chi_1$ and $\chi_2$ < 0.05; higher spins require further testing)

# Usage

## Generating a Waveform

Generating waveforms with Ripple is straightforward. Below is an example using the IMRPhenomXAS model to compute the $h_+$ and $h_\times$ polarizations.

Start with the basic imports:

```python
import jax.numpy as jnp

from ripple.waveforms import IMRPhenomXAS
from ripple import ms_to_Mc_eta
```

And now we can just set the parameters and call the waveform!

```python
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
# Note: JAX does not raise index errors, so ensure the array is correctly ordered
theta_ripple = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination])

# Generate frequency grid
f_l = 24                 # Lower frequency bound (Hz)
f_u = 512                # Upper frequency bound (Hz)
del_f = 0.01             # Frequency resolution (Hz)
fs = jnp.arange(f_l, f_u, del_f)
f_ref = f_l              # Reference frequency

# Generate the waveform
hp_ripple, hc_ripple = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(fs, theta_ripple, f_ref)

# For better performance, we recommend JIT-compiling the waveform function.
# This avoids recompilation overhead when the frequency array length changes:

import jax

@jax.jit
def waveform(theta):
    return IMRPhenomXAS.gen_IMRPhenomXAS_hphc(fs, theta)
```

# Attribution

If you use Ripple in your research, please cite the accompanying paper:

```
@article{Edwards:2023sak,
    author = "Edwards, Thomas D. P. and Wong, Kaze W. K. and Lam, Kelvin K. H. and Coogan, Adam and Foreman-Mackey, Daniel and Isi, Maximiliano and Zimmerman, Aaron",
    title = "{Differentiable and hardware-accelerated waveforms for gravitational wave data analysis}",
    eprint = "2302.05329",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1103/PhysRevD.110.064028",
    journal = "Phys. Rev. D",
    volume = "110",
    number = "6",
    pages = "064028",
    year = "2024"
}
```
