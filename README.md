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

Ripple is now maintained by the GW-JAX-Team organization. Originally developed by Thomas Edwards and Adam Coogan, with significant contributions from Kaze Wong and the community. For questions or comments, please open an issue on the [GitHub repository](https://github.com/GW-JAX-Team/ripple).

# Installation

The simplest way to install the package is to do it through pip

```
pip install ripplegw
```

This will install the latest stable release and its dependencies.
Ripple is based on [Jax](https://github.com/google/jax).
By default, installing Ripple will automatically install Jax available on [PyPI](https://pypi.org).
By default this installs the CPU version of Jax. If you have a GPU and want to use it, you can install the GPU version of Jax by running:

```
pip install ripplegw[cuda]
```

If you want to install the latest version of Ripple, you can clone this repo and install it locally:

```
git clone https://github.com/GW-JAX-Team/ripple.git
cd ripple
pip install -e .
```

**Note:** By default we do not enable float64 in `jax` since we want to allow users to use float32 to improve performance.
If you require float64, please include the following code at the start of the script:

```python
from jax import config
config.update("jax_enable_x64", True)
```

See https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html for other common `jax` gotchas.

# Supported waveforms

Both waveforms have been tested extensively and match `lalsuite` implementations to machine precision across all the parameter space.

- IMRPhenomXAS (aligned spin)
- IMRPhenomD (aligned spin)
- IMRPhenomPv2 (Still finalizing sampling checks)
- TaylorF2 with tidal effects
- IMRPhenomD_NRTidalv2, verified for the low spin regime (chi1, chi2 < 0.05), further testing is required for higher spins

### Generating a waveform and its derivative

Generating a waveform is incredibly easy. Below is an example of calling the PhenomXAS waveform model
to get the h_+ and h_x polarizations of the waveform model

We start with some basic imports:

```python
import jax.numpy as jnp

from ripple.waveforms import IMRPhenomXAS
from ripple import ms_to_Mc_eta
```

And now we can just set the parameters and call the waveform!

```python
# Get a frequency domain waveform
# source parameters

m1_msun = 20.0 # In solar masses
m2_msun = 19.0
chi1 = 0.5 # Dimensionless spin
chi2 = -0.5
tc = 0.0 # Time of coalescence in seconds
phic = 0.0 # Time of coalescence
dist_mpc = 440 # Distance to source in Mpc
inclination = 0.0 # Inclination Angle

# The PhenomD waveform model is parameterized with the chirp mass and symmetric mass ratio
Mc, eta = ms_to_Mc_eta(jnp.array([m1_msun, m2_msun]))

# These are the parametrs that go into the waveform generator
# Note that JAX does not give index errors, so if you pass in the
# the wrong array it will behave strangely
theta_ripple = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination])

# Now we need to generate the frequency grid
f_l = 24
f_u = 512
del_f = 0.01
fs = jnp.arange(f_l, f_u, del_f)
f_ref = f_l

# And finally lets generate the waveform!
hp_ripple, hc_ripple = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(fs, theta_ripple, f_ref)

# Note that we have not internally jitted the functions since this would
# introduce an annoying overhead each time the user evaluated the function with a different length frequency array
# We therefore recommend that the user jit the function themselves to accelerate evaluations. For example:

import jax

@jax.jit
def waveform(theta):
    return IMRPhenomXAS.gen_IMRPhenomXAS_hphc(fs, theta)
```

# Attribution

If you used Ripple in your research, we would really appreciate it if you could cite the accompanying paper:

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
