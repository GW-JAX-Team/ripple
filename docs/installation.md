# Installation

The simplest way to install ripple is through pip:

```bash
pip install rippleGW
```

This will install the latest stable release and its dependencies.
ripple is built on [JAX](https://github.com/jax-ml/jax).
By default, this installs the CPU version of JAX.
If you have an NVIDIA GPU, install the CUDA-enabled version:

```bash
pip install "rippleGW[cuda]"
```

If you want to install the latest version of ripple, you can clone this repo and install it locally:

```bash
git clone https://github.com/GW-JAX-Team/ripple.git
cd ripple
pip install -e .
```

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment.
After cloning the repository, run `uv sync` to create a virtual environment with all dependencies installed.

## Continuous-wave ephemeris files

The continuous-wave models (`ExactPulsarSignal`, `PulsarSignal`, `BinaryPulsarSignal`) need a JPL solar-system ephemeris to barycenter the signal, e.g. `earth00-40-DE405.dat.gz` (and `sun00-40-DE405.dat.gz` for the full and binary models).
**You don't need to fetch these yourself:** pass a standard LALPulsar name and ripple downloads and caches it on first use, in `$XDG_CACHE_HOME/ripplegw/ephemeris` (`$RIPPLEGW_CACHE_DIR` if set — point this at shared/project storage on HPC systems so every job reuses the same cache).
If you already have a copy (e.g. from an installed LALSuite), pass its path directly instead, with no network access.
Make sure your observation span lies within the file's coverage (the standard `earth00-40-*`/`sun00-40-*` files cover GPS years 2000–2040).
