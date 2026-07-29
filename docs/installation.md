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

## Continuous-wave ephemeris data

The CW waveform models (`PulsarSignal`, `ExactPulsarSignal`, `BinaryPulsarSignal`) need a LALPulsar Earth/Sun ephemeris file, which isn't bundled with ripple.
By default this is downloaded automatically and cached locally the first time it's needed, so most users don't need to do anything.

If you're running on a machine without internet access -- e.g. an HPC compute node -- that automatic download will fail.
Run this once on a machine that does have internet access (e.g. the cluster's login node):

```bash
ripplegw-fetch-ephemeris
```

The cache directory (`$RIPPLEGW_CACHE_DIR`, or `~/.cache/ripplegw/ephemeris` by default) is normally on shared/home storage, so a later job on a node without internet access will find the file already cached and never attempt a download.
