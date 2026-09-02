# ripple 🌊

### A JAX-based package for differentiable gravitational-wave waveform generation

[![docs](https://img.shields.io/badge/docs-online-blue)](https://gw-jax-team.github.io/ripple/) [![license](https://img.shields.io/badge/License-MIT-blue)](https://github.com/GW-JAX-Team/ripple/blob/main/LICENSE) [![coverage](https://img.shields.io/coveralls/github/GW-JAX-Team/ripple/main)](https://coveralls.io/github/GW-JAX-Team/ripple?branch=main) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GW-JAX-Team/ripple/main.svg)](https://results.pre-commit.ci/latest/github/GW-JAX-Team/ripple/main)

ripple is a JAX-based package for differentiable gravitational-wave waveform generation.
By implementing waveform models as differentiable JAX functions, ripple enables gradient-based inference and runs natively on GPU, making it well-suited for use within modern probabilistic inference pipelines such as [Jim](https://github.com/GW-JAX-Team/jim).

**Supported waveforms:**

- TaylorF2 — fast post-Newtonian waveform, aligned spins, tidal-capable
- IMRPhenomXAS — aligned-spin binary black holes
- IMRPhenomXPHM — precessing spins with higher-order modes
- IMRPhenomXAS_NRTidalv3 — tidal corrections for binary neutron star mergers
- IMRPhenomXP_NRTidalv3 — precessing spins with tidal corrections
- SineGaussian — time-domain burst waveform
- PulsarSignal — continuous-wave signal from a spinning neutron star
- etc.

See the [Waveform Catalogue](https://gw-jax-team.github.io/ripple/stable/guides/catalogue/) for the full list of supported waveforms.

For a quick introduction, see the [Quick Start guide](https://gw-jax-team.github.io/ripple/stable/quickstart/).

> [!WARNING]
> ripple has not yet reached v1.0.0 and the API may change.
> Use at your own risk.
> Consider pinning to a specific version if you need API stability.

## Installation

The simplest way to install ripple is through pip:

```bash
pip install rippleGW
```

This will install the latest stable release and its dependencies.
ripple is built on [JAX](https://github.com/jax-ml/jax).
By default, this installs the CPU version of JAX.
If you have an NVIDIA GPU, install the CUDA-enabled version:

```bash
pip install rippleGW[cuda]
```

If you want to install the latest version of ripple, you can clone this repo and install it locally:

```bash
git clone https://github.com/GW-JAX-Team/ripple.git
cd ripple
pip install -e .
```

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment.
After cloning the repository, run `uv sync` to create a virtual environment with all dependencies installed.

## Origins

ripple was originally developed as [tedwards2412/ripple](https://github.com/tedwards2412/ripple) by [Thomas Edwards](https://github.com/tedwards2412) and others.
The original repository is no longer actively maintained; this fork is the active continuation of the project.

## Attribution

If you use ripple in your research, please cite the accompanying paper:

```bibtex
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
