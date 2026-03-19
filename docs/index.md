# Ripple :ocean:

**A lightweight JAX package for differentiable and fast gravitational wave data analysis**

[![doc](https://badgen.net/badge/Read/the%20doc/blue)](https://ripplegw.readthedocs.io/) [![license](https://badgen.net/badge/License/MIT/blue)](https://github.com/GW-JAX-Team/ripple/blob/main/LICENSE) [![coverage](https://badgen.net/coveralls/c/github/GW-JAX-Team/ripple/main)](https://coveralls.io/github/GW-JAX-Team/ripple?branch=main) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GW-JAX-Team/ripple/main.svg)](https://results.pre-commit.ci/latest/github/GW-JAX-Team/ripple/main)

ripple is a JAX-based package for differentiable and hardware-accelerated gravitational wave data analysis. It is maintained by the [GW-JAX-Team](https://github.com/GW-JAX-Team) and was originally developed by Thomas Edwards and Adam Coogan, with significant contributions from Kaze Wong and the community.

See the accompanying paper, [Edwards et al. (2024)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.064028), for more details.

## Key Features

- **Differentiable waveforms** — All waveform models are fully differentiable via JAX autodiff
- **Hardware acceleration** — Native GPU support through JAX
- **Validated implementations** — Extensively tested against `lalsuite` to machine precision
- **Lightweight** — Minimal dependencies, focused on waveform generation

## Supported Waveforms

- **IMRPhenomXAS** — Aligned spin
- **IMRPhenomD** — Aligned spin
- **IMRPhenomPv2** — Precessing spin
- **IMRPhenomXPHM** — Precessing spin with higher-order modes
- **TaylorF2** — With tidal effects
- **IMRPhenomD_NRTidalv2** — Tidal (verified for low spin)
- **IMRPhenomXAS_NRTidalv3** — Tidal

## Getting Started

1. Head to the **[Getting Started](quickstart.md)** page for installation and a basic example.
2. Explore the **[Tutorials](tutorials/Generating_waveforms.ipynb)** for detailed waveform generation and validation notebooks.
3. Check the **[API Reference](api/)** for full API documentation.
