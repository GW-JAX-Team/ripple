# ripple 🌊

## A JAX-based package for differentiable gravitational-wave waveform generation

[![license](https://img.shields.io/badge/License-MIT-blue)](https://github.com/GW-JAX-Team/ripple/blob/main/LICENSE) [![coverage](https://img.shields.io/coveralls/github/GW-JAX-Team/ripple/main)](https://coveralls.io/github/GW-JAX-Team/ripple?branch=main) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GW-JAX-Team/ripple/main.svg)](https://results.pre-commit.ci/latest/github/GW-JAX-Team/ripple/main)

ripple is a JAX-based package for differentiable gravitational-wave waveform generation. By implementing waveform models as differentiable JAX functions, ripple enables gradient-based inference and runs natively on GPU, making it well-suited for use within modern probabilistic inference pipelines such as [Jim](https://github.com/GW-JAX-Team/jim).

**Supported waveforms:**

- TaylorF2
- IMRPhenomD
- IMRPhenomD_NRTidalv2
- IMRPhenomXAS
- IMRPhenomXAS_NRTidalv3
- IMRPhenomPv2
- IMRPhenomXPHM (MSA)

!!! warning
    ripple has not yet reached v1.0.0 and the API may change. Use at your own risk. Consider pinning to a specific version if you need API stability.

## Documentation

- **[Installation](installation.md)** — How to install ripple
- **[Quick Start](quickstart.md)** — A basic example to get started
- **[Tutorials](tutorials/index.md)** — Step-by-step guides and worked examples
- **[FAQ](FAQ.md)** — Answers to common questions
