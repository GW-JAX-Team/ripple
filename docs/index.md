# ripple 🌊

## A JAX-based package for differentiable gravitational-wave waveform generation

ripple is a JAX-based package for differentiable gravitational-wave waveform generation.
By implementing waveform models as differentiable JAX functions, ripple enables gradient-based inference and runs natively on GPU, making it well-suited for use within modern probabilistic inference pipelines such as [Jim](https://github.com/GW-JAX-Team/jim).

**Supported waveforms:** a range of frequency- and time-domain compact-binary, burst, and continuous-wave (pulsar) models, covering aligned-spin, precessing, and tidal physics.
See the [Waveform Catalogue](guides/catalogue.md) for the full, always-up-to-date list with each model's parameters and capabilities.
The continuous-wave models additionally require a JPL ephemeris file (e.g. `earth00-40-DE405.dat.gz`); see the [Installation](installation.md#continuous-wave-ephemeris-files) page.

!!! warning
    ripple has not yet reached v1.0.0 and the API may change.
    Use at your own risk.
    Consider pinning to a specific version if you need API stability.

## Documentation

- **[Installation](installation.md)** — How to install ripple
- **[Quick Start](quickstart.md)** — A basic example to get started
- **[Guides](guides/index.md)** — Working with waveforms, parameters, JAX transformations, and the waveform catalogue
- **[Developer Guide](dev/index.md)** — Contributing to ripple, especially adding a new waveform
- **[FAQ](FAQ.md)** — Answers to common questions
- **[Citing ripple](citing.md)** — How to cite ripple
- **[Contributing to ripple](contributing.md)** — How to contribute to ripple
