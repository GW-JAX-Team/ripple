# ripple 🌊

## A JAX-based package for differentiable gravitational-wave waveform generation

ripple is a JAX-based package for differentiable gravitational-wave waveform generation.
By implementing waveform models as differentiable JAX functions, ripple enables gradient-based inference and runs natively on GPU, making it well-suited for use within modern probabilistic inference pipelines such as [Jim](https://github.com/GW-JAX-Team/jim).

See the [Waveform Catalogue](notes/catalogue.md) for the list of supported waveforms.

!!! warning
    ripple has not yet reached v1.0.0 and the API may change.
    Use at your own risk.
    Consider pinning to a specific version if you need API stability.

## Documentation

- **[Installation](installation.md)** — How to install ripple
- **[Quick Start](quickstart.md)** — A basic example to get started
- **[Tutorials](tutorials/index.md)** — Hands-on worked examples
- **[Guides](guides/index.md)** — Working with waveforms and JAX transformations
- **[How-to](how-to/index.md)** — Short recipes for specific tasks
- **[Technical Notes](notes/index.md)** — Parameters and conventions, and the waveform catalogue
- **[Development](dev/index.md)** — Contributing to ripple, especially adding a new waveform
- **[FAQ](FAQ.md)** — Answers to common questions
- **[Citing ripple](citing.md)** — How to cite ripple
- **[Contributing to ripple](contributing.md)** — How to contribute to ripple
