# Developer Guide

This guide covers contributing to ripple's internals — in particular, implementing a new waveform model.
For the process side of contributing (bug reports, pull requests, feature principles), see [Contributing](../contributing.md).

## Setting up

```bash
git clone https://github.com/GW-JAX-Team/ripple.git
cd ripple
uv sync --group test --group doc
uv run pre-commit install
```

`uv run ruff check src/` and `uv run pyright` should both pass cleanly before you open a PR; `pre-commit` runs them (and formatting) automatically on commit.

## In this guide

- **[Architecture](architecture.md)** — Why ripple is organised the way it is: the design goals, the registry, and the `Waveform` class hierarchy.
- **[Adding a Waveform](adding_a_waveform.md)** — Step-by-step: implement, register, and validate a new model, with a full worked example.
- **[Testing](testing.md)** — The two test tiers (CI, accuracy), how to run each, and what a new waveform needs.
- **[LAL Agreement](lal_agreement.md)** — What's known about the overlap between ripple and LALSuite for each supported waveform.
