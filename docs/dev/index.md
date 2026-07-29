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

`pre-commit` should pass cleanly before you open a PR.

## In this guide

- **[Architecture](architecture.md)** — Why ripple is organised the way it is: the design goals, the registry, and the `Waveform` class hierarchy.
- **[Adding a Waveform](adding_a_waveform.md)** — Step by step: implement, register, and validate a new model.
- **[Test Coverage](test_coverage.md)** — What the test suite protects, what it checks automatically, and what a new waveform needs.
- **[Run Reference Checks](run_reference_checks.md)** — Select one waveform and run its reference checks locally, on Slurm, or on HTCondor.
- **[Reference Comparisons and Limits](reference_comparisons_and_limits.md)** — How ripple is compared with external references for each supported waveform, including thresholds and known limits.
