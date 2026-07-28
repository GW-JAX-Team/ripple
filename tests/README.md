# ripple test suite

This directory is maintained for contributors.
For the developer workflow, including the single local/Slurm/HTCondor cross-validation launcher, read [Testing](../docs/dev/testing.md) and [Cross-validation tests](../docs/dev/cross_validation.md).

## Layout

```
tests/
├── unit/                 # focused registry and utility checks
├── integration/          # public behaviour of every registered waveform
├── cross_validation/     # reference tests and their launcher
├── helpers/              # parameter, grid, metric, and configuration helpers
└── psds/                 # PSD data used by frequency-domain validation
```

The fast suite is CI-owned.
Cross-validation is selected explicitly by waveform through:

```bash
python -m tests.cross_validation.submit --scheduler local --waveform IMRPhenomD --n-samples 10
```

Use `--scheduler slurm` or `--scheduler condor` to submit the same test to HPC, and `--plots` to retain figures.
See the developer guide for prerequisites, CW ephemerides, result locations, and troubleshooting.
