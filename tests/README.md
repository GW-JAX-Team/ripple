# ripple test suite

This directory is maintained for contributors.
For the developer workflow, including the single local/Slurm/HTCondor cross-validation launcher, read [Test Coverage](../docs/dev/test_coverage.md) and [Run Reference Checks](../docs/dev/run_reference_checks.md).

## Layout

```
tests/
├── unit/                 # mirrors `src/ripplegw/`
├── integration/          # public behaviour of every registered waveform
├── cross_validation/     # reference tests that need the cross-validation group, and checks of their tooling
├── helpers/              # parameter, grid, metric, and configuration helpers
└── psds/                 # PSD data used by frequency-domain validation
```

The fast suite is CI-owned.
Cross-validation is selected explicitly by waveform through:

```bash
python -m tests.cross_validation.submit --scheduler local --waveform IMRPhenomD --n-samples 10
```

Use `--scheduler slurm` or `--scheduler condor` to submit the same test to HPC, and `--plots` to retain figures.
See the Development section of the docs for prerequisites, CW ephemerides, result locations, and troubleshooting.
