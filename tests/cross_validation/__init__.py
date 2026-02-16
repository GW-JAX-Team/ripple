"""Cross-validation tests comparing ripple waveforms against LALSuite.

This directory contains tests that compare ripple's waveform implementations
against LALSuite's reference implementations. These tests compute noise-weighted
mismatches and verify that ripple matches LAL to machine precision.

These tests are EXCLUDED from CI because they require LALSuite, which is
not always available. They should be run manually during development and
before releases.

To run these tests:
    uv run pytest tests/cross_validation -v

To run with specific parameters:
    uv run pytest tests/cross_validation::test_lal_mismatch -v -k imrphenomd
"""
