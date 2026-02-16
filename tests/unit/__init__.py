"""Unit tests for ripple components.

This directory contains unit tests for individual ripple components, such as:
- Conversion functions (ms_to_Mc_eta, lambdas_to_lambda_tildes, etc.)
- Individual amplitude/phase components of waveform models
- Utility functions (get_eff_pads, get_match_arr, etc.)
- Waveform model internal utilities

These tests run in CI and should be fast (< 1 second each).

To add unit tests, create test files here with the naming convention test_*.py.
Each test function should be prefixed with test_ to be discovered by pytest.

Example:
    # tests/unit/test_conversions.py
    from ripplegw import ms_to_Mc_eta, Mc_eta_to_ms
    import jax.numpy as jnp

    def test_mass_conversion_roundtrip():
        m1, m2 = 1.4, 1.3
        Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
        m1_recovered, m2_recovered = Mc_eta_to_ms(jnp.array([Mc, eta]))
        assert jnp.allclose(m1, m1_recovered)
        assert jnp.allclose(m2, m2_recovered)
"""
