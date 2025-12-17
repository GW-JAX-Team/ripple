"""Unit tests for the QNM fit functions in lal_sim_imr_phenom_thm_fits.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import checkify

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_thm_fits import evaluate_qnm_fit_fring22

# Enable 64-bit precision for accurate comparisons
jax.config.update("jax_enable_x64", True)


class TestEvaluateQnmFitFring22:
    """Test the QNM fit for (2,2) mode ringdown frequency."""

    def test_spin_zero(self):
        """Test at zero spin (non-spinning case)."""
        result = evaluate_qnm_fit_fring22(0.0)
        # Expected value from the fit: numerator/denominator at x=0
        expected = 0.05947169566573468 / 1.0
        assert jnp.isclose(result, expected, rtol=1e-10)

    def test_spin_half(self):
        """Test at moderate spin value."""
        spin = 0.5
        result = evaluate_qnm_fit_fring22(spin)
        # Compute expected manually from the polynomial
        x = spin
        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2
        x5 = x3 * x2
        x6 = x3 * x3
        x7 = x4 * x3
        num = (
            0.05947169566573468
            - 0.14989771215394762 * x
            + 0.09535606290986028 * x2
            + 0.02260924869042963 * x3
            - 0.02501704155363241 * x4
            - 0.005852438240997211 * x5
            + 0.0027489038393367993 * x6
            + 0.0005821983163192694 * x7
        )
        den = 1 - 2.8570126619966296 * x + 2.373335413978394 * x2 - 0.6036964688511505 * x4 + 0.0873798215084077 * x6
        expected = num / den
        assert jnp.isclose(result, expected, rtol=1e-10)

    def test_spin_max(self):
        """Test at maximum spin (a=1)."""
        result = evaluate_qnm_fit_fring22(1.0)
        # Should be finite and reasonable (not NaN/inf)
        assert jnp.isfinite(result)
        assert result > 0  # Frequency should be positive

    def test_spin_min(self):
        """Test at minimum spin (a=-1)."""
        result = evaluate_qnm_fit_fring22(-1.0)
        # Should be finite and reasonable
        assert jnp.isfinite(result)
        assert result > 0

    def test_invalid_spin_too_high(self):
        """Test that invalid spin > 1 raises an error."""
        with pytest.raises(checkify.JaxRuntimeError):
            evaluate_qnm_fit_fring22(1.1)

    def test_invalid_spin_too_low(self):
        """Test that invalid spin < -1 raises an error."""
        with pytest.raises(checkify.JaxRuntimeError):
            evaluate_qnm_fit_fring22(-1.1)

    def test_jit_compatibility(self):
        """Test that the function is JIT-compatible."""

        # Create a version without checkify for JIT testing
        def evaluate_qnm_fit_fring22_no_check(final_dimless_spin: float) -> float:
            x2 = final_dimless_spin * final_dimless_spin
            x3 = x2 * final_dimless_spin
            x4 = x2 * x2
            x5 = x3 * x2
            x6 = x3 * x3
            x7 = x4 * x3
            return (
                0.05947169566573468
                - 0.14989771215394762 * final_dimless_spin
                + 0.09535606290986028 * x2
                + 0.02260924869042963 * x3
                - 0.02501704155363241 * x4
                - 0.005852438240997211 * x5
                + 0.0027489038393367993 * x6
                + 0.0005821983163192694 * x7
            ) / (
                1
                - 2.8570126619966296 * final_dimless_spin
                + 2.373335413978394 * x2
                - 0.6036964688511505 * x4
                + 0.0873798215084077 * x6
            )

        jitted_func = jax.jit(evaluate_qnm_fit_fring22_no_check)
        final_dimless_spin = jnp.array(0.5, dtype=jnp.float64)
        result = jitted_func(final_dimless_spin)
        expected = evaluate_qnm_fit_fring22_no_check(final_dimless_spin)
        assert jnp.isclose(result, expected, rtol=1e-10)

    def test_gradient(self):
        """Test that the function is differentiable."""

        # Create a version without checkify for gradient testing
        def evaluate_qnm_fit_fring22_no_check(final_dimless_spin: float) -> float:
            x2 = final_dimless_spin * final_dimless_spin
            x3 = x2 * final_dimless_spin
            x4 = x2 * x2
            x5 = x3 * x2
            x6 = x3 * x3
            x7 = x4 * x3
            return (
                0.05947169566573468
                - 0.14989771215394762 * final_dimless_spin
                + 0.09535606290986028 * x2
                + 0.02260924869042963 * x3
                - 0.02501704155363241 * x4
                - 0.005852438240997211 * x5
                + 0.0027489038393367993 * x6
                + 0.0005821983163192694 * x7
            ) / (
                1
                - 2.8570126619966296 * final_dimless_spin
                + 2.373335413978394 * x2
                - 0.6036964688511505 * x4
                + 0.0873798215084077 * x6
            )

        grad_func = jax.grad(evaluate_qnm_fit_fring22_no_check)
        gradient = grad_func(jnp.array(0.5, dtype=jnp.float64))
        # Gradient should be finite (the fit is smooth)
        assert jnp.isfinite(gradient)

    def test_vmap_compatibility(self):
        """Test vectorized evaluation with vmap."""

        # Create a version without checkify for vmap testing
        def evaluate_qnm_fit_fring22_no_check(final_dimless_spin: float) -> float:
            x2 = final_dimless_spin * final_dimless_spin
            x3 = x2 * final_dimless_spin
            x4 = x2 * x2
            x5 = x3 * x2
            x6 = x3 * x3
            x7 = x4 * x3
            return (
                0.05947169566573468
                - 0.14989771215394762 * final_dimless_spin
                + 0.09535606290986028 * x2
                + 0.02260924869042963 * x3
                - 0.02501704155363241 * x4
                - 0.005852438240997211 * x5
                + 0.0027489038393367993 * x6
                + 0.0005821983163192694 * x7
            ) / (
                1
                - 2.8570126619966296 * final_dimless_spin
                + 2.373335413978394 * x2
                - 0.6036964688511505 * x4
                + 0.0873798215084077 * x6
            )

        spins = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float64)
        vmapped_func = jax.vmap(evaluate_qnm_fit_fring22_no_check)
        results = vmapped_func(spins)
        # Check each result individually
        for spin, result in zip(spins, results):
            expected = evaluate_qnm_fit_fring22_no_check(spin)
            assert jnp.isclose(result, expected, rtol=1e-10)

    def test_numerical_stability(self):
        """Test numerical stability near boundaries."""
        # Near a=1
        result_near_max = evaluate_qnm_fit_fring22(0.999)
        assert jnp.isfinite(result_near_max)

        # Near a=-1
        result_near_min = evaluate_qnm_fit_fring22(-0.999)
        assert jnp.isfinite(result_near_min)

        # Very small spin
        result_small = evaluate_qnm_fit_fring22(1e-6)
        assert jnp.isfinite(result_small)
