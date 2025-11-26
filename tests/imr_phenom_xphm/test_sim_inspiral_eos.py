"""Unit tests for sim_inspiral_eos.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ripplegw.waveforms.imr_phenom_xphm.sim_inspiral_eos import (
    xlal_sim_inspiral_eos_q_from_lambda,
)

try:
    import lalsimulation as lalsim

    HAS_LALSIM = True
except ImportError:
    HAS_LALSIM = False


class TestXLALSimInspiralEOSQFromLambda:
    """Test suite for xlal_sim_inspiral_eos_q_from_lambda function."""

    def test_below_tolerance(self):
        """Test that function returns 1.0 for lambda_ < tolerance."""
        lambda_val = 0.3  # Below tolerance of 0.5
        result = xlal_sim_inspiral_eos_q_from_lambda(lambda_val)
        assert jnp.isclose(result, 1.0), f"Expected 1.0 for lambda_={lambda_val}, got {result}"

    def test_at_tolerance(self):
        """Test that function handles lambda_ near tolerance."""
        lambda_val = 0.5  # At tolerance
        result = xlal_sim_inspiral_eos_q_from_lambda(lambda_val)
        # Should compute polynomial (not exactly 1.0)
        assert result > 0.0, f"Expected positive result for lambda_={lambda_val}, got {result}"

    def test_above_tolerance(self):
        """Test that function computes polynomial for lambda_ > tolerance."""
        lambda_val = 2.0  # Above tolerance
        result = xlal_sim_inspiral_eos_q_from_lambda(lambda_val)
        # For lambda=2, should compute polynomial
        assert result > 1.0, f"Expected result > 1.0 for lambda_={lambda_val}, got {result}"

    def test_large_lambda(self):
        """Test with large lambda value."""
        lambda_val = 100.0
        result = xlal_sim_inspiral_eos_q_from_lambda(lambda_val)
        assert jnp.isfinite(result), f"Expected finite result for lambda_={lambda_val}, got {result}"
        assert result > 1.0, "Expected result > 1.0 for large lambda"

    def test_array_input(self):
        """Test that function works with array input via vmap."""
        lambdas = jnp.array([0.1, 0.3, 0.5, 1.0, 2.0, 10.0])
        results = jax.vmap(xlal_sim_inspiral_eos_q_from_lambda)(lambdas)

        # Check that all results are finite and positive
        assert jnp.all(jnp.isfinite(results)), "Some results are not finite"
        assert jnp.all(results > 0.0), "Some results are not positive"

        # Check that results below tolerance are ~1.0
        below_tol = results[lambdas < 0.5]
        assert jnp.allclose(below_tol, 1.0, atol=1e-5), f"Below tolerance results: {below_tol}"

        # Check monotonicity: generally, q increases with lambda for large lambda
        above_tol = results[lambdas > 0.5]
        for i in range(len(above_tol) - 1):
            if above_tol[i + 1] < above_tol[i]:
                # Non-monotonic is okay near the transition, but let's check magnitudes
                assert jnp.abs(above_tol[i + 1] - above_tol[i]) < 0.5, "Large jump in results"

    def test_jit_compatibility(self):
        """Test that the function is JIT-compatible."""
        jit_q = jax.jit(xlal_sim_inspiral_eos_q_from_lambda)

        # Test with single scalar
        result_scalar = jit_q(2.0)
        assert jnp.isfinite(result_scalar), "JIT scalar result is not finite"

        # Test with vmap
        lambdas = jnp.array([0.1, 0.5, 2.0])
        jit_vmap_q = jax.jit(jax.vmap(xlal_sim_inspiral_eos_q_from_lambda))
        results_vmap = jit_vmap_q(lambdas)
        assert jnp.all(jnp.isfinite(results_vmap)), "JIT vmap results contain non-finite values"

    def test_differentiability(self):
        """Test that the function is differentiable."""
        # Gradient should exist for lambda_ > tolerance
        lambda_val = 2.0
        grad_q = jax.grad(xlal_sim_inspiral_eos_q_from_lambda)(lambda_val)
        assert jnp.isfinite(grad_q), f"Gradient is not finite at lambda_={lambda_val}"

    @pytest.mark.skipif(not HAS_LALSIM, reason="lalsimulation not available")
    def test_cross_check_with_lalsim(self):
        """Cross-check results against lalsimulation library."""
        # Test points to check
        test_lambdas = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]

        for lambda_val in test_lambdas:
            jax_result = xlal_sim_inspiral_eos_q_from_lambda(lambda_val)
            lal_result = lalsim.SimInspiralEOSQfromLambda(lambda_val)

            # Allow small relative tolerance due to numerical differences
            rel_tol = 1e-5
            abs_tol = 1e-8

            if jnp.isclose(jax_result, lal_result, rtol=rel_tol, atol=abs_tol):
                assert True, f"Results match for lambda_={lambda_val}"
            else:
                # Provide detailed error message
                error_msg = (
                    f"Mismatch at lambda_={lambda_val}: "
                    f"JAX={float(jax_result)}, LAL={lal_result}, "
                    f"relative_diff={abs(float(jax_result) - lal_result) / abs(lal_result)}"
                )
                pytest.fail(error_msg)

    @pytest.mark.skipif(not HAS_LALSIM, reason="lalsimulation not available")
    def test_cross_check_array_with_lalsim(self):
        """Cross-check array results against lalsimulation library."""
        lambdas = jnp.array([0.2, 0.5, 1.5, 3.0, 8.0])

        # Compute with JAX
        jax_results = jax.vmap(xlal_sim_inspiral_eos_q_from_lambda)(lambdas)

        # Compute with LAL
        lal_results = jnp.array([lalsim.SimInspiralEOSQfromLambda(float(lam)) for lam in lambdas])

        # Check element-wise
        rel_tol = 1e-5
        abs_tol = 1e-8

        for i, (jax_val, lal_val) in enumerate(zip(jax_results, lal_results)):
            rel_diff = abs(jax_val - lal_val) / abs(lal_val) if lal_val != 0 else 0
            assert jnp.isclose(jax_val, lal_val, rtol=rel_tol, atol=abs_tol), (
                f"Array element {i} (lambda={lambdas[i]}): "
                f"JAX={float(jax_val)}, LAL={lal_val}, "
                f"relative_diff={rel_diff}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
