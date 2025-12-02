"""Unit tests for lal_sim_inspiral.py."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_inspiral import (
    xlal_sim_inspiral_set_quad_mon_params_from_lambdas,
)
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import (
    IMRPhenomXPHMParameterDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.sim_inspiral_eos import (
    xlal_sim_inspiral_eos_q_from_lambda,
)


class TestXLALSimInspiralSetQuadMonParamsFromLambdas:
    """Test suite for xlal_sim_inspiral_set_quad_mon_params_from_lambdas function."""

    @pytest.fixture
    def sample_params(self):
        """Fixture for sample parameter dataclass."""
        return IMRPhenomXPHMParameterDataClass(
            lambda1=10.0,  # Tidal deformability for body 1
            lambda2=20.0,  # Tidal deformability for body 2
            d_quad_mon1=0.0,  # Will be computed
            d_quad_mon2=0.0,  # Will be computed
        )

    def test_no_update_when_quad_params_nonzero(self, sample_params):
        """Test that function doesn't update if d_quad_mon params are already nonzero."""
        # Create params with nonzero quadrupole params
        params = dataclasses.replace(sample_params, d_quad_mon1=0.5, d_quad_mon2=0.3)
        result = xlal_sim_inspiral_set_quad_mon_params_from_lambdas(params)

        assert result.d_quad_mon1 == 0.5, "d_quad_mon1 should not change"
        assert result.d_quad_mon2 == 0.3, "d_quad_mon2 should not change"

    def test_no_update_when_lambda_zero(self, sample_params):
        """Test that function doesn't update if lambdas are zero."""
        # Create params with zero lambdas
        params = dataclasses.replace(sample_params, lambda1=0.0, lambda2=0.0, d_quad_mon1=0.0, d_quad_mon2=0.0)
        result = xlal_sim_inspiral_set_quad_mon_params_from_lambdas(params)

        assert result.d_quad_mon1 == 0.0, "d_quad_mon1 should remain 0.0"
        assert result.d_quad_mon2 == 0.0, "d_quad_mon2 should remain 0.0"

    def test_update_quad_param1_when_lambda1_positive(self, sample_params):
        """Test that d_quad_mon1 is computed from lambda1 when conditions are met."""
        # lambda1 > 0, d_quad_mon1 == 0
        params = dataclasses.replace(sample_params, lambda1=10.0, d_quad_mon1=0.0, lambda2=0.0, d_quad_mon2=0.5)
        result = xlal_sim_inspiral_set_quad_mon_params_from_lambdas(params)

        # Expected: d_quad_mon1 = q(lambda1) - 1.0
        expected_q = xlal_sim_inspiral_eos_q_from_lambda(10.0)
        expected_d_quad_mon1 = expected_q - 1.0

        assert jnp.isclose(
            result.d_quad_mon1, expected_d_quad_mon1
        ), f"d_quad_mon1 should be {expected_d_quad_mon1}, got {result.d_quad_mon1}"
        assert result.d_quad_mon2 == 0.5, "d_quad_mon2 should remain unchanged"

    def test_update_quad_param2_when_lambda2_positive(self, sample_params):
        """Test that d_quad_mon2 is computed from lambda2 when conditions are met."""
        # lambda2 > 0, d_quad_mon2 == 0
        params = dataclasses.replace(sample_params, lambda2=20.0, d_quad_mon2=0.0, lambda1=0.0, d_quad_mon1=0.3)
        result = xlal_sim_inspiral_set_quad_mon_params_from_lambdas(params)

        # Expected: d_quad_mon2 = q(lambda2) - 1.0
        expected_q = xlal_sim_inspiral_eos_q_from_lambda(20.0)
        expected_d_quad_mon2 = expected_q - 1.0

        assert jnp.isclose(
            result.d_quad_mon2, expected_d_quad_mon2
        ), f"d_quad_mon2 should be {expected_d_quad_mon2}, got {result.d_quad_mon2}"
        assert result.d_quad_mon1 == 0.3, "d_quad_mon1 should remain unchanged"

    def test_update_both_quad_params(self, sample_params):
        """Test that both d_quad_mon1 and d_quad_mon2 are computed when both conditions are met."""
        # Both lambda1 and lambda2 > 0, both d_quad_mon params == 0
        params = dataclasses.replace(sample_params, lambda1=10.0, lambda2=20.0, d_quad_mon1=0.0, d_quad_mon2=0.0)
        result = xlal_sim_inspiral_set_quad_mon_params_from_lambdas(params)

        # Expected values
        expected_q1 = xlal_sim_inspiral_eos_q_from_lambda(10.0)
        expected_d_quad_mon1 = expected_q1 - 1.0
        expected_q2 = xlal_sim_inspiral_eos_q_from_lambda(20.0)
        expected_d_quad_mon2 = expected_q2 - 1.0

        assert jnp.isclose(
            result.d_quad_mon1, expected_d_quad_mon1
        ), f"d_quad_mon1 should be {expected_d_quad_mon1}, got {result.d_quad_mon1}"
        assert jnp.isclose(
            result.d_quad_mon2, expected_d_quad_mon2
        ), f"d_quad_mon2 should be {expected_d_quad_mon2}, got {result.d_quad_mon2}"

    def test_partial_update_lambda1_only(self, sample_params):
        """Test that only d_quad_mon1 is updated when only lambda1 condition is met."""
        params = dataclasses.replace(sample_params, lambda1=15.0, lambda2=0.0, d_quad_mon1=0.0, d_quad_mon2=0.0)
        result = xlal_sim_inspiral_set_quad_mon_params_from_lambdas(params)

        expected_q1 = xlal_sim_inspiral_eos_q_from_lambda(15.0)
        expected_d_quad_mon1 = expected_q1 - 1.0

        assert jnp.isclose(result.d_quad_mon1, expected_d_quad_mon1)
        assert result.d_quad_mon2 == 0.0, "d_quad_mon2 should remain 0.0"

    def test_jit_compatibility(self, sample_params):
        """Test that the function is JIT-compatible."""
        jit_fn = jax.jit(xlal_sim_inspiral_set_quad_mon_params_from_lambdas)

        # Test with sample params
        result_jit = jit_fn(sample_params)
        result_eager = xlal_sim_inspiral_set_quad_mon_params_from_lambdas(sample_params)

        # Results should match
        assert jnp.isclose(result_jit.d_quad_mon1, result_eager.d_quad_mon1)
        assert jnp.isclose(result_jit.d_quad_mon2, result_eager.d_quad_mon2)

    def test_jit_with_different_inputs(self, sample_params):
        """Test that JIT-compiled function works correctly with different inputs."""
        jit_fn = jax.jit(xlal_sim_inspiral_set_quad_mon_params_from_lambdas)

        # Test case 1: Both lambdas positive
        params1 = dataclasses.replace(sample_params, lambda1=10.0, lambda2=20.0, d_quad_mon1=0.0, d_quad_mon2=0.0)
        result1 = jit_fn(params1)
        assert result1.d_quad_mon1 != 0.0
        assert result1.d_quad_mon2 != 0.0

        # Test case 2: Only lambda1 positive
        params2 = dataclasses.replace(sample_params, lambda1=10.0, lambda2=0.0, d_quad_mon1=0.0, d_quad_mon2=0.0)
        result2 = jit_fn(params2)
        assert result2.d_quad_mon1 != 0.0
        assert result2.d_quad_mon2 == 0.0

        # Test case 3: Quadrupole params already set
        params3 = dataclasses.replace(sample_params, lambda1=10.0, lambda2=20.0, d_quad_mon1=0.5, d_quad_mon2=0.3)
        result3 = jit_fn(params3)
        assert result3.d_quad_mon1 == 0.5
        assert result3.d_quad_mon2 == 0.3

    def test_immutability_of_input(self, sample_params):
        """Test that the input dataclass is not modified."""
        params_original = dataclasses.replace(
            sample_params, lambda1=10.0, lambda2=20.0, d_quad_mon1=0.0, d_quad_mon2=0.0
        )
        # Store original values
        orig_d_quad_mon1 = params_original.d_quad_mon1
        orig_d_quad_mon2 = params_original.d_quad_mon2

        # Call function
        _ = xlal_sim_inspiral_set_quad_mon_params_from_lambdas(params_original)

        # Verify input wasn't modified (frozen dataclass should prevent this anyway)
        assert params_original.d_quad_mon1 == orig_d_quad_mon1
        assert params_original.d_quad_mon2 == orig_d_quad_mon2

    def test_consistency_with_eos_function(self, sample_params):
        """Test that results are consistent with the EOS function."""
        lambda1 = 15.0
        lambda2 = 25.0

        params = dataclasses.replace(sample_params, lambda1=lambda1, lambda2=lambda2, d_quad_mon1=0.0, d_quad_mon2=0.0)
        result = xlal_sim_inspiral_set_quad_mon_params_from_lambdas(params)

        # Manually compute expected values
        q1 = xlal_sim_inspiral_eos_q_from_lambda(lambda1)
        q2 = xlal_sim_inspiral_eos_q_from_lambda(lambda2)
        expected_d_quad_mon1 = q1 - 1.0
        expected_d_quad_mon2 = q2 - 1.0

        assert jnp.isclose(result.d_quad_mon1, expected_d_quad_mon1)
        assert jnp.isclose(result.d_quad_mon2, expected_d_quad_mon2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
