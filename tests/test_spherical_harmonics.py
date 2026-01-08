"""Unit tests for spherical harmonics helper functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lal import SpinWeightedSphericalHarmonic

from ripplegw.waveforms.spherical_harmonics import (
    compute_sminus2_l2,
    compute_sminus2_l3,
    compute_sminus2_l4,
)

jax.config.update("jax_enable_x64", True)


class TestSminus2l2:
    """Validation tests for spin -2 weighted spherical harmonics with l=2."""

    @pytest.mark.parametrize("m", [-2, -1, 0, 1, 2])
    def test_against_lal(self, m: int) -> None:
        """Values agree with LALSuite's implementation for representative inputs."""

        theta_samples = jnp.linspace(0.0, jnp.pi, num=10, dtype=jnp.float64)

        computed = compute_sminus2_l2(theta_samples, m)
        expected = jnp.array(
            [SpinWeightedSphericalHarmonic(theta, 0.0, -2, 2, m) for theta in np.asarray(theta_samples)],
            dtype=jnp.complex128,
        )

        np.testing.assert_allclose(computed, expected, rtol=1e-12, atol=1e-12)

    def test_jit_compatibility(self) -> None:
        """Function is compatible with JAX JIT compilation."""

        theta_samples = jnp.linspace(0.0, jnp.pi, num=10, dtype=jnp.float64)
        m = -1

        jit_fn = jax.jit(compute_sminus2_l2)
        jit_result = jit_fn(theta_samples, m)

        assert jnp.allclose(jit_result, compute_sminus2_l2(theta_samples, m))


class TestSminus2l3:
    """Validation tests for spin -2 weighted spherical harmonics with l=3."""

    @pytest.mark.parametrize("m", [-3, -2, -1, 0, 1, 2, 3])
    def test_against_lal(self, m: int) -> None:
        """Values agree with LALSuite's implementation for representative inputs."""

        theta_samples = jnp.linspace(0.0, jnp.pi, num=10, dtype=jnp.float64)

        computed = compute_sminus2_l3(theta_samples, m)
        expected = jnp.array(
            [SpinWeightedSphericalHarmonic(theta, 0.0, -2, 3, m) for theta in np.asarray(theta_samples)],
            dtype=jnp.complex128,
        )

        np.testing.assert_allclose(computed, expected, rtol=1e-12, atol=1e-12)

    def test_jit_compatibility(self) -> None:
        """Function is compatible with JAX JIT compilation."""

        theta_samples = jnp.linspace(0.0, jnp.pi, num=10, dtype=jnp.float64)
        m = 2

        jit_fn = jax.jit(compute_sminus2_l3)
        jit_result = jit_fn(theta_samples, m)

        assert jnp.allclose(jit_result, compute_sminus2_l3(theta_samples, m))


class TestSminus2l4:
    """Validation tests for spin -2 weighted spherical harmonics with l=4."""

    @pytest.mark.parametrize("m", [-4, -3, -2, -1, 0, 1, 2, 3, 4])
    def test_against_lal(self, m: int) -> None:
        """Values agree with LALSuite's implementation for representative inputs."""

        theta_samples = jnp.linspace(0.0, jnp.pi, num=10, dtype=jnp.float64)

        computed = compute_sminus2_l4(theta_samples, m)
        expected = jnp.array(
            [SpinWeightedSphericalHarmonic(theta, 0.0, -2, 4, m) for theta in np.asarray(theta_samples)],
            dtype=jnp.complex128,
        )

        np.testing.assert_allclose(computed, expected, rtol=1e-12, atol=1e-12)

    def test_jit_compatibility(self) -> None:
        """Function is compatible with JAX JIT compilation."""

        theta_samples = jnp.linspace(0.0, jnp.pi, num=10, dtype=jnp.float64)
        m = -3

        jit_fn = jax.jit(compute_sminus2_l4)
        jit_result = jit_fn(theta_samples, m)

        assert jnp.allclose(jit_result, compute_sminus2_l4(theta_samples, m))


if __name__ == "__main__":
    pytest.main([__file__])
