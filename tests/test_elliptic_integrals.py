"""Unit tests for elliptic integral helper functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ripplegw.gsl_ellint import ellipfinc, gsl_sf_elljac_e, rf

jax.config.update("jax_enable_x64", True)

scipy_special = pytest.importorskip("scipy.special")

ATOL = 1.0e-12
RTOL = 1.0e-12


class TestCarlsonRF:
    """Validation tests for Carlson's ``R_F`` implementation."""

    @pytest.fixture
    def sample_points(self) -> np.ndarray:
        """Provide representative (x, y, z) tuples within the valid domain."""

        return np.array(
            [
                [0.5, 1.0, 1.5],
                [0.2, 0.3, 0.4],
                [10.0, 12.5, 1.1],
                [1.0e-3, 2.0e-3, 3.0e-3],
                [2.5, 0.75, 1.1],
            ],
            dtype=np.float64,
        )

    def test_matches_scipy(self, sample_points: np.ndarray) -> None:
        """Values agree with SciPy's ``elliprf`` for representative inputs."""

        x, y, z = (jnp.asarray(component) for component in sample_points.T)
        result = np.asarray(rf(x, y, z))
        expected = scipy_special.elliprf(np.asarray(x), np.asarray(y), np.asarray(z))
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_jitted_matches_scipy(self, sample_points: np.ndarray) -> None:
        """JIT-compiled calls stay numerically aligned with SciPy."""

        x, y, z = (jnp.asarray(component) for component in sample_points.T)
        compiled = jax.jit(lambda a, b, c: rf(a, b, c))
        result = np.asarray(compiled(x, y, z))
        expected = scipy_special.elliprf(np.asarray(x), np.asarray(y), np.asarray(z))
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_batched_inputs(self, sample_points: np.ndarray) -> None:
        """Verify batched inputs produce correct results for each batch element."""

        x, y, z = (jnp.asarray(component) for component in sample_points.T)
        result = np.asarray(rf(x, y, z))

        # Compare each batch element individually
        for i in range(len(sample_points)):
            single_result = np.asarray(rf(x[i], y[i], z[i]))
            np.testing.assert_allclose(result[i], single_result, rtol=RTOL, atol=ATOL)

    def test_vmap_compatibility(self, sample_points: np.ndarray) -> None:
        """Ensure vmap produces consistent results with manual batching."""

        x, y, z = (jnp.asarray(component) for component in sample_points.T)

        # Manual batched call
        batch_result = rf(x, y, z)

        # vmap over first axis
        vmapped = jax.vmap(lambda xi, yi, zi: rf(xi, yi, zi))
        vmap_result = vmapped(x, y, z)

        np.testing.assert_allclose(np.asarray(batch_result), np.asarray(vmap_result), rtol=RTOL, atol=ATOL)


class TestIncompleteEllipticF:
    """Tests for the incomplete elliptic integral of the first kind."""

    @pytest.fixture
    def sample_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate valid ``phi``/``k`` combinations away from singularities."""

        phi = np.linspace(0.2, 1.2, num=6, dtype=np.float64)
        k = np.linspace(0.0, 0.85, num=6, dtype=np.float64)
        return phi, k

    def test_matches_scipy(self, sample_inputs: tuple[np.ndarray, np.ndarray]) -> None:
        """Results match SciPy's ``ellipkinc`` (parameter ``m = k**2``)."""

        phi, k = sample_inputs
        phi_jnp, k_jnp = jnp.asarray(phi), jnp.asarray(k)
        result = np.asarray(ellipfinc(phi_jnp, k_jnp))
        expected = scipy_special.ellipkinc(phi, k**2)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_jitted_matches_scipy(self, sample_inputs: tuple[np.ndarray, np.ndarray]) -> None:
        """JIT-compiled evaluations stay aligned with SciPy."""

        phi, k = sample_inputs
        compiled = jax.jit(lambda a, b: ellipfinc(a, b))
        result = np.asarray(compiled(jnp.asarray(phi), jnp.asarray(k)))
        expected = scipy_special.ellipkinc(phi, k**2)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_batched_inputs(self, sample_inputs: tuple[np.ndarray, np.ndarray]) -> None:
        """Verify batched inputs produce correct results for each batch element."""

        phi, k = sample_inputs
        phi_jnp, k_jnp = jnp.asarray(phi), jnp.asarray(k)
        result = np.asarray(ellipfinc(phi_jnp, k_jnp))

        # Compare each batch element individually
        for i in range(len(phi)):
            single_result = np.asarray(ellipfinc(phi_jnp[i], k_jnp[i]))
            np.testing.assert_allclose(result[i], single_result, rtol=RTOL, atol=ATOL)

    def test_vmap_compatibility(self, sample_inputs: tuple[np.ndarray, np.ndarray]) -> None:
        """Ensure vmap produces consistent results with manual batching."""

        phi, k = sample_inputs
        phi_jnp, k_jnp = jnp.asarray(phi), jnp.asarray(k)

        # Manual batched call
        batch_result = ellipfinc(phi_jnp, k_jnp)

        # vmap over first axis
        vmapped = jax.vmap(lambda p, k: ellipfinc(p, k))
        vmap_result = vmapped(phi_jnp, k_jnp)

        np.testing.assert_allclose(np.asarray(batch_result), np.asarray(vmap_result), rtol=RTOL, atol=ATOL)


class TestJacobiElliptic:
    """Expected comparisons against SciPy's Jacobi elliptic functions."""

    @pytest.fixture
    def sample_arguments(self) -> tuple[np.ndarray, np.ndarray]:
        """Return representative ``u`` and ``m`` values."""

        u = np.linspace(0.1, 1.0, num=5, dtype=np.float64)
        m = np.linspace(0.05, 0.95, num=5, dtype=np.float64)
        return u, m

    def test_matches_scipy(self, sample_arguments: tuple[np.ndarray, np.ndarray]) -> None:
        """Placeholder comparison against SciPy's ``ellipj`` implementation."""

        u, m = sample_arguments
        sn, cn, dn = gsl_sf_elljac_e(jnp.asarray(u[0]), jnp.asarray(m[0]))
        expected = scipy_special.ellipj(u[0], m[0])
        np.testing.assert_allclose(np.asarray(sn), expected[0], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.asarray(cn), expected[1], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.asarray(dn), expected[2], rtol=RTOL, atol=ATOL)

    def test_vectorized_matches_scipy(self, sample_arguments: tuple[np.ndarray, np.ndarray]) -> None:
        """Placeholder vectorized comparison against SciPy's ``ellipj``."""

        u, m = sample_arguments
        sn, cn, dn = gsl_sf_elljac_e(jnp.asarray(u), jnp.asarray(m))
        expected = scipy_special.ellipj(u, m)
        np.testing.assert_allclose(np.asarray(sn), expected[0], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.asarray(cn), expected[1], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.asarray(dn), expected[2], rtol=RTOL, atol=ATOL)

    def test_batched_inputs(self, sample_arguments: tuple[np.ndarray, np.ndarray]) -> None:
        """Verify batched inputs produce correct results for each batch element."""

        u, m = sample_arguments
        u_jnp, m_jnp = jnp.asarray(u), jnp.asarray(m)
        sn, cn, dn = gsl_sf_elljac_e(u_jnp, m_jnp)

        # Compare each batch element individually
        for i in range(len(u)):
            sn_single, cn_single, dn_single = gsl_sf_elljac_e(u_jnp[i], m_jnp[i])
            np.testing.assert_allclose(np.asarray(sn[i]), np.asarray(sn_single), rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(np.asarray(cn[i]), np.asarray(cn_single), rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(np.asarray(dn[i]), np.asarray(dn_single), rtol=RTOL, atol=ATOL)

    def test_vmap_compatibility(self, sample_arguments: tuple[np.ndarray, np.ndarray]) -> None:
        """Ensure vmap produces consistent results with manual batching."""

        u, m = sample_arguments
        u_jnp, m_jnp = jnp.asarray(u), jnp.asarray(m)

        # Manual batched call
        sn_batch, cn_batch, dn_batch = gsl_sf_elljac_e(u_jnp, m_jnp)

        # vmap over first axis
        vmapped = jax.vmap(lambda ui, mi: gsl_sf_elljac_e(ui, mi))
        sn_vmap, cn_vmap, dn_vmap = vmapped(u_jnp, m_jnp)

        np.testing.assert_allclose(np.asarray(sn_batch), np.asarray(sn_vmap), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.asarray(cn_batch), np.asarray(cn_vmap), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.asarray(dn_batch), np.asarray(dn_vmap), rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    pytest.main([__file__])
