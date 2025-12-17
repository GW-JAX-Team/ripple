"""Unit tests for lal_sim_imr_phenom_x_pnr_deviations.py"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from lalsimulation import (
    SimIMRPhenomXCP_MU1_l2m2,
    SimIMRPhenomXCP_MU2_l2m2,
    SimIMRPhenomXCP_MU3_l2m2,
    SimIMRPhenomXCP_NU0_l2m2,
    SimIMRPhenomXCP_NU4_l2m2,
    SimIMRPhenomXCP_NU5_l2m2,
    SimIMRPhenomXCP_NU6_l2m2,
    SimIMRPhenomXCP_ZETA1_l2m2,
    SimIMRPhenomXCP_ZETA2_l2m2,
)

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_pnr_deviations import (
    xlal_sim_imr_phenom_xcp_mu1_l2m2,
    xlal_sim_imr_phenom_xcp_mu2_l2m2,
    xlal_sim_imr_phenom_xcp_mu3_l2m2,
    xlal_sim_imr_phenom_xcp_nu0_l2m2,
    xlal_sim_imr_phenom_xcp_nu4_l2m2,
    xlal_sim_imr_phenom_xcp_nu5_l2m2,
    xlal_sim_imr_phenom_xcp_nu6_l2m2,
    xlal_sim_imr_phenom_xcp_zeta1_l2m2,
    xlal_sim_imr_phenom_xcp_zeta2_l2m2,
)

jax.config.update("jax_enable_x64", True)


class Testl2m2PNRFunctions:
    """Unit tests for l2m2 PNR deviation functions."""

    @pytest.mark.parametrize(
        ("func_jax, func_lal"),
        [
            (xlal_sim_imr_phenom_xcp_mu1_l2m2, SimIMRPhenomXCP_MU1_l2m2),
            (xlal_sim_imr_phenom_xcp_mu2_l2m2, SimIMRPhenomXCP_MU2_l2m2),
            (xlal_sim_imr_phenom_xcp_mu3_l2m2, SimIMRPhenomXCP_MU3_l2m2),
            (xlal_sim_imr_phenom_xcp_nu4_l2m2, SimIMRPhenomXCP_NU4_l2m2),
            (xlal_sim_imr_phenom_xcp_nu5_l2m2, SimIMRPhenomXCP_NU5_l2m2),
            (xlal_sim_imr_phenom_xcp_nu6_l2m2, SimIMRPhenomXCP_NU6_l2m2),
            (xlal_sim_imr_phenom_xcp_zeta1_l2m2, SimIMRPhenomXCP_ZETA1_l2m2),
            (xlal_sim_imr_phenom_xcp_zeta2_l2m2, SimIMRPhenomXCP_ZETA2_l2m2),
            (xlal_sim_imr_phenom_xcp_nu0_l2m2, SimIMRPhenomXCP_NU0_l2m2),
        ],
    )
    def test_consistency_with_lal(
        self,
        func_jax,
        func_lal,
    ):
        """Test JAX PNR deviation functions against LAL implementations."""

        # Sample inputs
        theta = 0.4
        eta = 0.16
        a1 = 0.3

        # Compute outputs
        out_jax = func_jax(theta, eta, a1)
        out_lal = func_lal(theta, eta, a1)

        # Compare - convert JAX scalar to float for comparison
        assert jnp.allclose(float(out_jax), out_lal, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize(
        "func_jax",
        [
            xlal_sim_imr_phenom_xcp_mu1_l2m2,
            xlal_sim_imr_phenom_xcp_mu2_l2m2,
            xlal_sim_imr_phenom_xcp_mu3_l2m2,
            xlal_sim_imr_phenom_xcp_nu4_l2m2,
            xlal_sim_imr_phenom_xcp_nu5_l2m2,
            xlal_sim_imr_phenom_xcp_nu6_l2m2,
            xlal_sim_imr_phenom_xcp_zeta1_l2m2,
            xlal_sim_imr_phenom_xcp_zeta2_l2m2,
            xlal_sim_imr_phenom_xcp_nu0_l2m2,
        ],
    )
    def test_jit_compatibility(
        self,
        func_jax,
    ):
        """Test JAX PNR deviation functions are JIT compatible."""

        # Sample inputs
        theta = 0.4
        eta = 0.16
        a1 = 0.3

        # JIT compile function
        func_jit = jax.jit(func_jax)

        # Compute outputs
        out_jit = func_jit(theta, eta, a1)
        out_eager = func_jax(theta, eta, a1)

        # Compare
        assert jnp.allclose(out_jit, out_eager, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
