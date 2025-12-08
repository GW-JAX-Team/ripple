from __future__ import annotations

import copy
import dataclasses

import jax
import jax.numpy as jnp
import pytest
from data_class_sample_data import WAVEFORM_DATA_CLASS_SAMPLE

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_angle_cases import (
    imr_phenom_x_initialize_msa_system,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import (
    IMRPhenomXPrecessionDataClass,
)

jax.config.update("jax_enable_x64", True)


def _build_waveform_struct() -> IMRPhenomXWaveformDataClass:
    """Create a waveform dataclass with self-consistent masses and spins."""

    wf_data = copy.deepcopy(WAVEFORM_DATA_CLASS_SAMPLE)

    m1 = 30.0
    m2 = 20.0
    m_tot = m1 + m2
    q_ge_1 = m1 / m2
    eta = (m1 * m2) / (m_tot * m_tot)
    delta = jnp.sqrt(1.0 - 4.0 * eta)

    chi1l = 0.30
    chi2l = -0.25
    chi_eff = ((m1 / m_tot) * chi1l) + ((m2 / m_tot) * chi2l)

    wf_data.update(
        {
            "m1": m1,
            "m2": m2,
            "m_tot": m_tot,
            "m1_si": m1,
            "m2_si": m2,
            "m_tot_si": m_tot,
            "q": q_ge_1,
            "eta": float(eta),
            "eta2": float(eta**2),
            "eta3": float(eta**3),
            "eta4": float(eta**4),
            "delta": float(delta),
            "chi1l": chi1l,
            "chi2l": chi2l,
            "chi_eff": float(chi_eff),
            "f_ref": 20.0,
            "phi0": 0.0,
        }
    )

    return IMRPhenomXWaveformDataClass(**wf_data)


def _build_precession_struct(p_wf: IMRPhenomXWaveformDataClass) -> IMRPhenomXPrecessionDataClass:
    """Create a precession dataclass compatible with the waveform struct."""

    m1 = p_wf.m1
    m2 = p_wf.m2
    eta = p_wf.eta
    eta2 = p_wf.eta2
    eta3 = p_wf.eta3
    eta4 = p_wf.eta4
    inveta = 1.0 / eta
    inveta2 = inveta * inveta
    inveta3 = inveta2 * inveta

    q_less_than_one = m2 / m1
    invqq = 1.0 / q_less_than_one

    chi1x = 0.10
    chi1y = 0.05
    chi1z = p_wf.chi1l
    chi2x = -0.07
    chi2y = 0.02
    chi2z = p_wf.chi2l

    s1_vec = jnp.array([chi1x * eta / q_less_than_one, chi1y * eta / q_less_than_one, chi1z * eta / q_less_than_one])
    s2_vec = jnp.array([chi2x * eta * q_less_than_one, chi2y * eta * q_less_than_one, chi2z * eta * q_less_than_one])

    s1_norm = float(jnp.linalg.norm(s1_vec))
    s2_norm = float(jnp.linalg.norm(s2_vec))
    s1_norm_sq = float(s1_norm**2)
    s2_norm_sq = float(s2_norm**2)

    dot_s1_l = float(s1_vec[2])
    dot_s2_l = float(s2_vec[2])
    dot_s1_s2 = float(jnp.inner(s1_vec, s2_vec))
    dot_s1_ln = float(dot_s1_l / s1_norm)
    dot_s2_ln = float(dot_s2_l / s2_norm)

    delta_qq = (1.0 - q_less_than_one) / (1.0 + q_less_than_one)
    delta2_qq = delta_qq * delta_qq
    delta3_qq = delta_qq * delta2_qq
    delta4_qq = delta_qq * delta3_qq

    pi_gm = 0.01

    p_prec = IMRPhenomXPrecessionDataClass()

    return dataclasses.replace(
        p_prec,
        imr_phenom_x_prec_version=223,
        imr_phenom_x_return_co_prec=0,
        eta=eta,
        eta2=eta2,
        eta3=eta3,
        eta4=eta4,
        inveta=inveta,
        inveta2=inveta2,
        inveta3=inveta3,
        chi1x=chi1x,
        chi1y=chi1y,
        chi1z=chi1z,
        chi2x=chi2x,
        chi2y=chi2y,
        chi2z=chi2z,
        pi_gm=pi_gm,
        two_pi_gm=2.0 * pi_gm,
        qq=q_less_than_one,
        inv_qq=invqq,
        delta_qq=delta_qq,
        delta2_qq=delta2_qq,
        delta3_qq=delta3_qq,
        delta4_qq=delta4_qq,
        s1x=float(s1_vec[0]),
        s1y=float(s1_vec[1]),
        s1z=float(s1_vec[2]),
        s2x=float(s2_vec[0]),
        s2y=float(s2_vec[1]),
        s2z=float(s2_vec[2]),
        s1_norm=s1_norm,
        s1_norm_2=s1_norm_sq,
        s2_norm=s2_norm,
        s2_norm_2=s2_norm_sq,
        dot_s1_l=dot_s1_l,
        dot_s2_l=dot_s2_l,
        dot_s1_s2=dot_s1_s2,
        dot_s1_ln=dot_s1_ln,
        dot_s2_ln=dot_s2_ln,
        l_hat_cos_theta=1.0,
        l_hat_phi=0.0,
        l_hat_theta=0.0,
    )


class TestIninitializeMsaSystem:
    """Test suite for imr_phenom_x_initialize_msa_system function."""

    @pytest.fixture
    def sample_structs(self) -> tuple[IMRPhenomXWaveformDataClass, IMRPhenomXPrecessionDataClass]:
        """Fixture providing matched waveform and precession structs."""

        p_wf = _build_waveform_struct()
        p_prec = _build_precession_struct(p_wf)
        return p_wf, p_prec

    def test_initialize_msa_system_success(self, sample_structs):
        """Initialization succeeds for a representative configuration."""

        p_wf, p_prec = sample_structs
        err, result = imr_phenom_x_initialize_msa_system(p_wf, p_prec, 5)
        err.throw()
        assert result is None

    def test_initialize_msa_system_invalid_prec_version(self, sample_structs):
        """Invalid precession version triggers the expected check failure."""

        p_wf, p_prec = sample_structs
        bad_prec = dataclasses.replace(p_prec, imr_phenom_x_prec_version=101)

        err, _ = imr_phenom_x_initialize_msa_system(p_wf, bad_prec, 5)
        with pytest.raises(Exception, match="MSA system requires"):
            err.throw()

    def test_initialize_msa_system_invalid_expansion_order(self, sample_structs):
        """Unsupported expansion order results in a check failure."""

        p_wf, p_prec = sample_structs

        err, _ = imr_phenom_x_initialize_msa_system(p_wf, p_prec, 0)
        with pytest.raises(Exception, match="Expansion order for MSA corrections"):
            err.throw()

    def test_initialize_msa_system_jit(self, sample_structs):
        """Function is JIT compatible when treating expansion_order as static."""

        p_wf, p_prec = sample_structs
        jit_fn = jax.jit(imr_phenom_x_initialize_msa_system, static_argnums=2)

        err, result = jit_fn(p_wf, p_prec, 5)
        err.throw()
        assert result is None

    def test_initialize_msa_system_jit_with_arrays(self, sample_structs):
        """Function is JIT compatible with batched waveform and precession structs."""

        p_wf, p_prec = sample_structs

        p_wf_batched = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), p_wf)
        p_prec_batched = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), p_prec)
        expansion_order = 5

        # Option 1: Use jax.lax.map with a fixed expansion_order
        @jax.jit
        def batched_initialize_msa_system_fixed_order(p_wf_batched, p_prec_batched, order):
            # jax.lax.map is like vmap but processes sequentially (still compiled efficiently)
            def process_single(args):
                p_wf_single, p_prec_single = args
                return imr_phenom_x_initialize_msa_system(p_wf_single, p_prec_single, order)

            return jax.lax.map(process_single, (p_wf_batched, p_prec_batched))

        # Test with a single expansion order for all batch elements
        err_batch, results_batch = batched_initialize_msa_system_fixed_order(
            p_wf_batched, p_prec_batched, expansion_order
        )

        for i in range(2):
            err_single = jax.tree_util.tree_map(lambda x: x[i], err_batch)
            err_single.throw()

        assert results_batch is None

    def test_initialize_msa_system_jit_with_different_orders(self, sample_structs):
        """Function with different expansion orders per batch element."""

        p_wf, p_prec = sample_structs

        p_wf_batched = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), p_wf)
        p_prec_batched = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), p_prec)

        # Use separate JIT-compiled functions for each order
        # This is the most practical approach for different expansion orders
        jit_fn_order5 = jax.jit(imr_phenom_x_initialize_msa_system, static_argnums=2)
        jit_fn_order4 = jax.jit(imr_phenom_x_initialize_msa_system, static_argnums=2)

        # Extract individual elements and call with appropriate order
        err_0, result_0 = jit_fn_order5(
            jax.tree_util.tree_map(lambda x: x[0], p_wf_batched),
            jax.tree_util.tree_map(lambda x: x[0], p_prec_batched),
            5,
        )
        err_1, result_1 = jit_fn_order4(
            jax.tree_util.tree_map(lambda x: x[1], p_wf_batched),
            jax.tree_util.tree_map(lambda x: x[1], p_prec_batched),
            4,
        )

        # Check errors
        err_0.throw()
        err_1.throw()

        assert result_0 is None
        assert result_1 is None


if __name__ == "__main__":
    pytest.main([__file__])
