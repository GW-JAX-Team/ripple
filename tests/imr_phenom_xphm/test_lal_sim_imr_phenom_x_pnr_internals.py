"""Unit tests for lal_sim_imr_phenom_x_pnr_internals.py"""

from __future__ import annotations

import copy
import dataclasses

import jax.numpy as jnp
import pytest
from data_class_sample_data import WAVEFORM_DATA_CLASS_SAMPLE

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_pnr_internals import (
    imr_phenom_x_pnr_get_and_set_pnr_variables,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import (
    IMRPhenomXPrecessionDataClass,
)


def _build_waveform_struct() -> IMRPhenomXWaveformDataClass:
    """Create a waveform dataclass with self-consistent masses and spins."""

    wf_data = copy.deepcopy(WAVEFORM_DATA_CLASS_SAMPLE)

    m1 = 20.0
    m2 = 5.0
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


class TestImrPhenomXPNRGetAndSetPNRVariables:
    """Test imr_phenom_x_pnr_get_and_set_pnr_variables function."""

    @pytest.fixture
    def sample_structs(self) -> tuple[IMRPhenomXWaveformDataClass, IMRPhenomXPrecessionDataClass]:
        """Fixture providing matched waveform and precession structs."""

        p_wf = _build_waveform_struct()
        p_prec = _build_precession_struct(p_wf)
        return p_wf, p_prec

    def test_basic_output_structure(self, sample_structs):
        """Test that the function returns a dataclass with expected fields."""
        p_wf, p_prec = sample_structs
        p_prec = imr_phenom_x_pnr_get_and_set_pnr_variables(p_wf, p_prec)

        assert isinstance(p_prec, IMRPhenomXPrecessionDataClass)
        assert hasattr(p_prec, "chi_single_spin")
        assert hasattr(p_prec, "chi_single_spin_antisymmetric")
        assert hasattr(p_prec, "cos_theta_single_spin")
        assert hasattr(p_prec, "theta_antisymmetric")
        assert hasattr(p_prec, "cos_theta_final_single_spin")
        assert hasattr(p_prec, "pnr_inspiral_scaling")

    def test_prec_version_330_uses_evolved_spins(self, sample_structs):
        """Test that version 330 uses evolved spin values."""
        p_wf, p_prec_330 = sample_structs

        p_prec_330 = dataclasses.replace(
            p_prec_330,
            imr_phenom_x_prec_version=330,
            chi1x=0.1,
            chi1y=0.2,
            chi1z=0.3,
            chi2x=-0.1,
            chi2y=-0.2,
            chi2z=-0.3,
            chi1x_evolved=0.5,  # Different from non-evolved
            chi1y_evolved=0.6,
            chi2x_evolved=-0.5,
            chi2y_evolved=-0.6,
            chi1z_evolved=0.7,
            chi2z_evolved=-0.7,
            chi_p=0.2,
            pnr_q_window_lower=8.5,
            pnr_q_window_upper=12.0,
            pnr_chi_window_lower=0.85,
            pnr_chi_window_upper=1.2,
        )

        p_prec_330 = imr_phenom_x_pnr_get_and_set_pnr_variables(p_wf, p_prec_330)

        # The result should be different from using non-evolved spins
        # due to the conditional logic based on version 330
        print(p_prec_330.chi_single_spin)
        assert p_prec_330.chi_single_spin != 0.0
        assert p_prec_330.cos_theta_single_spin != 0.0

    def test_prec_version_non_330_uses_regular_spins(self, sample_structs):
        """Test that non-330 versions use regular spin values."""
        p_wf, p_prec_310 = sample_structs
        p_prec_310 = dataclasses.replace(
            p_prec_310,
            imr_phenom_x_prec_version=310,
            chi1x=0.1,
            chi1y=0.2,
            chi1z=0.3,
            chi2x=-0.1,
            chi2y=-0.2,
            chi2z=-0.3,
            chi1x_evolved=0.5,  # Different from non-evolved
            chi1y_evolved=0.6,
            chi2x_evolved=-0.5,
            chi2y_evolved=-0.6,
            chi1z_evolved=0.7,
            chi2z_evolved=-0.7,
            chi_p=0.2,
            pnr_q_window_lower=8.5,
            pnr_q_window_upper=12.0,
            pnr_chi_window_lower=0.85,
            pnr_chi_window_upper=1.2,
        )

        p_prec_310 = imr_phenom_x_pnr_get_and_set_pnr_variables(p_wf, p_prec_310)

        # Should use non-evolved spins
        assert p_prec_310.chi_single_spin >= 0.0

    def test_low_mass_ratio_behavior(self, sample_structs):
        """Test behavior for mass ratios <= 1.5."""
        p_wf, p_prec = sample_structs
        p_wf = dataclasses.replace(
            p_wf,
            m1=15.0,
            m2=10.0,
            m_tot=25.0,
            q=1.5,
            eta=(15.0 * 10.0) / (25.0 * 25.0),
        )

        p_prec = dataclasses.replace(
            p_prec,
            imr_phenom_x_prec_version=310,
            chi1x=0.1,
            chi1y=0.2,
            chi1z=0.3,
            chi2x=-0.1,
            chi2y=-0.2,
            chi2z=-0.3,
            chi_p=0.25,
            pnr_q_window_lower=8.5,
            pnr_q_window_upper=12.0,
            pnr_chi_window_lower=0.85,
            pnr_chi_window_upper=1.2,
        )

        p_prec = imr_phenom_x_pnr_get_and_set_pnr_variables(p_wf, p_prec)

        # For q <= 1.5, the function uses a blending formula
        assert p_prec.chi_single_spin >= 0.0
        assert -1.0 <= p_prec.cos_theta_single_spin <= 1.0

    def test_pnr_inspiral_scaling_outside_window(self, sample_structs):
        """Test that PNR inspiral scaling is set correctly outside the window."""
        # High mass ratio, outside window
        p_wf, p_prec = sample_structs

        m1 = 100.0
        m2 = 1.0
        p_wf = dataclasses.replace(
            p_wf,
            m1=m1,
            m2=m2,
            m_tot=m1 + m2,
            q=m1 / m2,
            eta=(m1 * m2) / ((m1 + m2) * (m1 + m2)),
        )

        p_prec = dataclasses.replace(
            p_prec,
            imr_phenom_x_prec_version=310,
            chi1x=0.1,
            chi1y=0.2,
            chi1z=0.3,
            chi2x=-0.1,
            chi2y=-0.2,
            chi2z=-0.3,
            chi_p=0.2,
            pnr_q_window_lower=8.5,
            pnr_q_window_upper=12.0,
            pnr_chi_window_lower=0.85,
            pnr_chi_window_upper=1.2,
        )

        p_prec = imr_phenom_x_pnr_get_and_set_pnr_variables(p_wf, p_prec)

        # High mass ratio (q > 12.0)
        assert p_prec.pnr_inspiral_scaling == 1

    def test_jax_jit_compatibility(self, sample_structs):
        """Test that the function is compatible with JAX JIT compilation."""
        import jax

        jitted_func = jax.jit(imr_phenom_x_pnr_get_and_set_pnr_variables)

        # Should not raise an error
        p_wf, p_prec = sample_structs
        p_prec = jitted_func(p_wf, p_prec)

        assert isinstance(p_prec, IMRPhenomXPrecessionDataClass)
        assert p_prec.chi_single_spin >= 0.0


if __name__ == "__main__":
    pytest.main([__file__])
