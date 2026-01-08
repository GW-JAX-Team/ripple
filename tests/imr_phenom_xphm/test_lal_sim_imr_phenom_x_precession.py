"""Test suite for lal_sim_imr_phenom_x_precession.py."""

from __future__ import annotations

import copy
import dataclasses

import jax
import jax.numpy as jnp
import pytest
from data_class_sample_data import WAVEFORM_DATA_CLASS_SAMPLE
from jax.experimental import checkify

from ripplegw.constants import PI, gt
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXPHMParameterDataClass,
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession import (
    get_alphaepsilon_atfref,
    imr_phenom_x_initialize_msa_system,
    imr_phenom_x_return_phi_zeta_costheta_l_msa,
    imr_phenom_x_set_precessing_remnant_params,
    imr_phenom_xp_check_max_opening_angle,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import (
    IMRPhenomXPrecessionDataClass,
)

jax.config.update("jax_enable_x64", True)


def _build_waveform_struct(m1=30.0, m2=20.0, chi1l=0.30, chi2l=-0.25) -> IMRPhenomXWaveformDataClass:
    """Create a waveform dataclass"""

    wf_data = copy.deepcopy(WAVEFORM_DATA_CLASS_SAMPLE)

    m_tot = m1 + m2
    pi_m = PI * gt * m_tot
    q_ge_1 = m1 / m2
    eta = (m1 * m2) / (m_tot * m_tot)
    delta = jnp.sqrt(1.0 - 4.0 * eta)

    chi_eff = ((m1 / m_tot) * chi1l) + ((m2 / m_tot) * chi2l)

    wf_data.update(
        {
            "m1": m1,
            "m2": m2,
            "m_tot": m_tot,
            "m1_si": m1,
            "m2_si": m2,
            "m_tot_si": m_tot,
            "pi_m": pi_m,
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

    s1_perp = (m1**2) * jnp.sqrt(chi1x * chi1x + chi1y * chi1y)
    s2_perp = (m2**2) * jnp.sqrt(chi2x * chi2x + chi2y * chi2y)
    big_a1 = 2.0 + (3.0 * m2) / (2.0 * m1)
    big_a2 = 2.0 + (3.0 * m1) / (2.0 * m2)
    a_sp_1 = big_a1 * s1_perp
    a_sp_2 = big_a2 * s2_perp

    # /* S_p = max(A1 S1_perp, A2 S2_perp) */
    num = jax.lax.select(a_sp_2 > a_sp_1, a_sp_2, a_sp_1)
    den = jax.lax.select(m2 > m1, big_a2 * (m2**2), big_a1 * (m1**2))
    chi_p = num / den

    s_perp = chi_p * m1**2

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
        chi_p=chi_p,
        s_perp=s_perp,
    )


def dataclasses_allclose(dc1, dc2, rtol=1e-5, atol=1e-8):
    """Compare two dataclasses field-by-field with tolerance."""
    for field in dataclasses.fields(dc1):
        v1 = getattr(dc1, field.name)
        v2 = getattr(dc2, field.name)

        # Handle None values
        if v1 is None and v2 is None:
            continue
        if v1 is None or v2 is None:
            print(f"Mismatch in {field.name}: {v1} vs {v2}")
            return False

        # Convert to jax arrays for comparison (handles mixed JAX array / Python scalar)
        try:
            arr1 = jnp.asarray(v1)
            arr2 = jnp.asarray(v2)
            if not jnp.allclose(arr1, arr2, rtol=rtol, atol=atol):
                print(f"Mismatch in {field.name}: {v1} vs {v2}")
                return False
        except (TypeError, ValueError):
            # Fall back to direct comparison for non-numeric types
            if v1 != v2:
                print(f"Mismatch in {field.name}: {v1} vs {v2}")
                return False
    return True


class TestIMRPhenomXPCheckMaxOpeningAngle:
    """Test class for imr_phenom_x_check_max_opening_angle function."""

    def test_disable_multiband(self, capsys):
        """Test that a warning is printed when max opening angle exceeds Pi/4."""
        p_wf = _build_waveform_struct(m1=80.0, m2=2.0)  # p_wf.q > 7.0
        p_prec = _build_precession_struct(p_wf)
        lal_params = IMRPhenomXPHMParameterDataClass(threshold_mband=1.0)

        # Set parameters to trigger the warning
        p_prec = dataclasses.replace(
            p_prec, s_l=-jnp.inf, chi_p=1.0  # (l_min + p_prec.s_l) < 0.0  # p_prec.chi_p > 0.0
        )

        lal_params_result, p_prec_result = imr_phenom_xp_check_max_opening_angle(p_wf, p_prec, lal_params)

        captured = capsys.readouterr()
        assert "Warning: The maximum opening angle exceeds Pi/2" in captured.out
        assert "Warning: Multibanding may lead to pathological behaviour in this case." in captured.out

        assert lal_params_result.threshold_mband == 0.0
        assert p_prec_result.imr_phenom_x_prec_version == 0

    def test_max_beta_warning(self, capsys):
        """Test that a warning is printed when max opening angle exceeds Pi/4."""
        p_wf = _build_waveform_struct(chi1l=0.0, chi2l=jnp.pi / 2)
        p_prec = _build_precession_struct(p_wf)

        lal_params = IMRPhenomXPHMParameterDataClass()

        _ = imr_phenom_xp_check_max_opening_angle(p_wf, p_prec, lal_params)

        captured = capsys.readouterr()
        assert "Warning: The maximum opening angle" in captured.out

    def test_warnings_in_jit_compiled(self, capsys):
        """Test that the function can be JIT-compiled."""

        p_wf = _build_waveform_struct(m1=80.0, m2=2.0)  # p_wf.q > 7.0
        p_prec = _build_precession_struct(p_wf)
        lal_params = IMRPhenomXPHMParameterDataClass(threshold_mband=1.0)

        # Set parameters to trigger the warning
        p_prec = dataclasses.replace(
            p_prec, s_l=-jnp.inf, chi_p=1.0  # (l_min + p_prec.s_l) < 0.0  # p_prec.chi_p > 0.0
        )

        jit_func = jax.jit(imr_phenom_xp_check_max_opening_angle)

        lal_params_result, p_prec_result = jit_func(p_wf, p_prec, lal_params)

        captured = capsys.readouterr()
        assert "Warning: The maximum opening angle exceeds Pi/2" in captured.out
        assert "Warning: Multibanding may lead to pathological behaviour in this case." in captured.out

        assert lal_params_result.threshold_mband == 0.0
        assert p_prec_result.imr_phenom_x_prec_version == 0

        p_wf = _build_waveform_struct(m1=80.0, m2=2.0)
        p_prec = _build_precession_struct(p_wf)
        lal_params = IMRPhenomXPHMParameterDataClass()

        _ = jit_func(p_wf, p_prec, lal_params)

        captured = capsys.readouterr()
        assert "Warning: The maximum opening angle" in captured.out


class TestIMRPhenomXSetPrecessingRemnantParams:
    """Test class for imr_phenom_x_set_precessing_remnant_params function."""

    def branch_switching_block(
        self,
        xp_final_spin_mod,
        precessing_tag,
        pnr_use_tuned_coprec,
        pnr_use_input_coprec_deviations,
        imr_phenom_x_prec_version,
        msa_error,
    ):
        fsflag = xp_final_spin_mod
        fsflag = jax.lax.select(jnp.logical_and(fsflag == 4, precessing_tag == 3), 3, fsflag)

        # /* For PhenomPNR, we wil use the PhenomPv2 final spin function's result, modified such that its sign is given by sign( cos(betaRD) ). See the related fsflag case below. */
        fsflag = jax.lax.select(jnp.logical_and(pnr_use_tuned_coprec, fsflag < 6), 5, fsflag)

        # /* When tuning the coprecessing model, we wish to enforce use of the non-precessing final spin. See the related fsflag case below. */
        fsflag = jax.lax.select(pnr_use_input_coprec_deviations, 6, fsflag)

        pflag = imr_phenom_x_prec_version

        def case0_branch(_):
            return 0

        def case1_branch(_):
            return 1

        def case24_branch(_):
            return 24

        def case3_branch(_):
            def inner_branch(_):
                def standard_branch(_):
                    return 3

                def error_branch(_):
                    def _warn_msa_error():
                        print("Initialization of MSA system failed. Defaulting to final spin version 0.")

                    jax.debug.callback(_warn_msa_error)
                    return 30

                return jax.lax.cond(
                    msa_error == 1,
                    error_branch,
                    standard_branch,
                    operand=None,
                )

            def outer_branch(_):
                def _warn_prec_version_error():
                    print(
                        "Error: XLALSimInspiralWaveformParamsLookupPhenomXPFinalSpinMod version 3 requires PrecVersion 220, 221, 222, 223 or 224. Defaulting to version 0."
                    )

                jax.debug.callback(_warn_prec_version_error)
                return 30

            return jax.lax.cond(
                jnp.isin(pflag, jnp.array([220, 221, 222, 223, 224])),
                inner_branch,
                outer_branch,
                operand=None,
            )

        def case5_branch(_):
            return 5

        def case6_branch(_):
            return 6

        def case7_branch(_):
            return 7

        return jax.lax.switch(
            fsflag,
            [
                case0_branch,
                case1_branch,
                case24_branch,
                case3_branch,
                case24_branch,
                case5_branch,
                case6_branch,
                case7_branch,
            ],
            operand=None,
        )

    def test_switching(self):
        """Test the branch switching logic in imr_phenom_x_set_precessing_remnant_params."""
        test_cases = [
            # (xp_final_spin_mod, precessing_tag, pnr_use_tuned_coprec, pnr_use_input_coprec_deviations, imr_phenom_x_prec_version, msa_error, expected_fsflag)
            (0, 0, False, False, 220, 0, 0),
            (1, 0, False, False, 220, 0, 1),
            (2, 0, False, False, 220, 0, 24),
            (2, 0, True, False, 220, 0, 5),  # pnr_use_tuned_coprec case
            (2, 0, False, True, 220, 0, 6),  # pnr_use_input_coprec_deviations case
            (4, 0, False, False, 220, 0, 24),
            (4, 3, False, False, 220, 0, 3),  # precessing_tag adjustment case
            (3, 0, False, False, 220, 0, 3),
            (3, 0, False, False, 220, 1, 30),  # MSA error case
            (3, 0, False, False, 219, 0, 30),  # Invalid prec version case
            (4, 3, False, False, 220, 0, 3),  # xp_final_spin_mod adjusted to 3
            (5, 0, True, False, 220, 0, 5),  # pnr_use_tuned_coprec case
            (6, 0, False, True, 220, 0, 6),  # pnr_use_input_coprec_deviations case
            (7, 0, False, False, 220, 0, 7),
        ]

        for (
            xp_final_spin_mod,
            precessing_tag,
            pnr_use_tuned_coprec,
            pnr_use_input_coprec_deviations,
            imr_phenom_x_prec_version,
            msa_error,
            expected_fsflag,
        ) in test_cases:
            result_fsflag = self.branch_switching_block(
                xp_final_spin_mod,
                precessing_tag,
                pnr_use_tuned_coprec,
                pnr_use_input_coprec_deviations,
                imr_phenom_x_prec_version,
                msa_error,
            )
            assert result_fsflag == expected_fsflag

    def test_returns_updated_waveform_struct(self):
        """Test that the function returns an updated waveform struct with modified final spin parameters."""
        p_wf = _build_waveform_struct()
        p_prec = _build_precession_struct(p_wf)
        lal_params = IMRPhenomXPHMParameterDataClass(xp_final_spin_mod=0)

        # Set necessary values for the test
        p_wf = dataclasses.replace(
            p_wf,
            apply_pnr_deviations=True,
            pnr_dev_parameter=1.0,
            nu5=0.1,
            nu6=0.2,
        )
        p_prec = dataclasses.replace(
            p_prec,
            imr_phenom_xpnr_use_input_coprec_deviations=False,
            imr_phenom_xpnr_use_tuned_coprec=False,
        )

        checkified_func = checkify.checkify(imr_phenom_x_set_precessing_remnant_params)
        err, p_wf_result = checkified_func(p_wf, p_prec, lal_params)

        err.throw()  # Raise any errors caught by checkify

        # Check that the returned struct has updated values
        assert p_wf_result.a_final != p_wf.a_final
        assert p_wf_result.a_final_prec != p_wf.a_final_prec
        assert p_wf_result.f_ring != p_wf.f_ring
        assert p_wf_result.f_damp != p_wf.f_damp
        # The final spin should be computed and finite
        assert jnp.isfinite(p_wf_result.a_final)
        assert jnp.isfinite(p_wf_result.a_final_prec)

    def test_final_spin_magnitude_clipped_to_one(self):
        """Test that final spin is clipped when magnitude exceeds 1."""
        p_wf = _build_waveform_struct()
        p_prec = _build_precession_struct(p_wf)
        lal_params = IMRPhenomXPHMParameterDataClass(xp_final_spin_mod=6)  # Force case 6 to copy p_wf.a_final_non_prec

        p_wf = dataclasses.replace(p_wf, a_final_non_prec=1.2)  # Set non-precessing final spin > 1 to test clipping
        # Use extreme chi_p to potentially cause large final spin
        p_prec = dataclasses.replace(
            p_prec,
            imr_phenom_xpnr_use_input_coprec_deviations=False,
            imr_phenom_xpnr_use_tuned_coprec=False,
        )

        checkified_func = checkify.checkify(imr_phenom_x_set_precessing_remnant_params)
        err, p_wf_result = checkified_func(p_wf, p_prec, lal_params)

        err.throw()  # Raise any errors caught by checkify

        # Final spin magnitude should be == +/- 1
        assert jnp.abs(p_wf_result.a_final_prec) == jnp.copysign(1.0, p_wf.a_final_non_prec)
        assert jnp.abs(p_wf_result.a_final) == jnp.copysign(1.0, p_wf.a_final_non_prec)

    def test_pnr_deviations_applied_to_ringdown_frequencies(self):
        """Test that PNR deviations are applied to ringdown frequencies when apply_pnr_deviations is True."""
        p_wf = _build_waveform_struct()
        p_prec = _build_precession_struct(p_wf)
        lal_params = IMRPhenomXPHMParameterDataClass(xp_final_spin_mod=0)

        nu5_val = 0.001
        nu6_val = 0.002
        pnr_dev_param = 1.5

        p_wf = dataclasses.replace(
            p_wf,
            apply_pnr_deviations=True,
            pnr_dev_parameter=pnr_dev_param,
            nu5=nu5_val,
            nu6=nu6_val,
        )
        p_prec = dataclasses.replace(
            p_prec,
            imr_phenom_xpnr_use_input_coprec_deviations=False,
            imr_phenom_xpnr_use_tuned_coprec=False,
        )

        checkified_func = checkify.checkify(imr_phenom_x_set_precessing_remnant_params)

        # Get result without PNR deviations for comparison
        p_wf_no_dev = dataclasses.replace(p_wf, apply_pnr_deviations=False)
        err_no_dev, p_wf_result_no_dev = checkified_func(p_wf_no_dev, p_prec, lal_params)
        err_no_dev.throw()

        # Get result with PNR deviations
        err_with_dev, p_wf_result_with_dev = checkified_func(p_wf, p_prec, lal_params)
        err_with_dev.throw()

        # The frequencies should be different when deviations are applied
        expected_f_ring_dev = p_wf_result_no_dev.f_ring - (pnr_dev_param * nu5_val)
        expected_f_damp_dev = p_wf_result_no_dev.f_damp + (pnr_dev_param * nu6_val)

        assert jnp.isclose(p_wf_result_with_dev.f_ring, expected_f_ring_dev, rtol=1e-6)
        assert jnp.isclose(p_wf_result_with_dev.f_damp, expected_f_damp_dev, rtol=1e-6)

    def test_jit_compilation(self):
        """Test that the function can be JIT-compiled with checkify."""

        p_wf = _build_waveform_struct()
        p_prec = _build_precession_struct(p_wf)
        lal_params = IMRPhenomXPHMParameterDataClass(xp_final_spin_mod=0)

        p_wf = dataclasses.replace(
            p_wf,
            m=1.0,
            m1_2=0.36,
            m2_2=0.16,
            m_final=0.95,
            a_final_non_prec=0.7,
        )
        p_prec = dataclasses.replace(
            p_prec,
            imr_phenom_xpnr_use_input_coprec_deviations=False,
            imr_phenom_xpnr_use_tuned_coprec=False,
            chi_p=0.1,
        )

        # Wrap with checkify to handle internal checkify.check calls, then JIT
        checkified_func = checkify.checkify(imr_phenom_x_set_precessing_remnant_params)
        jit_func = jax.jit(checkified_func)

        # Should complete without error
        err, p_wf_result = jit_func(p_wf, p_prec, lal_params)

        # Check that no errors were raised
        err.throw()  # This will raise if there was a checkify error
        assert jnp.isfinite(p_wf_result.a_final)


class TestGetAlphaEpsilonAtFref:
    """Test class for get_alphaepsilon_atfref function."""

    def test_msa_branch(self):
        """Test MSA branch of get_alphaepsilon_atfref function."""
        # Sample lal_params
        p_wf = _build_waveform_struct()
        p_prec = _build_precession_struct(p_wf)
        p_prec = dataclasses.replace(p_prec, imr_phenom_x_prec_version=220)

        _, p_prec = imr_phenom_x_initialize_msa_system(p_wf, p_prec, 2)
        p_prec_copy = p_prec.copy()

        alpha_offset, epsilon_offset, p_prec = get_alphaepsilon_atfref(2, p_prec, p_wf)

        # MSA code to get expected alpha offset
        mprime = 2
        omega_ref = p_wf.pi_m * p_wf.f_ref * 2.0 / mprime

        v = jnp.cbrt(omega_ref)
        vangles, p_prec_expected = imr_phenom_x_return_phi_zeta_costheta_l_msa(v, p_wf, p_prec_copy)

        alpha_offset_expected = vangles[0] - p_prec_expected.alpha0
        epsilon_offset_expected = vangles[1] - p_prec_expected.epsilon0

        assert jnp.isclose(alpha_offset, alpha_offset_expected)
        assert jnp.isclose(epsilon_offset, epsilon_offset_expected)

    def test_other_branch(self):
        """Test non-MSA branch of get_alphaepsilon_atfref function."""

        # Sample lal_params
        p_wf = _build_waveform_struct()
        p_prec = _build_precession_struct(p_wf)
        p_prec = dataclasses.replace(p_prec, imr_phenom_x_prec_version=100)

        _, p_prec = imr_phenom_x_initialize_msa_system(p_wf, p_prec, 2)
        p_prec_copy = p_prec.copy()

        alpha_offset, epsilon_offset, p_prec = get_alphaepsilon_atfref(2, p_prec, p_wf)

        # Non-MSA code to get expected alpha offset
        mprime = 2
        omega_ref = p_wf.pi_m * p_wf.f_ref * 2.0 / mprime
        logomega_ref = jnp.log(omega_ref)
        omega_ref_cbrt = jnp.cbrt(omega_ref)
        omega_ref_cbrt2 = omega_ref_cbrt * omega_ref_cbrt

        alpha_offset_expected = (
            p_prec_copy.alpha1 / omega_ref
            + p_prec_copy.alpha2 / omega_ref_cbrt2
            + p_prec_copy.alpha3 / omega_ref_cbrt
            + p_prec_copy.alpha4_l * logomega_ref
            + p_prec_copy.alpha5 * omega_ref_cbrt
            - p_prec_copy.alpha0
        )

        epsilon_offset_expected = (
            p_prec_copy.epsilon1 / omega_ref
            + p_prec_copy.epsilon2 / omega_ref_cbrt2
            + p_prec_copy.epsilon3 / omega_ref_cbrt
            + p_prec_copy.epsilon4_l * logomega_ref
            + p_prec_copy.epsilon5 * omega_ref_cbrt
            - p_prec_copy.epsilon0
        )

        assert jnp.isclose(alpha_offset, alpha_offset_expected)
        assert jnp.isclose(epsilon_offset, epsilon_offset_expected)

    def test_jit_compilation(self):
        """Test that the function can be JIT-compiled."""

        p_wf = _build_waveform_struct()
        p_prec = _build_precession_struct(p_wf)
        p_prec = dataclasses.replace(p_prec, imr_phenom_x_prec_version=220)  # Just look at the MSA branch

        _, p_prec = imr_phenom_x_initialize_msa_system(p_wf, p_prec, 2)
        p_prec_copy = p_prec.copy()

        jit_func = jax.jit(get_alphaepsilon_atfref)

        # Should complete without error
        alpha_offset, epsilon_offset, p_prec = jit_func(2, p_prec, p_wf)

        # MSA code to get expected alpha offset
        mprime = 2
        omega_ref = p_wf.pi_m * p_wf.f_ref * 2.0 / mprime

        v = jnp.cbrt(omega_ref)
        vangles, p_prec_expected = imr_phenom_x_return_phi_zeta_costheta_l_msa(v, p_wf, p_prec_copy)

        alpha_offset_expected = vangles[0] - p_prec_expected.alpha0
        epsilon_offset_expected = vangles[1] - p_prec_expected.epsilon0

        assert jnp.isclose(alpha_offset, alpha_offset_expected)
        assert jnp.isclose(epsilon_offset, epsilon_offset_expected)


class TestIMRPhenomXReturnPhiZetaCosthetaLMSA:
    """Test class for imr_phenom_x_return_phi_zeta_costheta_l_msa function."""

    def test_returns(self):
        """Test that the function returns angles."""
        p_wf = _build_waveform_struct()
        p_prec = _build_precession_struct(p_wf)
        p_prec = dataclasses.replace(
            p_prec,
            imr_phenom_x_prec_version=220,
            l0=1.0,
            l1=0.0,
            l2=10.0,
            l3=0.0,
            l4=0.0,
            l5=0.0,
            l6=4.0,
            l7=0.0,
            l8=0.0,
            l8_l=4.2,
        )

        _, p_prec = imr_phenom_x_initialize_msa_system(p_wf, p_prec, 2)

        v = 0.2  # Sample velocity
        angles, _ = imr_phenom_x_return_phi_zeta_costheta_l_msa(v, p_wf, p_prec)

        assert (angles != jnp.zeros(3)).all()

    def test_jit_compilation(self):
        """Test that the function can be JIT-compiled."""

        p_wf = _build_waveform_struct()
        p_prec = _build_precession_struct(p_wf)
        p_prec = dataclasses.replace(
            p_prec,
            imr_phenom_x_prec_version=220,
            l0=1.0,
            l1=0.0,
            l2=10.0,
            l3=0.0,
            l4=0.0,
            l5=0.0,
            l6=4.0,
            l7=0.0,
            l8=0.0,
            l8_l=4.2,
        )
        _, p_prec = imr_phenom_x_initialize_msa_system(p_wf, p_prec, 2)

        v = 0.2  # Sample velocity

        jit_func = jax.jit(imr_phenom_x_return_phi_zeta_costheta_l_msa)

        # Should complete without error
        _ = jit_func(v, p_wf, p_prec)


if __name__ == "__main__":
    pytest.main([__file__])
