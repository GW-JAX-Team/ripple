"""Unit tests for dataclasses defined in lal_sim_imr_phenom_x_internals_dataclass.py."""

from __future__ import annotations

import dataclasses

import jax
import pytest

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import IMRPhenomXWaveformDataClass


class TestIMRPhenomXWaveformDataClass:
    """Test suite for IMRPhenomXWaveformDataClass."""

    @pytest.fixture
    def sample_data(self):
        """Fixture for sample data to create the dataclass."""
        return {
            "debug": 0,
            "imr_phenom_x_inspiral_phase_version": 104,
            "imr_phenom_x_intermediate_phase_version": 105,
            "imr_phenom_x_ringdown_phase_version": 105,
            "imr_phenom_xpnr_use_tuned_coprec": 1,
            "imr_phenom_xpnr_use_tuned_coprec_33": 0,
            "imr_phenom_x_return_co_prec": 0,
            "phenom_x_only_return_phase": 0,
            "mu1": 0.0,
            "mu2": 0.0,
            "mu3": 0.0,
            "mu4": 0.0,
            "nu0": 0.0,
            "nu4": 0.0,
            "nu5": 0.0,
            "nu6": 0.0,
            "zeta1": 0.0,
            "zeta2": 0.0,
            "pnr_dev_parameter": 0.0,
            "pnr_window": 0.0,
            "apply_pnr_deviations": 0,
            "m1_si": 1.0,
            "m2_si": 1.0,
            "m_tot_si": 2.0,
            "m1": 1.0,
            "m2": 1.0,
            "m_tot": 2.0,
            "mc": 1.0,
            "q": 1.0,
            "eta": 0.25,
            "delta": 0.0,
            "chi1l": 0.0,
            "chi2l": 0.0,
            "chi_eff": 0.0,
            "chi_pn_hat": 0.0,
            "s_tot_r": 0.0,
            "dchi": 0.0,
            "dchi_half": 0.0,
            "sl": 0.0,
            "sigma_l": 0.0,
            "chi_tot_perp": 0.0,
            "chi_p": 0.0,
            "theta_ls": 0.0,
            "a1": 0.0,
            "m": 1.0,
            "m1_2": 1.0,
            "m2_2": 1.0,
            "lambda1": 0.0,
            "lambda2": 0.0,
            "quad_param1": 0.0,
            "quad_param2": 0.0,
            "kappa2_t": 0.0,
            "f_merger": 0.0,
            "eta2": 0.0625,
            "eta3": 0.015625,
            "eta4": 0.00390625,
            "chi1l2": 0.0,
            "chi1l3": 0.0,
            "chi2l2": 0.0,
            "chi2l3": 0.0,
            "chi1l2l": 0.0,
            "dphase0": 0.0,
            "amp0": 1.0,
            "amp_norm": 1.0,
            "f_meco": 0.0,
            "f_isco": 0.0,
            "beta_rd": 0.0,
            "f_ring22_prec": 0.0,
            "f_ring_eff_shift_divided_by_emm": 0.0,
            "f_ring_cp": 0.0,
            "f_ring": 0.0,
            "f_damp": 0.0,
            "f_ring21": 0.0,
            "f_damp21": 0.0,
            "f_ring33": 0.0,
            "f_damp33": 0.0,
            "f_ring32": 0.0,
            "f_damp32": 0.0,
            "f_ring44": 0.0,
            "f_damp44": 0.0,
            "f_min": 0.0,
            "f_max": 0.0,
            "m_f_max": 0.0,
            "f_max_prime": 0.0,
            "delta_f": 0.0,
            "delta_mf": 0.0,
            "f_cut": 0.0,
            "f_cut_def": 0.0,
            "f_ref": 0.0,
            "m_f_ref": 0.0,
            "m_sec": 0.0,
            "phi_ref_in": 0.0,
            "phi0": 0.0,
            "phi_f_ref": 0.0,
            "pi_m": 0.0,
            "v_ref": 0.0,
            "e_rad": 0.0,
            "a_final": 0.0,
            "m_final": 0.0,
            "a_final_prec": 0.0,
            "a_final_non_prec": 0.0,
            "distance": 1.0,
            "inclination": 0.0,
            "beta": 0.0,
            "lal_params": {},
            "pnr_single_spin": 0,
            "f_inspiral_align": 0.0,
            "xas_dphase_at_f_inspiral_align": 0.0,
            "xas_phase_at_f_inspiral_align": 0.0,
            "xhm_dphase_at_f_inspiral_align": 0.0,
            "xhm_phase_at_f_inspiral_align": 0.0,
            "imr_phenom_xpnr_force_xhm_alignment": 0,
        }

    def test_instantiation_and_field_access(self, sample_data):
        """Test that the dataclass can be instantiated and fields accessed."""
        p_wf = IMRPhenomXWaveformDataClass(**sample_data)
        assert p_wf.m1 == 1.0
        assert p_wf.eta == 0.25
        assert p_wf.lal_params == {}
        assert isinstance(p_wf.debug, int)
        assert isinstance(p_wf.m1_si, float)

    def test_immutability(self, sample_data):
        """Test that the dataclass is immutable (if frozen=True)."""
        p_wf = IMRPhenomXWaveformDataClass(**sample_data)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p_wf.m1 = 2.0  # Should fail if frozen

    def test_functional_update(self, sample_data):
        """Test updating fields with dataclasses.replace()."""
        p_wf = IMRPhenomXWaveformDataClass(**sample_data)
        updated = dataclasses.replace(p_wf, m1=2.0, eta=0.2)
        assert updated.m1 == 2.0
        assert updated.eta == 0.2
        assert p_wf.m1 == 1.0  # Original unchanged

    def test_jax_tree_operations(self, sample_data):
        """Test JAX tree flattening and unflattening."""
        p_wf = IMRPhenomXWaveformDataClass(**sample_data)
        flat, treedef = jax.tree_util.tree_flatten(p_wf)
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)
        assert reconstructed == p_wf

    def test_jit_compatibility(self, sample_data):
        """Test that the dataclass works with jax.jit."""

        @jax.jit
        def add_masses(p_wf):
            return p_wf.m1 + p_wf.m2

        p_wf = IMRPhenomXWaveformDataClass(**sample_data)
        result = add_masses(p_wf)
        assert result == 2.0

    def test_equality_and_copy(self, sample_data):
        """Test equality and copying."""
        p_wf1 = IMRPhenomXWaveformDataClass(**sample_data)
        p_wf2 = IMRPhenomXWaveformDataClass(**sample_data)
        assert p_wf1 == p_wf2
        copied = dataclasses.replace(p_wf1)  # Shallow copy
        assert copied == p_wf1
        assert copied is not p_wf1  # Different object
