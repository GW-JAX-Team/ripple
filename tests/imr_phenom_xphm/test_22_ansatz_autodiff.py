"""Unit tests for auto- and analytical diff for 22-mode phase ansatz"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from test_lal_sim_imr_phenom_x_pnr_internals import _build_precession_struct

from ripplegw.constants import MSUN
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_inspiral import (
    imr_phenom_x_inspiral_phase_22_ansatz,
    imr_phenom_x_inspiral_phase_22_ansatz_int,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_intermediate import (
    imr_phenom_x_intermediate_phase_22_ansatz,
    imr_phenom_x_intermediate_phase_22_ansatz_int,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals import (
    imr_phenom_x_get_phase_coefficients,
    imr_phenom_x_initialize_powers,
    imr_phenom_x_set_waveform_variables,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXPhaseCoefficientsDataClass,
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import (
    IMRPhenomXPrecessionDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_ringdown import (
    imr_phenom_x_ringdown_phase_22_ansatz,
    imr_phenom_x_ringdown_phase_22_ansatz_int,
)
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import IMRPhenomXPHMParameterDataClass

jax.config.update("jax_enable_x64", True)


def sample_structs() -> (
    tuple[IMRPhenomXWaveformDataClass, IMRPhenomXPrecessionDataClass, IMRPhenomXPhaseCoefficientsDataClass]
):
    """Fixture providing matched waveform and precession structs."""

    m1 = 20.0
    m2 = 5.0
    chi1l_in = 0.3
    chi2l_in = -0.25
    delta_f = 0.1
    f_ref = 20.0
    phi0 = 0.0
    f_min = 10.0
    f_max = 1024.0
    distance = 500.0
    inclination = 0.0

    _, powers_of_pi = imr_phenom_x_initialize_powers(jnp.pi)
    _, p_wf = imr_phenom_x_set_waveform_variables(
        m1_si=m1 * MSUN,
        m2_si=m2 * MSUN,
        chi1l_in=chi1l_in,
        chi2l_in=chi2l_in,
        delta_f=delta_f,
        f_ref=f_ref,
        phi0=phi0,
        f_min=f_min,
        f_max=f_max,
        distance=distance,
        inclination=inclination,
        lal_params=IMRPhenomXPHMParameterDataClass(),
        powers_of_lalpi=powers_of_pi,
    )
    p_prec = _build_precession_struct(p_wf)
    p_phase = IMRPhenomXPhaseCoefficientsDataClass()
    _, p_phase = imr_phenom_x_get_phase_coefficients(p_wf, p_phase)

    return p_wf, p_prec, p_phase


class TestInspiralPhaseAutodiff:
    """Test suite comparing JAX autodiff of ansatz to analytical ansatz_int for inspiral phase."""

    sample_structs = sample_structs()

    def test_autodiff_matches_analytical_inspiral(self):
        """Test that autodiff of ansatz_int matches the analytical ansatz."""
        p_wf, _, p_phase = self.sample_structs

        mf = p_wf.m_f_ref

        _, powers_of_mf = imr_phenom_x_initialize_powers(mf)

        # Compute derivative using ansatz (phase derivative)
        phase_deriv_analytical = imr_phenom_x_inspiral_phase_22_ansatz(mf, powers_of_mf, p_phase)

        # Compute derivative using autodiff of phase_22_ansatz_int
        def phase_int_fn(f):
            _, powers = imr_phenom_x_initialize_powers(f)
            return imr_phenom_x_inspiral_phase_22_ansatz_int(f, powers, p_phase)

        phase_deriv_autodiff = jax.grad(phase_int_fn)(mf)

        assert jnp.isclose(
            phase_deriv_analytical, phase_deriv_autodiff, rtol=1e-10, atol=1e-12
        ), f"Mismatch: analytical={phase_deriv_analytical}, autodiff={phase_deriv_autodiff}"


class TestIntermediatePhaseAutodiff:
    """Test suite comparing JAX autodiff of ansatz to analytical ansatz_int for intermediate phase."""

    sample_structs = sample_structs()

    def test_autodiff_matches_analytical_intermediate(self):
        """Test that autodiff of ansatz_int matches the analytical ansatz."""
        p_wf, _, p_phase = self.sample_structs

        mf = p_wf.m_f_ref

        _, powers_of_mf = imr_phenom_x_initialize_powers(mf)

        # Compute derivative using ansatz (phase derivative)
        _, phase_deriv_analytical = imr_phenom_x_intermediate_phase_22_ansatz(mf, powers_of_mf, p_wf, p_phase)

        # Compute derivative using autodiff of phase_22_ansatz_int
        def phase_int_fn(f):
            _, powers = imr_phenom_x_initialize_powers(f)
            return imr_phenom_x_intermediate_phase_22_ansatz_int(f, powers, p_wf, p_phase)[1]

        phase_deriv_autodiff = jax.grad(phase_int_fn)(mf)

        assert jnp.isclose(
            phase_deriv_analytical, phase_deriv_autodiff, rtol=1e-10, atol=1e-12
        ), f"Mismatch: analytical={phase_deriv_analytical}, autodiff={phase_deriv_autodiff}"


class TestRingdownPhaseAutodiff:
    """Test suite comparing JAX autodiff of ansatz to analytical ansatz_int for ringdown phase."""

    sample_structs = sample_structs()

    def test_autodiff_matches_analytical_ringdown(self):
        """Test that autodiff of ansatz_int matches the analytical ansatz."""
        p_wf, _, p_phase = self.sample_structs

        mf = p_wf.m_f_ref

        _, powers_of_mf = imr_phenom_x_initialize_powers(mf)

        # Compute derivative using ansatz (phase derivative)
        _, phase_deriv_analytical = imr_phenom_x_ringdown_phase_22_ansatz(mf, powers_of_mf, p_wf, p_phase)

        # Compute derivative using autodiff of phase_22_ansatz_int
        def phase_int_fn(f):
            _, powers = imr_phenom_x_initialize_powers(f)
            return imr_phenom_x_ringdown_phase_22_ansatz_int(f, powers, p_wf, p_phase)[1]

        phase_deriv_autodiff = jax.grad(phase_int_fn)(mf)

        assert jnp.isclose(
            phase_deriv_analytical, phase_deriv_autodiff, rtol=1e-10, atol=1e-12
        ), f"Mismatch: analytical={phase_deriv_analytical}, autodiff={phase_deriv_autodiff}"


if __name__ == "__main__":
    pytest.main([__file__])
