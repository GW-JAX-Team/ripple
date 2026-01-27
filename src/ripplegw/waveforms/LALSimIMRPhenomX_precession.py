import jax.numpy as jnp
import math
from typing import Any
from ..typing import Array
from ..constants import G, MSUN, C, MTSUN, GAMMA
import jax
from .spherical_harmonics import *
from .IMRPhenomXPHM_utils import *
from .LALSimInspiralSpinTaylor import XLALSimInspiralSpinTaylorPNEvolveOrbit
from dataclasses import dataclass, field
from jax_dataclasses import pytree_dataclass

from .LALSimIMRPhenomX_PNR_internals import (IMRPhenomX_PNR_HMInterpolationDeltaF, IMRPhenomX_PNR_GetAndSetPNRVariables, IMRPhenomX_PNR_GetAndSetCoPrecParams)
from .initialise_MSA_system import IMRPhenomX_Initialize_MSA_System, IMRPhenomX_Return_phi_zeta_costhetaL_MSA

from .LALSimIMRPhenomX_PNR_alpha import IMRPhenomX_PNR_precompute_alpha_coefficients
from .LALSimIMRPhenomX_PNR_beta import (
    IMRPhenomX_PNR_precompute_beta_coefficients,
    IMRPhenomX_PNR_BetaConnectionFrequencies
)

from .LALSimIMRPhenomXHM_internals import IMRPhenomXHM_GenerateRingdownFrequency

from .LALSimIMRPhenomX_PNR_internals import (XLALSimIMRPhenomXFinalSpin2017, XLALSimIMRPhenomXFinalMass2017)
from .LALSimIMRPhenomTHM_fits import evaluate_QNMfit_fring21

from .LALSimIMRPhenomX_qnm import (evaluate_QNMfit_fring22, evaluate_QNMfit_fdamp22)

@pytree_dataclass
class CommonConstants:
    sqrt2: float = 1.4142135623730951
    sqrt5: float = 2.23606797749978981
    sqrt6: float = 2.44948974278317788
    sqrt7: float = 2.64575131106459072
    sqrt10: float = 3.16227766016838
    sqrt14: float = 3.74165738677394133
    sqrt15: float = 3.87298334620741702
    sqrt70: float = 8.36660026534075563
    sqrt30: float = 5.477225575051661
    sqrt2p5: float = 1.58113883008419
    log16: float = 2.772588722239781
    power_of_lalpi_2: float = 9.869604401089358
    MAX_TOL_ATAN: float = 1.0e-15


@pytree_dataclass
class IMRPhenomXGetAndSetPrecessionVariables:
    """
    lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#af089ef2586c52b12016c0d791b176121

    Parameters
    ----------
    pWF : dict
        Some useful waveform parameters
    m1_SI : float
        Mass of object 1 (heavier one) in SI units
    m2_SI : float
        Mass of object 2 (lighter one) in SI units
    chi1x : float
        Object 1 spin x-component
    chi1y : float
        Object 1 spin y-component
    chi1z : float
        Object 1 spin z-component
    chi2x : float
        Object 2 spin x-component
    chi2y : float
        Object 2 spin y-component
    chi2z : float
        Object 2 spin z-component
    lalParams : dict
        LAL parameters
    debug_flag : bool
        Debug flag

    Note: We assume m1 > m2, q > 1, dm = m1 - m2 = delta = sqrt(1-4eta) > 0
    """
    # Input parameters
    pWF: dict
    m1_SI: float
    m2_SI: float
    chi1x: float
    chi1y: float
    chi1z: float
    chi2x: float
    chi2y: float
    chi2z: float
    lalParams: dict
    debug_flag: bool

    # Common constants
    common_constants: CommonConstants = field(default_factory=CommonConstants)

    # Computed attributes (initialized in __post_init__)
    m1: float = field(init=False)
    m2: float = field(init=False)
    M: float = field(init=False)
    q: float = field(init=False)

    # Mass powers
    m1_2: float = field(init=False)
    m1_3: float = field(init=False)
    m1_4: float = field(init=False)
    m1_5: float = field(init=False)
    m1_6: float = field(init=False)
    m1_7: float = field(init=False)
    m1_8: float = field(init=False)
    m2_2: float = field(init=False)

    # Eta (symmetric mass ratio) powers
    eta: float = field(init=False)
    eta2: float = field(init=False)
    eta3: float = field(init=False)
    eta4: float = field(init=False)
    eta5: float = field(init=False)
    eta6: float = field(init=False)

    # Delta powers
    delta: float = field(init=False)
    delta2: float = field(init=False)
    delta3: float = field(init=False)

    # Inverse eta
    inveta: float = field(init=False)
    inveta2: float = field(init=False)
    inveta3: float = field(init=False)
    inveta4: float = field(init=False)
    sqrt_inveta: float = field(init=False)

    # Effective spin
    chi_eff: float = field(init=False)

    # Gravitational constants
    twopiGM: float = field(init=False)
    piGM: float = field(init=False)

    # Spin norms
    chi1_norm: float = field(init=False)
    chi2_norm: float = field(init=False)

    # Dimensionful spins
    S1x: float = field(init=False)
    S1y: float = field(init=False)
    S1z: float = field(init=False)
    S1_norm: float = field(init=False)
    S2x: float = field(init=False)
    S2y: float = field(init=False)
    S2z: float = field(init=False)
    S2_norm: float = field(init=False)

    # Spin norm powers
    S1_norm_2: float = field(init=False)
    S2_norm_2: float = field(init=False)

    # Perpendicular spin components
    chi1_perp: float = field(init=False)
    chi2_perp: float = field(init=False)
    S1_perp: float = field(init=False)
    S2_perp: float = field(init=False)

    # Total perpendicular spin
    STot_perp: float = field(init=False)
    chiTot_perp: float = field(init=False)

    # Effective precessing spin parameters (Schmidt et al, PRD 91, 024043, 2015)
    A1: float = field(init=False)
    A2: float = field(init=False)
    ASp1: float = field(init=False)
    ASp2: float = field(init=False)
    chip: float = field(init=False)
    chi1L: float = field(init=False)
    chi2L: float = field(init=False)
    chi_p: float = field(init=False)
    phi0_aligned: float = field(init=False)
    SL: float = field(init=False)
    Sperp: float = field(init=False)
    pWF22AS: None = field(init=False, default=None)

    # Version-specific flags
    IMRPhenomXPrecVersion: int = field(init=False)
    PolarizationSymmetry: float = field(init=False)

    # MSA system fields (set by IMRPhenomX_Initialize_MSA_System)
    qq: float = field(init=False, default=0.0)
    invqq: float = field(init=False, default=0.0)
    delta_qq: float = field(init=False, default=0.0)
    delta2_qq: float = field(init=False, default=0.0)
    delta3_qq: float = field(init=False, default=0.0)
    delta4_qq: float = field(init=False, default=0.0)
    Lhat_cos_theta: float = field(init=False, default=0.0)
    Lhat_phi: float = field(init=False, default=0.0)
    Lhat_theta: float = field(init=False, default=0.0)
    S1_0: Any = field(init=False, default=None)
    S2_0: Any = field(init=False, default=None)
    v_0: float = field(init=False, default=0.0)
    v_0_2: float = field(init=False, default=0.0)
    L_0: Any = field(init=False, default=None)
    dotS1L: float = field(init=False, default=0.0)
    dotS2L: float = field(init=False, default=0.0)
    dotS1S2: float = field(init=False, default=0.0)
    dotS1Ln: float = field(init=False, default=0.0)
    dotS2Ln: float = field(init=False, default=0.0)
    constants_L: Any = field(init=False, default=None)
    Seff: float = field(init=False, default=0.0)
    Seff2: float = field(init=False, default=0.0)
    S_0: Any = field(init=False, default=None)
    J_0: Any = field(init=False, default=None)
    S_0_norm: float = field(init=False, default=0.0)
    S_0_norm_2: float = field(init=False, default=0.0)
    L_0_norm: float = field(init=False, default=0.0)
    J_0_norm: float = field(init=False, default=0.0)
    J_0_norm_2: float = field(init=False, default=0.0)
    L_0_norm_2: float = field(init=False, default=0.0)
    SAv2: float = field(init=False, default=0.0)
    SAv: float = field(init=False, default=0.0)
    invSAv2: float = field(init=False, default=0.0)
    invSAv: float = field(init=False, default=0.0)
    c1: float = field(init=False, default=0.0)
    c12: float = field(init=False, default=0.0)
    c1_over_eta: float = field(init=False, default=0.0)
    S1L_pav: float = field(init=False, default=0.0)
    S2L_pav: float = field(init=False, default=0.0)
    S1S2_pav: float = field(init=False, default=0.0)
    S1Lsq_pav: float = field(init=False, default=0.0)
    S2Lsq_pav: float = field(init=False, default=0.0)
    S1LS2L_pav: float = field(init=False, default=0.0)
    beta3: float = field(init=False, default=0.0)
    beta5: float = field(init=False, default=0.0)
    beta6: float = field(init=False, default=0.0)
    beta7: float = field(init=False, default=0.0)
    sigma4: float = field(init=False, default=0.0)
    a0_2: float = field(init=False, default=0.0)
    a0_3: float = field(init=False, default=0.0)
    a2_2: float = field(init=False, default=0.0)
    psi0: float = field(init=False, default=0.0)
    psi1: float = field(init=False, default=0.0)
    psi2: float = field(init=False, default=0.0)
    Delta: float = field(init=False, default=0.0)
    Omegaz0: float = field(init=False, default=0.0)
    Omegaz1: float = field(init=False, default=0.0)
    Omegaz2: float = field(init=False, default=0.0)
    Omegaz3: float = field(init=False, default=0.0)
    Omegaz4: float = field(init=False, default=0.0)
    Omegaz5: float = field(init=False, default=0.0)
    MSA_ERROR: int = field(init=False, default=0)
    Omegaz0_coeff: float = field(init=False, default=0.0)
    Omegaz1_coeff: float = field(init=False, default=0.0)
    Omegaz2_coeff: float = field(init=False, default=0.0)
    Omegaz3_coeff: float = field(init=False, default=0.0)
    Omegaz4_coeff: float = field(init=False, default=0.0)
    Omegaz5_coeff: float = field(init=False, default=0.0)
    Omegazeta0: float = field(init=False, default=0.0)
    Omegazeta1: float = field(init=False, default=0.0)
    Omegazeta2: float = field(init=False, default=0.0)
    Omegazeta3: float = field(init=False, default=0.0)
    Omegazeta4: float = field(init=False, default=0.0)
    Omegazeta5: float = field(init=False, default=0.0)
    Omegazeta0_coeff: float = field(init=False, default=0.0)
    Omegazeta1_coeff: float = field(init=False, default=0.0)
    Omegazeta2_coeff: float = field(init=False, default=0.0)
    Omegazeta3_coeff: float = field(init=False, default=0.0)
    Omegazeta4_coeff: float = field(init=False, default=0.0)
    Omegazeta5_coeff: float = field(init=False, default=0.0)
    phiz_0: float = field(init=False, default=0.0)
    zeta_0: float = field(init=False, default=0.0)
    zeta_polarization: float = field(init=False, default=0.0)


    def __post_init__(self):
        """Compute all derived quantities."""
        self._setup_version_flags()
        self._compute_masses()
        self._compute_mass_powers()
        self._update_pWF_dict()
        self._compute_eta_quantities()
        self._compute_gravitational_constants()
        self._compute_spin_quantities()
        self._compute_effective_spin_parameters()

        #self._validate_kerr_bound()
        #self.compute_evolved_spin_using_spintaylor() # Function that uses Spin Taylor approximantion to evolve spins
        
        self.compute_evolved_spin_using_msa()
        self.compute_and_set_spherical_harmonics()
        

    def _compute_masses(self):
        """Compute normalized masses and mass ratio."""
        object.__setattr__(self, 'm1', self.m1_SI / self.pWF['Mtot_SI'])
        object.__setattr__(self, 'm2', self.m2_SI / self.pWF['Mtot_SI'])
        object.__setattr__(self, 'M', self.m1 + self.m2)
        object.__setattr__(self, 'q', self.m1 / self.m2)

    def _compute_mass_powers(self):
        """Compute useful powers of masses."""
        # Powers of m1
        object.__setattr__(self, 'm1_2', self.m1 * self.m1)
        object.__setattr__(self, 'm1_3', self.m1 * self.m1_2)
        object.__setattr__(self, 'm1_4', self.m1 * self.m1_3)
        object.__setattr__(self, 'm1_5', self.m1 * self.m1_4)
        object.__setattr__(self, 'm1_6', self.m1 * self.m1_5)
        object.__setattr__(self, 'm1_7', self.m1 * self.m1_6)
        object.__setattr__(self, 'm1_8', self.m1 * self.m1_7)

        # Powers of m2
        object.__setattr__(self, 'm2_2', self.m2 * self.m2)

    def _update_pWF_dict(self):
        """Update pWF dictionary with computed mass values."""
        self.pWF['M'] = self.M
        self.pWF['m1_2'] = self.m1_2
        self.pWF['m2_2'] = self.m2_2

    def _compute_eta_quantities(self):
        """Compute eta (symmetric mass ratio) and delta related quantities."""
        # Powers of eta
        object.__setattr__(self, 'eta', self.pWF['eta'])
        object.__setattr__(self, 'eta2', self.eta * self.eta)
        object.__setattr__(self, 'eta3', self.eta * self.eta2)
        object.__setattr__(self, 'eta4', self.eta * self.eta3)
        object.__setattr__(self, 'eta5', self.eta * self.eta4)
        object.__setattr__(self, 'eta6', self.eta * self.eta5)

        # Delta in terms of q > 1
        object.__setattr__(self, 'delta', self.pWF['delta'])
        object.__setattr__(self, 'delta2', self.delta * self.delta)
        object.__setattr__(self, 'delta3', self.delta * self.delta2)

        # Inverse eta (cached for efficiency)
        object.__setattr__(self, 'inveta', 1.0 / self.eta)
        object.__setattr__(self, 'inveta2', 1.0 / self.eta2)
        object.__setattr__(self, 'inveta3', 1.0 / self.eta3)
        object.__setattr__(self, 'inveta4', 1.0 / self.eta4)
        object.__setattr__(self, 'sqrt_inveta', 1.0 / jnp.sqrt(self.eta))

        # Effective aligned spin
        object.__setattr__(self, 'chi_eff', self.pWF['chiEff'])

    def _compute_gravitational_constants(self):
        """Compute gravitational constants."""
        object.__setattr__(self, 'twopiGM', 2 * jnp.pi * G * (self.m1_SI + self.m2_SI) / C / C / C)
        object.__setattr__(self, 'piGM', jnp.pi * (self.m1_SI + self.m2_SI) * (G / C) / (C * C))

    def _compute_spin_quantities(self):
        """Compute all spin-related quantities."""
        # Spin norms
        object.__setattr__(self, 'chi1_norm', jnp.sqrt(self.chi1x * self.chi1x + self.chi1y * self.chi1y + self.chi1z * self.chi1z))
        object.__setattr__(self, 'chi2_norm', jnp.sqrt(self.chi2x * self.chi2x + self.chi2y * self.chi2y + self.chi2z * self.chi2z))

        # Dimensionful spins
        object.__setattr__(self, 'S1x', self.chi1x * self.m1_2)
        object.__setattr__(self, 'S1y', self.chi1y * self.m1_2)
        object.__setattr__(self, 'S1z', self.chi1z * self.m1_2)
        object.__setattr__(self, 'S1_norm', jnp.abs(self.chi1_norm) * self.m1_2)

        object.__setattr__(self, 'S2x', self.chi2x * self.m2_2)
        object.__setattr__(self, 'S2y', self.chi2y * self.m2_2)
        object.__setattr__(self, 'S2z', self.chi2z * self.m2_2)
        object.__setattr__(self, 'S2_norm', jnp.abs(self.chi2_norm) * self.m2_2)

        # Spin norm powers
        object.__setattr__(self, 'S1_norm_2', self.S1_norm * self.S1_norm)
        object.__setattr__(self, 'S2_norm_2', self.S2_norm * self.S2_norm)

        # Perpendicular spin components
        object.__setattr__(self, 'chi1_perp', jnp.sqrt(self.chi1x * self.chi1x + self.chi1y * self.chi1y))
        object.__setattr__(self, 'chi2_perp', jnp.sqrt(self.chi2x * self.chi2x + self.chi2y * self.chi2y))

        # Spin projections
        object.__setattr__(self, 'S1_perp', self.m1_2 * jnp.sqrt(self.chi1x * self.chi1x + self.chi1y * self.chi1y))
        object.__setattr__(self, 'S2_perp', self.m2_2 * jnp.sqrt(self.chi2x * self.chi2x + self.chi2y * self.chi2y))

        # Total perpendicular spin (norm of in-plane vector sum: Norm[S1perp + S2perp])
        object.__setattr__(self, 'STot_perp', jnp.sqrt((self.S1x + self.S2x) * (self.S1x + self.S2x) + (self.S1y + self.S2y) * (self.S1y + self.S2y)))

        # chiTot_perp (distinguishes from Sperp used in construction of chi_p)
        # For normalization, see Sec. IV D of arXiv:2004.06503
        object.__setattr__(self, 'chiTot_perp', self.STot_perp * (self.M * self.M) / self.m1_2)

        # Store chiTot_perp to pWF for use in XCP modifications (PNRUseTunedCoprec)
        self.pWF['chiTot_perp'] = self.chiTot_perp

    def _setup_version_flags(self):
        """Setup version-specific flags and configuration parameters."""
        # Get IMRPhenomX precession version from LAL dictionary
        object.__setattr__(self, 'IMRPhenomXPrecVersion', self.lalParams['IMRPhenomXPrecVersion'])

        # Convert version 300 to 223
        version = jnp.where(self.IMRPhenomXPrecVersion == 300, 223, self.IMRPhenomXPrecVersion)
        object.__setattr__(self, 'IMRPhenomXPrecVersion', version)

        # Calculate in-plane spin magnitude
        chi_in_plane = jnp.sqrt(
            self.chi1x * self.chi1x + self.chi1y * self.chi1y +
            self.chi2x * self.chi2x + self.chi2y * self.chi2y
        )

        # Default to NNLO angles if in-plane spins are negligible and version 330 is selected
        # The solutions would be dominated by numerical noise
        version = jnp.where(
            (chi_in_plane < 1e-6) & (self.IMRPhenomXPrecVersion == 330),
            102,
            self.IMRPhenomXPrecVersion
        )
        object.__setattr__(self, 'IMRPhenomXPrecVersion', version)

        # Default to NNLO if in-plane spins are negligible and SpinTaylor option is selected
        version = jnp.where(
            (chi_in_plane < 1e-7) & (self.IMRPhenomXPrecVersion//100 == 3),
            102,
            self.IMRPhenomXPrecVersion
        )
        object.__setattr__(self, 'IMRPhenomXPrecVersion', version)
        object.__setattr__(self, 'PolarizationSymmetry', 1.0)

        #Line 245-255
        # Disable tuned PNR angles, tuned coprec and mode asymmetries in low in-plane spin limit
        cond = (chi_in_plane<1e-7) & (self.lalParams['PNRUseTunedAngles'] == 1) & (self.pWF['PNR_SINGLE_SPIN']!=1)
        self.lalParams['PNRUseTunedAngles'] = jnp.where(cond, False, self.lalParams['PNRUseTunedAngles'])
        self.lalParams['AntisymmetricWaveform'] = jnp.where(cond, False, self.lalParams['AntisymmetricWaveform'])
        self.lalParams['PNRUseTunedCoprec'] = jnp.where(cond, False, self.lalParams['PNRUseTunedCoprec'])

    def _compute_effective_spin_parameters(self):
        """
        Calculate the effective precessing spin parameter.
        Reference: Schmidt et al, PRD 91, 024043, 2015
        Note: m1 > m2, so body 1 is the larger black hole
        """
        # Compute A1 and A2 coefficients
        object.__setattr__(self, 'A1', 2.0 + (3.0 * self.m2) / (2.0 * self.m1))
        object.__setattr__(self, 'A2', 2.0 + (3.0 * self.m1) / (2.0 * self.m2))
        object.__setattr__(self, 'ASp1', self.A1 * self.S1_perp)
        object.__setattr__(self, 'ASp2', self.A2 * self.S2_perp)

        # S_p = max(A1 S1_perp, A2 S2_perp)
        num = jnp.where(self.ASp2 > self.ASp1, self.ASp2, self.ASp1)
        den = jnp.where(self.m2 > self.m1, self.A2 * self.m2_2, self.A1 * self.m1_2)

        # chi_p = max(A1 * Sp1, A2 * Sp2) / (A_i * m_i^2) where i is the index of the larger BH
        object.__setattr__(self, 'chip', num / den)
        object.__setattr__(self, 'chi1L', self.chi1z)
        object.__setattr__(self, 'chi2L', self.chi2z)

        object.__setattr__(self, 'chi_p', self.chip)
        # Store chi_p to pWF (used in PNRUseTunedCoprec)
        self.pWF['chi_p'] = self.chi_p
        object.__setattr__(self, 'phi0_aligned', self.pWF['phi0'])

        # Effective (dimensionful) aligned spin
        object.__setattr__(self, 'SL', self.chi1L * self.m1_2 + self.chi2L * self.m2_2)

        # Effective (dimensionful) in-plane spin (m1 > m2)
        object.__setattr__(self, 'Sperp', self.chi_p * self.m1_2)

        # Initialize pWF22AS for SpinTaylor code
        object.__setattr__(self, 'pWF22AS', None)

    def _validate_kerr_bound(self):
        """Validate that spin magnitudes are within Kerr bound."""
        kerr_bound_flag = check_kerr_bound(
            self.lalParams['PNRUseTunedAngles'],
            self.pWF['PNR_SINGLE_SPIN'],
            self.chi1_norm,
            self.chi2_norm
        )



    def _initialize_mode_arrays_and_frequencies(self):
        """Initialize mode arrays and handle deltaF."""
        object.__setattr__(self, 'L_MAX_PNR', jnp.max(jnp.array(self.lalParams['ModeArray'])))
        object.__setattr__(self, 'M_MAX', jnp.max(jnp.array(self.lalParams['ModeArray'][:, 1])))

        # Handle deltaF == 0 case
        deltaMF = jnp.where(
            self.pWF['deltaF'] == 0,
            get_deltaF_from_wfstruct(self.pWF),
            -1  # FIXME
        )
        self.pWF['deltaMF'] = deltaMF

    def _compute_path1_parameters(self):
        """Compute integration parameters for PNRUseTunedAngles == False."""
        integration_buffer_path1 = jnp.where(self.pWF['deltaF'] > 0., 3. * self.pWF['deltaF'], 0.5)
        flow_path1 = (self.pWF['fMin'] - integration_buffer_path1) * 2 / self.M_MAX
        return integration_buffer_path1, flow_path1

    def _compute_path2_parameters(self):
        """Compute integration parameters for PNRUseTunedAngles == True."""
        flow_temp_path2, fmin_HM_inspiral = self._compute_path2_initial_frequencies()
        Mf_low_cut, MF_high_cut = self._compute_frequency_cutoffs()
        flow_path2_intermediate = self._compute_path2_intermediate_flow(
            flow_temp_path2, fmin_HM_inspiral, Mf_low_cut, MF_high_cut
        )
        integration_buffer_path2, flow_path2 = self._finalize_path2_parameters(flow_path2_intermediate)
        return integration_buffer_path2, flow_path2

    def _compute_path2_initial_frequencies(self):
        """Compute initial frequencies for path 2."""
        iStart_here_path2 = jnp.where(
            self.pWF['deltaF'] == 0.,
            0,
            jnp.floor(self.pWF['fMin'] / self.pWF['deltaF']).astype(int)
        )
        flow_temp_path2 = jnp.where(
            self.pWF['deltaF'] == 0.,
            0.,
            iStart_here_path2 * self.pWF['deltaF']
        )
        fmin_HM_inspiral = flow_temp_path2 * 2.0 / self.M_MAX
        return flow_temp_path2, fmin_HM_inspiral

    def _compute_frequency_cutoffs(self):
        """Compute frequency cutoffs using PNR coefficients."""
        # Temporarily set version to 223 for PNR variable computation
        precVersion_save = self.IMRPhenomXPrecVersion
        object.__setattr__(self, 'IMRPhenomXPrecVersion', 223)
        IMRPhenomX_PNR_GetAndSetPNRVariables(self, self.pWF) ## First precversion is set to 223

        alphaParams = IMRPhenomX_PNR_precompute_alpha_coefficients(self.pWF, self)
        betaParams = IMRPhenomX_PNR_precompute_beta_coefficients(self.pWF, self)
        Mf_beta_lower, Mf_beta_upper = IMRPhenomX_PNR_BetaConnectionFrequencies(betaParams)

        # Restore version
        object.__setattr__(self, 'IMRPhenomXPrecVersion', precVersion_save)

        # Compute cutoff frequencies
        Mf_alpha_upper = alphaParams.A4 / 3.0
        Mf_low_cut = (3.0 / 3.5) * Mf_alpha_upper
        MF_high_cut = Mf_beta_lower

        self.pWF['fCutDef'] = jnp.where(self.pWF['chiEff']>0.99, 0.33, 0.3)
        self.pWF['IMRPhenomXPNRUseTunedCoprec'] = False
        self.pWF['fRing'] = IMRPhenomXHM_GenerateRingdownFrequency(2, 2, self.pWF)

        # Adjust high cutoff
        MF_high_cut = jnp.where(
            jnp.logical_or(
                MF_high_cut > self.pWF['fCutDef'],
                MF_high_cut < 0.1 * self.pWF['fRING']
            ),
            self.pWF['fRING'],
            MF_high_cut
        )

        # Adjust low cutoff
        Mf_low_cut = jnp.where(
            jnp.logical_or(
                Mf_low_cut > self.pWF['fCutDef'],
                MF_high_cut < Mf_low_cut
            ),
            MF_high_cut / 2.0,
            Mf_low_cut
        )

        return Mf_low_cut, MF_high_cut

    def _compute_path2_intermediate_flow(self, flow_temp_path2, fmin_HM_inspiral, Mf_low_cut, MF_high_cut):
        """Compute intermediate flow frequency for path 2."""
        flow_alpha = XLALSimIMRPhenomXUtilsMftoHz(
            Mf_low_cut * 0.65 * self.M_MAX / 2.0,
            self.pWF['Mtot']
        )

        # Compute ringdown frequency adjustment
        Mf_RD_22 = self.pWF['fRING']
        Mf_RD_lm = IMRPhenomXHM_GenerateRingdownFrequency(self.L_MAX_PNR, self.M_MAX, self.pWF)

        fmin_HM_ringdown = XLALSimIMRPhenomXUtilsMftoHz(
            XLALSimIMRPhenomXUtilsHztoMf(flow_temp_path2, self.pWF['Mtot']) - (Mf_RD_lm - Mf_RD_22),
            self.pWF['Mtot']
        )
        else_branch_result = jnp.where(
            jnp.logical_and(fmin_HM_ringdown < fmin_HM_inspiral, fmin_HM_ringdown > 0.0),
            fmin_HM_ringdown,
            fmin_HM_inspiral
        )

        # Main conditional
        flow_path2_intermediate = jnp.where(
            flow_alpha < flow_temp_path2,
            fmin_HM_inspiral / 1.5,
            else_branch_result
        )

        return flow_path2_intermediate

    def _finalize_path2_parameters(self, flow_path2_intermediate):
        """Finalize path 2 integration buffer and flow."""
        pnr_interpolation_deltaf = IMRPhenomX_PNR_HMInterpolationDeltaF(
            flow_path2_intermediate, self.pWF, self
        )

        integration_buffer_path2 = 1.4 * pnr_interpolation_deltaf
        flow_path2 = jnp.where(
            flow_path2_intermediate - 2.0 * pnr_interpolation_deltaf < 0,
            flow_path2_intermediate / 2.0,
            flow_path2_intermediate - 2.0 * pnr_interpolation_deltaf
        )

        iStart_here_path2_final = jnp.floor(flow_path2 / pnr_interpolation_deltaf).astype(int)
        flow_path2 = iStart_here_path2_final * pnr_interpolation_deltaf

        return integration_buffer_path2, flow_path2

    def _select_integration_path(self, integration_buffer_path1, integration_buffer_path2,
                                  flow_path1, flow_path2):
        """Select between path 1 and path 2 based on PNRUseTunedAngles."""
        integration_buffer = jnp.where(
            self.lalParams['PNRUseTunedAngles'],
            integration_buffer_path2,
            integration_buffer_path1
        )
        flow = jnp.where(self.lalParams['PNRUseTunedAngles'], flow_path2, flow_path1)

        object.__setattr__(self, 'integration_buffer', integration_buffer)
        return integration_buffer, flow

    def _run_spintaylor_evolution(self, flow):
        """Run SpinTaylor evolution and return PN arrays."""
        PNarrays, fmin_integration = IMRPhenomX_InspiralAngles_SpinTaylor(
            self.chi1x, self.chi1y, self.chi1z,
            self.chi2x, self.chi2y, self.chi2z,
            flow, self.IMRPhenomXPrecVersion,
            self.pWF, self.lalParams
        )
        object.__setattr__(
            self, 'Mfmin_integration',
            XLALSimIMRPhenomXUtilsHztoMf(fmin_integration, self.pWF['Mtot'])
        )
        return PNarrays, fmin_integration

    def _extract_and_set_evolved_spins(self, PNarrays):
        """Extract evolved spins from PNarrays and apply rotation if needed."""
        chi1_evolved, chi2_evolved = self._compute_rotated_spins(PNarrays)
        self._set_evolved_spin_attributes(chi1_evolved, chi2_evolved)

    def _compute_rotated_spins(self, PNarrays):
        """Compute rotated spin vectors from PNarrays."""
        # Extract final values from PNarrays (last element)
        lenPN = len(PNarrays[0])

        chi1x_temp = PNarrays[1][lenPN-1]
        chi1y_temp = PNarrays[2][lenPN-1]
        chi1z_temp = PNarrays[3][lenPN-1]

        chi2x_temp = PNarrays[4][lenPN-1]
        chi2y_temp = PNarrays[5][lenPN-1]
        chi2z_temp = PNarrays[6][lenPN-1]

        Lx = PNarrays[7][lenPN-1]
        Ly = PNarrays[8][lenPN-1]
        Lz = PNarrays[9][lenPN-1]

        # Calculate rotation angles from angular momentum
        phi = jnp.arctan2(Ly, Lx)
        L_mag = jnp.sqrt(Lx*Lx + Ly*Ly + Lz*Lz)
        theta = jnp.arccos(Lz / L_mag)

        # Rotate chi1 vector
        _v1 = IMRPhenomX_rotate_z(-phi, jnp.array([chi1x_temp, chi1y_temp, chi1z_temp]))
        chi1_rotated = IMRPhenomX_rotate_y(-theta, _v1)

        # Rotate chi2 vector
        _v2 = IMRPhenomX_rotate_z(-phi, jnp.array([chi2x_temp, chi2y_temp, chi2z_temp]))
        chi2_rotated = IMRPhenomX_rotate_y(-theta, _v2)

        # Conditionally use rotated or original values based on IMRPhenomXPrecVersion
        is_version_330 = (self.IMRPhenomXPrecVersion == 330)

        chi1_evolved = jnp.array([
            jnp.where(is_version_330, chi1_rotated[0], self.chi1x),
            jnp.where(is_version_330, chi1_rotated[1], self.chi1y),
            jnp.where(is_version_330, chi1_rotated[2], self.chi1z)
        ])

        chi2_evolved = jnp.array([
            jnp.where(is_version_330, chi2_rotated[0], self.chi2x),
            jnp.where(is_version_330, chi2_rotated[1], self.chi2y),
            jnp.where(is_version_330, chi2_rotated[2], self.chi2z)
        ])

        return chi1_evolved, chi2_evolved

    def _set_evolved_spin_attributes(self, chi1_evolved, chi2_evolved):
        """Set evolved spin attributes."""
        object.__setattr__(self, 'chi1x_evolved', chi1_evolved[0])
        object.__setattr__(self, 'chi1y_evolved', chi1_evolved[1])
        object.__setattr__(self, 'chi1z_evolved', chi1_evolved[2])
        object.__setattr__(self, 'chi2x_evolved', chi2_evolved[0])
        object.__setattr__(self, 'chi2y_evolved', chi2_evolved[1])
        object.__setattr__(self, 'chi2z_evolved', chi2_evolved[2])



    def flag_222_223_twoPN_non_spinning_orbitan_angular_momentum(self):
        L0   = 1.0
        L1   = 0.0
        L2   = 3.0/2. + self.eta/6.0
        L3   = (-7*(self.chi1L + self.chi2L + self.chi1L*self.delta - self.chi2L*self.delta) + 5*(self.chi1L + self.chi2L)*self.eta)/6.
        L4   = (81 + (-57 + self.eta)*self.eta)/24.
        L5   = (-1650*(self.chi1L + self.chi2L + self.chi1L*self.delta - self.chi2L*self.delta) + 1336*(self.chi1L + self.chi2L)*self.eta + 511*(self.chi1L - self.chi2L)*self.delta*self.eta + 28*(self.chi1L + self.chi2L)*self.eta2)/600.
        L6   = (10935 + self.eta*(-62001 + 1674*self.eta + 7*self.eta2 + 2214*self.common_constants.power_of_lalpi_2))/1296.
        L7   = 0.0
        L8   = 0.0
        L8L = 0.0
        return L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L


    def compute_evolved_spin_using_msa(self):

        """
        What is compute_evolved_spin_using_msa function supposed to return?
        """

        phenom_xp_convention = 1

        #Line 569
        self = IMRPhenomX_PNR_GetAndSetPNRVariables(self, self.pWF)
        #What is the output of this function?

        #Line 580
        self = IMRPhenomX_PNR_GetAndSetCoPrecParams(self, self.pWF, self.lalParams)


        #if pflag in 220, 221, 222, 223, 224...
        #Line 597
        self, Omegazeta0_coeff, Omegazeta1_coeff, Omegazeta2_coeff, Omegazeta3_coeff, Omegazeta4_coeff, Omegazeta5_coeff, g0 = IMRPhenomX_Initialize_MSA_System(self, self.pWF, self.lalParams['ExpansionOrder'])


        object.__setattr__(self, 'Omegazeta0_coeff', Omegazeta0_coeff)
        object.__setattr__(self, 'Omegazeta1_coeff', Omegazeta1_coeff)
        object.__setattr__(self, 'Omegazeta2_coeff', Omegazeta2_coeff)
        object.__setattr__(self, 'Omegazeta3_coeff', Omegazeta3_coeff)
        object.__setattr__(self, 'Omegazeta4_coeff', Omegazeta4_coeff)
        object.__setattr__(self, 'Omegazeta5_coeff', Omegazeta5_coeff)
        object.__setattr__(self, 'Omegazeta5_coeff', Omegazeta5_coeff)
        object.__setattr__(self, 'g0', g0)
        #TODO if MSA_ERROR: switch to NNLO


        #Mfinal, afinal, fRING, fDAMP = IMRPhenomX_SetPrecessingRemnantParams(self, self.pWF, self.lalParams)
        # The output of this function should be Mfinal, afinal, fring, and fdamp
        # To be checked


        # case 223: Line 691 compute orbital angular momentum
        L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L = self.flag_222_223_twoPN_non_spinning_orbitan_angular_momentum()

        LRef = self.M * self.M * XLALSimIMRPhenomXLPNAnsatz(self.pWF['v_ref'], self.pWF['eta'] / self.pWF['v_ref'], L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L) 

        J0x_Sf = (self.m1_2)*self.chi1x + (self.m2_2)*self.chi2x
        J0y_Sf = (self.m1_2)*self.chi1y + (self.m2_2)*self.chi2y
        J0z_Sf = (self.m1_2)*self.chi1z + (self.m2_2)*self.chi2z + LRef

        J0_Sf = jnp.array([J0x_Sf, J0y_Sf, J0z_Sf])
        J0     = jnp.sqrt(J0x_Sf*J0x_Sf + J0y_Sf*J0y_Sf + J0z_Sf*J0z_Sf)

        # Compress line 772 - 781
        #/* Get angle between J0 and LN (z-direction) */
        thetaJ_Sf = jax.lax.cond(J0<1e-10, lambda _: 0.0, lambda _:jnp.acos(J0z_Sf / J0), operand = None)

        # Line 783
        phiRef = self.pWF['phiRef_In']
        # Line 785
        MAX_TOL_ATAN = 1.0e-15

        tol_condition = (jnp.abs(J0x_Sf) < MAX_TOL_ATAN) & (jnp.abs(J0y_Sf) < MAX_TOL_ATAN)
        # Compress line 797-825
        #Get azimuthal angle of J0 in the source frame
        phiJ_Sf = get_phiJ_Sf(tol_condition, J0_Sf)

        #phi0_aligned = phiJ_Sf

        #Compress line 828 - 846 #FIXME in function set_phi0 I am not sure what to do for cases 5, 6, 7. What is the old value?
        phi0 = 0 #phenom_xp_convention=1 it is zero

        #Determine kappa via rotations, as above */
        Nx_Sf = jnp.sin(self.pWF['inclination'])*jnp.cos((jnp.pi / 2.0) - phiRef)
        Ny_Sf = jnp.sin(self.pWF['inclination'])*jnp.sin((jnp.pi / 2.0) - phiRef)
        Nz_Sf = jnp.cos(self.pWF['inclination'])
        N_Sf = jnp.array([Nx_Sf, Ny_Sf, Nz_Sf])

        v_in = jnp.array([Nx_Sf, Ny_Sf, Nz_Sf])

        vout = IMRPhenomX_rotate_z(-phiJ_Sf, v_in)
        vout = IMRPhenomX_rotate_y(-thetaJ_Sf, vout)

        #/* Note difference in overall - sign w.r.t PhenomPv2 code */
        kappa = XLALSimIMRPhenomXatan2tol(vout[1],vout[0], MAX_TOL_ATAN)

        #/* Now determine alpha0 by rotating LN. In the source frame, LN = {0,0,1} */
        tmp_x = 0.0
        tmp_y = 0.0
        tmp_z = 1.0
        v_in = jnp.array([tmp_x, tmp_y, tmp_z])
        vout = IMRPhenomX_rotate_z(-phiJ_Sf,   v_in)
        vout = IMRPhenomX_rotate_y(-thetaJ_Sf, vout)
        vout = IMRPhenomX_rotate_z(-kappa,     vout)

        # Compress line 887 - 930
        tol_condition = (jnp.abs(vout[0]) < MAX_TOL_ATAN) & (jnp.abs(vout[1]) < MAX_TOL_ATAN)
        alpha0 = jnp.pi - kappa # For phenom_xp_convention = 1
        

        # Compress line 931-966
        thetaJN, Nz_Jf, Nx_Jf = thetaJN_Nz_Nx_1_6_7(N_Sf, J0_Sf, J0)
        object.__setattr__(self, 'thetaJN', thetaJN)

        '''
        Define the polarizations used. This follows the conventions adopted for IMRPhenomPv2.

        The IMRPhenomP polarizations are defined following the conventions in Arun et al (arXiv:0810.5336),
        i.e. projecting the metric onto the P, Q, N triad defining where: P = (N x J) / |N x J|.

        However, the triad X,Y,N used in LAL (the "waveframe") follows the definition in the
        NR Injection Infrastructure (Schmidt et al, arXiv:1703.01076).

        The triads differ from each other by a rotation around N by an angle zeta. We therefore need to rotate 
        the polarizations by an angle 2 zeta.
        '''

        Xx_Sf = -jnp.cos(self.pWF['inclination']) * jnp.sin(phiRef)
        Xy_Sf = -jnp.cos(self.pWF['inclination']) * jnp.cos(phiRef)
        Xz_Sf = +jnp.sin(self.pWF['inclination'])

        v = jnp.array([Xx_Sf, Xy_Sf, Xz_Sf])
        vout = IMRPhenomX_rotate_z(-phiJ_Sf, v)
        vout = IMRPhenomX_rotate_y(-thetaJ_Sf, vout)
        vout = IMRPhenomX_rotate_z(-kappa, vout)

        '''

            The components tmp_i are now the components of X in the J frame.

            We now need the polar angle of this vector in the P, Q basis of Arun et al:

                P = (N x J) / |NxJ|

            Note, that we put N in the (pos x)z half plane of the J frame 

        '''

        #Compress line 1002-1034
        PArun_Jf, QArun_Jf = PQ_Arun_1_6_7(Nx_Jf, Nz_Jf)

        #As it is line 1035-1043
        #(X . P)
        XdotPArun = (vout[0] * PArun_Jf[0]) + (vout[1] * PArun_Jf[1]) + (vout[2] * PArun_Jf[2])

        #(X . Q)
        XdotQArun = (vout[0] * QArun_Jf[0]) + (vout[1] * QArun_Jf[1]) + (vout[2] * QArun_Jf[2])

        #Now get the angle zeta
        zeta_polarization = jnp.atan2(XdotQArun, XdotPArun)
        object.__setattr__(self, 'zeta_polarization', zeta_polarization)

        #/* ********** PN Euler Angle Coefficients ********** */
        #/*
        #    This uses the single spin PN Euler angles as per IMRPhenomPv2
        #*/  

        #/* ********** PN Euler Angle Coefficients ********** */
        # Compress line 1050-1143
        alpha1, alpha2, alpha3, alpha4L, alpha5, epsilon1, epsilon2, epsilon3, epsilon4L, epsilon5 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  
        #self.compute_alpha_epsilon_220_330()

        # Compressed line 1163-1177
        epsilon0 = set_epsilon0(phenom_xp_convention, phiJ_Sf)

        ## Compression line 1178-1202
        alpha_offset, epsilon_offset, alpha_offset_1, epsilon_offset_1, alpha_offset_3, epsilon_offset_3, alpha_offset_4, epsilon_offset_4 = convention_five_or_seven_false(
            self.pWF['piM'], self.pWF['fRef'], alpha0, epsilon0,
            self.eta, self.eta2, self.eta3, self.eta4,
            self.inveta, self.c1, self.c1_over_eta,
            self.SAv, self.SAv2, self.invSAv, self.invSAv2,
            self.constants_L, self.S1_norm_2, self.S2_norm_2,
            self.qq, self.delta_qq, self.Seff, self.dotS1Ln, self.dotS2Ln, self.S_0_norm,
            self.psi0, self.psi1, self.psi2, self.g0,
            self.Omegaz0_coeff, self.Omegaz1_coeff, self.Omegaz2_coeff, self.Omegaz3_coeff, self.Omegaz4_coeff, self.Omegaz5_coeff, self.phiz_0,
            self.Omegazeta0_coeff, self.Omegazeta1_coeff, self.Omegazeta2_coeff, self.Omegazeta3_coeff, self.Omegazeta4_coeff, self.Omegazeta5_coeff, self.zeta_0)


        object.__setattr__(self, 'alpha_offset', alpha_offset)
        object.__setattr__(self, 'epsilon_offset', epsilon_offset)
        object.__setattr__(self, 'alpha_offset_1', alpha_offset_1)
        object.__setattr__(self, 'epsilon_offset_1', epsilon_offset_1)
        object.__setattr__(self, 'alpha_offset_3', alpha_offset_3)
        object.__setattr__(self, 'epsilon_offset_3', epsilon_offset_3)
        object.__setattr__(self, 'alpha_offset_4', alpha_offset_4)
        object.__setattr__(self, 'epsilon_offset_4', epsilon_offset_4)

        #object.__setattr__(self, 'cexp_i_alpha', cexp_i_alpha)
        #object.__setattr__(self, 'cexp_i_epsilon', cexp_i_epsilon)
        #object.__setattr__(self, 'cexp_i_betah', cexp_i_betah)


        # When L + SL < 0 and q>7, we disable multibanding NH: I will skip this function
        #self.IMRPhenomXPCheckMaxOpeningAngle()

        # Activate multibanding for Euler angles it threshold !=0. Only for PhenomXPHM. */
        #MBandPrecVersion = jax.lax.cond(self.lalParams['PhenomXPHMThresholdMband']==0, lambda _: 0, lambda _: 1, operand = None)
        ## NH: I do not implement PhenomXPHMThresholdMband==1 option. The output of the above line will always be self.MBandPrecVersion = 0. 


        # At high mass ratios, we find there can be numerical instabilities in the model, although the waveforms continue to be well behaved.
        # We warn to user of the possibility of these instabilities.

        return None

    def compute_and_set_spherical_harmonics(self):
        """
        Compute all required spin-weighted spherical harmonics and assign them to self.

        This method computes Y_{l,m}^{-2}(theta, phi=0) for:
        - l=2, m in [-2, -1, 0, 1, 2]
        - l=3, m in [-3, -2, -1, 0, 1, 2, 3]
        - l=4, m in [-4, -3, -2, -1, 0, 1, 2, 3, 4]

        The spherical harmonics are evaluated at theta = self.thetaJN and phi = 0.
        """
        # l=2 modes
        object.__setattr__(self, 'Y2m2', compute_sminus2_l2(theta=self.thetaJN, m=-2))
        object.__setattr__(self, 'Y2m1', compute_sminus2_l2(theta=self.thetaJN, m=-1))
        object.__setattr__(self, 'Y20', compute_sminus2_l2(theta=self.thetaJN, m=0))
        object.__setattr__(self, 'Y21', compute_sminus2_l2(theta=self.thetaJN, m=1))
        object.__setattr__(self, 'Y22', compute_sminus2_l2(theta=self.thetaJN, m=2))

        # l=3 modes
        object.__setattr__(self, 'Y3m3', compute_sminus2_l3(theta=self.thetaJN, m=-3))
        object.__setattr__(self, 'Y3m2', compute_sminus2_l3(theta=self.thetaJN, m=-2))
        object.__setattr__(self, 'Y3m1', compute_sminus2_l3(theta=self.thetaJN, m=-1))
        object.__setattr__(self, 'Y30', compute_sminus2_l3(theta=self.thetaJN, m=0))
        object.__setattr__(self, 'Y31', compute_sminus2_l3(theta=self.thetaJN, m=1))
        object.__setattr__(self, 'Y32', compute_sminus2_l3(theta=self.thetaJN, m=2))
        object.__setattr__(self, 'Y33', compute_sminus2_l3(theta=self.thetaJN, m=3))

        # l=4 modes
        object.__setattr__(self, 'Y4m4', compute_sminus2_l4(theta=self.thetaJN, m=-4))
        object.__setattr__(self, 'Y4m3', compute_sminus2_l4(theta=self.thetaJN, m=-3))
        object.__setattr__(self, 'Y4m2', compute_sminus2_l4(theta=self.thetaJN, m=-2))
        object.__setattr__(self, 'Y4m1', compute_sminus2_l4(theta=self.thetaJN, m=-1))
        object.__setattr__(self, 'Y40', compute_sminus2_l4(theta=self.thetaJN, m=0))
        object.__setattr__(self, 'Y41', compute_sminus2_l4(theta=self.thetaJN, m=1))
        object.__setattr__(self, 'Y42', compute_sminus2_l4(theta=self.thetaJN, m=2))
        object.__setattr__(self, 'Y43', compute_sminus2_l4(theta=self.thetaJN, m=3))
        object.__setattr__(self, 'Y44', compute_sminus2_l4(theta=self.thetaJN, m=4))




def PQ_Arun_1_6_7(Nx_Jf, Nz_Jf):
    # Get polar angle of X vector in J frame in the P,Q basis of Arun et al
    PArunx_Jf = Nz_Jf
    PAruny_Jf = 0.0
    PArunz_Jf = -Nx_Jf

    QArunx_Jf = 0.0
    QAruny_Jf = 1.0
    QArunz_Jf = 0.0

    return jnp.array([PArunx_Jf, PAruny_Jf, PArunz_Jf]), jnp.array([QArunx_Jf, QAruny_Jf, QArunz_Jf])

def thetaJN_Nz_Nx_1_6_7(N_Sf, J0_Sf, J0):
    # Line 957-962

    J0dotN     = (J0_Sf[0] * N_Sf[0]) + (J0_Sf[1] * N_Sf[1]) + (J0_Sf[2] * N_Sf[2])
    thetaJN = jnp.acos( J0dotN / J0 )
    Nz_Jf     = jnp.cos(thetaJN)
    Nx_Jf     = jnp.sin(thetaJN)

    return thetaJN, Nz_Jf, Nx_Jf

def get_phiJ_Sf(tol_condition, J0_Sf):
    """
    Compute phiJ_Sf based on tolerance condition.

    Since convention_condition is always False, this simplifies to:
    - If tol_condition is True: return 0.0
    - Otherwise: return atan2(J0_Sf[1], J0_Sf[0])
    """
    phiJ_Sf = jax.lax.cond(
        tol_condition,
        lambda _: 0.0,
        lambda _: jnp.atan2(J0_Sf[1], J0_Sf[0]),
        operand=None
    )

    return phiJ_Sf


def convention_five_or_seven_false(piM, fRef, alpha0, epsilon0,
                                   eta, eta2, eta3, eta4,
                                   inveta, c1, c1_over_eta,
                                   SAv, SAv2, invSAv, invSAv2,
                                   constants_L, S1_norm_2, S2_norm_2,
                                   qq, delta_qq, Seff, dotS1Ln, dotS2Ln, S_0_norm,
                                   psi0, psi1, psi2, g0,
                                   Omegaz0_coeff, Omegaz1_coeff, Omegaz2_coeff, Omegaz3_coeff, Omegaz4_coeff, Omegaz5_coeff, phiz_0,
                                   Omegazeta0_coeff, Omegazeta1_coeff, Omegazeta2_coeff, Omegazeta3_coeff, Omegazeta4_coeff, Omegazeta5_coeff, zeta_0):
    # Get initial Get \alpha and \epsilon offsets at \omega = pi * M * f_{Ref} */
    mprime = 2 #FIXME why always 2?

    alpha_offset, epsilon_offset = Get_alphaepsilon_atfref(
        mprime, piM, fRef, alpha0, epsilon0,
        eta, eta2, eta3, eta4,
        inveta, c1, c1_over_eta,
        SAv, SAv2, invSAv, invSAv2,
        constants_L, S1_norm_2, S2_norm_2,
        qq, delta_qq, Seff, dotS1Ln, dotS2Ln, S_0_norm,
        psi0, psi1, psi2, g0,
        Omegaz0_coeff, Omegaz1_coeff, Omegaz2_coeff, Omegaz3_coeff, Omegaz4_coeff, Omegaz5_coeff, phiz_0,
        Omegazeta0_coeff, Omegazeta1_coeff, Omegazeta2_coeff, Omegazeta3_coeff, Omegazeta4_coeff, Omegazeta5_coeff, zeta_0)
    return alpha_offset, epsilon_offset, alpha_offset, epsilon_offset, alpha_offset, epsilon_offset, alpha_offset, epsilon_offset



def Get_alphaepsilon_atfref(mprime, piM, fRef, alpha0, epsilon0,
                            eta, eta2, eta3, eta4,
                            inveta, c1, c1_over_eta,
                            SAv, SAv2, invSAv, invSAv2,
                            constants_L, S1_norm_2, S2_norm_2,
                            qq, delta_qq, Seff, dotS1Ln, dotS2Ln, S_0_norm,
                            psi0, psi1, psi2, g0,
                            Omegaz0_coeff, Omegaz1_coeff, Omegaz2_coeff, Omegaz3_coeff, Omegaz4_coeff, Omegaz5_coeff, phiz_0,
                            Omegazeta0_coeff, Omegazeta1_coeff, Omegazeta2_coeff, Omegazeta3_coeff, Omegazeta4_coeff, Omegazeta5_coeff, zeta_0):
    omega_ref = piM * fRef * 2 / mprime

    alpha_offset, epsilon_offset = Get_alphaepsilon_atfref_pflag_true(
        omega_ref, alpha0, epsilon0,
        eta, eta2, eta3, eta4,
        inveta, c1, c1_over_eta,
        SAv, SAv2, invSAv, invSAv2,
        constants_L, S1_norm_2, S2_norm_2,
        qq, delta_qq, Seff, dotS1Ln, dotS2Ln, S_0_norm,
        psi0, psi1, psi2, g0,
        Omegaz0_coeff, Omegaz1_coeff, Omegaz2_coeff, Omegaz3_coeff, Omegaz4_coeff, Omegaz5_coeff, phiz_0,
        Omegazeta0_coeff, Omegazeta1_coeff, Omegazeta2_coeff, Omegazeta3_coeff, Omegazeta4_coeff, Omegazeta5_coeff, zeta_0)

    return alpha_offset, epsilon_offset


def Get_alphaepsilon_atfref_pflag_true(omega_ref, alpha0, epsilon0,
                                       eta, eta2, eta3, eta4,
                                       inveta, c1, c1_over_eta,
                                       SAv, SAv2, invSAv, invSAv2,
                                       constants_L, S1_norm_2, S2_norm_2,
                                       qq, delta_qq, Seff, dotS1Ln, dotS2Ln, S_0_norm,
                                       psi0, psi1, psi2, g0,
                                       Omegaz0_coeff, Omegaz1_coeff, Omegaz2_coeff, Omegaz3_coeff, Omegaz4_coeff, Omegaz5_coeff, phiz_0,
                                       Omegazeta0_coeff, Omegazeta1_coeff, Omegazeta2_coeff, Omegazeta3_coeff, Omegazeta4_coeff, Omegazeta5_coeff, zeta_0):

    v = jnp.cbrt(omega_ref)
    vangles = IMRPhenomX_Return_phi_zeta_costhetaL_MSA(v, eta, eta2, eta3, eta4,
                                                       inveta, c1, c1_over_eta,
                                                       SAv, SAv2, invSAv, invSAv2,
                                                       constants_L, S1_norm_2, S2_norm_2,
                                                       qq, delta_qq, Seff, dotS1Ln, dotS2Ln, S_0_norm,
                                                       psi0, psi1, psi2, g0,
                                                       Omegaz0_coeff, Omegaz1_coeff, Omegaz2_coeff, Omegaz3_coeff, Omegaz4_coeff, Omegaz5_coeff, phiz_0,
                                                       Omegazeta0_coeff, Omegazeta1_coeff, Omegazeta2_coeff, Omegazeta3_coeff, Omegazeta4_coeff, Omegazeta5_coeff, zeta_0)

    alpha_offset = vangles[0] - alpha0
    epsilon_offset = vangles[1] - epsilon0
    return alpha_offset, epsilon_offset
    


def XLALSimIMRPhenomXLPNAnsatz(v: float, LNorm: float, L0: float, L1: float, L2: float, 
                               L3: float, L4: float, L5: float, L6: float, L7: float, 
                               L8: float, L8L: float) -> float:
    """
    Compute orbital angular momentum using post-Newtonian expansion
    
    Args:
        v: Input velocity (float)
        LNorm: Orbital angular momentum normalization (float)
        L0: Newtonian orbital angular momentum (float)
        L1: 0.5PN Orbital angular momentum (float)
        L2: 1.0PN Orbital angular momentum (float)
        L3: 1.5PN Orbital angular momentum (float)
        L4: 2.0PN Orbital angular momentum (float)
        L5: 2.5PN Orbital angular momentum (float)
        L6: 3.0PN Orbital angular momentum (float)
        L7: 3.5PN Orbital angular momentum (float)
        L8: 4.0PN Orbital angular momentum (float)
        L8L: 4.0PN logarithmic orbital angular momentum term (float)
        
    Returns:
        float: Orbital angular momentum
    """
    
    x = v * v
    x2 = x * x
    x3 = x * x2
    x4 = x * x3
    sqx = jnp.sqrt(x)
    
    # Here LN is the Newtonian pre-factor: LN = \eta / \sqrt{x} :
    # L = L_N \sum_a L_a x^{a/2}
    #   = L_N [ L0 + L1 x^{1/2} + L2 x^{2/2} + L3 x^{3/2} + ... ]
    
    return LNorm * (L0 + L1*sqx + L2*x + L3*(x*sqx) + L4*x2 + L5*(x2*sqx) + 
                    L6*x3 + L7*(x3*sqx) + L8*x4 + L8L*x4*jnp.log(x))







def set_epsilon0(phenom_xp_convention, phiJ_Sf):

    epsilon0 = jax.lax.cond(
        jnp.isin(phenom_xp_convention, jnp.array([1, 6])),
        lambda _: phiJ_Sf - jnp.pi,
        lambda _: 0.0,
        operand=None,
    )
    
    return epsilon0







    
def get_deltaF_from_wfstruct(pWF: dict) -> float:
    """
    Get deltaF from waveform structure
    
    Args:
        pWF: Waveform structure dictionary (dict)
        
    Returns:
        float: Delta frequency in dimensionless units
    """
    
    seglen = XLALSimInspiralChirpTimeBound(
        pWF['fRef'], pWF['m1_SI'], pWF['m2_SI'], pWF['chi1L'], pWF['chi2L']
    )
    
    deltaFv1 = 1.0 / jnp.maximum(4.0, jnp.power(2, jnp.ceil(jnp.log(seglen)/jnp.log(2))))
    deltaF = jnp.minimum(deltaFv1, 0.1)
    deltaMF = XLALSimIMRPhenomXUtilsHztoMf(deltaF, pWF['Mtot'])
    
    return deltaMF







def XLALSimInspiralChirpTimeBound(fstart: float, m1: float, m2: float, s1: float, s2: float) -> float:
    """
    Calculate chirp time bound for inspiral
    
    Args:
        fstart: Starting frequency (float)
        m1: Mass of object 1 (float)
        m2: Mass of object 2 (float)
        s1: Spin of object 1 (float)
        s2: Spin of object 2 (float)
        
    Returns:
        float: Chirp time bound
    """
    
    M = m1 + m2  # total mass
    mu = m1 * m2 / M  # reduced mass
    eta = mu / M  # symmetric mass ratio
    
    # chi = (s1*m1 + s2*m2)/M <= max(|s1|,|s2|)
    # over-estimate of chi
    chi = jnp.abs(jnp.where(jnp.abs(s1) > jnp.abs(s2), s1, s2))
    
    # note: for some reason these coefficients are named wrong...
    # "2PN" should be "1PN", "4PN" should be "2PN", etc.
    c0 = jnp.abs(XLALSimInspiralTaylorT2Timing_0PNCoeff(M, eta))
    c2 = XLALSimInspiralTaylorT2Timing_2PNCoeff(eta)
    
    # the 1.5pN spin term is in TaylorT2 is 8*beta/5 [Citation ??]
    # where beta = (113/12 + (25/4)(m2/m1))*(s1*m1^2/M^2) + 2 <-> 1
    # [Cutler & Flanagan, Physical Review D 49, 2658 (1994), Eq. (3.21)]
    # which can be written as (113/12)*chi - (19/6)(s1 + s2)
    # and we drop the negative contribution
    c3 = (226.0/15.0) * chi
    
    # there is also a 1.5PN term with eta, but it is negative so do not include it
    c4 = XLALSimInspiralTaylorT2Timing_4PNCoeff(eta)
    
    v = jnp.power(jnp.pi * G * M * fstart, 1.0/3.0) / C
    
    return c0 * jnp.power(v, -8) * (1.0 + (c2 + (c3 + c4 * v) * v) * v * v)

def XLALSimInspiralTaylorT2Timing_0PNCoeff(totalmass: float, eta: float) -> float:
    """
    Calculate 0PN coefficient for TaylorT2 timing
    
    Args:
        totalmass: Total mass in kilograms (float)
        eta: Symmetric mass ratio (float)
        
    Returns:
        float: 0PN timing coefficient
    """
    
    # convert totalmass from kilograms to seconds
    totalmass *= G / jnp.power(C, 3.0)
    
    return -5.0 * totalmass / (256.0 * eta)


def XLALSimInspiralTaylorT2Timing_2PNCoeff(eta: float) -> float:
    """
    Calculate 2PN coefficient for TaylorT2 timing
    
    Args:
        eta: Symmetric mass ratio (float)
        
    Returns:
        float: 2PN timing coefficient
    """
    
    return 7.43/2.52 + 11.0/3.0 * eta


def XLALSimInspiralTaylorT2Timing_4PNCoeff(eta: float) -> float:

    return 30.58673/5.08032 + 54.29/5.04*eta + 61.7/7.2*eta*eta



def XLALSimIMRPhenomXUtilsHztoMf(fHz: float, Mtot_Msun: float) -> float:
    """
    Convert frequency from Hz to geometric units (Mf).

    Parameters
    ----------
    fHz : float
        Frequency in Hz
    Mtot_Msun : float
        Total mass in solar masses

    Returns
    -------
    float
        Geometric frequency Mf
    """
    return fHz * Mtot_Msun * MTSUN


def XLALSimIMRPhenomXUtilsMftoHz(Mf: float, Mtot_Msun: float) -> float:
    """
    Convert frequency from geometric units (Mf) to Hz.

    Parameters
    ----------
    Mf : float
        Geometric frequency
    Mtot_Msun : float
        Total mass in solar masses

    Returns
    -------
    float
        Frequency in Hz
    """
    return Mf / (Mtot_Msun * MTSUN)


def check_kerr_bound(pnr_use_tuned_angles, pnr_single_spin, chi1_norm, chi2_norm):
    """
    Check if spin magnitudes violate the Kerr bound.

    Parameters
    ----------
    pnr_use_tuned_angles : bool
        Whether PNR tuned angles are used
    pnr_single_spin : int
        PNR single spin flag
    chi1_norm : float
        Magnitude of spin 1
    chi2_norm : float
        Magnitude of spin 2

    Returns
    -------
    bool
        True if valid, False if Kerr bound is violated
    """
    # Condition to apply check
    should_check = jnp.logical_or(
        jnp.logical_not(pnr_use_tuned_angles),
        pnr_single_spin != 1)

    # Compute violations
    chi1_violation = jnp.abs(chi1_norm) > 1.0
    chi2_violation = jnp.abs(chi2_norm) > 1.0

    # Only raise error if we should check AND there's a violation
    error_condition = jnp.logical_and(
        should_check,
        jnp.logical_or(chi1_violation, chi2_violation)
    )

    # Return success/error flag
    return jnp.where(
        error_condition,
        False,  # Error case
        True    # Success case
    )




def IMRPhenomX_InspiralAngles_SpinTaylor(chi1x: float, chi1y: float, chi1z: float, 
                                         chi2x: float, chi2y: float, chi2z: float,
                                         fmin: float, PrecVersion: int, pWF: dict, lalParams: dict):
    '''
    Output: PhenomXPInspiralArrays [out] Struct containing solutions returned by PNEvolveOrbit 
    Output: fmin_PN [out] Minimum frequency in PN solutions array
    '''


    fRef = pWF['fRef']
    m1_SI = pWF['m1_SI']
    m2_SI = pWF['m2_SI']


    s1x=chi1x 
    s1y=chi1y
    s1z=chi1z

    s2x=chi2x
    s2y=chi2y
    s2z=chi2z

    piGM = jnp.pi * (pWF['m1_SI'] + pWF['m2_SI']) * (G / C) / (C * C)


    quadparam1=pWF["quadparam1"]
    quadparam2=pWF["quadparam2"]
    lambda1=pWF["lambda1"]
    lambda2=pWF["lambda2"]

    PrecVersion_cond = (PrecVersion==311) | (PrecVersion==321)
    quadparam1 = jnp.where(PrecVersion_cond, 1, quadparam1)
    quadparam2 = jnp.where(PrecVersion_cond, 1, quadparam2)
    lambda1 = jnp.where(PrecVersion_cond, 0, lambda1)
    lambda2 = jnp.where(PrecVersion_cond, 0, lambda2)

    #Compress line 4634-4637
    phaseO = jnp.where(lalParams['phaseO']==-1, 7, lalParams['phaseO'])
    spinO = jnp.where(lalParams['spinO']==-1, 6, lalParams['spinO'])
    tideO = jnp.where(lalParams['tideO']==-1, 12, lalParams['tideO'])
    lscorr = 0.0
    
    #Skip 4638-4655

    lnhatx = 0.0
    lnhaty = 0.0
    lnhatz = 1.0

    e1x = 1.0
    e1y = 0.0
    e1z = 0.0
    

    """
    If PhenomXPSpinTaylorVersion is None: set it to "SpinTaylorT4"
    """

    approx = lalParams['approx_name']


    fMECO_Hz = XLALSimIMRPhenomXUtilsMftoHz(pWF['fMECO'], pWF['Mtot'])
    fmin_condition = (fmin > fMECO_Hz) & ((PrecVersion==320) | (PrecVersion==321))
    fmin = jnp.where(fmin_condition, fMECO_Hz, fmin)

    fCut = XLALSimIMRPhenomXUtilsMftoHz(pWF['fRING']+8 * pWF['fDAMP'], pWF['Mtot'])
    

    deltaT_coarse = .5 * lalParams['coarse_fac'] / fCut

    
    #Line 4681
    #if(coarse_fac  < 1) { XLAL_ERROR(XLAL_EDOM, "Coarse factor must be >= 1!\n")}

    #Line 4685-4686
    #fS = fmin
    #fE = fCut

    fref_zero_or_same_to_fmin = (fRef < 1e-10) | (jnp.abs(fRef-fmin) < 1e-10)

    #Compress line 4688-4780
    PhenomXPInspiralArrays = jax.lax.cond(fref_zero_or_same_to_fmin, integrate_forward, integrate_both_sides, fRef, fmin, fCut, deltaT_coarse, m1_SI, m2_SI, s1x,s1y,s1z,s2x,s2y,s2z,lnhatx,lnhaty,lnhatz,e1x,e1y,e1z,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx)
    #V_PN, Phi_PN, S1x_PN, S1y_PN, S1z_PN, S2x_PN, S2y_PN, S2z_PN, LNhatx_PN, LNhaty_PN, LNhatz_PN, E1x_PN, E1y_PN, E1z_PN

    #Line 4782
    #if lalParams['coarse_fac'] > 1: # ignoring this flag. I force it to be ==1.  

    ## copy coarse-grid data to fine-grid
    ## destroy coarse-grid

    #check that the first frequency node returned is indeed below the fmin requested, to avoid interpolation errors. If not return an error which will trigger the fallback to MSA

    fminPN=jnp.power(PhenomXPInspiralArrays[0][0],3.)/piGM

    spin_taylor_success_check = (fminPN<0.0) | (fminPN>fmin)
    status = jnp.where(spin_taylor_success_check, 0, 1)
    return PhenomXPInspiralArrays, status




def integrate_forward(fRef, fmin, fCut, deltaT_coarse, m1_SI, m2_SI, s1x, s1y, s1z, s2x, s2y, s2z, lnhatx, lnhaty, lnhatz, e1x, e1y, e1z, lambda1, lambda2, quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx):
    '''
    If fRef is zero or is equal to fmin, we only need to integrate from fmin to fCut, i.e., forward. 
    This function is called to perform forward integration. 
    Line 4690-4697
    '''

    fS = fmin
    fE = fCut

    V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z = XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT_coarse, m1_SI, m2_SI,fS,fE,s1x,s1y,s1z,s2x,s2y,s2z,lnhatx,lnhaty,lnhatz,e1x,e1y,e1z,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr)
    return V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z


def integrate_both_sides(fRef, fmin, fCut, deltaT_coarse, m1_SI, m2_SI,fS,fE,s1x,s1y,s1z,s2x,s2y,s2z,lnhatx,lnhaty,lnhatz,e1x,e1y,e1z,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx):
    '''
    If fRef > fmin, we first integrate from fRef to fmin and then fRef to fCut. 
    This function is called to integrate on both sides
    FIXME: We may want to get rid of jnp.append by making arrays of zeros and populating them. 
    Line 4701-4773
    '''

    fS =  fRef
    fE = fmin - 0.5

    # Backward integration
    V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z = XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT_coarse, m1_SI, m2_SI,fS,fE,s1x,s1y,s1z,s2x,s2y,s2z,lnhatx,lnhaty,lnhatz,e1x,e1y,e1z,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr)


    fS = fRef
    fE = fCut
    #Skipping the sanity check of if...else. Just jump to forward integration. 
    V_forward, Phi_forward, S1x_forward, S1y_forward, S1z_forward, S2x_forward, S2y_forward, S2z_forward, LNhatx_forward, LNhaty_forward, LNhatz_forward, E1x_forward, E1y_forward, E1z_forward = XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT_coarse, 
                                                                                                                                            m1_SI, m2_SI, fS, fE, s1x, s1y, s1z, s2x, s2y,
                                                                                                                                            s2z, lnhatx, lnhaty, lnhatz, e1x, e1y, e1z, lambda1,lambda2, quadparam1, quadparam2, spinO, tideO, phaseO, lscorr)
    V = jnp.append(V, V_forward)
    Phi = jnp.append(Phi, Phi_forward)
    S1x = jnp.append(S1x, S1x_forward)
    S1y = jnp.append(S1y, S1y_forward)
    S1z = jnp.append(S1z, S1z_forward)

    S2x = jnp.append(S2x, S2x_forward)
    S2y = jnp.append(S2y, S2y_forward)
    S2z = jnp.append(S2z, S2z_forward)

    LNhatx = jnp.append(LNhatx, LNhatx_forward)
    LNhaty = jnp.appnd(LNhaty, LNhaty_forward)
    LNhatz = jnp.append(LNhatz, LNhatz_forward)
    
    E1x = jnp.append(E1x, E1x_forward)
    E1y = jnp.append(E1y, E1y_forward)
    E1z = jnp.append(E1z, E1z_forward)


    return V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z




def IMRPhenomX_rotate_z(angle, v): 
    """
    Rotate a 3D vector v = (vx, vy, vz) about the z-axis by given angle.
    Args:
        angle: scalar angle in radians (JAX array or float)
        v: array-like of shape (3,) representing [vx, vy, vz]
    Returns:
        rotated vector as a JAX array of shape (3,)
    """
    cosa = jnp.cos(angle)
    sina = jnp.sin(angle)
    vx = v[0]
    vy = v[1]
    vz = v[2]

    vx_rot = vx * cosa - vy * sina
    vy_rot = vx * sina + vy * cosa
    vz_rot = vz  # unchanged

    return jnp.array([vx_rot, vy_rot, vz_rot])




def IMRPhenomX_rotate_y(angle, v):
    """
    Rotate a 3D vector v = (vx, vy, vz) about the y-axis by a given angle.
    Args:
        angle: scalar angle in radians (JAX array or float)
        v: array-like of shape (3,) representing [vx, vy, vz]
    Returns:
        rotated vector as a JAX array of shape (3,)
    """

    cosa = jnp.cos(angle)
    sina = jnp.sin(angle)
    vx = v[0]
    vy = v[1]
    vz = v[2]

    vx_rot =  vx * cosa + vz * sina
    vy_rot =  vy  # unchanged
    vz_rot = -vx * sina + vz * cosa

    return jnp.array([vx_rot, vy_rot, vz_rot])


def XLALSimIMRPhenomXPrecessingFinalSpin2017(
    eta: float,
    chi1L: float,
    chi2L: float,
    chi_perp: float
) -> float:
    """
    Calculate precessing final spin using the 2017 fitting formula.
    This is essentially the PhenomPv2 final spin prescription.

    Args:
        eta: Symmetric mass ratio
        chi1L: Aligned spin component of BH 1
        chi2L: Aligned spin component of BH 2
        chi_perp: Perpendicular spin component

    Returns:
        float: Final dimensionless spin including precession effects
    """
    # Get mass ratio from eta
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + delta)
    # m2 = 0.5 * (1.0 - delta)  # Not used, but kept for reference

    # Compute parallel component of final spin (non-precessing)
    af_parallel = XLALSimIMRPhenomXFinalSpin2017(eta, chi1L, chi2L)

    # Compute perpendicular component contribution
    # Weight by appropriate mass factor (larger BH dominates)
    q_factor = m1  # m1 is already normalized, m1 > m2 by convention
    Sperp = chi_perp * q_factor * q_factor

    # Total final spin magnitude
    af = jnp.copysign(1.0, af_parallel) * jnp.sqrt(Sperp * Sperp + af_parallel * af_parallel)

    return af


def IMRPhenomX_PNR_GenerateRingdownPNRBeta(pWF: dict, pPrec) -> float:
    """
    Generate ringdown value of precession angle beta for PNR.
    This is used to set the sign of the final spin and calculate effective ringdown frequency.

    Args:
        pWF: Waveform structure dictionary
        pPrec: Precession structure

    Returns:
        float: Ringdown beta angle in radians
    """
    # This function would compute beta at ringdown frequency
    # For now, we provide a placeholder that should be replaced with actual PNR beta computation
    # The actual implementation requires the full PNR beta angle model

    # Import beta computation if available
    from .LALSimIMRPhenomX_PNR_beta import IMRPhenomX_PNR_precompute_beta_coefficients

    # Compute beta coefficients
    betaParams = IMRPhenomX_PNR_precompute_beta_coefficients(pWF, pPrec)

    # Evaluate beta at ringdown frequency
    # This is a simplified version - actual implementation evaluates the PNR beta model at fRING
    Mf_RD = pWF.get('fRING', 0.3)  # Use computed fRING or default

    # Simplified beta evaluation (actual implementation would use the full PNR beta expression)
    betaRD = betaParams.B0 + betaParams.B1 * Mf_RD + betaParams.B2 * Mf_RD**2

    return betaRD
