import jax.numpy as jnp
import math
from ..typing import Array
from ..constants import G, MSUN, C, MTSUN, GAMMA
import jax
from .spherical_harmonics import *
from .IMRPhenomXPHM_utils import *
from .LALSimInspiralSpinTaylor import XLALSimInspiralSpinTaylorPNEvolveOrbit
from .LALSimIMRPhenomX_PNR_internals import IMRPhenomX_PNR_HMInterpolationDeltaF
from dataclasses import dataclass
from jax_dataclasses import pytree_dataclass
from .LALSimIMRPhenomX_PNR_alpha import IMRPhenomX_PNR_precompute_alpha_coefficients
from .LALSimIMRPhenomX_PNR_beta import (IMRPhenomX_PNR_precompute_beta_coefficients, IMRPhenomX_PNR_BetaConnectionFrequencies)



@pytree_dataclass
class IMRPhenomXGetAndSetPrecessionVariables:
    # Basic parameters

    def __init__(self, pWF: dict, m1_SI: float, m2_SI: float, chi1x: float, chi1y: float, chi1z: float, chi2x: float, chi2y: float, chi2z: float, lalParams: dict, debug_flag: bool):
        """
        lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#af089ef2586c52b12016c0d791b176121

        Functions to Perform Frame Transformations and Populate Structs
        - Calculates frame transformation
        - PN Euler angles
        - Frame transformations
        - Orbital angular momenta etc

        
        pWF: Some useful waveform parameters

        m1_SI: Mass of object 1 (heavier one) in SI units
        m2_SI: Mass of object 2 (lighter one) in SI units


        chi1x: Object 1 spin. x-component.
        chi1y: Object 1 spin. y-component.
        chi1z: Object 1 spin. z-component.

        chi2x: Object 2 spin. x-component.
        chi2y: Object 2 spin. y-component.
        chi2z: Object 2 spin. z-component.

        lalParams: FIXME
        debug_flag: FIXME

        Note to self:
        1: NNLO
        2: MSA
        3: Spin-Taylor

        precversionTag  or precessing_tag means the first digit
        pflag means the full number
        """

        # Here we assume m1 > m2, q > 1, dm = m1 - m2 = delta = sqrt(1-4eta) > 0
        self.pWF = pWF
        #self.pWF['LALparams'] = lalParams
        self.lalParams = lalParams
        self.MAX_TOL_ATAN = 1.0e-15
        self.check_convention_array = jnp.array([0, 5])
        #Pre-cache useful powers here
        self.sqrt2   = 1.4142135623730951
        self.sqrt5   = 2.23606797749978981
        self.sqrt6   = 2.44948974278317788
        self.sqrt7   = 2.64575131106459072
        self.sqrt10  = 3.16227766016838
        self.sqrt14  = 3.74165738677394133
        self.sqrt15  = 3.87298334620741702
        self.sqrt70  = 8.36660026534075563
        self.sqrt30  = 5.477225575051661
        self.sqrt2p5 = 1.58113883008419
        self.log16    = 2.772588722239781
        self.power_of_lalpi_2 = jnp.power(jnp.pi, 2)
        self.debug_prec = debug_flag

        # Sort out version-specific flags
        # Get IMRPhenomX precession version from LAL dictionary
        self.IMRPhenomXPrecVersion = self.lalParams['IMRPhenomXPrecVersion']
        self.IMRPhenomXPrecVersion = jnp.where(self.IMRPhenomXPrecVersion == 300, 223, self.IMRPhenomXPrecVersion)
        
        self.manual_prescription_tag = self.check_prescription_tag(self.IMRPhenomXPrecVersion)# 1 for NNLO, 2 for MSA, 3 for SpinTaylor
        # Line 109
        # default to NNLO angles if in-plane spins are negligible and one of the SpinTaylor options has been selected. The solutions would be dominated by numerical noise.
        chi_in_plane = jnp.sqrt(chi1x*chi1x+chi1y*chi1y+chi2x*chi2x+chi2y*chi2y)

        self.IMRPhenomXPrecVersion = jnp.where((chi_in_plane<1e-6) & (self.IMRPhenomXPrecVersion == 330), 102, self.IMRPhenomXPrecVersion)
        
        #Compress line 114
        self.IMRPhenomXPrecVersion = jnp.where((chi_in_plane<1e-7) & (self.manual_prescription_tag==3), 102, self.IMRPhenomXPrecVersion)


        self.ExpansionOrder = lalParams['ExpansionOrder']
        self.PNRUseTunedAngles = lalParams['PNRUseTunedAngles']
        self.IMRPhenomXPNRInterpTolerance = lalParams['IMRPhenomXPNRInterpTolerance']
        self.AntisymmetricWaveform = lalParams['AntisymmetricWaveform']
        self.PolarizationSymmetry = 1.0

        #### Skipping the multibanding bookkeeping around line 136-142

        self.m1_SI = m1_SI
        self.m2_SI = m2_SI
        # Define a number of convenient local parameters #############
        self.m1        = self.m1_SI / self.pWF['Mtot_SI']   #/* Normalized mass of larger companion:   m1_SI / Mtot_SI */
        self.m2        = self.m2_SI / self.pWF['Mtot_SI']   #/* Normalized mass of smaller companion:  m2_SI / Mtot_SI */
        self.M         = (self.m1 + self.m2)              #/* Total mass in solar units */
        
        # Useful powers of mass
        self.m1_2      = self.m1 * self.m1
        self.m1_3      = self.m1 * self.m1_2
        self.m1_4      = self.m1 * self.m1_3
        self.m1_5      = self.m1 * self.m1_4
        self.m1_6      = self.m1 * self.m1_5
        self.m1_7      = self.m1 * self.m1_6
        self.m1_8      = self.m1 * self.m1_7
        
        self.m2_2      = self.m2 * self.m2
        
        self.pWF['M'] = self.M
        self.pWF['m1_2'] = self.m1_2
        self.pWF['m2_2'] = self.m2_2

        self.q = self.m1/self.m2


        # Powers of eta
        self.eta       = self.pWF['eta']
        self.eta2      = self.eta*self.eta
        self.eta3      = self.eta*self.eta2
        self.eta4      = self.eta*self.eta3
        self.eta5      = self.eta*self.eta4
        self.eta6      = self.eta*self.eta5

        # \delta in terms of q > 1
        self.delta     = self.pWF['delta']
        self.delta2    = self.delta*self.delta
        self.delta3    = self.delta*self.delta2

        # Cache these powers, as we use them regularly

        self.inveta         = 1.0 / self.eta
        self.inveta2        = 1.0 / self.eta2
        self.inveta3        = 1.0 / self.eta3
        self.inveta4        = 1.0 / self.eta4
        self.sqrt_inveta    = 1.0 / jnp.sqrt(self.eta)

        self.chi_eff   = self.pWF['chiEff']

        self.twopiGM        = 2*jnp.pi * G * (self.m1_SI + self.m2_SI) / C / C / C
        self.piGM           = jnp.pi * (self.m1_SI + self.m2_SI) * (G / C) / (C * C)

        ####  Set spin variables in pPrec struct  ######
        self.chi1x          = chi1x
        self.chi1y          = chi1y
        self.chi1z          = chi1z
        self.chi1_norm      = jnp.sqrt(chi1x*chi1x + chi1y*chi1y + chi1z*chi1z)

        self.chi2x          = chi2x
        self.chi2y          = chi2y
        self.chi2z          = chi2z
        self.chi2_norm      = jnp.sqrt(chi2x*chi2x + chi2y*chi2y + chi2z*chi2z)

        ###Check that spins obey Kerr bound */ ####
        kerr_bound_satisfied = self.check_kerr_bound(self.lalParams['PNRUseTunedAngles'],  self.pWF['PNR_SINGLE_SPIN'], self.chi1_norm, self.chi2_norm)

        ###/* Calculate dimensionful spins */
        self.S1x        = self.chi1x * self.m1_2
        self.S1y        = self.chi1y * self.m1_2
        self.S1z        = self.chi1z * self.m1_2
        self.S1_norm    = jnp.abs(self.chi1_norm) * self.m1_2

        self.S2x        = chi2x * self.m2_2
        self.S2y        = chi2y * self.m2_2
        self.S2z        = chi2z * self.m2_2
        self.S2_norm    = jnp.abs(self.chi2_norm) * self.m2_2

        ###// Useful powers
        self.S1_norm_2  = self.S1_norm * self.S1_norm
        self.S2_norm_2  = self.S2_norm * self.S2_norm

        self.chi1_perp  = jnp.sqrt(chi1x*chi1x + chi1y*chi1y)
        self.chi2_perp  = jnp.sqrt(chi2x*chi2x + chi2y*chi2y)

        ###/* Get spin projections */
        self.S1_perp    = (self.m1_2) * jnp.sqrt(self.chi1x*self.chi1x + self.chi1y*self.chi1y)
        self.S2_perp    = (self.m2_2) * jnp.sqrt(self.chi2x*self.chi2x + self.chi2y*self.chi2y)

        ###/* Norm of in-plane vector sum: Norm[ S1perp + S2perp ] */
        self.STot_perp     = jnp.sqrt( (self.S1x+self.S2x)*(self.S1x+self.S2x) + (self.S1y+self.S2y)*(self.S1y+self.S2y) )

        ##/* This is called chiTot_perp to distinguish from Sperp used in contrusction of chi_p. For normalization, see Sec. IV D of arXiv:2004.06503 */
        self.chiTot_perp   = self.STot_perp * (self.M*self.M) / self.m1_2
        ##/* Store self.chiTot_perp to pWF so that it can be used in XCP modifications (PNRUseTunedCoprec) */
        self.pWF['chiTot_perp'] = self.chiTot_perp

        # Line 245-255
        #disable tuned PNR angles, tuned coprec and mode asymmetries in low in-plane spin limit */
        cond = (chi_in_plane<1e-7) & (self.PNRUseTunedAngles == 1) & (self.pWF['PNR_SINGLE_SPIN']!=1)

        self.PNRUseTunedAngles = jnp.where(cond, False, self.PNRUseTunedAngles)
        self.lalParams['PNRUseTunedAngles'] = jnp.where(cond, False, self.lalParams['PNRUseTunedAngles'])

        self.AntisymmetricWaveform = jnp.where(cond, False, self.AntisymmetricWaveform)
        self.lalParams['AntisymmetricWaveform'] = jnp.where(cond, False, self.lalParams['AntisymmetricWaveform'])

        self.lalParams['PNRUseTunedCoprec'] = jnp.where(cond, False, self.lalParams['PNRUseTunedCoprec'])

        #Calculate the effective precessing spin parameter (Schmidt et al, PRD 91, 024043, 2015): m1 > m2, so body 1 is the larger black hole
        self.A1             = 2.0 + (3.0 * self.m2) / (2.0 * self.m1)
        self.A2             = 2.0 + (3.0 * self.m1) / (2.0 * self.m2)
        self.ASp1           = self.A1 * self.S1_perp
        self.ASp2           = self.A2 * self.S2_perp

        #/* S_p = max(A1 S1_perp, A2 S2_perp) */
        num       = jnp.where(self.ASp2 > self.ASp1, self.ASp2, self.ASp1)
        den       = jnp.where(self.m2 > self.m1 , self.A2*self.m2_2, self.A1*self.m1_2)

        #/* chi_p = max(A1 * Sp1 , A2 * Sp2) / (A_i * m_i^2) where i is the index of the larger BH */
        self.chip      = num / den
        self.chi1L     = self.chi1z
        self.chi2L     = self.chi2z


        self.chi_p          = self.chip
        #// (PNRUseTunedCoprec)
        self.pWF['chi_p']        = self.chi_p
        self.phi0_aligned   = self.pWF['phi0']

        #/* Effective (dimensionful) aligned spin */
        self.SL             = self.chi1L*self.m1_2 + self.chi2L*self.m2_2

        #/* Effective (dimensionful) in-plane spin */
        self.Sperp          = self.chi_p  * self.m1_2                 # /* m1 > m2 */

        #self.MSA_ERROR      = 0

        self.pWF22AS = None
        #start of SpinTaylor code

        if self.manual_prescription_tag==3:
            print("Executing Spin Taylor code")
            self.L_MAX_PNR = max(self.lalParams['ModeArray'])
            flow = self.pWF['fMin']
            if self.pWF['deltaF']==0:
                self.pWF['deltaMF'] = get_deltaF_from_wfstruct(self.pWF)
            
            #if PNR angles are disabled, step back accordingly to the waveform's frequency grid step
            if self.lalParams['PNRUseTunedAngles'] == False:
                self.integration_buffer = jnp.where(self.pWF['deltaF'] > 0., 3. * self.pWF['deltaF'], 0.5)
                flow = (self.pWF['fMin'] - self.integration_buffer) * 2 / self.M_MAX
            
            #if PNR angles are enabled, adjust buffer to the requirements of IMRPhenomX_PNR_GeneratePNRAngleInterpolants

            else:
                #Compress line 336-340
                iStart_here = jnp.where(self.pWF['deltaF'] == 0., 0, jnp.floor(self.pWF['fMin'] / self.pWF['deltaF']).astype(int))
                flow = jnp.where(self.pWF['deltaF'] == 0., 0., iStart_here * self.pWF['deltaF'])
                fmin_HM_inspiral = flow * 2.0 / self.M_MAX

                precVersion = self.IMRPhenomXPrecVersion
                #fill in a fake value to allow the next code to work
                self.IMRPhenomXPrecVersion = 223
                IMRPhenomX_PNR_GetAndSetPNRVariables(self, pWF)
                #XLAL_CHECK(XLAL_SUCCESS == status, XLAL_EFUNC, "Error: IMRPhenomX_PNR_GetAndSetPNRVariables failed in IMRPhenomXGetAndSetPrecessionVariables.\n")

                alphaParams = IMRPhenomX_PNR_precompute_alpha_coefficients(self.pWF, self.pPrec)
                betaParams = IMRPhenomX_PNR_precompute_beta_coefficients(self.pWF, self.pPrec)
                connection_frequencies = IMRPhenomX_PNR_BetaConnectionFrequencies(betaParams)

                self.IMRPhenomXPrecVersion = precVersion

                Mf_alpha_upper = alphaParams['A4'] / 3.0
                Mf_low_cut = (3.0 / 3.5) * Mf_alpha_upper
                MF_high_cut = betaParams['Mf_beta_lower']

                #Compress line 375-380
                # First conditional assignment
                MF_high_cut = jnp.where(
                    jnp.logical_or(
                        MF_high_cut > self.pWF['fCutDef'],
                        MF_high_cut < 0.1 * self.pWF['fRING']
                    ),
                    self.pWF['fRING'],
                    MF_high_cut
                )

                # Second conditional assignment  
                Mf_low_cut = jnp.where(
                    jnp.logical_or
                        Mf_low_cut > self.pWF['fCutDef'],
                        MF_high_cut < Mf_low_cut
                    ),
                    MF_high_cut / 2.0,
                    Mf_low_cut
                )


                flow_alpha = XLALSimIMRPhenomXUtilsMftoHz(Mf_low_cut * 0.65 * self.M_MAX / 2.0, self.pWF['Mtot'])
                
                #flow is approximately in the intermediate region of the frequency map
                #conservatively reduce flow to account for potential problems in this region

                # Compute values for the else branch
                Mf_RD_22 = self.pWF['fRING']
                Mf_RD_lm = IMRPhenomXHM_GenerateRingdownFrequency(self.pPrec['L_MAX_PNR'], self.pPrec['M_MAX'], self.pWF)
                fmin_HM_ringdowm = XLALSimIMRPhenomXUtilsMftoHz(
                    XLALSimIMRPhenomXUtilsHztoMf(flow, self.pWF['Mtot']) - (Mf_RD_lm - Mf_RD_22), 
                    self.pWF['Mtot']
                )
                else_branch_result = jnp.where(
                    jnp.logical_and(fmin_HM_ringdowm < fmin_HM_inspiral, fmin_HM_ringdowm > 0.0),
                    fmin_HM_ringdowm,
                    fmin_HM_inspiral
                )

                # Main conditional
                flow = jnp.where(flow_alpha < flow, fmin_HM_inspiral / 1.5, else_branch_result)

                ## Line 397 - 402
                pnr_interpolation_deltaf = IMRPhenomX_PNR_HMInterpolationDeltaF(flow, pWF, pPrec)
                self.integration_buffer = 1.4*pnr_interpolation_deltaf
                flow = jnp.where(flow - 2.0 * pnr_interpolation_deltaf < 0, flow / 2.0, flow - 2.0 * pnr_interpolation_deltaf)

                iStart_here = jnp.floor(flow / pnr_interpolation_deltaf).astype(int)
                flow = iStart_here * pnr_interpolation_deltaf
                
                
            #XLAL_CHECK(flow>0.,XLAL_EDOM,"Error in %s: starting frequency for SpinTaylor angles must be positive!",__func__)
            PNarrays, fmin_integration = IMRPhenomX_InspiralAngles_SpinTaylor(chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, flow,self.IMRPhenomXPrecVersion, pWF, lalParams)                 
            #   // convert the min frequency of integration to geometric units for later convenience
            self.Mfmin_integration = XLALSimIMRPhenomXUtilsHztoMf(self.fmin_integration, self.pWF['Mtot'])

            if self.IMRPhenomXPrecVersion == 330:
                chi1x_evolved = chi1x
                chi1y_evolved = chi1y
                chi1z_evolved = chi1z
                chi2x_evolved = chi2x
                chi2y_evolved = chi2y
                chi2z_evolved = chi2z

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

                phi = jnp.atan2( Ly, Lx )
                theta = jnp.acos( Lz / jnp.sqrt(Lx*Lx + Ly*Ly + Lz*Lz) )

                _v = IMRPhenomX_rotate_z(-phi, jnp.array([chi1x_temp, chi1y_temp, chi1z_temp]))
                chi1x_temp, chi1y_temp, chi1z_temp = IMRPhenomX_rotate_y(-theta, _v)

                _v = IMRPhenomX_rotate_z(-phi, jnp.array([chi2x_temp, chi2y_temp, chi2z_temp]))
                chi2x_temp, chi2y_temp, chi2z_temp = IMRPhenomX_rotate_y(-theta, jnp.array([chi2x_temp, chi2y_temp, chi2z_temp]))

                chi1x_evolved = chi1x_temp
                chi1y_evolved = chi1y_temp
                chi1z_evolved = chi1z_temp


                chi2x_evolved = chi2x_temp
                chi2y_evolved = chi2y_temp
                chi2z_evolved = chi2z_temp


                self.chi1x_evolved = chi1x_evolved
                self.chi1y_evolved = chi1y_evolved
                self.chi1z_evolved = chi1z_evolved

                self.chi2x_evolved = chi2x_evolved
                self.chi2y_evolved = chi2y_evolved
                self.chi2z_evolved = chi2z_evolved

            #  // if PN numerical integration fails, default to MSA+fallback to NNLO
            if "Failure":
                "Place holder"
                print("Warning: due to a failure in the SpinTaylor routines, the model will default to MSA angles.")
                self.IMRPhenomXPrecVersion = 223

            #  // end of SpinTaylor code

        self.IMRPhenomXPrecVersion = jax.lax.cond(self.IMRPhenomXPrecVersion==self.manual_prescription_tag, self.spin_taylor_code, lambda _: 223, operand = None)

        # Compress line 486-490
        pflag = jnp.int32(self.IMRPhenomXPrecVersion)
        self.manual_prescription_tag = self.check_prescription_tag(self.IMRPhenomXPrecVersion) 

        # Skipping 491-494. Sanity check if pflag is in the allowd values [101, 102, 103, 104, 220, 221, 222, 223, 224, 310, 311, 320, 321, 330]
        # Skipping line  496-544. Looks like sanity check. FIXME: Source of error. 
        
        #/* Calculate parameter for two-spin to single-spin map used in PNR and XCP */
        #/* Initialize PNR variables */
        self.chi_singleSpin = 0.0
        self.costheta_singleSpin = 0.0
        self.costheta_final_singleSpin = 0.0
        self.chi_singleSpin_antisymmetric = 0.0
        self.theta_antisymmetric = 0.0
        self.PNR_HM_Mflow = 0.0
        self.PNR_HM_Mfhigh = 0.0

        self.PNR_q_window_lower = 0.0
        self.PNR_q_window_upper = 0.0
        self.PNR_chi_window_lower = 0.0
        self.PNR_chi_window_upper = 0.0
        #self.PNRInspiralScaling = 0

        PNR_variables = IMRPhenomX_PNR_GetAndSetPNRVariables(self.pWF, self.lalParams)

        self.alphaPNR = 0.0
        self.betaPNR = 0.0
        self.gammaPNR = 0.0

        #/*...#...#...#...#...#...#...#...#...#...#...#...#...#...#.../
        #/      Get and/or store CoPrec params into pWF and pPrec     /
        #/...#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/

        PNR_co_prec_params = IMRPhenomX_PNR_GetAndSetCoPrecParams(self.pWF, self.lalParams)

        #/*..#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Logic used in compressing 585 to 609
        # If flag is in [220, 221, 222, 223, 224], try MSA. 
            # If it fails, 
            #   check if the flag is in [220, 223, 224]
                # If yes, updated flag to 102
                # If no, terminal failure
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # Compress line 585 to 609 #FIXME: NH possible source of error
        MSA_ERROR = 0
        MSA_ERROR = self.compute_MSA_and_check_MSA_error(pflag)
        pflag = self.switch_to_NNLO(pflag, MSA_ERROR)
        
        # Skipping DEBUG line 610-616

        #/*...#...#...#...#...#...#...#...#...#...#...#...#...#...#.../
        #/      Compute and set final spin and RD frequency           /
        #/...#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/
        
        precessing_remnant_params = IMRPhenomX_SetPrecessingRemnantParams(self.pWF, self.lalParams)
        #/*..#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/


        # /* Useful powers of \chi_p */
        self.chip2    = self.chip * self.chip

        #/* Useful powers of spins aligned with L */
        self.chi1L2   = self.chi1L * self.chi1L
        self.chi2L2   = self.chi2L * self.chi2L

        #Cache the orbital angular momentum coefficients for future use.

        #References:
        #- Kidder, PRD, 52, 821-847, (1995), arXiv:gr-qc/9506022
        #- Blanchet, LRR, 17, 2, (2014), arXiv:1310.1528
        #- Bohe et al, 1212.5520v2
        #- Marsat, CQG, 32, 085008, (2015), arXiv:1411.4118

        # Compress line 641 - 755
        self.L0, self.L1, self.L2, self.L3, self.L4, self.L5, self.L6, self.L7, self.L8, self.L8L = self.twoPN_non_spinning_orbitan_angular_momentum(pflag)

        #Reference orbital angular momentum
        self.LRef = self.M * self.M * XLALSimIMRPhenomXLPNAnsatz(self.pWF['v_ref'], self.pWF['eta'] / self.pWF['v_ref'], self.L0, self.L1, self.L2, self.L3, self.L4, self.L5, self.L6, self.L7, self.L8, self.L8L) 

        '''
            In the following code block we construct the convetions that relate the source frame and the LAL frame.

            A detailed discussion of the conventions can be found in Appendix C and D of arXiv:2004.06503 and https://dcc.ligo.org/LIGO-T1500602
        '''

        #Get source frame (*_Sf) J = L + S1 + S2. This is an instantaneous frame in which L is aligned with z */
        self.J0x_Sf = (self.m1_2)*self.chi1x + (self.m2_2)*self.chi2x
        self.J0y_Sf = (self.m1_2)*self.chi1y + (self.m2_2)*self.chi2y
        self.J0z_Sf = (self.m1_2)*self.chi1z + (self.m2_2)*self.chi2z + self.LRef

        self.J0_Sf = jnp.array([self.J0x_Sf, self.J0y_Sf, self.J0z_Sf])


        self.J0     = jnp.sqrt(self.J0x_Sf*self.J0x_Sf + self.J0y_Sf*self.J0y_Sf + self.J0z_Sf*self.J0z_Sf)

        # Compress line 772 - 781
        #/* Get angle between J0 and LN (z-direction) */
        self.thetaJ_Sf = jax.lax.cond(self.J0<1e-10, lambda _: 0.0, lambda _:jnp.acos(self.J0z_Sf / self.J0), operand = None)

        # Line 783
        self.phiRef = self.pWF['phiRef_In']
        # Line 785
        phenom_xp_convention     = self.lalParams['PhenomXPConvention']

        tol_condition = (jnp.abs(self.J0x_Sf) < self.MAX_TOL_ATAN) & (jnp.abs(self.J0y_Sf) < self.MAX_TOL_ATAN)
        # Compress line 797-825
        #Get azimuthal angle of J0 in the source frame
        self.phiJ_Sf = self.get_phiJ_Sf(tol_condition, self.phiRef, phenom_xp_convention, self.J0_Sf)

        self.phi0_aligned = - self.phiJ_Sf

        #Compress line 828 - 846 #FIXME in function set_phi0 I am not sure what to do for cases 5, 6, 7. What is the old value?
        self.phi0 = self.set_phi0(phenom_xp_convention, self.phi0_aligned)

        '''
        Here we follow the same prescription as in IMRPhenomPv2:

        Now rotate from SF to J frame to compute alpha0, the azimuthal angle of LN, as well as
        thetaJ, the angle between J and N.

        The J frame is defined by imposing that J points in the z-direction and the line of sight N is in the xz-plane
        (with positive projection along x).

        The components of any vector in the (new) J-frame can be obtained by rotation from the (old) source frame (SF).
        This is done by multiplying by: RZ[-kappa].RY[-thetaJ].RZ[-phiJ]

        Note that kappa is determined by rotating N with RY[-thetaJ].RZ[-phiJ], which brings J to the z-axis, and
        taking the opposite of the azimuthal angle of the rotated N.
        '''

        #Determine kappa via rotations, as above */
        self.Nx_Sf = jnp.sin(self.pWF['inclination'])*jnp.cos((jnp.pi / 2.0) - self.phiRef)
        self.Ny_Sf = jnp.sin(self.pWF['inclination'])*jnp.sin((jnp.pi / 2.0) - self.phiRef)
        self.Nz_Sf = jnp.cos(self.pWF['inclination'])
        self.N_Sf = jnp.array([self.Nx_Sf, self.Ny_Sf, self.Nz_Sf])

        v_in = jnp.array([self.Nx_Sf, self.Ny_Sf, self.Nz_Sf])

        vout = IMRPhenomX_rotate_z(-self.phiJ_Sf, v_in)
        vout = IMRPhenomX_rotate_y(-self.thetaJ_Sf, vout)

        #/* Note difference in overall - sign w.r.t PhenomPv2 code */
        self.kappa = XLALSimIMRPhenomXatan2tol(vout[1],vout[0], self.MAX_TOL_ATAN)

        #/* Now determine alpha0 by rotating LN. In the source frame, LN = {0,0,1} */
        tmp_x = 0.0
        tmp_y = 0.0
        tmp_z = 1.0
        v_in = jnp.array([tmp_x, tmp_y, tmp_z])
        vout = IMRPhenomX_rotate_z(-self.phiJ_Sf,   v_in)
        vout = IMRPhenomX_rotate_y(-self.thetaJ_Sf, vout)
        vout = IMRPhenomX_rotate_z(-self.kappa,     vout)

        # Compress line 887 - 930
        tol_condition = (jnp.abs(vout[0]) < self.MAX_TOL_ATAN) & (jnp.abs(vout[1]) < self.MAX_TOL_ATAN)
        self.alpha0 = self.set_alpha0(tol_condition, phenom_xp_convention, vout[0], vout[1], self.kappa)
        

        # Compress line 931-966
        self.thetaJN, self.Nz_Jf, self.Nx_Jf = jax.lax.cond(jnp.isin(phenom_xp_convention, jnp.array([0, 5])), self.thetaJN_Nz_Nx_0_5, self.thetaJN_Nz_Nx_1_6_7, v_in, self.N_Sf, self.J0_Sf, self.phiJ_Sf, self.thetaJ_Sf, self.kappa)


        '''
        Define the polarizations used. This follows the conventions adopted for IMRPhenomPv2.

        The IMRPhenomP polarizations are defined following the conventions in Arun et al (arXiv:0810.5336),
        i.e. projecting the metric onto the P, Q, N triad defining where: P = (N x J) / |N x J|.

        However, the triad X,Y,N used in LAL (the "waveframe") follows the definition in the
        NR Injection Infrastructure (Schmidt et al, arXiv:1703.01076).

        The triads differ from each other by a rotation around N by an angle zeta. We therefore need to rotate 
        the polarizations by an angle 2 zeta.
        '''
        #Compressed line 983  to 991
        self.Xx_Sf = -jnp.cos(pWF['inclination']) * jnp.sin(self.phiRef)
        self.Xy_Sf = -jnp.cos(pWF['inclination']) * jnp.cos(self.phiRef)
        self.Xz_Sf = +jnp.sin(pWF['inclination'])


        v = jnp.array([self.Xx_Sf, self.Xy_Sf, self.Xz_Sf])
        vout = IMRPhenomX_rotate_z(-self.phiJ_Sf, v)
        vout = IMRPhenomX_rotate_y(-self.thetaJ_Sf, vout)
        vout = IMRPhenomX_rotate_z(-self.kappa, vout)


        '''

            The components tmp_i are now the components of X in the J frame.

            We now need the polar angle of this vector in the P, Q basis of Arun et al:

                P = (N x J) / |NxJ|

            Note, that we put N in the (pos x)z half plane of the J frame 

        '''
        #Compress line 1002-1034
        self.PArun_Jf, self.QArun_Jf = jax.lax.cond(jnp.isin(phenom_xp_convention, jnp.array([0, 5])), self.PQ_Arun_0_5, self.PQ_Arun_1_6_7, self.Nx_Jf, self.Nz_Jf)

        #As it is line 1035-1043
        #(X . P)
        self.XdotPArun = (vout[0] * self.PArun_Jf[0]) + (vout[1] * self.PArun_Jf[1]) + (vout[2] * self.PArun_Jf[2])

        #(X . Q)
        self.XdotQArun = (vout[0] * self.QArun_Jf[0]) + (vout[1] * self.QArun_Jf[1]) + (vout[2] * self.QArun_Jf[2])

        #Now get the angle zeta
        self.zeta_polarization = jnp.atan2(self.XdotQArun, self.XdotPArun)

        #/* ********** PN Euler Angle Coefficients ********** */
        #/*
        #    This uses the single spin PN Euler angles as per IMRPhenomPv2
        #*/  

        #/* ********** PN Euler Angle Coefficients ********** */
        # Compress line 1050-1143
        cond = jnp.isin(pflag, jnp.array([101, 102, 103, 104]))
        self.alpha1, self.alpha2, self.alpha3, self.alpha4L, self.alpha5, self.epsilon1, self.epsilon2, self.epsilon3, self.epsilon4L, self.epsilon5 = jax.lax.cond(cond, self.compute_alpha_epsilon_101_104, self.compute_alpha_epsilon_220_330, operand = None)

        #Skipping the #if DEBUG==1 lines 1144-1162
        
        # Compressed line 1163-1177
        self.epsilon0 = self.set_epsilon0(phenom_xp_convention, self.phiJ_Sf)

        ## Compression line 1178-1202
        cond = (phenom_xp_convention == 5) | (phenom_xp_convention==7)
        self.alpha_offset, self.epsilon_offset, self.alpha_offset_1, self.epsilon_offset_1, self.alpha_offset_3, self.epsilon_offset_3, self.alpha_offset_4, self.epsilon_offset_4 =  jax.lax.cond(cond, self.convention_five_or_seven_true, self.convention_five_or_seven_false, operand = self.alpha0)

        self.cexp_i_alpha   = 0.
        self.cexp_i_epsilon = 0.
        self.cexp_i_betah   = 0.

        # When L + SL < 0 and q>7, we disable multibanding NH: I will skip this function
        #self.IMRPhenomXPCheckMaxOpeningAngle()

        # Activate multibanding for Euler angles it threshold !=0. Only for PhenomXPHM. */
        self.MBandPrecVersion = jax.lax.cond(self.lalParams['PhenomXPHMThresholdMband']==0, lambda _: 0, lambda _: 1, operand = None)
        ## NH: I do not implement PhenomXPHMThresholdMband==1 option. The output of the above line will always be self.MBandPrecVersion = 0. 


        # At high mass ratios, we find there can be numerical instabilities in the model, although the waveforms continue to be well behaved.
        # We warn to user of the possibility of these instabilities.
        # printf(pWF->q)
        jax.lax.cond(pWF["q"] > 80, 
                     lambda _: jax.debug.print("Very high mass ratio, possibility of numerical instabilities. Waveforms remain well behaved."), 
                     lambda _: None, 
                     operand = None)


        self.Y2m2         = compute_sminus2_l2(theta = self.thetaJN, m = -2)
        self.Y2m1         = compute_sminus2_l2(theta = self.thetaJN, m = -1)
        self.Y20          = compute_sminus2_l2(theta = self.thetaJN, m = 0)
        self.Y21          = compute_sminus2_l2(theta = self.thetaJN, m = 1)
        self.Y22          = compute_sminus2_l2(theta = self.thetaJN, m = 2)

        self.Y3m3         = compute_sminus2_l3(theta = self.thetaJN, m = -3)
        self.Y3m2         = compute_sminus2_l3(theta = self.thetaJN, m = -2)
        self.Y3m1         = compute_sminus2_l3(theta = self.thetaJN, m = -1)
        self.Y30          = compute_sminus2_l3(theta = self.thetaJN, m = 0)
        self.Y31          = compute_sminus2_l3(theta = self.thetaJN, m = 1)
        self.Y32          = compute_sminus2_l3(theta = self.thetaJN, m = 2)
        self.Y33          = compute_sminus2_l3(theta = self.thetaJN, m = 3)

        self.Y4m4         = compute_sminus2_l4(theta = self.thetaJN, m = -4)
        self.Y4m3         = compute_sminus2_l4(theta = self.thetaJN, m = -3)
        self.Y4m2         = compute_sminus2_l4(theta = self.thetaJN, m = -2)
        self.Y4m1         = compute_sminus2_l4(theta = self.thetaJN, m = -1)
        self.Y40          = compute_sminus2_l4(theta = self.thetaJN, m = 0)
        self.Y41          = compute_sminus2_l4(theta = self.thetaJN, m = 1)
        self.Y42          = compute_sminus2_l4(theta = self.thetaJN, m = 2)
        self.Y43          = compute_sminus2_l4(theta = self.thetaJN, m = 3)
        self.Y44          = compute_sminus2_l4(theta = self.thetaJN, m = 4)

    def check_kerr_bound(self, pnr_use_tuned_angles, pnr_single_spin, chi1_norm, chi2_norm):
        """Function to compress line 209-213"""
        
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
        
        # Use where to conditionally raise error or return success
        # In practice, you might want to return the error condition
        # and handle it upstream
        return jnp.where(
            error_condition,
            False,  # Error case
            True    # Success case
            )

    def convention_five_or_seven_true(self, alpha0):
        return -alpha0, 0, -alpha0, 0, -alpha0, 0, -alpha0, 0
    
    def convention_five_or_seven_false(self, alpha0):
        # Get initial Get \alpha and \epsilon offsets at \omega = pi * M * f_{Ref} */
        alpha_offset, epsilon_offset = self.Get_alphaepsilon_atfref(2)
        return alpha_offset, epsilon_offset, alpha_offset, epsilon_offset, alpha_offset, epsilon_offset, alpha_offset, epsilon_offset
    
    def Get_alphaepsilon_atfref(self, mprime):
        omega_ref = self.pWF['piM'] * self.pWF['fRef'] * 2 / mprime
        pflag = self.IMRPhenomXPrecVersion

        #/* Explicitly enumerate MSA flags */
        cond = (pflag == 220) | (pflag == 221) | (pflag == 222) | (pflag == 223) | (pflag == 224)

        alpha_offset, epsilon_offset = jax.lax.cond(cond, self.Get_alphaepsilon_atfref_pflag_true, self.Get_alphaepsilon_atfref_pflag_false, omega_ref)

        return alpha_offset, epsilon_offset
    
    def Get_alphaepsilon_atfref_pflag_true(self, omega_ref):

        v = jnp.cbrt(omega_ref)
        vangles  = IMRPhenomX_Return_phi_zeta_costhetaL_MSA(self, v, self.pWF) # FIXME

        alpha_offset = vangles['x'] - self.alpha0
        epsilon_offset = vangles['x'] - self.epsilon0
        return alpha_offset, epsilon_offset
    
    def Get_alphaepsilon_atfref_pflag_false(self, omega_ref):
        logomega_ref    = jnp.log(omega_ref)
        omega_ref_cbrt  = jnp.cbrt(omega_ref)
        omega_ref_cbrt2 = omega_ref_cbrt * omega_ref_cbrt

        alpha_offset = (self.alpha1  / omega_ref 
                        + self.alpha2  / omega_ref_cbrt2 
                        + self.alpha3  / omega_ref_cbrt 
                        + self.alpha4L * logomega_ref 
                        + self.alpha5  * omega_ref_cbrt - self.alpha0)

        epsilon_offset = (self.epsilon1  / omega_ref 
                          + self.epsilon2  / omega_ref_cbrt2 
                          + self.epsilon3  / omega_ref_cbrt 
                          + self.epsilon4L * logomega_ref 
                          + self.epsilon5  * omega_ref_cbrt - self.epsilon0)

        return alpha_offset, epsilon_offset
    
    def set_epsilon0(self, phenom_xp_convention, phiJ_Sf):

        epsilon0 = jax.lax.cond(
            jnp.isin(phenom_xp_convention, jnp.array([1, 6])),
            lambda _: phiJ_Sf - jnp.pi,
            lambda _: 0.0,
            operand=None,
        )
        
        return epsilon0
    

    def compute_alpha_epsilon_101_104(self, _):
        #This uses the single spin PN Euler angles as per IMRPhenomPv2
        #Post-Newtonian Euler Angles: alpha */
        chiL = (1.0 + self.q) * (self.chi_eff / self.q)
        chiL2 = chiL * chiL

        alpha1 = -35/192. + (5*self.delta)/(64.*self.m1)
        alpha2 = ((15*chiL*self.delta*self.m1)/128. - (35*chiL*self.m1_2)/128.) / self.eta
        alpha3 = (-5515/3072. + self.eta*(-515/384. - (15*self.delta2)/(256.*self.m1_2) 
                                          + (175*self.delta)/(256.*self.m1)) + (4555*self.delta)/(7168.*self.m1) 
                                          + ((15*self.chip2*self.delta*self.m1_3)/128. - (35*self.chip2*self.m1_4)/128.) / self.eta2)
        #This is the term proportional to log(w) */

        alpha4L = ((5*chiL*self.delta2)/16. - (5*chiL*self.delta*self.m1)/3. + (2545*chiL*self.m1_2)/1152.
                   + ((-2035*chiL*self.delta*self.m1)/21504. 
                      + (2995*chiL*self.m1_2)/9216.)/self.eta
                   + ((5*chiL*self.chip2*self.delta*self.m1_5)/128. 
                      - (35*chiL*self.chip2*self.m1_6)/384.) / self.eta3
                   - (35*jnp.pi)/48. + (5*self.delta*jnp.pi)/(16.*self.m1))
        
        alpha5 = (5*(-190512*self.delta3*self.eta6 + 2268*self.delta2*self.eta3*self.m1*(self.eta2*(323 + 784*self.eta) + 336*(25*chiL2 + self.chip2)*self.m1_4)
                + 7*self.m1_3*(8024297*self.eta4 + 857412*self.eta5 + 3080448*self.eta6
                               + 143640*self.chip2*self.eta2*self.m1_4 - 127008*self.chip2*(-4*chiL2 + self.chip2)*self.m1_8
                               + 6048*self.eta3*((2632*chiL2 + 115*self.chip2)*self.m1_4 - 672*chiL*self.m1_2*jnp.pi))
                + 3*self.delta*self.m1_2*(-5579177*self.eta4 + 80136*self.eta5 - 3845520*self.eta6
                                           + 146664*self.chip2*self.eta2*self.m1_4 + 127008*self.chip2*(-4*chiL2 + self.chip2)*self.m1_8
                                           - 42336*self.eta3*((726*chiL2 + 29*self.chip2)*self.m1_4 - 96*chiL*self.m1_2*jnp.pi)))) / (6.5028096e7*self.eta4*self.m1_3)


        epsilon1 = -35/192. + (5*self.delta)/(64.*self.m1)
        epsilon2 = ((15*chiL*self.delta*self.m1)/128. - (35*chiL*self.m1_2)/128.) / self.eta
        epsilon3 = -5515/3072. + self.eta*(-515/384. - (15*self.delta2)/(256.*self.m1_2) + (175*self.delta)/(256.*self.m1)) + (4555*self.delta)/(7168.*self.m1)
        epsilon4L = (5*chiL*self.delta2)/16. - (5*chiL*self.delta*self.m1)/3. + (2545*chiL*self.m1_2)/1152. \
                    + ((-2035*chiL*self.delta*self.m1)/21504. + (2995*chiL*self.m1_2)/9216.) / self.eta \
                    - (35*jnp.pi)/48. + (5*self.delta*jnp.pi)/(16.*self.m1)
        epsilon5 = (5*(-190512*self.delta3*self.eta3 + 2268*self.delta2*self.m1*(self.eta2*(323 + 784*self.eta) + 8400*chiL2*self.m1_4)
                        - 3*self.delta*self.m1_2*(self.eta*(5579177 + 504*self.eta*(-159 + 7630*self.eta)) + 254016*chiL*self.m1_2*(121*chiL*self.m1_2 - 16*jnp.pi))
                        + 7*self.m1_3*(self.eta*(8024297 + 36*self.eta*(23817 + 85568*self.eta)) + 338688*chiL*self.m1_2*(47*chiL*self.m1_2 - 12*jnp.pi)))) / (6.5028096e7*self.eta*self.m1_3)

        return alpha1, alpha2, alpha3, alpha4L, alpha5, epsilon1, epsilon2, epsilon3, epsilon4L, epsilon5
    
    def compute_alpha_epsilon_220_330(self, _):
        return 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  
    

    def PQ_Arun_0_5(self, Nx_Jf, Nz_Jf):
        #Get polar angle of X vector in J frame in the P,Q basis of Arun et al
        PArunx_Jf = 0.0
        PAruny_Jf = -1.0
        PArunz_Jf = 0.0

        #Q = (N x P) by construction
        QArunx_Jf = Nz_Jf
        QAruny_Jf = 0.0
        QArunz_Jf = -Nx_Jf

        return jnp.array([PArunx_Jf, PAruny_Jf, PArunz_Jf]), jnp.array([QArunx_Jf, QAruny_Jf, QArunz_Jf])

    def PQ_Arun_1_6_7(self, Nx_Jf, Nz_Jf):
        # Get polar angle of X vector in J frame in the P,Q basis of Arun et al
        PArunx_Jf = Nz_Jf
        PAruny_Jf = 0.0
        PArunz_Jf = -Nx_Jf

        QArunx_Jf = 0.0
        QAruny_Jf = 1.0
        QArunz_Jf = 0.0

        return jnp.array([PArunx_Jf, PAruny_Jf, PArunz_Jf]), jnp.array([QArunx_Jf, QAruny_Jf, QArunz_Jf])
    

    def thetaJN_Nz_Nx_0_5(self, v_in, N_Sf, J0_Sf, phiJ_Sf, thetaJ_Sf, kappa):
        # Line 937-952 #
        #Now determine thetaJN by rotating N
        
        v = IMRPhenomX_rotate_z(phiJ_Sf,   v_in)
        v = IMRPhenomX_rotate_y(thetaJ_Sf, v)
        v = IMRPhenomX_rotate_z(kappa,     v)

        # We don't need the y-component but we will store it anyway

        Nz_Jf = v[2]
        # This is a unit vector, so no normalization
        thetaJN = jnp.acos(Nz_Jf)

        return thetaJN, v[2], v[0]

    def thetaJN_Nz_Nx_1_6_7(self, v_in, N_Sf, J0_Sf, phiJ_Sf, thetaJ_Sf, kappa):
        # Line 957-962

        J0dotN     = (J0_Sf[0] * N_Sf[0]) + (J0_Sf[1] * N_Sf[1]) + (J0_Sf[2] * N_Sf[2])
        thetaJN = jnp.acos( J0dotN / self.J0 )
        Nz_Jf     = jnp.cos(thetaJN)
        Nx_Jf     = jnp.sin(thetaJN)

        return thetaJN, Nz_Jf, Nx_Jf

    def set_phi0(self, phenom_xp_convention, phi0_aligned):
        phi0 = jnp.where(phenom_xp_convention == 0, phi0_aligned, 0.0)
        phi0 = jnp.where(phenom_xp_convention == 1, 0.0, phi0) 
        return phi0
    
    def set_alpha0(self, tol_condition, phenom_xp_convention, tmp_x, tmp_y, kappa):
        convention_condition = jnp.isin(phenom_xp_convention, self.check_convention_array)

        alpha0 = jax.lax.cond(tol_condition,
            lambda _: jax.lax.cond(convention_condition, lambda _2: jnp.pi, lambda _2: jnp.pi - kappa, operand = None),
            lambda _: jax.lax.cond(convention_condition, lambda _2: jnp.atan2(tmp_y, tmp_x), lambda _2: jnp.pi - kappa, operand = None),
            operand=None)
        
        return alpha0
    
    def get_phiJ_Sf(self, tol_condition, phiRef, phenom_xp_convention, J0_Sf):
        convention_condition = jnp.isin(phenom_xp_convention, self.check_convention_array)

        phiJ_Sf = jax.lax.cond(tol_condition, 
                     lambda _: jax.lax.cond(convention_condition, lambda xx: jnp.pi/2.0 - phiRef, lambda yy: 0.0, operand = None), 
                     lambda _: jnp.atan2(J0_Sf[1], J0_Sf[0]), 
                     operand = None)
        
        return phiJ_Sf

    def twoPN_non_spinning_orbitan_angular_momentum(self, pflag):

        # Branch functions
        def case_101(_):
            return self.flag_101_twoPN_non_spinning_orbitan_angular_momentum()

        def case_102_330(_):
            return self.flag_102_330_twoPN_non_spinning_orbitan_angular_momentum()

        def case_222_223(_):
            return self.flag_222_223_twoPN_non_spinning_orbitan_angular_momentum()

        def case_103(_):
            return self.flag_103_twoPN_non_spinning_orbitan_angular_momentum()

        def case_104(_):
            return self.flag_104_twoPN_non_spinning_orbitan_angular_momentum()

        branches = (
            case_101,
            case_102_330,
            case_222_223,
            case_103,
            case_104,
        )

        idx = jnp.where(pflag == 101, 0,
            jnp.where(jnp.isin(pflag, jnp.array([102, 220, 221, 224, 310, 311, 320, 321, 330])), 1,
            jnp.where(jnp.isin(pflag, jnp.array([222, 223])), 2,
            jnp.where(pflag == 103, 3,
            jnp.where(pflag == 104, 4, -1)))))

        return jax.lax.switch(idx, branches, operand=None)

    def flag_101_twoPN_non_spinning_orbitan_angular_momentum(self):
    
        L0   = 1.0
        L1   = 0.0
        L2   = ((3.0/2.0) + (self.eta/6.0))
        L3   = 0.0
        L4   = (81.0 + (-57.0 + self.eta)*self.eta)/24.
        L5   = 0.0
        L6   = 0.0
        L7   = 0.0
        L8   = 0.0
        L8L  = 0.0

        return  L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L
    
    def flag_102_330_twoPN_non_spinning_orbitan_angular_momentum(self):
        # 3PN orbital angular momentum 
        L0   = 1.0
        L1   = 0.0
        L2   = 3.0/2. + self.eta/6.0
        L3   = (5*(self.chi1L*(-2 - 2*self.delta + self.eta) + self.chi2L*(-2 + 2*self.delta + self.eta)))/6.
        L4   = (81 + (-57 + self.eta)*self.eta)/24.
        L5   = (-7*(self.chi1L*(72 + self.delta*(72 - 31*self.eta) + self.eta*(-121 + 2*self.eta)) + self.chi2L*(72 + self.eta*(-121 + 2*self.eta) + self.delta*(-72 + 31*self.eta))))/144.
        L6   = (10935 + self.eta*(-62001 + self.eta*(1674 + 7*self.eta) + 2214*self.power_of_lalpi_2))/1296.
        L7   = 0.0
        L8   = 0.0

        #This is the log(x) term
        L8L  = 0.0
        return L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L
    
    def flag_222_223_twoPN_non_spinning_orbitan_angular_momentum(self):
        L0   = 1.0
        L1   = 0.0
        L2   = 3.0/2. + self.eta/6.0
        L3   = (-7*(self.chi1L + self.chi2L + self.chi1L*self.delta - self.chi2L*self.delta) + 5*(self.chi1L + self.chi2L)*self.eta)/6.
        L4   = (81 + (-57 + self.eta)*self.eta)/24.
        L5   = (-1650*(self.chi1L + self.chi2L + self.chi1L*self.delta - self.chi2L*self.delta) + 1336*(self.chi1L + self.chi2L)*self.eta + 511*(self.chi1L - self.chi2L)*self.delta*self.eta + 28*(self.chi1L + self.chi2L)*self.eta2)/600.
        L6   = (10935 + self.eta*(-62001 + 1674*self.eta + 7*self.eta2 + 2214*self.power_of_lalpi_2))/1296.
        L7   = 0.0
        L8   = 0.0

        #This is the log(x) term
        L8L  = 0.0
        #break
        return L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L
    
    def flag_103_twoPN_non_spinning_orbitan_angular_momentum(self):

        L0   = 1.0
        L1   = 0.0
        L2   = 3.0/2. + self.eta/6.0
        L3   = (5*(self.chi1L*(-2 - 2*self.delta + self.eta) + self.chi2L*(-2 + 2*self.delta + self.eta)))/6.
        L4   = (81 + (-57 + self.eta)*self.eta)/24.
        L5   = (-7*(self.chi1L*(72 + self.delta*(72 - 31*self.eta) + self.eta*(-121 + 2*self.eta)) + self.chi2L*(72 + self.eta*(-121 + 2*self.eta) + self.delta*(-72 + 31*self.eta))))/144.
        L6   = (10935 + self.eta*(-62001 + self.eta*(1674 + 7*self.eta) + 2214*self.power_of_lalpi_2))/1296.
        L7   = (self.chi2L*(-324 + self.eta*(1119 - 2*self.eta*(172 + self.eta)) + self.delta*(324 + self.eta*(-633 + 14*self.eta)))
                            - self.chi1L*(324 + self.eta*(-1119 + 2*self.eta*(172 + self.eta)) + self.delta*(324 + self.eta*(-633 + 14*self.eta))))/32.
        L8   = 2835/128. - (self.eta*(-10677852 + 100*self.eta*(-640863 + self.eta*(774 + 11*self.eta))
                        + 26542080*GAMMA + 675*(3873 + 3608*self.eta)*self.power_of_lalpi_2))/622080. - (64*self.eta*self.log16)/3.

        L8L  = -(64.0/3.0) * self.eta


        return  L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L
    
    def flag_104_twoPN_non_spinning_orbitan_angular_momentum(self):
        L0   = 1.0
        L1   = 0.0
        L2   = 3.0/2. + self.eta/6.0
        L3   = (5*(self.chi1L*(-2 - 2*self.delta + self.eta) + self.chi2L*(-2 + 2*self.delta + self.eta)))/6.
        L4   = (81 + (-57 + self.eta)*self.eta)/24.
        L5   = (-7*(self.chi1L*(72 + self.delta*(72 - 31*self.eta) + self.eta*(-121 + 2*self.eta)) + self.chi2L*(72 + self.eta*(-121 + 2*self.eta) + self.delta*(-72 + 31*self.eta))))/144.
        L6   = (10935 + self.eta*(-62001 + self.eta*(1674 + 7*self.eta) + 2214*self.power_of_lalpi_2))/1296.
        L7   = (self.chi2L*(-324 + self.eta*(1119 - 2*self.eta*(172 + self.eta)) + self.delta*(324 + self.eta*(-633 + 14*self.eta)))
                            - self.chi1L*(324 + self.eta*(-1119 + 2*self.eta*(172 + self.eta)) + self.delta*(324 + self.eta*(-633 + 14*self.eta))))/32.
        L8   = 2835/128. - (self.eta*(-10677852 + 100*self.eta*(-640863 + self.eta*(774 + 11*self.eta))
                        + 26542080*GAMMA + 675*(3873 + 3608*self.eta)*self.power_of_lalpi_2))/622080. - (64*self.eta*self.log16)/3.

        #This is the log(x) term at 4PN, x^4/2 * log(x)
        L8L  = -(64.0/3.0) * self.eta

        #Leading order in spin at all PN orders, note that the 1.5PN terms are already included. Here we have additional 2PN and 3.5PN corrections.
        L4  += (self.chi1L2*(1 + self.delta - 2*self.eta) + 4*self.chi1L*self.chi2L*self.eta - self.chi2L2*(-1 + self.delta + 2*self.eta))/2.
        L7  +=  (3*(self.chi1L + self.chi2L)*self.eta*(self.chi1L2*(1 + self.delta - 2*self.eta) + 4*self.chi1L*self.chi2L*self.eta - self.chi2L2*(-1 + self.delta + 2*self.eta)))/4.

        return  L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L


    def compute_MSA_and_check_MSA_error(self, pflag:int)->int:

        MSA_conditions = jnp.isin(pflag, jnp.array([220, 221, 222, 223, 224]))

        MSA_ERROR = 0

        MSA_ERROR = jax.lax.cond(MSA_conditions,
                                      IMRPhenomX_Initialize_MSA_System,
                                      lambda _: MSA_ERROR,
                                      operand = None)
        return MSA_ERROR
    
    def switch_to_NNLO(self, pflag:int, MSA_ERROR:int)->int:

        sub_condition = jnp.isin(pflag, jnp.array([220, 223, 224]))
        condition = (MSA_ERROR==1) & (sub_condition)
        
        return jax.lax.cond(condition,
                             lambda _: 102,
                             lambda _: pflag,
                             operand = None)

    
    def check_prescription_tag(self, n:int)->int:
        '''
        Retuns the first digit of a three digit number
        1 means NNLO
        2 means MSA
        3 means Spin Taylor
        '''
        return jnp.int32(n/100)
        

    def compute_flow_spin_taylor(self, PNRUseTunedAngles: int)->float:
        """
        Substitute function for line 324-340 in LALSimIMRPhenomX_precession.c script
        """
        
        def PNRTuned_true(_):
            return jnp.where(self.pWF['deltaF']==0.0, self.pWF['fMin'], jnp.floor_divide(self.pWF['fMin'], self.pWF['deltaF'])*self.pWF['deltaF'])
        
        def PNRTuned_false(_):
            M_MAX = 1.0 #FIXME this is definietly not the right value
            integration_buffer = jnp.where(self.pWF['deltaF']>0.0, 3*self.pWF['deltaF'], 0.5) 
            return (self.pWF['fMin'] - integration_buffer)*2 / M_MAX
        
        return jax.lax.cond(self.PNRUseTunedAngles, PNRTuned_true, PNRTuned_false, operand = None)


    def spin_taylor_code():
        '''

        Output of the spin taylor code is chi1_evolved and chi_2_evolved

        '''
        return 330

'''
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
End of IMRPhenomXGetAndSetPrecessionVaraibles
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
'''




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

#FIXME
def IMRPhenomX_SetPrecessingRemnantParams(pWF, lalParams):
    return None


#FIXME
def IMRPhenomX_PNR_GetAndSetPNRVariables(*args):
    #https://github.com/GW-JAX-Team/ripple/blob/f46fe3610bce9b0c82927f1de966190cf5e0ae53/src/ripplegw/waveforms/imr_phenom_xphm/lal_sim_imr_phenom_x_pnr_internals.py#L25
    return None


#FIXME
def IMRPhenomX_PNR_GetAndSetCoPrecParams(*args):
    return None


def XLALSimIMRPhenomXUtilsMftoHz(Mf, Mtot_Msun):
    return Mf / (MTSUN*Mtot_Msun)




#FIXME
def IMRPhenomX_GetandSetModes(ModeArray: list, IMRPhenomXPrecessionStruct: dict):
    return None



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




def IMRPhenomX_Return_phi_zeta_costhetaL_MSA(pPrec, v, pWF):
    # Wrapper to generate \f$\phi_z\f$, \f$\zeta\f$ and \f$\cos \theta_L\f$ at a given frequency

    vout = jnp.array([0, 0, 0])
    pPrec = None


    L_norm = pWF['eta']/v
    J_norm = IMRPhenomX_JNorm_MSA(L_norm, pPrec)

    L_norm3PN       = 0.0

    # Compressing line 2212 - 2220
    cond = (pPrec.IMRPhenomXPrecVersion == 222) | (pPrec.IMRPhenomXPrecVersion == 223)
    L_norm3PN = jax.lax.cond(cond, IMRPhenomX_L_norm_3PN_of_v, XLALSimIMRPhenomXLPNAnsatz, v, L_norm, pPrec)

    '''
    if (pPrec.IMRPhenomXPrecVersion == 222) | (pPrec.IMRPhenomXPrecVersion == 223):
        L_norm3PN = IMRPhenomX_L_norm_3PN_of_v(v, v*v, L_norm, pPrec)

    else:
        L_norm3PN = XLALSimIMRPhenomXLPNAnsatz(v, L_norm, pPrec.L0, pPrec.L1, pPrec.L2, pPrec.L3, pPrec.L4, pPrec.L5, pPrec.L6, pPrec.L7, pPrec.L8, pPrec.L8L)
    '''
    

    J_norm3PN = IMRPhenomX_JNorm_MSA(L_norm3PN, pPrec)
    vRoots    = IMRPhenomX_Return_Roots_MSA(L_norm, J_norm, pPrec)


    pPrec.S32  = vRoots.x
    pPrec.Smi2 = vRoots.y
    pPrec.Spl2 = vRoots.z

    pPrec.Spl2mSmi2   = pPrec.Spl2 - pPrec.Smi2
    pPrec.Spl2pSmi2   = pPrec.Spl2 + pPrec.Smi2
    pPrec.Spl         = jnp.sqrt(pPrec.Spl2)
    pPrec.Smi         = jnp.sqrt(pPrec.Smi2)

    SNorm = IMRPhenomX_Return_SNorm_MSA(v, pPrec)
    pPrec.S_norm      = SNorm
    pPrec.S_norm_2    = SNorm * SNorm

    vMSA = {0.,0.,0.}

    # Compressing line 2245-2249
    vMSA_correction = IMRPhenomX_Return_MSA_Corrections_MSA(v, L_norm, J_norm, pPrec)
    cond = (jnp.abs(pPrec.Smi2 - pPrec.Spl2) > 1.e-5)
    vMSA = jnp.where(cond, vMSA_correction, vMSA)
    '''
    if(jnp.abs(pPrec.Smi2 - pPrec.Spl2) > 1.e-5):
    
        #Get phiz_0_MSA and zeta_0_MSA
        vMSA = IMRPhenomX_Return_MSA_Corrections_MSA(v, L_norm, J_norm, pPrec)
    '''

    phiz_MSA     = vMSA.x
    zeta_MSA     = vMSA.y

    phiz         = IMRPhenomX_Return_phiz_MSA(v, J_norm, pPrec)
    zeta         = IMRPhenomX_Return_zeta_MSA(v, pPrec)
    cos_theta_L        = IMRPhenomX_costhetaLJ(L_norm3PN, J_norm3PN, SNorm)

    vout[0] = phiz + phiz_MSA
    vout[1] = zeta + zeta_MSA
    vout[2] = cos_theta_L

    return vout


def IMRPhenomX_JNorm_MSA(LNorm:float, pPrec)->float:
    JNorm2 = (LNorm * LNorm + 2.0 * LNorm * pPrec.c1_over_eta + pPrec.SAv2)
    return jnp.sqrt(JNorm2)


def IMRPhenomX_L_norm_3PN_of_v(v: jax.Array, L_norm: float, pPrec)->float:
    v2 = v*v
    term_4 = pPrec.constants_L[4]
    term_3 = pPrec.constants_L[3]
    term_2 = pPrec.constants_L[2]
    term_1 = pPrec.constants_L[1]
    term_0 = pPrec.constants_L[0]
    L_norm3PN = L_norm*(1. + v2*(term_0 + v*term_1 + v2*(term_2 + v*term_3 + v2*(term_4))))

    return L_norm3PN


def IMRPhenomX_Return_SNorm_MSA(v, pPrec):

    v2 = v * v

    cancel_condition = jnp.abs(pPrec.Smi2 - pPrec.Spl2) < 1e-5

    def sn_zero(_):
        sn = jnp.array(0.0)
        return sn

    def sn_jacobi(_):
        # Equation 25 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
        m = (pPrec.Smi2 - pPrec.Spl2) / (pPrec.S32 - pPrec.Spl2)

        psi = IMRPhenomX_psiofv(
            v, v2,
            pPrec.psi0, pPrec.psi1, pPrec.psi2,
            pPrec
        )

        # Jacobi elliptic functions
        sn, cn, dn = gsl_sf_elljac_e(psi, m) # FIXME
        return sn

    sn = jax.lax.cond(cancel_condition, sn_zero, sn_jacobi, operand=None)

    # Equation 23 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    SNorm2 = pPrec.Spl2 + (pPrec.Smi2 - pPrec.Spl2) * sn * sn

    return jnp.sqrt(SNorm2)



def gsl_sf_elljac_e(x, y):
    """
    TODO: Not yet implemented
    """
    return jnp.array(1.), jnp.array(2.), jnp.array(3.)



def IMRPhenomX_costhetaLJ(
    L_norm: float, 
    J_norm: float, 
    S_norm: float
    ) -> float:
    costhetaLJ = 0.5 * (J_norm**2 + L_norm**2 - S_norm**2) / L_norm * J_norm

    # Clamp the value to the interval [-1.0, 1.0]
    costhetaLJ = jnp.clip(costhetaLJ, -1.0, 1.0)

    return costhetaLJ




def IMRPhenomXHM_GenerateRingdownFrequency(ell: int, emm: int, wf22: dict) -> float:
    """
    Wrapper function to return ringdown frequency
    
    Args:
        ell: Spherical harmonic l index (int)
        emm: Spherical harmonic m index (int) - guaranteed to be positive
        wf22: Waveform structure dictionary (dict)
        
    Returns:
        float: Ringdown frequency
    """
    # emm is guaranteed to be positive
    modeTag = ell * 10 + emm
    
    # if the tuned coprecessing tuning is activated, use the precessing final spin
    afinal = jnp.where(
        wf22['IMRPhenomXPNRUseTunedCoprec'],
        wf22['afinal_prec'],
        wf22['afinal']
    )
    
    def case_21():
        return evaluate_QNMfit_fring21(afinal) / wf22['Mfinal']
    
    def case_22():
        return wf22['fRING']
    
    def case_33():
        return evaluate_QNMfit_fring33(afinal) / wf22['Mfinal']
    
    def case_32():
        return evaluate_QNMfit_fring32(afinal) / wf22['Mfinal']
    
    def case_44():
        return evaluate_QNMfit_fring44(afinal) / wf22['Mfinal']
    
    def default_case():
        # In JAX, we can't raise errors in jit-compiled code
        # Return NaN or handle error upstream
        return jnp.nan
    
    # Create branches for each case
    fRING = jax.lax.switch(
        jnp.where(modeTag == 21, 0,
        jnp.where(modeTag == 22, 1,
        jnp.where(modeTag == 33, 2,
        jnp.where(modeTag == 32, 3,
        jnp.where(modeTag == 44, 4, 5))))),
        [case_21, case_22, case_33, case_32, case_44, default_case]
    )
    
    # if the coprecessing tuning is activated, return Effective RD frequency
    fRING = jnp.where(
        jnp.logical_and(
            wf22['IMRPhenomXPNRUseTunedCoprec'],
            jnp.logical_or(ell != 2, emm != 2)
        ),
        fRING - emm * wf22['fRINGEffShiftDividedByEmm'],
        fRING
    )
    
    return fRING

def XLALSimIMRPhenomXUtilsHztoMf(fHz: float, Mtot_Msun: float) -> float:

    """
    Convert frequency from Hz to dimensionless units
    
    Args:
        fHz: Frequency in Hz (float)
        Mtot_Msun: Total mass in solar masses (float)
        
    Returns:
        float: Frequency in dimensionless units
    """
    return fHz * (MTSUN * Mtot_Msun)

def evaluate_QNMfit_fring21(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fring21
    
    Args:
        finalDimlessSpin: Final dimensionless spin (float)
        
    Returns:
        float: QNM frequency fit result
    """
    
    # Check bounds - return NaN for invalid input
    # (In JAX, we can't raise errors in jit-compiled code)
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0
    
    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    
    numerator = (0.059471695665734674 - 0.07585416297991414*finalDimlessSpin + 
                 0.021967909664591865*x2 - 0.0018964744613388146*x3 + 
                 0.001164879406179587*x4 - 0.0003387374454044957*x5)
    
    denominator = (1 - 1.4437415542456158*finalDimlessSpin + 0.49246920313191234*x2)
    
    return_val = numerator / denominator
    
    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)

def evaluate_QNMfit_fring33(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fring33
    
    Args:
        finalDimlessSpin: Final dimensionless spin (float)
        
    Returns:
        float: QNM frequency fit result
    """
    
    # Check bounds - return NaN for invalid input
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0
    
    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    
    numerator = (0.09540436245212061 - 0.22799517865876945*finalDimlessSpin + 
                 0.13402916709362475*x2 + 0.03343753057911253*x3 - 
                 0.030848060170259615*x4 - 0.006756504382964637*x5 + 
                 0.0027301732074159835*x6)
    
    denominator = (1 - 2.7265947806178334*finalDimlessSpin + 2.144070539525238*x2 - 
                   0.4706873667569393*x4 + 0.05321818246993958*x6)
    
    return_val = numerator / denominator
    
    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)

def evaluate_QNMfit_fring32(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fring32
    
    Args:
        finalDimlessSpin: Final dimensionless spin (float)
        
    Returns:
        float: QNM frequency fit result
    """
    
    # Check bounds - return NaN for invalid input
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0
    
    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    
    numerator = (0.09540436245212061 - 0.13628306966373951*finalDimlessSpin + 
                 0.030099881830507727*x2 - 0.000673589757007597*x3 + 
                 0.0118277880067919*x4 + 0.0020533816327907334*x5 - 
                 0.0015206141948469621*x6)
    
    denominator = (1 - 1.6531854335715193*finalDimlessSpin + 0.5634705514193629*x2 + 
                   0.12256204148002939*x4 - 0.027297817699401976*x6)
    
    return_val = numerator / denominator
    
    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)

def evaluate_QNMfit_fring44(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fring44
    
    Args:
        finalDimlessSpin: Final dimensionless spin (float)
        
    Returns:
        float: QNM frequency fit result
    """
    
    # Check bounds - return NaN for invalid input
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0
    
    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    
    numerator = (0.1287821193485683 - 0.21224284094693793*finalDimlessSpin + 
                 0.0710926778043916*x2 + 0.015487322972031054*x3 - 
                 0.002795401084713644*x4 + 0.000045483523029172406*x5 + 
                 0.00034775290179000503*x6)
    
    denominator = (1 - 1.9931645124693607*finalDimlessSpin + 1.0593147376898773*x2 - 
                   0.06378640753152783*x4)
    
    return_val = numerator / denominator
    
    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)


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


#/** This function initializes all the core variables required for the MSA system. This will be called first. */
def IMRPhenomX_Initialize_MSA_System(pWF: dict, pPrec: dict, ExpansionOrder: int):

    #Sanity check on the precession version
    pflag = pPrec.IMRPhenomXPrecVersion
    if pflag not in [220, 221, 222, 223, 224]:
        raise ValueError("Error: MSA system requires IMRPhenomXPrecVersion 220, 221, 222, 223 or 224.")
    
    '''
      First initialize the system of variables needed for Chatziioannou et al, PRD, 88, 063011, (2013), arXiv:1307.4418:
        - Racine et al, PRD, 80, 044010, (2009), arXiv:0812.4413
        - Favata, PRD, 80, 024002, (2009), arXiv:0812.0069
        - Blanchet et al, PRD, 84, 064041, (2011), arXiv:1104.5659
        - Bohe et al, CQG, 30, 135009, (2013), arXiv:1303.7412
    '''

    eta = pPrec['eta']
    eta2 = pPrec['eta2']
    eta3 = pPrec['eta3']
    eta4 = pPrec['eta4']

    m1 = pWF['m1']
    m2 = pWF['m2']

    # PN Coefficients for d \omega / d t as per LALSimInspiralFDPrecAngles_internals.c 
    LAL_LN2 = jnp.log(2.0)
    domegadt_constants_NS = [
        96./5., -1486./35., -264./5., 384.*jnp.pi/5., 34103./945., 13661./105., 944./15., 
        jnp.pi*(-4159./35.), jnp.pi*(-2268./5.), 
        (16447322263./7276500. + jnp.pi*jnp.pi*512./5. - LAL_LN2*109568./175. - GAMMA*54784./175.),
        (-56198689./11340. + jnp.pi*jnp.pi*902./5.), 1623./140., -1121./27., -54784./525., 
        -jnp.pi*883./42., jnp.pi*71735./63., jnp.pi*73196./63.
    ]

    domegadt_constants_SO = [
        -904./5., -120., -62638./105., 4636./5., -6472./35., 3372./5., -jnp.pi*720., 
        -jnp.pi*2416./5., -208520./63., 796069./105., -100019./45., -1195759./945., 
        514046./105., -8709./5., -jnp.pi*307708./105., jnp.pi*44011./7., 
        -jnp.pi*7992./7., jnp.pi*151449./35.
    ]

    domegadt_constants_SS = [-494./5., -1442./5., -233./5., -719./5.]

    L_csts_nonspin = [
        3./2., 1./6., 27./8., -19./8., 1./24., 135./16., 
        -6889./144. + 41./24.*jnp.pi*jnp.pi, 31./24., 7./1296.
    ]

    L_csts_spinorbit = [-14./6., -3./2., -11./2., 133./72., -33./8., 7./4.]

    '''
        Note that Chatziioannou et al use q = m2/m1, where m1 > m2 and therefore q < 1
        IMRPhenomX assumes m1 > m2 and q > 1. For the internal MSA code, flip q and
        dump this to pPrec->qq, where qq explicitly dentoes that this is 0 < q < 1.
    '''

    q = m2 / m1  # m2 / m1, q < 1, m1 > m2
    invq = 1.0 / q  # m1 / m2, invq > 1, m1 > m2
    pPrec['qq'] = q
    pPrec['invqq'] = invq

    mu = (m1 * m2) / (m1 + m2)

    #    /* \delta and powers of \delta in terms of q < 1, should just be m1 - m2 */
    pPrec['delta_qq'] = (1.0 - pPrec['qq']) / (1.0 + pPrec['qq'])
    pPrec['delta2_qq'] = pPrec['delta_qq'] * pPrec['delta_qq']
    pPrec['delta3_qq'] = pPrec['delta_qq'] * pPrec['delta2_qq']
    pPrec['delta4_qq'] = pPrec['delta_qq'] * pPrec['delta3_qq']

    # Initialize empty vectors (using dictionaries to represent vectors)
    S1v = jnp.array([0.0, 0.0, 0.0])
    S2v =jnp.array([0.0, 0.0, 0.0])

    # Define source frame such that \hat{L} = {0,0,1} with L_z pointing along \hat{z}
    Lhat =jnp.array([0.0, 0.0, 1.0])

    # Set LHat variables - these are fixed
    pPrec['Lhat_cos_theta'] = 1.0  # Cosine of Polar angle of orbital angular momentum
    pPrec['Lhat_phi'] = 0.0        # Azimuthal angle of orbital angular momentum
    pPrec['Lhat_theta'] = 0.0      # Polar angle of orbital angular momentum

    # Dimensionful spin vectors, note eta = m1 * m2 and q = m2/m1
    S1v['x'] = pPrec['chi1x'] * eta/q  # eta / q = m1^2
    S1v['y'] = pPrec['chi1y'] * eta/q
    S1v['z'] = pPrec['chi1z'] * eta/q

    S2v['x'] = pPrec['chi2x'] * eta*q  # eta * q = m2^2
    S2v['y'] = pPrec['chi2y'] * eta*q
    S2v['z'] = pPrec['chi2z'] * eta*q

    S1_0_norm = IMRPhenomX_vector_L2_norm(S1v)  # stub
    S2_0_norm = IMRPhenomX_vector_L2_norm(S2v)  # stub


    # Initial dimensionful spin vectors at reference frequency
    # S1 = {S1x,S1y,S1z}
    pPrec['S1_0'] = jnp.array([S1v[0], S1v[1], S1v[2]])

    # S2 = {S2x,S2y,S2z}  
    pPrec['S2_0'] = jnp.array([S2v[0], S2v[1], S2v[2]])

    # Reference velocity v and v^2
    pPrec['v_0'] = jnp.power(pPrec['piGM'] * pWF['fRef'], 1.0/3.0)
    pPrec['v_0_2'] = pPrec['v_0'] * pPrec['v_0']

    # Reference orbital angular momenta
    L_0 = jnp.array([0.0, 0.0, 0.0])
    L_0 = IMRPhenomX_vector_scalar(Lhat, pPrec['eta'] / pPrec['v_0'])  # stub
    pPrec['L_0'] = L_0    

    # Inner products used in MSA system
    dotS1L = IMRPhenomX_vector_dot_product(S1v, Lhat)  # stub
    dotS2L = IMRPhenomX_vector_dot_product(S2v, Lhat)  # stub
    dotS1S2 = IMRPhenomX_vector_dot_product(S1v, S2v)  # stub
    dotS1Ln = dotS1L / S1_0_norm
    dotS2Ln = dotS2L / S2_0_norm

    # Add dot products to struct
    pPrec['dotS1L'] = dotS1L
    pPrec['dotS2L'] = dotS2L
    pPrec['dotS1S2'] = dotS1S2
    pPrec['dotS1Ln'] = dotS1Ln
    pPrec['dotS2Ln'] = dotS2Ln


    # Coefficients for PN orbital angular momentum at 3PN, as per LALSimInspiralFDPrecAngles_internals.c
    pPrec['constants_L'] = jnp.zeros(5)  # Initialize array for 5 coefficients

    pPrec['constants_L'] = pPrec['constants_L'].at[0].set(
        L_csts_nonspin[0] + eta * L_csts_nonspin[1]
    )
    pPrec['constants_L'] = pPrec['constants_L'].at[1].set(
        IMRPhenomX_Get_PN_beta(L_csts_spinorbit[0], L_csts_spinorbit[1], pPrec)  # stub
    )
    pPrec['constants_L'] = pPrec['constants_L'].at[2].set(
        L_csts_nonspin[2] + eta * L_csts_nonspin[3] + eta * eta * L_csts_nonspin[4]
    )
    pPrec['constants_L'] = pPrec['constants_L'].at[3].set(
        IMRPhenomX_Get_PN_beta(
            (L_csts_spinorbit[2] + L_csts_spinorbit[3] * eta), 
            (L_csts_spinorbit[4] + L_csts_spinorbit[5] * eta), 
            pPrec
        )  # stub
    )
    pPrec['constants_L'] = pPrec['constants_L'].at[4].set(
        L_csts_nonspin[5] + L_csts_nonspin[6] * eta + L_csts_nonspin[7] * eta * eta + L_csts_nonspin[8] * eta * eta * eta
    )

    # Effective total spin
    Seff = (1.0 + q) * pPrec['dotS1L'] + (1 + (1.0/q)) * pPrec['dotS2L']
    Seff2 = Seff * Seff

    pPrec['Seff'] = Seff
    pPrec['Seff2'] = Seff2

    #Line 2347
    # Initial total spin, S = S1 + S2
    S0 = jnp.array([0.0, 0.0, 0.0])
    S0 = IMRPhenomX_vector_sum(S1v, S2v)  # stub

    # Cache total spin in the precession struct
    pPrec['S_0'] = S0
    
    #    /* Initial total angular momentum, J = L + S1 + S2 */
    pPrec['J_0'] = IMRPhenomX_vector_sum(pPrec['L_0'],pPrec['S_0'])

    # Norm of total initial spin
    pPrec['S_0_norm'] = IMRPhenomX_vector_L2_norm(S0)
    pPrec['S_0_norm_2'] = pPrec['S_0_norm'] * pPrec['S_0_norm']

    # Norm of orbital and total angular momenta
    pPrec['L_0_norm'] = IMRPhenomX_vector_L2_norm(pPrec['L_0'])
    pPrec['J_0_norm'] = IMRPhenomX_vector_L2_norm(pPrec['J_0'])

    L0norm = pPrec['L_0_norm']
    J0norm = pPrec['J_0_norm']

    # Useful powers
    pPrec['S_0_norm_2'] = pPrec['S_0_norm'] * pPrec['S_0_norm']
    pPrec['J_0_norm_2'] = pPrec['J_0_norm'] * pPrec['J_0_norm']
    pPrec['L_0_norm_2'] = pPrec['L_0_norm'] * pPrec['L_0_norm']

    # Vector for obtaining B, C, D coefficients
    vBCD = IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(
        pPrec['L_0_norm'], pPrec['J_0_norm'], pPrec)
    

    vRoots = jnp.array([0.0, 0.0, 0.0])

    vRoots = IMRPhenomX_Return_Roots_MSA(pPrec['L_0_norm'],pPrec['J_0_norm'],pPrec)
    
    #Line 2500

    pPrec['Spl2'] = vRoots[2]
    pPrec['Smi2'] = vRoots[1]
    pPrec['S32'] = vRoots[0]

    # S^2_+ + S^2_-
    pPrec['Spl2pSmi2'] = pPrec['Spl2'] + pPrec['Smi2']

    # S^2_+ - S^2_-
    pPrec['Spl2mSmi2'] = pPrec['Spl2'] - pPrec['Smi2']

    # S_+ and S_-
    pPrec['Spl'] = jnp.sqrt(pPrec['Spl2'])
    pPrec['Smi'] = jnp.sqrt(pPrec['Smi2'])

    # Eq. 45 of PRD 95, 104004, (2017), arXiv:1703.03967, set from initial conditions
    pPrec['SAv2'] = 0.5 * (pPrec['Spl2pSmi2'])
    pPrec['SAv'] = jnp.sqrt(pPrec['SAv2'])
    pPrec['invSAv2'] = 1.0 / pPrec['SAv2']
    pPrec['invSAv'] = 1.0 / pPrec['SAv']


    # c_1 is determined by Eq. 41 of PRD, 95, 104004, (2017), arXiv:1703.03967
    c_1 = 0.5 * (J0norm*J0norm - L0norm*L0norm - pPrec['SAv2']) / pPrec['L_0_norm'] * eta
    c1_2 = c_1 * c_1

    # Useful powers and combinations of c_1
    pPrec['c1'] = c_1
    pPrec['c12'] = c_1 * c_1
    pPrec['c1_over_eta'] = c_1 / eta

    # Average spin couplings over one precession cycle: A9 - A14 of arXiv:1703.03967
    omqsq = (1.0 - q) * (1.0 - q) + 1e-16
    omq2 = (1.0 - q * q) + 1e-16

    # Precession averaged spin couplings, Eq. A9 - A14 of arXiv:1703.03967, note that we only use the initial values
    pPrec['S1L_pav'] = (c_1 * (1.0 + q) - q * eta * Seff) / (eta * omq2)
    pPrec['S2L_pav'] = -q * (c_1 * (1.0 + q) - eta * Seff) / (eta * omq2)
    pPrec['S1S2_pav'] = 0.5 * pPrec['SAv2'] - 0.5 * (pPrec['S1_norm_2'] + pPrec['S2_norm_2'])
    pPrec['S1Lsq_pav'] = (pPrec['S1L_pav']*pPrec['S1L_pav'] + 
                        ((pPrec['Spl2mSmi2'])*(pPrec['Spl2mSmi2']) * pPrec['v_0_2']) / (32.0 * eta2 * omqsq))
    pPrec['S2Lsq_pav'] = (pPrec['S2L_pav']*pPrec['S2L_pav'] + 
                        (q*q*(pPrec['Spl2mSmi2'])*(pPrec['Spl2mSmi2']) * pPrec['v_0_2']) / (32.0 * eta2 * omqsq))
    pPrec['S1LS2L_pav'] = (pPrec['S1L_pav']*pPrec['S2L_pav'] - 
                        q * (pPrec['Spl2mSmi2'])*(pPrec['Spl2mSmi2'])*pPrec['v_0_2'] / (32.0 * eta2 * omqsq))
    
    # Spin couplings in arXiv:1703.03967
    pPrec['beta3'] = (((113./12.) + (25./4.)*(m2/m1)) * pPrec['S1L_pav'] + 
                    ((113./12.) + (25./4.)*(m1/m2)) * pPrec['S2L_pav'])

    pPrec['beta5'] = (((31319./1008.) - (1159./24.)*eta) + (m2/m1)*((809./84) - (281./8.)*eta)) * pPrec['S1L_pav'] + \
                    (((31319./1008.) - (1159./24.)*eta) + (m1/m2)*((809./84) - (281./8.)*eta)) * pPrec['S2L_pav']

    pPrec['beta6'] = jnp.pi * (((75./2.) + (151./6.)*(m2/m1))*pPrec['S1L_pav'] + 
                            ((75./2.) + (151./6.)*(m1/m2))*pPrec['S2L_pav'])

    pPrec['beta7'] = (((130325./756) - (796069./2016)*eta + (100019./864.)*eta2) + 
                    (m2/m1)*((1195759./18144) - (257023./1008.)*eta + (2903/32.)*eta2)) * pPrec['S1L_pav'] + \
                    (((130325./756) - (796069./2016)*eta + (100019./864.)*eta2) + 
                    (m1/m2)*((1195759./18144) - (257023./1008.)*eta + (2903/32.)*eta2)) * pPrec['S2L_pav']

    pPrec['sigma4'] = ((1.0/mu) * ((247./48.)*pPrec['S1S2_pav'] - (721./48.)*pPrec['S1L_pav']*pPrec['S2L_pav']) + 
                    (1.0/(m1*m1)) * ((233./96.)*pPrec['S1_norm_2'] - (719./96.)*pPrec['S1Lsq_pav']) + 
                    (1.0/(m2*m2)) * ((233./96.)*pPrec['S2_norm_2'] - (719./96.)*pPrec['S2Lsq_pav']))
    

    # Compute PN coefficients using precession-averaged spin couplings
    pPrec['a0'] = 96.0 * eta / 5.0

    # These are all normalized by a factor of a0
    pPrec['a2'] = -(743./336.) - (11.0/4.)*eta
    pPrec['a3'] = 4.0 * jnp.pi - pPrec['beta3']
    pPrec['a4'] = (34103./18144.) + (13661./2016.)*eta + (59./18.)*eta2 - pPrec['sigma4']
    pPrec['a5'] = -(4159./672.)*jnp.pi - (189./8.)*jnp.pi*eta - pPrec['beta5']
    pPrec['a6'] = ((16447322263./139708800.) + (16./3.)*jnp.pi*jnp.pi - (856./105)*jnp.log(16.) - 
                (1712./105.)*GAMMA - pPrec['beta6'] + 
                eta*((451./48)*jnp.pi*jnp.pi - (56198689./217728.)) + 
                eta2*(541./896.) - eta3*(5605./2592.))
    pPrec['a7'] = (-(4415./4032.)*jnp.pi + (358675./6048.)*jnp.pi*eta + 
                (91495./1512.)*jnp.pi*eta2 - pPrec['beta7'])

    # Coefficients are weighted by an additional factor of a_0
    pPrec['a2'] *= pPrec['a0']
    pPrec['a3'] *= pPrec['a0']
    pPrec['a4'] *= pPrec['a0']
    pPrec['a5'] *= pPrec['a0']
    pPrec['a6'] *= pPrec['a0']
    pPrec['a7'] *= pPrec['a0']


    #Line 2597
    if pflag == 222 or pflag == 223:
        pPrec['a0'] = eta * domegadt_constants_NS[0]
        pPrec['a2'] = eta * (domegadt_constants_NS[1] + eta * (domegadt_constants_NS[2]))
        pPrec['a3'] = eta * (domegadt_constants_NS[3] + 
                            IMRPhenomX_Get_PN_beta(domegadt_constants_SO[0], domegadt_constants_SO[1], pPrec))
        pPrec['a4'] = eta * (domegadt_constants_NS[4] + eta * (domegadt_constants_NS[5] + eta * (domegadt_constants_NS[6])) + 
                            IMRPhenomX_Get_PN_sigma(domegadt_constants_SS[0], domegadt_constants_SS[1], pPrec) +  # stub
                            IMRPhenomX_Get_PN_tau(domegadt_constants_SS[2], domegadt_constants_SS[3], pPrec))    # stub
        pPrec['a5'] = eta * (domegadt_constants_NS[7] + eta * (domegadt_constants_NS[8]) + 
                            IMRPhenomX_Get_PN_beta((domegadt_constants_SO[2] + eta * (domegadt_constants_SO[3])), 
                                                    (domegadt_constants_SO[4] + eta * (domegadt_constants_SO[5])), 
                                                    pPrec))

    

    # Useful powers of a_0
    pPrec['a0_2'] = pPrec['a0'] * pPrec['a0']
    pPrec['a0_3'] = pPrec['a0_2'] * pPrec['a0']
    pPrec['a2_2'] = pPrec['a2'] * pPrec['a2']

    # Calculate g coefficients as in Appendix A of Chatziioannou et al, PRD, 95, 104004, (2017), arXiv:1703.03967.
    # These constants are used in TaylorT2 where domega/dt is expressed as an inverse polynomial
    pPrec['g0'] = 1.0 / pPrec['a0']

    # Eq. A2 (1703.03967)
    pPrec['g2'] = -(pPrec['a2'] / pPrec['a0_2'])

    # Eq. A3 (1703.03967)
    pPrec['g3'] = -(pPrec['a3'] / pPrec['a0_2'])

    # Eq.A4 (1703.03967)
    pPrec['g4'] = -(pPrec['a4'] * pPrec['a0'] - pPrec['a2_2']) / pPrec['a0_3']

    # Eq. A5 (1703.03967)
    pPrec['g5'] = -(pPrec['a5'] * pPrec['a0'] - 2.0 * pPrec['a3'] * pPrec['a2']) / pPrec['a0_3']

    # Useful powers of delta
    delta = pPrec['delta_qq']
    delta2 = delta * delta
    delta3 = delta * delta2
    delta4 = delta * delta3

    # These are the phase coefficients of Eq. 51 of PRD, 95, 104004, (2017), arXiv:1703.03967
    pPrec['psi0'] = 0.0
    pPrec['psi1'] = 0.0
    pPrec['psi2'] = 0.0

    # \psi_1 is defined in Eq. C1 of Appendix C in PRD, 95, 104004, (2017), arXiv:1703.03967
    pPrec['psi1'] = 3.0 * (2.0 * eta2 * Seff - c_1) / (eta * delta2)

    c_1_over_nu = pPrec['c1_over_eta']
    c_1_over_nu_2 = c_1_over_nu * c_1_over_nu
    one_p_q_sq = (1.0 + q) * (1.0 + q)
    Seff_2 = Seff * Seff
    q_2 = q * q
    one_m_q_sq = (1.0 - q) * (1.0 - q)
    one_m_q2_2 = (1.0 - q_2) * (1.0 - q_2)
    one_m_q_4 = one_m_q_sq * one_m_q_sq

    # This implements the Delta term as in LALSimInspiralFDPrecAngles.c
    # c.f. https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiralFDPrecAngles_internals.c#L145
    if pflag == 222 or pflag == 223:
        Del1 = 4.0 * c_1_over_nu_2 * one_p_q_sq
        Del2 = 8.0 * c_1_over_nu * q * (1.0 + q) * Seff
        Del3 = 4.0 * (one_m_q2_2 * pPrec['S1_norm_2'] - q_2 * Seff_2)
        Del4 = 4.0 * c_1_over_nu_2 * q_2 * one_p_q_sq
        Del5 = 8.0 * c_1_over_nu * q_2 * (1.0 + q) * Seff
        Del6 = 4.0 * (one_m_q2_2 * pPrec['S2_norm_2'] - q_2 * Seff_2)
        pPrec['Delta'] = jnp.sqrt(jnp.abs((Del1 - Del2 - Del3) * (Del4 - Del5 - Del6)))
    else:
        # Coefficients of \Delta as defined in Eq. C3 of Appendix C in PRD, 95, 104004, (2017), arXiv:1703.03967.
        term1 = c1_2 * eta / (q * delta4)
        term2 = -2.0 * c_1 * eta3 * (1.0 + q) * Seff / (q * delta4)
        term3 = -eta2 * (delta2 * pPrec['S1_norm_2'] - eta2 * Seff_2) / delta4
        
        # Is this 1) (c1_2 * q * eta / delta4) or 2) c1_2*eta2/delta4?
        # - In paper.pdf, the expression 1) is used.
        # Using eta^2 leads to higher frequency oscillations, use q * eta
        term4 = c1_2 * eta * q / delta4
        term5 = -2.0 * c_1 * eta3 * (1.0 + q) * Seff / delta4
        term6 = -eta2 * (delta2 * pPrec['S2_norm_2'] - eta2 * Seff_2) / delta4
        pPrec['Delta']  = jnp.sqrt( jnp.abs( (term1 + term2 + term3) * (term4 + term5 + term6) ) )

    # Line 2706

    if pflag == 222 or pflag == 223:
        u1 = 3.0 * pPrec['g2'] / pPrec['g0']
        u2 = 0.75 * one_p_q_sq / one_m_q_4
        u3 = -20.0 * c_1_over_nu_2 * q_2 * one_p_q_sq
        u4 = 2.0 * one_m_q2_2 * (q * (2.0 + q) * pPrec['S1_norm_2'] + (1.0 + 2.0 * q) * pPrec['S2_norm_2'] - 2.0 * q * pPrec['SAv2'])
        u5 = 2.0 * q_2 * (7.0 + 6.0 * q + 7.0 * q_2) * 2.0 * c_1_over_nu * Seff
        u6 = 2.0 * q_2 * (3.0 + 4.0 * q + 3.0 * q_2) * Seff_2
        u7 = q * pPrec['Delta']
        
        # Eq. C2 (1703.03967)
        pPrec['psi2'] = u1 + u2 * (u3 + u4 + u5 - u6 + u7)
    else:
        # \psi_2 is defined in Eq. C2 of Appendix C in PRD, 95, 104004, (2017). Here we implement system of equations as in paper.pdf
        term1 = 3.0 * pPrec['g2'] / pPrec['g0']
        
        # q^2 or no q^2 in term2? Consensus on retaining q^2 term: https://git.ligo.org/waveforms/reviews/phenompv3hm/issues/7
        term2 = 3.0 * q * q / (2.0 * eta3)
        term3 = 2.0 * pPrec['Delta']
        term4 = -2.0 * eta2 * pPrec['SAv2'] / delta2
        term5 = -10.0 * eta * c1_2 / delta4
        term6 = 2.0 * eta2 * (7.0 + 6.0 * q + 7.0 * q * q) * c_1 * Seff / (omqsq * delta2)
        term7 = -eta3 * (3.0 + 4.0 * q + 3.0 * q * q) * Seff_2 / (omqsq * delta2)
        term8 = eta * (q * (2.0 + q) * pPrec['S1_norm_2'] + (1.0 + 2.0 * q) * pPrec['S2_norm_2']) / omqsq
        
        # \psi_2, C2 of Appendix C of PRD, 95, 104004, (2017)
        pPrec['psi2'] = term1 + term2 * (term3 + term4 + term5 + term6 + term7 + term8)


    # Eq. D1 of PRD, 95, 104004, (2017), arXiv:1703.03967
    Rm = pPrec['Spl2'] - pPrec['Smi2']
    Rm_2 = Rm * Rm

    # Eq. D2 and D3 Appendix D of PRD, 95, 104004, (2017), arXiv:1703.03967
    cp = pPrec['Spl2'] * eta2 - c1_2
    cm = pPrec['Smi2'] * eta2 - c1_2

    # Check if cm goes negative, this is likely pathological. If so, set MSA_ERROR to 1, so that waveform generator can handle
    # the error appropriately
    # if cm < 0.0:
    #     pPrec['MSA_ERROR'] = 1
    #     print(f"Error, coefficient cm = {cm:.16f}, which is negative and likely to be pathological. Triggering MSA failure.")

    # jnp.abs is here to help enforce positive definite cpcm
    cpcm = jnp.abs(cp * cm)
    sqrt_cpcm = jnp.sqrt(cpcm)

    # Eq. D4 in PRD, 95, 104004, (2017), arXiv:1703.03967 ; Note difference to published version.
    a1dD = 0.5 + 0.75/eta

    # Eq. D5 in PRD, 95, 104004, (2017), arXiv:1703.03967
    a2dD = -0.75*Seff/eta

    # Eq. E3 in PRD, 95, 104004, (2017), arXiv:1703.03967 ; Note that this is Rm * D2
    D2RmSq = (cp - sqrt_cpcm) / eta2

    # Eq. E4 in PRD, 95, 104004, (2017), arXiv:1703.03967 ; Note that this is Rm^2 * D4
    D4RmSq = -0.5*Rm*sqrt_cpcm/eta2 - cp/eta4*(sqrt_cpcm - cp)

    S0m = pPrec['S1_norm_2'] - pPrec['S2_norm_2']

    # Difference of spin norms squared, as used in Eq. D6 of PRD, 95, 104004, (2017), arXiv:1703.03967
    aw = (-3.0*(1.0 + q)/q*(2.0*(1.0 + q)*eta2*Seff*c_1 - (1.0 + q)*c1_2 + (1.0 - q)*eta2*S0m))
    cw = 3.0/32.0/eta*Rm_2
    dw = 4.0*cp - 4.0*D2RmSq*eta2
    hw = -2.0*(2.0*D2RmSq - Rm)*c_1
    fw = Rm*D2RmSq - D4RmSq - 0.25*Rm_2

    adD = aw / dw
    hdD = hw / dw
    cdD = cw / dw
    fdD = fw / dw

    gw = 3.0/16.0/eta2/eta*Rm_2*(c_1 - eta2*Seff)
    gdD = gw / dw

    # Useful powers of the coefficients
    hdD_2 = hdD * hdD
    adDfdD = adD * fdD
    adDfdDhdD = adDfdD * hdD
    adDhdD_2 = adD * hdD_2
    
    #Line 2800


    # Eq. D10 in PRD, 95, 104004, (2017), arXiv:1703.03967
    pPrec['Omegaz0'] = a1dD + adD

    # Eq. D11 in PRD, 95, 104004, (2017), arXiv:1703.03967
    pPrec['Omegaz1'] = a2dD - adD*Seff - adD*hdD

    # Eq. D12 in PRD, 95, 104004, (2017), arXiv:1703.03967
    pPrec['Omegaz2'] = adD*hdD*Seff + cdD - adD*fdD + adD*hdD_2

    # Eq. D13 in PRD, 95, 104004, (2017), arXiv:1703.03967
    pPrec['Omegaz3'] = (adDfdD - cdD - adDhdD_2)*(Seff + hdD) + adDfdDhdD

    # Eq. D14 in PRD, 95, 104004, (2017), arXiv:1703.03967
    pPrec['Omegaz4'] = (cdD + adDhdD_2 - 2.0*adDfdD)*(hdD*Seff + hdD_2 - fdD) - adD*fdD*fdD

    # Eq. D15 in PRD, 95, 104004, (2017), arXiv:1703.03967
    pPrec['Omegaz5'] = ((cdD - adDfdD + adDhdD_2) * fdD * (Seff + 2.0*hdD) - 
                        (cdD + adDhdD_2 - 2.0*adDfdD) * hdD_2 * (Seff + hdD) - 
                        adDfdD*fdD*hdD)
        
    # If Omegaz5 > 1000, this is larger than we expect and the system may be pathological.
    # Set MSA_ERROR = 1 to trigger an error
    if jnp.abs(pPrec['Omegaz5']) > 1000.0:
        pPrec['MSA_ERROR'] = 1
        print(f"Warning, |Omegaz5| = {pPrec['Omegaz5']:.16f}, which is larger than expected and may be pathological. Triggering MSA failure.")

    g0 = pPrec['g0']

    # Coefficients of Eq. 65, as defined in Equations D16 - D21 of PRD, 95, 104004, (2017), arXiv:1703.03967
    pPrec['Omegaz0_coeff'] = 3.0 * g0 * pPrec['Omegaz0']
    pPrec['Omegaz1_coeff'] = 3.0 * g0 * pPrec['Omegaz1']
    pPrec['Omegaz2_coeff'] = 3.0 * (g0 * pPrec['Omegaz2'] + pPrec['g2']*pPrec['Omegaz0'])
    pPrec['Omegaz3_coeff'] = 3.0 * (g0 * pPrec['Omegaz3'] + pPrec['g2']*pPrec['Omegaz1'] + pPrec['g3']*pPrec['Omegaz0'])
    pPrec['Omegaz4_coeff'] = 3.0 * (g0 * pPrec['Omegaz4'] + pPrec['g2']*pPrec['Omegaz2'] + pPrec['g3']*pPrec['Omegaz1'] + pPrec['g4']*pPrec['Omegaz0'])
    pPrec['Omegaz5_coeff'] = 3.0 * (g0 * pPrec['Omegaz5'] + pPrec['g2']*pPrec['Omegaz3'] + pPrec['g3']*pPrec['Omegaz2'] + pPrec['g4']*pPrec['Omegaz1'] + pPrec['g5']*pPrec['Omegaz0'])

    # Coefficients of zeta: in Appendix E of PRD, 95, 104004, (2017), arXiv:1703.03967
    c1oveta2 = c_1 / eta2
    pPrec['Omegazeta0'] = pPrec['Omegaz0']
    pPrec['Omegazeta1'] = pPrec['Omegaz1'] + pPrec['Omegaz0'] * c1oveta2
    pPrec['Omegazeta2'] = pPrec['Omegaz2'] + pPrec['Omegaz1'] * c1oveta2
    pPrec['Omegazeta3'] = pPrec['Omegaz3'] + pPrec['Omegaz2'] * c1oveta2 + gdD
    pPrec['Omegazeta4'] = pPrec['Omegaz4'] + pPrec['Omegaz3'] * c1oveta2 - gdD*Seff - gdD*hdD
    pPrec['Omegazeta5'] = pPrec['Omegaz5'] + pPrec['Omegaz4'] * c1oveta2 + gdD*hdD*Seff + gdD*(hdD_2 - fdD)

    pPrec['Omegazeta0_coeff'] = -pPrec['g0'] * pPrec['Omegazeta0']
    pPrec['Omegazeta1_coeff'] = -1.5 * pPrec['g0'] * pPrec['Omegazeta1']
    pPrec['Omegazeta2_coeff'] = -3.0*(pPrec['g0'] * pPrec['Omegazeta2'] + pPrec['g2']*pPrec['Omegazeta0'])
    pPrec['Omegazeta3_coeff'] = 3.0*(pPrec['g0'] * pPrec['Omegazeta3'] + pPrec['g2']*pPrec['Omegazeta1'] + pPrec['g3']*pPrec['Omegazeta0'])
    pPrec['Omegazeta4_coeff'] = 3.0*(pPrec['g0'] * pPrec['Omegazeta4'] + pPrec['g2']*pPrec['Omegazeta2'] + pPrec['g3']*pPrec['Omegazeta1'] + pPrec['g4']*pPrec['Omegazeta0'])
    pPrec['Omegazeta5_coeff'] = 1.5*(pPrec['g0']*pPrec['Omegazeta5'] + pPrec['g2']*pPrec['Omegazeta3'] + pPrec['g3']*pPrec['Omegazeta2'] + pPrec['g4']*pPrec['Omegazeta1'] + pPrec['g5']*pPrec['Omegazeta0'])
        
    #Line 2887 - 2943 compressed

    pPrec = apply_expansion_order(pPrec, ExpansionOrder)

    #Line 2960 - 3004 compressed
    # Get psi0 term
    psi_of_v0 = 0.0
    mm = 0.0
    tmpB = 0.0
    volume_element = 0.0
    vol_sign = 0.0

    pPrec['psi0'] = compute_psi0(pPrec, L_0, S1v, S2v) 

    vMSA = jnp.array([0.0, 0.0, 0.0])

    phiz_0 = 0.0
    phiz_0_MSA = 0.0  # UNUSED in original

    zeta_0 = 0.0
    zeta_0_MSA = 0.0  # UNUSED in original

    # Tolerance chosen to be consistent with implementation in LALSimInspiralFDPrecAngles
    condition = jnp.abs(pPrec['Spl2'] - pPrec['Smi2']) > 1e-5

    def compute_msa_corrections():
        return IMRPhenomX_Return_MSA_Corrections_MSA(pPrec['v_0'], pPrec['L_0_norm'], pPrec['J_0_norm'], pPrec)  # stub

    def no_msa_corrections():
        return jnp.array([0.0, 0.0, 0.0])

    vMSA = jax.lax.cond(condition, compute_msa_corrections, no_msa_corrections)

    phiz_0_MSA = vMSA[0]
    zeta_0_MSA = vMSA[1]

    # Initial \phi_z
    pPrec['phiz_0'] = 0.0
    phiz_0 = IMRPhenomX_Return_phiz_MSA(pPrec['v_0'], pPrec['J_0_norm'], pPrec)  # stub

    # Initial \zeta
    pPrec['zeta_0'] = 0.0
    zeta_0 = IMRPhenomX_Return_zeta_MSA(pPrec['v_0'], pPrec)  # stub

    pPrec['phiz_0'] = -phiz_0 - vMSA[0]
    pPrec['zeta_0'] = -zeta_0 - vMSA[1]

    return None



def IMRPhenomX_Return_phiz_MSA(
    v: float, 
    JNorm: float, 
    pPrec
    ) -> float:
    
    invv = 1.0 / v
    invv2 = invv * invv
    LNewt = pPrec.eta / v

    c1 = pPrec.c1
    c12 = c1 * c1

    SAv2 = pPrec.SAv2
    SAv = pPrec.SAv
    invSAv = pPrec.invSAv
    invSAv2 = pPrec.invSAv2

    # These are log functions defined in Eq. D27 and D28 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    log1 = jnp.log(jnp.abs(c1 + JNorm * pPrec.eta + pPrec.eta * LNewt))
    log2 = jnp.log(jnp.abs(c1 + JNorm * SAv * v + SAv2 * v))

    # Eq. D22-D27 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    phiz_0_coeff = (JNorm * pPrec.inveta**4) * (
        0.5 * c12 - (c1 * pPrec.eta2 * invv) / 6.0 - (SAv2 * pPrec.eta2) / 3.0 - (pPrec.eta4 * invv2) / 3.0
    ) - (0.5 * c1 * pPrec.inveta) * (
        c12 * pPrec.inveta**4 - SAv2 * pPrec.inveta**2
    ) * log1

    phiz_1_coeff = (
        -0.5 * JNorm * pPrec.inveta**2 * (c1 + pPrec.eta * LNewt)
        + 0.5 * pPrec.inveta**3 * (c12 - pPrec.eta2 * SAv2) * log1
    )

    phiz_2_coeff = -JNorm + SAv * log2 - c1 * log1 * pPrec.inveta

    phiz_3_coeff = JNorm * v - pPrec.eta * log1 + c1 * log2 * invSAv

    phiz_4_coeff = (
        0.5 * JNorm * invSAv2 * v * (c1 + v * SAv2)
        - 0.5 * invSAv2 * invSAv * (c12 - pPrec.eta2 * SAv2) * log2
    )

    phiz_5_coeff = (
        -JNorm * v * (
            0.5 * c12 * invSAv2 * invSAv2
            - c1 * v * invSAv2 / 6.0
            - v * v / 3.0
            - pPrec.eta2 * invSAv2 / 3.0
        )
        + 0.5 * c1 * invSAv2 * invSAv2 * invSAv * (c12 - pPrec.eta2 * SAv2) * log2
    )

    # Eq. 66 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
 
    # \phi_{z,-1} = \sum^5_{n=0} <\Omega_z>^(n) \phi_z^(n) + \phi_{z,-1}^0
 
    # Note that the <\Omega_z>^(n) are given by self.Omegazn_coeff's as in Eqs. D15-D20
    phiz_out = (
        phiz_0_coeff * pPrec.Omegaz0_coeff
        + phiz_1_coeff * pPrec.Omegaz1_coeff
        + phiz_2_coeff * pPrec.Omegaz2_coeff
        + phiz_3_coeff * pPrec.Omegaz3_coeff
        + phiz_4_coeff * pPrec.Omegaz4_coeff
        + phiz_5_coeff * pPrec.Omegaz5_coeff
        + pPrec.phiz_0
    )

    # Ensure no NaN (replace with 0.0 if NaN)
    phiz_out = jnp.nan_to_num(phiz_out, nan=0.0)

    return phiz_out


    
def IMRPhenomX_Return_zeta_MSA(
    v: float, 
    pPrec
    ) -> float:
    invv = 1.0 / v
    invv2 = invv * invv
    invv3 = invv * invv2
    v2 = v * v
    logv = jnp.log(v)

    # Compute zeta using precession coefficients
    zeta_out = pPrec.eta * (
        pPrec.Omegazeta0_coeff * invv3 +
        pPrec.Omegazeta1_coeff * invv2 +
        pPrec.Omegazeta2_coeff * invv +
        pPrec.Omegazeta3_coeff * logv +
        pPrec.Omegazeta4_coeff * v +
        pPrec.Omegazeta5_coeff * v2
    ) + pPrec.zeta_0

    # Replace NaNs with 0 using jnp.nan_to_num
    zeta_out = jnp.nan_to_num(zeta_out, nan=0.0)

    return zeta_out



def IMRPhenomX_vector_L2_norm(v1: jnp.ndarray) -> float:
    """
    Calculate L2 norm of a 3D vector
    
    Args:
        v1: 3D vector as JAX array [x, y, z] (jnp.ndarray)
        
    Returns:
        float: L2 norm of the vector
    """
    return jnp.linalg.norm(v1)


def IMRPhenomX_vector_scalar(v1: jnp.ndarray, a: float) -> jnp.ndarray:
    """
    Multiply a vector by a scalar
    
    Args:
        v1: 3D vector as JAX array [x, y, z] (jnp.ndarray)
        a: Scalar multiplier (float)
        
    Returns:
        jnp.ndarray: Scaled vector
    """
    v2 = jnp.array([a * v1[0], a * v1[1], a * v1[2]])
    return v2


def IMRPhenomX_vector_dot_product(v1: jnp.ndarray, v2: jnp.ndarray) -> float:
    """
    Calculate dot product of two 3D vectors
    
    Args:
        v1: First 3D vector as JAX array (jnp.ndarray)
        v2: Second 3D vector as JAX array (jnp.ndarray)
        
    Returns:
        float: Dot product
    """
    return jnp.dot(v1, v2)


def IMRPhenomX_Get_PN_beta(a: float, b: float, pPrec: dict) -> float:
    """
    Calculate PN beta coefficient
    
    Args:
        a: First coefficient (float)
        b: Second coefficient (float)
        pPrec: Precession structure dictionary (dict)
        
    Returns:
        float: PN beta value
    """
    return (pPrec['dotS1L'] * (a + b * pPrec['qq']) + 
            pPrec['dotS2L'] * (a + b / pPrec['qq']))


def IMRPhenomX_vector_sum(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate sum of two 3D vectors
    
    Args:
        v1: First 3D vector as JAX array (jnp.ndarray)
        v2: Second 3D vector as JAX array (jnp.ndarray)
        
    Returns:
        jnp.ndarray: Sum of the vectors
    """
    return v1 + v2


def IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(LNorm, JNorm, pPrec):
    JNorm2 = JNorm * JNorm
    LNorm2 = LNorm * LNorm

    S1Norm2 = pPrec.S1_norm_2
    S2Norm2 = pPrec.S2_norm_2
    q       = pPrec.qq
    eta     = pPrec.eta
    delta   = pPrec.delta_qq
    deltaSq = delta * delta
    Seff    = pPrec.Seff

    J2mL2   = JNorm2 - LNorm2
    J2mL2Sq = J2mL2 * J2mL2

    # B coefficient (Eq. B2)
    B_coeff = ((LNorm2 + S1Norm2) * q +
               2.0 * LNorm * Seff -
               2.0 * JNorm2 -
               S1Norm2 - S2Norm2 +
               (LNorm2 + S2Norm2) / q)

    # C coefficient (Eq. B3)
    C_coeff = (J2mL2Sq -
               2.0 * LNorm * Seff * J2mL2 -
               2.0 * ((1.0 - q) / q) * LNorm2 * (S1Norm2 - q * S2Norm2) +
               4.0 * eta * LNorm2 * Seff * Seff -
               2.0 * delta * (S1Norm2 - S2Norm2) * Seff * LNorm +
               2.0 * ((1.0 - q) / q) * (q * S1Norm2 - S2Norm2) * JNorm2)

    # D coefficient (Eq. B4)
    D_coeff = (((1.0 - q) / q) * (S2Norm2 - q * S1Norm2) * J2mL2Sq +
               deltaSq * (S1Norm2 - S2Norm2)**2 * LNorm2 / eta +
               2.0 * delta * LNorm * Seff * (S1Norm2 - S2Norm2) * J2mL2)

    return jnp.array([B_coeff, C_coeff, D_coeff])


def IMRPhenomX_Return_Roots_MSA(LNorm, JNorm, pPrec):
    vBCD = IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(LNorm, JNorm, pPrec)  
    B, C, D = vBCD[0], vBCD[1], vBCD[2]

    B2 = B * B
    B3 = B2 * B
    BC = B * C

    p = C - B2 / 3.0
    qc = (2.0 / 27.0) * B3 - BC / 3.0 + D

    sqrtarg = jnp.sqrt(-p / 3.0)
    acosarg = 1.5 * qc / (p * sqrtarg)
    acosarg = jnp.clip(acosarg, -1.0, 1.0)

    theta = jnp.arccos(acosarg) / 3.0
    cos_theta = jnp.cos(theta)
    
    print(f'{p=}, {sqrtarg=}, {theta=}, {B=}, {B2=}, {C=}')
    
    vector_condition = jnp.logical_or(jnp.isnan(theta),
                                                   (jnp.isnan(sqrtarg)))
    scalar_condition = jnp.logical_or.reduce(jnp.array([(pPrec.dotS1Ln == 1.0),
                                                   (pPrec.dotS2Ln == 1.0),
                                                   (pPrec.dotS1Ln == -1.0),
                                                   (pPrec.dotS2Ln == -1.0),
                                                   (pPrec.S1_norm_2 == 0.0),
                                                   (pPrec.S2_norm_2 == 0.0)]))
    invalid_case = jnp.logical_or(vector_condition, scalar_condition)

    def roots_when_valid():
        tmp1 = 2.0 * sqrtarg * jnp.cos(theta - 4.0 * jnp.pi / 3.0) - B / 3.0
        tmp2 = 2.0 * sqrtarg * jnp.cos(theta - 2.0 * jnp.pi / 3.0) - B / 3.0
        tmp3 = 2.0 * sqrtarg * cos_theta - B / 3.0

        tmp4 = jnp.maximum(jnp.maximum(tmp1, tmp2), tmp3)
        tmp5 = jnp.minimum(jnp.minimum(tmp1, tmp2), tmp3)

        tmp6 = jnp.where(
            (tmp4 - tmp3 > 0.0) & (tmp5 - tmp3 < 0.0),
            tmp3,
            jnp.where((tmp4 - tmp1 > 0.0) & (tmp5 - tmp1 < 0.0), tmp1, tmp2)
        )

        S32 = tmp5
        Smi2 = jnp.abs(tmp6)
        Spl2 = jnp.abs(tmp4)
        return jnp.array([S32, Smi2, Spl2])

    def roots_when_invalid():
        Smi2 = pPrec.S_0_norm**2 * jnp.ones_like(LNorm)
        Spl2 = Smi2 + 1e-9
        S32 = jnp.zeros_like(LNorm)
        return jnp.array([S32, Smi2, Spl2])

    roots_array = jnp.where(
        jnp.atleast_1d(invalid_case),
        roots_when_invalid(),
        roots_when_valid()
    )
    
    print(f'{roots_array=}')

    return roots_array

def IMRPhenomX_Get_PN_sigma(a: float, b: float, pPrec: dict) -> float:
    """
    Calculate PN sigma coefficient
    
    Args:
        a: First coefficient (float)
        b: Second coefficient (float)
        pPrec: Precession structure dictionary (dict)
        
    Returns:
        float: PN sigma value
    """
    return pPrec['inveta'] * (a * pPrec['dotS1S2'] - b * pPrec['dotS1L'] * pPrec['dotS2L'])

def IMRPhenomX_Get_PN_tau(a: float, b: float, pPrec: dict) -> float:
    """
    Internal function to computes PN spin-spin couplings. As in LALSimInspiralFDPrecAngles.c
    
    Args:
        a: First coefficient (float)
        b: Second coefficient (float)
        pPrec: Precession structure dictionary (dict)
        
    Returns:
        float: PN tau value
    """
    return ((pPrec['qq'] * ((pPrec['S1_norm_2'] * a) - b * pPrec['dotS1L'] * pPrec['dotS1L']) + 
             (a * pPrec['S2_norm_2'] - b * pPrec['dotS2L'] * pPrec['dotS2L']) / pPrec['qq']) / 
            pPrec['eta'])




# Expansion order of corrections to retain
def apply_expansion_order(pPrec: dict, ExpansionOrder: int) -> dict:
    """
    Apply expansion order corrections in a JAX-friendly way
    
    Args:
        pPrec: Precession structure dictionary (dict)
        ExpansionOrder: Order of expansion (-1 for all orders, 1-5 for specific cutoffs) (int)
        
    Returns:
        dict: Updated pPrec dictionary
    """
    
    # Create masks for which coefficients to zero out based on expansion order
    zero_1_and_higher = ExpansionOrder <= 1
    zero_2_and_higher = ExpansionOrder <= 2
    zero_3_and_higher = ExpansionOrder <= 3
    zero_4_and_higher = ExpansionOrder <= 4
    zero_5_and_higher = ExpansionOrder <= 5
    
    # Apply corrections based on expansion order
    # For expansion order -1, keep all coefficients (no changes)
    # For higher orders, zero out coefficients beyond the specified order
    
    pPrec['Omegaz1_coeff'] = jnp.where(zero_1_and_higher, 0.0, pPrec['Omegaz1_coeff'])
    pPrec['Omegazeta1_coeff'] = jnp.where(zero_1_and_higher, 0.0, pPrec['Omegazeta1_coeff'])
    
    pPrec['Omegaz2_coeff'] = jnp.where(zero_2_and_higher, 0.0, pPrec['Omegaz2_coeff'])
    pPrec['Omegazeta2_coeff'] = jnp.where(zero_2_and_higher, 0.0, pPrec['Omegazeta2_coeff'])
    
    pPrec['Omegaz3_coeff'] = jnp.where(zero_3_and_higher, 0.0, pPrec['Omegaz3_coeff'])
    pPrec['Omegazeta3_coeff'] = jnp.where(zero_3_and_higher, 0.0, pPrec['Omegazeta3_coeff'])
    
    pPrec['Omegaz4_coeff'] = jnp.where(zero_4_and_higher, 0.0, pPrec['Omegaz4_coeff'])
    pPrec['Omegazeta4_coeff'] = jnp.where(zero_4_and_higher, 0.0, pPrec['Omegazeta4_coeff'])
    
    pPrec['Omegaz5_coeff'] = jnp.where(zero_5_and_higher, 0.0, pPrec['Omegaz5_coeff'])
    pPrec['Omegazeta5_coeff'] = jnp.where(zero_5_and_higher, 0.0, pPrec['Omegazeta5_coeff'])
    
    return pPrec


# Tolerance chosen to be consistent with implementation in LALSimInspiralFDPrecAngles
def compute_psi0(pPrec, L_0, S1v, S2v):
    condition = jnp.abs(pPrec['Smi2'] - pPrec['Spl2']) < 1.0e-5
    
    def psi0_zero():
        return 0.0
    
    def psi0_nonzero():
        mm = jnp.sqrt((pPrec['Smi2'] - pPrec['Spl2']) / (pPrec['S32'] - pPrec['Spl2']))
        tmpB = (pPrec['S_0_norm']*pPrec['S_0_norm'] - pPrec['Spl2']) / (pPrec['Smi2'] - pPrec['Spl2'])
        
        volume_element = IMRPhenomX_vector_dot_product(
            IMRPhenomX_vector_cross_product(L_0, S1v), S2v  
        )
        vol_sign = jnp.sign(volume_element)  # equivalent to (volume_element > 0) - (volume_element < 0)
        
        psi_of_v0 = IMRPhenomX_psiofv(pPrec['v_0'], pPrec['v_0_2'], 0.0, pPrec['psi1'], pPrec['psi2'], pPrec)  
        
        # Handle boundary cases for tmpB
        def handle_boundary_cases():
            # If tmpB > 1.0 and close to 1
            case1_condition = jnp.logical_and(tmpB > 1.0, (tmpB - 1.0) < 0.00001)
            case1_result = ellint_F(jnp.arcsin(vol_sign * jnp.sqrt(1.0)), mm) - psi_of_v0  # stub
            
            # If tmpB < 0.0 and close to 0
            case2_condition = jnp.logical_and(tmpB < 0.0, tmpB > -0.00001)
            case2_result = ellint_F(jnp.arcsin(vol_sign * jnp.sqrt(0.0)), mm) - psi_of_v0  # stub
            
            # Normal case
            normal_result = ellint_F(jnp.arcsin(vol_sign * jnp.sqrt(tmpB)), mm) - psi_of_v0  # stub
            
            return jnp.where(
                case1_condition, case1_result,
                jnp.where(case2_condition, case2_result, normal_result)
            )
        
        def normal_case():
            return ellint_F(jnp.arcsin(vol_sign * jnp.sqrt(tmpB)), mm) - psi_of_v0  # stub
        
        # Check if we're in boundary case
        boundary_condition = jnp.logical_or(tmpB < 0.0, tmpB > 1.0)
        
        return jax.lax.cond(boundary_condition, handle_boundary_cases, normal_case)
    
    return jax.lax.cond(condition, psi0_zero, psi0_nonzero)


def IMRPhenomX_psiofv(v, v2, psi0, psi1, psi2, pPrec):
    # Equation 51 in arXiv:1703.03967
    return psi0 - 0.75 * pPrec.g0 * pPrec.delta_qq * (1.0 + psi1 * v + psi2 * v2) / (v2 * v)

def IMRPhenomX_vector_cross_product(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate cross product of two 3D vectors
    
    Args:
        v1: First 3D vector as JAX array [x, y, z] (jnp.ndarray)
        v2: Second 3D vector as JAX array [x, y, z] (jnp.ndarray)
        
    Returns:
        jnp.ndarray: Cross product vector
    """
    return jnp.cross(v1, v2)




def IMRPhenomX_Return_MSA_Corrections_MSA(
    v, 
    LNorm, 
    JNorm, 
    pPrec
    ):
    
    v2 = v * v

    # Sets c0, c2 and c4 in pPrec as per Eq. B6-B8 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    c_vec = IMRPhenomX_Return_Constants_c_MSA(v, JNorm, pPrec)
    # Sets d0, d2 and d4 in pPrec as per Eq. B9-B11 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    d_vec = IMRPhenomX_Return_Constants_d_MSA(LNorm, JNorm, pPrec)  

    c0, c2, c4 = c_vec
    d0, d2, d4 = d_vec

    two_d0 = 2.0 * d0
    
    # Eq. B20 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    sd = jnp.sqrt(jnp.abs(d2 * d2 - 4.0 * d0 * d4))

    # Eq. F20-21 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    A_theta_L = 0.5 * ((JNorm / LNorm) + (LNorm / JNorm) - (pPrec.Spl2 / (JNorm * LNorm)))
    B_theta_L = 0.5 * pPrec.Spl2mSmi2 / (JNorm * LNorm)

    nc_num = 2.0 * (d0 + d2 + d4)
    nc_denom = two_d0 + d2 + sd

    nc = nc_num / nc_denom
    nd = nc_denom / two_d0

    sqrt_nc = jnp.sqrt(jnp.abs(nc))
    sqrt_nd = jnp.sqrt(jnp.abs(nd))

    psi = IMRPhenomX_Return_Psi_MSA(v, v2, pPrec) + pPrec.psi0
    psi_dot = IMRPhenomX_Return_Psi_dot_MSA(v, pPrec) 

    tan_psi = jnp.tan(psi)
    atan_psi = jnp.arctan(tan_psi)

    C1 = -0.5 * (c0 / d0 - 2.0 * (c0 + c2 + c4) / nc_num)
    C2num = (c0 * (-2.0 * d0 * d4 + d2 * d2 + d2 * d4) -
             c2 * d0 * (d2 + 2.0 * d4) +
             c4 * d0 * (two_d0 + d2))
    C2den = 2.0 * d0 * sd * (d0 + d2 + d4)
    C2 = C2num / C2den

    Cphi = C1 + C2
    Dphi = C1 - C2

    def compute_Cphi_term():
        
        return jnp.abs((
            (c4 * d0 * ((2 * d0 + d2) + sd) -
                c2 * d0 * ((d2 + 2.0 * d4) - sd) -
                c0 * ((2 * d0 * d4) - (d2 + d4) * (d2 - 
                sd))) / C2den) * (sqrt_nc / (nc - 1.0)) * (atan_psi - jnp.arctan(sqrt_nc * tan_psi))) / psi_dot
        
    def compute_Dphi_term():
            return jnp.abs((
                (-c4 * d0 * ((2 * d0 + d2) - sd) +
                 c2 * d0 * ((d2 + 2.0 * d4) + sd) -
                 c0 * (-(2 * d0 * d4) + (d2 + d4) * (d2 + sd))) / C2den
            ) * (sqrt_nd / (nd - 1.0)) * (atan_psi - jnp.arctan(sqrt_nd * tan_psi))) / psi_dot

    phiz_0_MSA_Cphi_term = jnp.where(nc == 1.0, 0.0, compute_Cphi_term())
    phiz_0_MSA_Dphi_term = jnp.where(nd == 1.0, 0.0, compute_Dphi_term())

    vMSA_x = phiz_0_MSA_Cphi_term + phiz_0_MSA_Dphi_term

    #####  restart from here
    vMSA_y = A_theta_L * vMSA_x + 2.0 * B_theta_L * d0 * (
                phiz_0_MSA_Cphi_term / (sd - d2) - phiz_0_MSA_Dphi_term / (sd + d2))

    vMSA_x = jnp.where(jnp.isnan(vMSA_x), 0.0, vMSA_x)
    vMSA_y = jnp.where(jnp.isnan(vMSA_y), 0.0, vMSA_y)

    return jnp.array([vMSA_x, vMSA_y, 0.0])



def IMRPhenomX_Return_Psi_MSA(v, v2, pPrec):
    return -0.75 * pPrec.g0 * pPrec.delta_qq * (1.0 + pPrec.psi1 * v + pPrec.psi2 * v2) / (v2 * v)



def IMRPhenomX_Return_Constants_c_MSA(v, JNorm, pPrec):
    v2 = v * v
    v3 = v * v2
    v4 = v2 * v2
    v6 = v3 * v3
    JNorm2 = JNorm * JNorm
    Seff = pPrec.Seff


    x = JNorm * (
        0.75 * (1.0 - Seff * v) * v2 * (
            pPrec.eta3
            + 4.0 * pPrec.eta3 * Seff * v
            - 2.0 * pPrec.eta * (
                JNorm2 - pPrec.Spl2 + 2.0 * (pPrec.S1_norm_2 - pPrec.S2_norm_2) * pPrec.delta_qq
            ) * v2
            - 4.0 * pPrec.eta * Seff * (JNorm2 - pPrec.Spl2) * v3
            + (JNorm2 - pPrec.Spl2) ** 2 * v4 * pPrec.inveta
        )
    )

    y = JNorm * (
        -1.5 * pPrec.eta * (pPrec.Spl2 - pPrec.Smi2)
        * (1.0 + 2.0 * Seff * v - (JNorm2 - pPrec.Spl2) * v2 * pPrec.inveta**2)
        * (1.0 - Seff * v) * v4
    )

    z = JNorm * (
        0.75 * pPrec.inveta * (pPrec.Spl2 - pPrec.Smi2) ** 2
        * (1.0 - Seff * v) * v6
    )

    return jnp.array([x, y, z])



def IMRPhenomX_Return_Constants_d_MSA(LNorm, JNorm, pPrec):
    LNorm2 = LNorm * LNorm
    JNorm2 = JNorm * JNorm

    x = - (JNorm2 - (LNorm + pPrec.Spl)) ** 2 * (JNorm2 - (LNorm - pPrec.Spl)) ** 2

    y = -2.0 * (pPrec.Spl2 - pPrec.Smi2) * (JNorm2 + LNorm2 - pPrec.Spl2)

    z = -(pPrec.Spl2 - pPrec.Smi2) ** 2

    return jnp.array([x, y, z])





def IMRPhenomX_Return_Psi_dot_MSA(v, pPrec):
    v2 = v * v

    A_coeff = -1.5 * v2 * v2 * v2 * (1.0 - v * pPrec.Seff) * jnp.sqrt(pPrec.inveta)
    psi_dot = 0.5 * A_coeff * jnp.sqrt(pPrec.Spl2 - pPrec.S32)

    return psi_dot