import jax
import jax.numpy as jnp

from ripplegw.constants import PI, C, G
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import (
    IMRPhenomXPrecessionDataClass,
)

def imr_phenom_x_pnr_get_and_set_pnr_variables(
    p_wf: IMRPhenomXWaveformDataClass,    
    p_prec: IMRPhenomXPrecessionDataClass
  ):

    #/* get needed quantities */
    m1 = p_wf.m1 * p_wf.m_tot
    m2 = p_wf.m2 * p_wf.m_tot
    q = p_wf.q

#     if(p_prec.imr_phenom_x_prec_version == 330)
#     {
#       chi1x = pPrec->chi1x_evolved;
#       chi1y = pPrec->chi1y_evolved;
#       chi2x = pPrec->chi2x_evolved;
#       chi2y = pPrec->chi2y_evolved;
#       chieff = XLALSimIMRPhenomXchiEff(pWF->eta,pPrec->chi1z_evolved,pPrec->chi2z_evolved);
#     }
#     else
#     {
#       chi1x = pPrec->chi1x;
#       chi1y = pPrec->chi1y;
#       chi2x = pPrec->chi2x;
#       chi2y = pPrec->chi2y;
#       chieff = XLALSimIMRPhenomXchiEff(pWF->eta,pPrec->chi1z,pPrec->chi2z);
#     }

#     chipar = pWF->Mtot * chieff / m1;
#     chiperp = 0.0;
#     costheta = 0.0;
#     chiperp_antisymmetric = 0.0;
#     theta_antisymmetric = 0.0;

#     /* compute effective in-plane spin contribution from Eq. 17 of arXiv:2107.08876 */
#     /* for XO4a this contribution is only used for mass ratios below 1.5 */
#     /* in versions of the model where we use the evolved spin values, we use this contribution at all mass ratios */
#     if(pPrec->IMRPhenomXPrecVersion == 330){
#           chis = sqrt((m1 * m1 * chi1x + m2 * m2 * chi2x) * (m1 * m1 * chi1x + m2 * m2 * chi2x) + (m1 * m1 * chi1y + m2 * m2 * chi2y) * (m1 * m1 * chi1y + m2 * m2 * chi2y)) / (m1 * m1);
# 	  chiperp = chis;
#     }
#     else{
#       if (q <= 1.5)
# 	{
# 	  chis = sqrt((m1 * m1 * chi1x + m2 * m2 * chi2x) * (m1 * m1 * chi1x + m2 * m2 * chi2x) + (m1 * m1 * chi1y + m2 * m2 * chi2y) * (m1 * m1 * chi1y + m2 * m2 * chi2y)) / (m1 * m1);
# 	  chiperp = sin((q - 1.0) * LAL_PI) * sin((q - 1.0) * LAL_PI) * pPrec->chi_p + cos((q - 1.0) * LAL_PI) * cos((q - 1.0) * LAL_PI) * chis;
# 	}
#       else
# 	{
# 	  chiperp = pPrec->chi_p;
# 	}
#     }

#     if (q <= 1.5)
#       {
# 	antisymmetric_chis = sqrt((m1 * m1 * chi1x - m2 * m2 * chi2x) * (m1 * m1 * chi1x - m2 * m2 * chi2x) + (m1 * m1 * chi1y - m2 * m2 * chi2y) * (m1 * m1 * chi1y - m2 * m2 * chi2y)) / (m1 * m1);
# 	chiperp_antisymmetric = sin((q - 1.0) * LAL_PI) * sin((q - 1.0) * LAL_PI) * pPrec->chi_p + cos((q - 1.0) * LAL_PI) * cos((q - 1.0) * LAL_PI) * antisymmetric_chis;
#       }
#     else
#       {
# 	chiperp_antisymmetric = pPrec->chi_p;
#       }

#     /* get the total magnitude, Eq. 18 of arXiv:2107.08876 */
#     chi_mag = sqrt(chipar * chipar + chiperp * chiperp);
#     pPrec->chi_singleSpin = chi_mag;

#     chi_mag_antisymmetric = sqrt(chipar * chipar + chiperp_antisymmetric * chiperp_antisymmetric);
#     pPrec->chi_singleSpin_antisymmetric = chi_mag_antisymmetric;

#     /* get the opening angle of the single spin, Eq. 19 of arXiv:2107.08876 */
#     if (chi_mag >= 1.0e-6)
#     {
#       costheta = chipar / chi_mag;
#     }
#     else
#     {
#       costheta = 0.;
#     }
#     if (chi_mag_antisymmetric >= 1.0e-6)
#     {
#       theta_antisymmetric = acos(chipar / chi_mag_antisymmetric);
#     }
#     else
#     {
#       theta_antisymmetric = 0.0;
#     }

#     pPrec->costheta_singleSpin = costheta;
#     pPrec->theta_antisymmetric = theta_antisymmetric;

#     /* compute an approximate final spin using single-spin mapping FIXME: add documentation */
#     chi1L = chi_mag * costheta;
#     chi2L = 0.0;

#     Xfparr = XLALSimIMRPhenomXFinalSpin2017(pWF->eta, chi1L, chi2L);

#     /* rescale Xfperp to use the final total mass of 1 */
#     qfactor = q / (1.0 + q);
#     Xfperp = qfactor * qfactor * chi_mag * sqrt(1.0 - costheta * costheta);
#     xf = sqrt(Xfparr * Xfparr + Xfperp * Xfperp);
#     if (xf > 1.0e-6)
#     {
#       pPrec->costheta_final_singleSpin = Xfparr / xf;
#     }
#     else
#     {
#       pPrec->costheta_final_singleSpin = 0.;
#     }

#     /* Initialize frequency values */
#     pPrec->PNR_HM_Mflow = 0.0;
#     pPrec->PNR_HM_Mfhigh = 0.0;

#     /* set angle window boundaries */
#     pPrec->PNR_q_window_lower = 8.5;
#     pPrec->PNR_q_window_upper = 12.0;
#     pPrec->PNR_chi_window_lower = 0.85;
#     pPrec->PNR_chi_window_upper = 1.2;

#     /* set inspiral scaling flag for HM frequency map */
#     pPrec->PNRInspiralScaling = 0;
#     if ((q > pPrec->PNR_q_window_upper) || (pPrec->chi_singleSpin > pPrec->PNR_chi_window_upper))
#     {
#       pPrec->PNRInspiralScaling = 1;
#     }

#     #if DEBUG == 1
#       printf("\nSetting PNR-related single-spin quantities:\n");
#       printf("chi_singleSpin                     : %e\n", pPrec->chi_singleSpin);
#       printf("chi_singleSpin_antisymmetric       : %e\n", pPrec->chi_singleSpin_antisymmetric);
#       printf("costheta_singleSpin                : %e\n", pPrec->costheta_singleSpin);
#       /* printf("theta_singleSpin_antisymmetric  : %e\n", pPrec->theta_singleSpin_antisymmetric); */
#       printf("theta_antisymmetric  : %e\n", pPrec->theta_antisymmetric);
#       printf("costheta_final_singleSpin          : %e\n\n", pPrec->costheta_final_singleSpin);
#     #endif

#     return XLAL_SUCCESS;
#   }