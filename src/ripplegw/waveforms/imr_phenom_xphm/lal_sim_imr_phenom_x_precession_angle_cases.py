import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw.constants import MRSUN, MSUN, PI, C, G, gt
from ripplegw.typing import Array
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXUsefulPowersDataClass,
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import IMRPhenomXPrecessionDataClass
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_inspiral_waveform_flags import (
    xlal_sim_inspiral_mode_array_is_mode_active,
)
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import IMRPhenomXPHMParameterDataClass


def SpinTaylor(
    pWF: IMRPhenomXWaveformDataClass,
    pPrec: IMRPhenomXPrecessionDataClass,
    lalParams: IMRPhenomXPHMParameterDataClass,
):

    # // check mode array to estimate frequency range over which splines will need to be evaluated
    ModeArray = lalParams.mode_array

    LMAX_PNR = jax.lax.select(
        ModeArray is not None,
        jax.lax.select(
            xlal_sim_inspiral_mode_array_is_mode_active(ModeArray, 4, 4),
            4,
            jax.lax.select(
                xlal_sim_inspiral_mode_array_is_mode_active(ModeArray, 3, 3) | xlal_sim_inspiral_mode_array_is_mode_active(ModeArray, 3, 2),
                3,
                2
            )
        ),
        2
    )
    L_MAX_PNR = LMAX_PNR

    
    # // buffer for GSL interpolation to succeed
    # // set first to fMin
    flow = pWF.f_min

    # if(pWF.delta_f==0.) pWF.delta_f = get_deltaF_from_wfstruct(pWF) 
    pWF = dataclasses.replace(
        pWF,
        delta_f = jax.lax.select(
            pWF.delta_f == 0.0,
            get_deltaF_from_wfstruct(pWF),
            pWF.delta_f
        )
    )

    # // if PNR angles are disabled, step back accordingly to the waveform's frequency grid step
    # if(PNRUseTunedAngles==false)
    # {

    # pPrec->integration_buffer = (pWF->deltaF>0.)? 3.*pWF->deltaF: 0.5;
    # flow = (pWF->fMin-pPrec->integration_buffer)*2./pPrec->M_MAX;

    # }
    # // if PNR angles are enabled, adjust buffer to the requirements of IMRPhenomX_PNR_GeneratePNRAngleInterpolants
    # else{

    # size_t iStart_here;

    # if (pWF->deltaF == 0.) iStart_here = 0;
    # else{
    # iStart_here= (size_t)(pWF->fMin / pWF->deltaF);
    # flow = iStart_here * pWF->deltaF;
    # }

    # REAL8 fmin_HM_inspiral = flow * 2.0 / pPrec->M_MAX;

    # INT4 precVersion = pPrec->IMRPhenomXPrecVersion;
    # // fill in a fake value to allow the next code to work
    # pPrec->IMRPhenomXPrecVersion = 223;
    # status = IMRPhenomX_PNR_GetAndSetPNRVariables(pWF, pPrec);
    # XLAL_CHECK(XLAL_SUCCESS == status, XLAL_EFUNC, "Error: IMRPhenomX_PNR_GetAndSetPNRVariables failed in IMRPhenomXGetAndSetPrecessionVariables.\n");

    # /* generate alpha parameters to catch edge cases */
    # IMRPhenomX_PNR_alpha_parameters *alphaParams = XLALMalloc(sizeof(IMRPhenomX_PNR_alpha_parameters));
    # IMRPhenomX_PNR_beta_parameters *betaParams = XLALMalloc(sizeof(IMRPhenomX_PNR_beta_parameters));
    # status = IMRPhenomX_PNR_precompute_alpha_coefficients(alphaParams, pWF, pPrec);
    # XLAL_CHECK(
    # XLAL_SUCCESS == status,
    # XLAL_EFUNC,
    # "Error: IMRPhenomX_PNR_precompute_alpha_coefficients failed.\n");
    # status = IMRPhenomX_PNR_precompute_beta_coefficients(betaParams, pWF, pPrec);
    # XLAL_CHECK(
    # XLAL_SUCCESS == status,
    # XLAL_EFUNC,
    # "Error: IMRPhenomX_PNR_precompute_beta_coefficients failed.\n");
    # status = IMRPhenomX_PNR_BetaConnectionFrequencies(betaParams);
    # XLAL_CHECK(
    # XLAL_SUCCESS == status,
    # XLAL_EFUNC,
    # "Error: IMRPhenomX_PNR_BetaConnectionFrequencies failed.\n");
    # pPrec->IMRPhenomXPrecVersion = precVersion;
    # REAL8 Mf_alpha_upper = alphaParams->A4 / 3.0;
    # REAL8 Mf_low_cut = (3.0 / 3.5) * Mf_alpha_upper;
    # REAL8 MF_high_cut = betaParams->Mf_beta_lower;
    # LALFree(alphaParams);
    # LALFree(betaParams);

    # if((MF_high_cut > pWF->fCutDef) || (MF_high_cut < 0.1 * pWF->fRING)){
    # MF_high_cut = pWF->fRING;
    # }
    # if((Mf_low_cut > pWF->fCutDef) || (MF_high_cut < Mf_low_cut)){
    # Mf_low_cut = MF_high_cut / 2.0;
    # }

    # REAL8 flow_alpha = XLALSimIMRPhenomXUtilsMftoHz(Mf_low_cut * 0.65 * pPrec->M_MAX / 2.0, pWF->Mtot);

    # if(flow_alpha < flow){
    # // flow is approximately in the intermediate region of the frequency map
    # // conservatively reduce flow to account for potential problems in this region
    # flow = fmin_HM_inspiral / 1.5;
    # }
    # else{
    # REAL8 Mf_RD_22 = pWF->fRING;
    # REAL8 Mf_RD_lm = IMRPhenomXHM_GenerateRingdownFrequency(pPrec->L_MAX_PNR, pPrec->M_MAX, pWF);
    # REAL8 fmin_HM_ringdowm = XLALSimIMRPhenomXUtilsMftoHz(XLALSimIMRPhenomXUtilsHztoMf(flow, pWF->Mtot) - (Mf_RD_lm - Mf_RD_22), pWF->Mtot);
    # flow = ((fmin_HM_ringdowm < fmin_HM_inspiral)&&(fmin_HM_ringdowm > 0.0)) ? fmin_HM_ringdowm : fmin_HM_inspiral;
    # }


    # double pnr_interpolation_deltaf = IMRPhenomX_PNR_HMInterpolationDeltaF(flow, pWF, pPrec);
    # pPrec->integration_buffer = 1.4*pnr_interpolation_deltaf;
    # flow = (flow - 2.0 * pnr_interpolation_deltaf < 0) ? flow / 2.0 : flow - 2.0 * pnr_interpolation_deltaf;

    # iStart_here = (size_t)(flow / pnr_interpolation_deltaf);
    # flow = iStart_here * pnr_interpolation_deltaf;
    # }

    # XLAL_CHECK(flow>0.,XLAL_EDOM,"Error in %s: starting frequency for SpinTaylor angles must be positive!",__func__);
    # status = IMRPhenomX_InspiralAngles_SpinTaylor(pPrec->PNarrays,&pPrec->fmin_integration,chi1x,chi1y,chi1z,chi2x,chi2y,chi2z,flow,pPrec->IMRPhenomXPrecVersion,pWF,lalParams);
    # // convert the min frequency of integration to geometric units for later convenience
    # pPrec->Mfmin_integration = XLALSimIMRPhenomXUtilsHztoMf(pPrec->fmin_integration,pWF->Mtot);

    # if (pPrec->IMRPhenomXPrecVersion == 330)
    # {

    # REAL8 chi1x_evolved = chi1x;
    # REAL8 chi1y_evolved = chi1y;
    # REAL8 chi1z_evolved = chi1z;
    # REAL8 chi2x_evolved = chi2x;
    # REAL8 chi2y_evolved = chi2y;
    # REAL8 chi2z_evolved = chi2z;

    # // in case that SpinTaylor angles generate, overwrite variables with evolved spins
    # if(status!=XLAL_FAILURE)  {
    # size_t lenPN = pPrec->PNarrays->V_PN->data->length;

    # REAL8 chi1x_temp = pPrec->PNarrays->S1x_PN->data->data[lenPN-1];
    # REAL8 chi1y_temp = pPrec->PNarrays->S1y_PN->data->data[lenPN-1];
    # REAL8 chi1z_temp = pPrec->PNarrays->S1z_PN->data->data[lenPN-1];

    # REAL8 chi2x_temp = pPrec->PNarrays->S2x_PN->data->data[lenPN-1];
    # REAL8 chi2y_temp = pPrec->PNarrays->S2y_PN->data->data[lenPN-1];
    # REAL8 chi2z_temp = pPrec->PNarrays->S2z_PN->data->data[lenPN-1];

    # REAL8 Lx = pPrec->PNarrays->LNhatx_PN->data->data[lenPN-1];
    # REAL8 Ly = pPrec->PNarrays->LNhaty_PN->data->data[lenPN-1];
    # REAL8 Lz = pPrec->PNarrays->LNhatz_PN->data->data[lenPN-1];

    # // orbital separation vector not stored in PN arrays
    # //REAL8 nx = pPrec->PNarrays->E1x->data->data[lenPN-1];
    # //REAL8 ny = pPrec->PNarrays->E1y->data->data[lenPN-1];

    # // rotate to get x,y,z components in L||z frame
    # REAL8 phi = atan2( Ly, Lx );
    # REAL8 theta = acos( Lz / sqrt(Lx*Lx + Ly*Ly + Lz*Lz) );
    # //REAL8 kappa = atan( ny/nx );

    # IMRPhenomX_rotate_z(-phi, &chi1x_temp, &chi1y_temp, &chi1z_temp);
    # IMRPhenomX_rotate_y(-theta, &chi1x_temp, &chi1y_temp, &chi1z_temp);
    # //IMRPhenomX_rotate_z(-kappa, &chi1x_temp, &chi1y_temp, &chi1z_temp);

    # IMRPhenomX_rotate_z(-phi, &chi2x_temp, &chi2y_temp, &chi2z_temp);
    # IMRPhenomX_rotate_y(-theta, &chi2x_temp, &chi2y_temp, &chi2z_temp);
    # //IMRPhenomX_rotate_z(-kappa, &chi2x_temp, &chi2y_temp, &chi2z_temp);

    # chi1x_evolved = chi1x_temp;
    # chi1y_evolved = chi1y_temp;
    # chi1z_evolved = chi1z_temp;

    # chi2x_evolved = chi2x_temp;
    # chi2y_evolved = chi2y_temp;
    # chi2z_evolved = chi2z_temp;
    # }

    # pPrec->chi1x_evolved = chi1x_evolved;
    # pPrec->chi1y_evolved = chi1y_evolved;
    # pPrec->chi1z_evolved = chi1z_evolved;
    # pPrec->chi2x_evolved = chi2x_evolved;
    # pPrec->chi2y_evolved = chi2y_evolved;
    # pPrec->chi2z_evolved = chi2z_evolved;

    # //printf("%f, %f, %f, %f, %f, %f\n", chi1x, chi1y, chi1z, chi2x, chi2y, chi2z);
    # //printf("%f, %f, %f, %f, %f, %f\n", chi1x_evolved, chi1y_evolved, chi1z_evolved, chi2x_evolved, chi2y_evolved, chi2z_evolved);
    # //printf("----\n");
    # }

    # // if PN numerical integration fails, default to MSA+fallback to NNLO
    # if(status==XLAL_FAILURE) {
    #                     LALFree(pPrec->PNarrays);
    #                     XLAL_PRINT_WARNING("Warning: due to a failure in the SpinTaylor routines, the model will default to MSA angles.");
    #                     pPrec->IMRPhenomXPrecVersion=223;
    #                     }
    # // end of SpinTaylor code
