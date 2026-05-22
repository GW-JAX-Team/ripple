#!/usr/bin/env python3
"""Check phase region at f_merger for the actual test parameter sets."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ripplegw.waveforms.IMRPhenomXAS import Phase
from ripplegw.waveforms.IMRPhenomX_utils import PhenomX_phase_coeff_table, get_cutoff_fMs
from ripplegw.waveforms.NRTidalv3_utils import _get_merger_frequency
from ripplegw.constants import MTSUN, PI

# Test parameter sets from the CSV (m1, m2, chi1, chi2, lambda1, lambda2)
samples = [
    (2.924775, 2.876786, -0.032948, -0.036051, 3875.664117, 2475.884551),  # worst
    (2.581107, 2.329985, -0.043495, -0.020786, 4697.494708,  171.942606),
    (2.665440, 1.811891,  0.001423, -0.040233, 2600.340106,  979.914312),
    (2.002788, 1.579863,  0.009241,  0.018423, 2733.551397,  226.136445),
    (2.270181, 1.228073, -0.045355, -0.005985,  924.272278, 1626.651654),
    (1.996646, 1.030848, -0.013364,  0.044889, 4546.602010, 4474.136752),
    (1.436350, 0.551461,  0.011185,  0.010754,  610.191174, 4847.923139),
    (0.954562, 0.890047,  0.046563, -0.004393, 2989.499894, 1293.899908),
    (0.958511, 0.889986,  0.030840,  0.028518, 4609.371175, 3312.611422),
    (1.260606, 0.645209, -0.019539, -0.030033,  442.462510, 1558.555380),  # best
]

overlap_losses = [2.83e-8, 1.69e-8, 8.22e-9, 4.44e-9, 3.53e-9, 3.00e-9, 4.15e-10, 2.92e-10, 2.41e-10, 2.03e-10]

T = 128.0
df = 1.0 / T

print(f"{'m1':>6} {'m2':>6} {'f_merger':>10} {'Mf_merger':>11} {'f1_Ms':>9} {'f2_Ms':>9} {'region':>12} {'Mf-f1_Ms':>10} {'df*d2phi':>10} {'loss':>10}")
print("-" * 120)

for (m1, m2, chi1, chi2, l1, l2), loss in zip(samples, overlap_losses):
    theta_i = jnp.array([m1, m2, chi1, chi2, l1, l2])
    M_s = (m1 + m2) * MTSUN

    f_merger = float(_get_merger_frequency(theta_i))
    Mf_merger = f_merger * M_s

    fMs_RD, _, fMs_MECO, fMs_ISCO = get_cutoff_fMs(m1, m2, chi1, chi2)
    fMs_MECO = float(fMs_MECO)
    fMs_ISCO = float(fMs_ISCO)
    fMs_RD = float(fMs_RD)
    fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
    deltafMs = (fMs_IMmatch - fMs_MECO) * 0.03
    f1_Ms = fMs_MECO - 1.0 * deltafMs
    f2_Ms = fMs_IMmatch + 0.5 * deltafMs

    if Mf_merger < f1_Ms:
        region = "inspiral"
    elif Mf_merger < f2_Ms:
        region = "intermediate"
    else:
        region = "ringdown"

    # Distance of Mf_merger from f1_Ms boundary
    dist_from_f1 = Mf_merger - f1_Ms

    # Compute second derivative of Phase at f_final using finite differences
    theta_bbh = jnp.array([m1, m2, chi1, chi2])
    f_final = f_merger  # since f_merger < 4096 Hz typically

    # PhaseDerivative at f_final and f_final - df
    dphi_at_final = float(jax.grad(Phase)(f_final, theta_bbh, PhenomX_phase_coeff_table))
    dphi_at_final_m1 = float(jax.grad(Phase)(f_final - df, theta_bbh, PhenomX_phase_coeff_table))

    # Backward secant approximation to dphiXAS at f_final
    secant = (float(Phase(f_final, theta_bbh, PhenomX_phase_coeff_table)) -
              float(Phase(f_final - df, theta_bbh, PhenomX_phase_coeff_table))) / (df * M_s)
    analytic = dphi_at_final / M_s

    # Error in dphiXAS = linb error = integrated effect on overlap
    linb_error = analytic - secant

    print(f"{m1:>6.2f} {m2:>6.2f} {f_merger:>10.2f} {Mf_merger:>11.6f} {f1_Ms:>9.6f} {f2_Ms:>9.6f} {region:>12} {dist_from_f1:>10.6f} {linb_error:>10.4e} {loss:>10.2e}")

    # More detailed check for worst case
    if m1 > 2.9:
        print(f"  >>> Worst case detail:")
        print(f"      f_final - df = {f_final - df:.6f} Hz, Mf(f-df) = {(f_final-df)*M_s:.8f}")
        print(f"      f1_Ms = {f1_Ms:.8f}, f_final*M_s = {f_final*M_s:.8f}")
        print(f"      Does f_final-df cross f1_Ms? {(f_final-df)*M_s < f1_Ms and f_final*M_s >= f1_Ms}")
        print(f"      secant  dphiXAS/dMf = {secant:.8e}")
        print(f"      analytic dphiXAS/dMf = {analytic:.8e}")
        print(f"      linb_error = analytic - secant = {linb_error:.8e}")
        # What phase error does this cause at f_ref vs f_merger?
        Mf_ref = 20.0 * M_s
        phase_error_at_fref = linb_error * (Mf_ref - Mf_merger)
        print(f"      Phase error at f_ref from linb_error: {phase_error_at_fref:.4e} rad")

        # Estimate overlap loss from linear phase error
        # delta_phi = linb_error * (Mf - Mf_merger)
        # For a noise-weighted integral with ET PSD, this gives a specific overlap loss
        print(f"      linb_error * (Mf_ref - Mf_merger) = {linb_error*(Mf_ref - Mf_merger):.4e} rad")
