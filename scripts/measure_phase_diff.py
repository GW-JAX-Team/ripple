#!/usr/bin/env python3
"""Measure ripple-vs-LAL phase difference on a physically relevant frequency band."""

import jax
import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ripplegw.constants import MTSUN
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3, _get_merger_frequency
from tests.utils import compute_overlap_loss, get_lal_waveform, get_nyquist_mask


def analyze_case(name, m1, m2, chi1, chi2, lambda1, lambda2, dist_mpc, phi_ref, incl):
    T = 128.0
    df = 1.0 / T
    f_l, f_u, f_ref_val = 20.0, 4096.0, 20.0
    freqs = np.arange(int(f_u / df) + 1) * df
    fs = freqs[(freqs > f_l) & (freqs < f_u)]
    M_s = (m1 + m2) * MTSUN

    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(
        jnp.array([lambda1, lambda2, m1, m2])
    )
    params_rip = jnp.array(
        [
            float(Mc),
            float(eta),
            chi1,
            chi2,
            float(lambda_tilde),
            float(delta_lambda_tilde),
            dist_mpc,
            0.0,
            phi_ref,
        ]
    )
    h_rip = np.array(gen_IMRPhenomXAS_NRTidalv3(jnp.array(fs), params_rip, f_ref_val))

    theta_lal = np.array(
        [m1, m2, chi1, chi2, lambda1, lambda2, dist_mpc, 0.0, phi_ref, incl], dtype=float
    )
    h_lal, _ = get_lal_waveform(
        theta_lal,
        "IMRPhenomXAS_NRTidalv3",
        f_l,
        f_u,
        df,
        f_ref_val,
        is_tidal=True,
        is_precessing=False,
    )

    nyquist_mask = np.array(get_nyquist_mask(jnp.array(fs), 2))
    h_rip = h_rip * nyquist_mask
    h_lal = np.array(h_lal) * nyquist_mask

    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    f_merger = float(_get_merger_frequency(theta_intrinsic))
    f_band_max = min(f_u - df, 1.35 * f_merger)

    amp_r = np.abs(h_rip)
    amp_l = np.abs(h_lal)
    amp_cut = 1e-10 * max(np.max(amp_r), np.max(amp_l))
    band = (fs <= f_band_max) & (amp_r > amp_cut) & (amp_l > amp_cut)
    if np.sum(band) < 10:
        raise RuntimeError(f"Insufficient bins in analysis band for {name}")

    ratio = h_rip[band] / h_lal[band]
    dphi = np.unwrap(np.angle(ratio))
    dphi0 = dphi - dphi[0]
    f_band = fs[band]
    Mf_band = f_band * M_s

    A = np.vstack([np.ones_like(Mf_band), Mf_band]).T
    coeffs, *_ = np.linalg.lstsq(A, dphi0, rcond=None)
    dphi_lin = coeffs[0] + coeffs[1] * Mf_band
    dphi_res = dphi0 - dphi_lin

    psd_data = np.loadtxt("tests/psds/ET_D_psd.txt")
    psd_interp = np.interp(fs, psd_data[:, 0], psd_data[:, 1])
    overlap_loss = float(compute_overlap_loss(h_rip, h_lal, psd_interp, fs))

    print(f"\n[{name}] m_total={m1+m2:.4f}, f_merger={f_merger:.2f} Hz, band<= {f_band_max:.2f} Hz")
    print(f"  overlap_loss={overlap_loss:.6e}")
    print(f"  bins in band={np.sum(band)} ({f_band[0]:.2f}..{f_band[-1]:.2f} Hz)")
    print(f"  linear slope b={coeffs[1]:.6e} rad/Mf")
    print(f"  residual max={np.max(np.abs(dphi_res)):.6e} rad, rms={np.sqrt(np.mean(dphi_res**2)):.6e} rad")
    for fc in [20, 50, 100, 200, 500, min(1000, f_band[-1]), f_band[-1]]:
        idx = np.argmin(np.abs(f_band - fc))
        print(
            f"    f={f_band[idx]:8.2f} Hz: dphi={dphi0[idx]: .6e}, dphi_res={dphi_res[idx]: .6e}"
        )


if __name__ == "__main__":
    analyze_case(
        "worst-sample",
        2.9247746304049858,
        2.87678576602479,
        -0.03294758763127085,
        -0.03605061393479582,
        3875.6641168055726,
        2475.884550556351,
        157.53404493373108,
        3.9162976324754704,
        0.624283725260845,
    )
    analyze_case(
        "zero-lambda-check",
        1.4,
        1.3,
        0.02,
        -0.01,
        0.0,
        0.0,
        100.0,
        0.0,
        np.pi / 3.0,
    )
