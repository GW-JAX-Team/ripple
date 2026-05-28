"""
diagnostic_mismatch.py
======================
Quantify the two mechanisms behind the mass-dependent overlap loss in
IMRPhenomXAS_NRTidalv3 (ripple vs LAL).

Assertions being tested
-----------------------
H1. Time-shift residual dominates at lambda=0.
    The NRTidalv3 code replaces the BBH time-alignment (linb_XAS) with
    linb_tidal = -dPhase/dMf(f_final), evaluated in JAX vs LAL C.
    Any floating-point discrepancy appears as a linear FD phase offset
    and is visible as overlap_loss_unopt >> overlap_loss_phase_opt
    (where phase-opt maximises over a single constant phase).

H2. Overlap_loss at lambda=0 correlates with the Planck taper spectral
    weight = PSD-weighted fraction of signal power above f_merger.
    Heavier systems have lower f_merger => larger taper weight => larger
    float-precision amplification of any taper discrepancy.

H3. Mismatch grows with lambda, approaching 1e-6 for realistic BNS
    parameters.  This quantifies the margin above the test threshold.

Output
------
Two figures in tests/cross_validation/figures/:
  diagnostic_lambda0.png       (H1 + H2 decomposition, lambda=0 batch)
  diagnostic_lambda_sweep.png  (H3: mismatch vs lambda_tilde)

Printed summary includes Pearson correlations that confirm/refute H1+H2.

Runtime note
-----------
Each ripple NRTidalv3 waveform takes ~7 s on CPU (521 K-point grid, T=128 s).
JIT warms up once (~15 s), then:
  Part A: N_samples × 7 s  → default 20 samples ≈  2.5 min
  Part B: 5 systems × 6 lambdas × 7 s              ≈  3.5 min
  Total default run ≈ 6–7 minutes.

Usage
-----
  cd /path/to/ripple
  python tests/cross_validation/diagnostic_mismatch.py [--n-samples N]
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE.parent))  # tests/ for utils


from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc
from ripplegw.waveforms.NRTidalv3_utils import _get_merger_frequency
from ripplegw.waveforms import IMRPhenomX_utils
from ripplegw.waveforms.IMRPhenomXAS import Amp
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes

from utils import (
    LAL_AVAILABLE,
    generate_random_params,
    get_freqs,
    compute_overlap_loss,
    get_lal_waveform,
    get_nyquist_mask,
    noise_weighted_inner_product,
    _noise_weighted_inner_product_complex,
)

if not LAL_AVAILABLE:
    sys.exit("LALSuite not available — install it and retry.")

# ── frequency / segment setup matching the BNS test ──────────────────────────
F_L = 20.0
F_U = 4096.0
F_SAMPLING = 8192.0
T = 128.0
F_REF = 20.0
DF = 1.0 / T

LAMBDA0_BOUNDS = {
    "m": [0.5, 3.0],
    "chi": [-0.05, 0.05],
    "lambda": [0.0, 0.0],
    "d_L": [100.0, 500.0],
}

# ── inner-product helpers ─────────────────────────────────────────────────────

@jax.jit
def _ol_unopt(h1, h2, psd, f):
    """Overlap loss with no time or phase freedom (same as test)."""
    return compute_overlap_loss(h1, h2, psd, f)


@jax.jit
def _ol_phase_opt(h1, h2, psd, f):
    """
    Overlap loss maximised over a single constant phase offset.
    Uses |<h1|h2>| instead of Re(<h1|h2>).

    Interpretation: if ol_unopt >> ol_phase_opt for a given pair, a
    constant phase offset dominates the mismatch.  If they are similar,
    the mismatch is from time-shift (frequency-dependent phase) or
    amplitude shape.
    """
    h1_sq = noise_weighted_inner_product(h1, h1, psd, f)
    h2_sq = noise_weighted_inner_product(h2, h2, psd, f)
    h1_h2_abs = jnp.abs(_noise_weighted_inner_product_complex(h1, h2, psd, f))
    denom = jnp.sqrt(h1_sq * h2_sq)
    ol = (h1_sq * h2_sq - h1_h2_abs ** 2) / (denom * (denom + h1_h2_abs))
    return jnp.clip(ol, 0.0)


def _time_offset_from_phase(h_rip, h_lal, f, psd):
    """
    Estimate the time offset Δt (seconds) between two FD waveforms using
    a SNR-weighted linear regression of the cross-phase vs frequency.

    For a pure time shift h_lal = h_rip * exp(2πi Δt f), the phase
    difference arg(h_rip* h_lal) = 2π Δt f + const.  The slope gives Δt.

    Returns Δt in milliseconds (|Δt| reported; sign encodes direction).
    """
    h_rip = np.asarray(h_rip)
    h_lal = np.asarray(h_lal)
    psd = np.asarray(psd)
    f = np.asarray(f)

    cross = np.conj(h_rip) * h_lal
    valid = (np.abs(cross) > 0) & np.isfinite(psd) & (psd > 0)
    if valid.sum() < 10:
        return 0.0

    cross_v = cross[valid]
    f_v = f[valid]

    # Phase difference (unwrapping not critical: offsets are << 2π here)
    phi = np.angle(cross_v)
    phi_unwrapped = np.unwrap(phi)

    # SNR-squared weights: |h_rip* h_lal| / PSD
    w = np.abs(cross_v) / psd[valid]
    w /= w.sum()

    # Weighted linear regression: phi = a + 2π Δt f  =>  slope = 2π Δt
    f_bar = np.dot(w, f_v)
    f2_bar = np.dot(w, f_v ** 2)
    pf_bar = np.dot(w, phi_unwrapped * f_v)
    p_bar = np.dot(w, phi_unwrapped)

    var_f = f2_bar - f_bar ** 2
    cov_pf = pf_bar - p_bar * f_bar

    if var_f < 1e-30:
        return 0.0

    slope = cov_pf / var_f  # = 2π Δt
    delta_t_s = slope / (2.0 * np.pi)
    return delta_t_s * 1e3  # → milliseconds


def _planck_weight(m1, m2, chi1, chi2, f, psd):
    """
    PSD-weighted fraction of BBH signal power lying above f_merger.
    This is the spectral region the Planck taper truncates; any
    float-precision taper discrepancy is amplified by this weight.
    Returns (weight_fraction, f_merger_Hz).
    """
    theta_int = jnp.array([float(m1), float(m2), float(chi1), float(chi2), 0.0, 0.0])
    f_merger = float(_get_merger_frequency(theta_int))
    amp_coeffs = IMRPhenomX_utils.PhenomX_amp_coeff_table
    A = np.abs(
        np.array(Amp(f, jnp.array([float(m1), float(m2), float(chi1), float(chi2)]),
                     amp_coeffs, D=1.0))
    )
    w = A ** 2 / np.asarray(psd)
    total = float(np.trapezoid(w, np.asarray(f)))
    above = float(np.trapezoid(np.where(np.asarray(f) > f_merger, w, 0.0), np.asarray(f)))
    weight = above / total if total > 0.0 else 0.0
    return weight, f_merger


def _to_ripple(theta_lal, l1, l2):
    """Pack LAL parameter array into ripple format."""
    m1, m2 = float(theta_lal[0]), float(theta_lal[1])
    chi1, chi2 = float(theta_lal[2]), float(theta_lal[3])
    dist = float(theta_lal[6])
    tc   = float(theta_lal[7])
    phic = float(theta_lal[8])
    inc  = float(theta_lal[9])
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lt, dlt = lambdas_to_lambda_tildes(jnp.array([float(l1), float(l2), m1, m2]))
    return jnp.array([float(Mc), float(eta), chi1, chi2,
                      float(lt), float(dlt), dist, tc, phic, inc])


# ── analysis: lambda=0 batch ──────────────────────────────────────────────────

def run_lambda0_batch(n_samples, fs, psd_interp, nyquist_mask, wf_jit):
    """
    Returns a dict of arrays (one entry per valid sample) with:
      M_total, q, ol_unopt, ol_phase_opt, t_offset_ms, planck_weight, f_merger
    wf_jit: pre-compiled JAX waveform function.
    """
    theta_batch = generate_random_params(
        n_samples, LAMBDA0_BOUNDS, is_tidal=True, seed=42
    )

    def _get_lal(args):
        i, th = args
        try:
            hp, _ = get_lal_waveform(
                th, "IMRPhenomXAS_NRTidalv3",
                F_L, F_U, DF, F_REF, is_tidal=True, is_precessing=False,
            )
            return i, hp
        except Exception:
            return i, None

    print(f"  Computing {n_samples} LAL waveforms in parallel...")
    with ThreadPoolExecutor() as pool:
        lal_results = list(pool.map(_get_lal, enumerate(theta_batch)))

    out = {k: [] for k in
           ("M_total", "q", "ol_unopt", "ol_phase_opt", "t_offset_ms",
            "planck_weight", "f_merger")}

    for i, hp_lal in lal_results:
        if hp_lal is None:
            continue
        th = theta_batch[i]
        m1, m2 = th[0], th[1]

        th_r = _to_ripple(th, 0.0, 0.0)
        hp_r = wf_jit(th_r)

        h1 = hp_r * nyquist_mask
        h2 = jnp.array(hp_lal) * nyquist_mask

        ol_u  = float(_ol_unopt(h1, h2, psd_interp, fs))
        ol_ph = float(_ol_phase_opt(h1, h2, psd_interp, fs))
        t_off = _time_offset_from_phase(np.array(h1), np.array(h2),
                                        np.array(fs), np.array(psd_interp))
        pw, f_m = _planck_weight(m1, m2, th[2], th[3], fs, psd_interp)

        out["M_total"].append(float(m1 + m2))
        out["q"].append(float(min(m1, m2) / max(m1, m2)))
        out["ol_unopt"].append(ol_u)
        out["ol_phase_opt"].append(ol_ph)
        out["t_offset_ms"].append(abs(t_off))
        out["planck_weight"].append(pw)
        out["f_merger"].append(f_m)

    return {k: np.array(v) for k, v in out.items()}


# ── analysis: lambda sweep ────────────────────────────────────────────────────

def run_lambda_sweep(m1, m2, chi1, chi2, lambda_values, fs, psd_interp, nyquist_mask,
                     wf_jit, label=""):
    """
    Sweep lambda1=lambda2=lambda and return (lambda_tilde_arr, overlap_loss_arr).
    wf_jit: pre-compiled JAX waveform function (avoids re-compilation per system).
    Skips values where LAL raises (e.g. NaN output).
    """
    theta_base_lal = np.array([m1, m2, chi1, chi2, 0.0, 0.0, 300.0, 0.0, 0.5, 0.4])

    lts_out, losses_out = [], []
    for lam in lambda_values:
        theta_lal = theta_base_lal.copy()
        theta_lal[4] = lam
        theta_lal[5] = lam
        try:
            hp_lal, _ = get_lal_waveform(
                theta_lal, "IMRPhenomXAS_NRTidalv3",
                F_L, F_U, DF, F_REF, is_tidal=True, is_precessing=False,
            )
        except Exception:
            continue  # skip this lambda value

        th_r = _to_ripple(theta_lal, lam, lam)
        hp_r = wf_jit(th_r)

        h1 = hp_r * nyquist_mask
        h2 = jnp.array(hp_lal) * nyquist_mask

        loss = float(_ol_unopt(h1, h2, psd_interp, fs))
        lt = float(lambdas_to_lambda_tildes(jnp.array([lam, lam, m1, m2]))[0])

        lts_out.append(lt)
        losses_out.append(loss)
        if label:
            print(f"    {label}: λ={lam:.0f}  λ̃={lt:.0f}  ol={loss:.2e}")

    return np.array(lts_out), np.array(losses_out)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_lambda0(r, out_dir):
    M      = r["M_total"]
    ol_u   = r["ol_unopt"]
    ol_ph  = r["ol_phase_opt"]
    t_ms   = r["t_offset_ms"]
    pw     = r["planck_weight"]
    f_mrg  = r["f_merger"]

    phase_contrib = np.maximum(ol_u - ol_ph, 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "IMRPhenomXAS_NRTidalv3 (λ=0): mismatch decomposition\n"
        "H1: time-shift residual  |  H2: Planck taper weight",
        fontsize=11,
    )

    # ── (0,0): overlap_loss decomposition vs M_total ─────────────────────────
    ax = axes[0, 0]
    ax.semilogy(M, ol_u,  "o", ms=5, alpha=0.7, label="Unoptimised (= test metric)")
    ax.semilogy(M, np.maximum(ol_ph, 1e-18), "s", ms=5, alpha=0.7,
                label="Phase-optimised (constant Δφ removed)")
    ax.semilogy(M, np.maximum(phase_contrib, 1e-18), "^", ms=5, alpha=0.6,
                label="ol_unopt − ol_phase_opt  (phase / time-shift residual)")
    ax.axhline(1e-6, color="red", ls="--", lw=1.5, label="threshold 1e-6")
    ax.set_xlabel(r"$M_{\rm total}\ [M_\odot]$")
    ax.set_ylabel(r"$1 - \mathcal{O}$")
    ax.set_title("Overlap loss decomposition\n"
                 "(H1: if time-shift dominates, "
                 "unopt ≫ phase-opt)")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # ── (0,1): Planck taper weight vs M_total ────────────────────────────────
    ax = axes[0, 1]
    sc = ax.scatter(M, pw * 100.0, c=np.log10(np.maximum(ol_u, 1e-18)),
                    cmap="viridis", s=30, alpha=0.85)
    plt.colorbar(sc, ax=ax, label=r"$\log_{10}(1-\mathcal{O}_{\rm unopt})$")
    # annotate f_merger for the highest-weight points
    top_idx = np.argsort(pw)[-5:]
    for i in top_idx:
        ax.annotate(f"{f_mrg[i]:.0f} Hz", (M[i], pw[i] * 100.0),
                    fontsize=6, alpha=0.65, ha="left", va="bottom")
    ax.set_xlabel(r"$M_{\rm total}\ [M_\odot]$")
    ax.set_ylabel(r"Planck spectral weight [%]" "\n" r"(SNR$^2$ fraction above $f_\mathrm{merger}$)")
    ax.set_title("Planck taper spectral weight vs mass\n"
                 "(H2: mismatch should correlate with weight)")
    ax.grid(True, alpha=0.3)

    # ── (1,0): measured time offset vs M_total ───────────────────────────────
    ax = axes[1, 0]
    ax.semilogy(M, np.maximum(t_ms, 1e-8), "o", ms=5, alpha=0.7,
                color="tab:orange")
    ax.set_xlabel(r"$M_{\rm total}\ [M_\odot]$")
    ax.set_ylabel(r"$|\Delta t|\ [\mathrm{ms}]$")
    ax.set_title(
        "SNR-weighted linear FD phase offset\n"
        "(slope of cross-phase vs frequency = $2\\pi\\Delta t$)"
    )
    ax.grid(True, alpha=0.3)

    # ── (1,1): scatter overlap_loss vs Planck weight ─────────────────────────
    ax = axes[1, 1]
    sc = ax.scatter(pw * 100.0, ol_u, c=M, cmap="plasma", s=30, alpha=0.85)
    plt.colorbar(sc, ax=ax, label=r"$M_{\rm total}\ [M_\odot]$")
    ax.set_yscale("log")
    ax.axhline(1e-6, color="red", ls="--", lw=1.5, label="threshold 1e-6")
    r_val = float(np.corrcoef(pw, ol_u)[0, 1]) if len(pw) > 2 else float("nan")
    ax.set_xlabel("Planck spectral weight [%]")
    ax.set_ylabel(r"$1 - \mathcal{O}_{\rm unopt}$")
    ax.set_title(f"Overlap loss vs taper weight  (Pearson r = {r_val:.3f})\n"
                 "(H2 confirmed if |r| is high)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "diagnostic_lambda0.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_lambda_sweep(sweep_results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        r"IMRPhenomXAS_NRTidalv3: overlap loss (ripple vs LAL) vs $\tilde{\Lambda}$"
        "\n(H3: quantify margin above threshold for realistic BNS parameters)",
        fontsize=11,
    )
    colors = plt.colormaps["tab10"](np.linspace(0, 0.6, len(sweep_results)))

    for (label, lts, losses), col in zip(sweep_results, colors):
        valid = np.isfinite(losses) & (losses > 0)
        if valid.sum() == 0:
            continue
        for ax in axes:
            ax.semilogy(lts[valid], losses[valid], "o-", color=col,
                        label=label, alpha=0.8, ms=5)

    for ax in axes:
        ax.axhline(1e-6, color="red", ls="--", lw=1.5, label="threshold 1e-6")
        ax.set_xlabel(r"$\tilde{\Lambda}$")
        ax.set_ylabel(r"$1 - \mathcal{O}$  (unoptimised)")
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Full range")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), fontsize=8)

    axes[1].set_title("Zoomed near threshold")
    axes[1].set_ylim([5e-13, 5e-6])
    axes[1].legend(by_label.values(), by_label.keys(), fontsize=8)

    fig.tight_layout()
    path = out_dir / "diagnostic_lambda_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of lambda=0 samples for the batch analysis (default 20, ~2.5 min)")
    args = parser.parse_args()

    out_dir = HERE / "figures"
    out_dir.mkdir(exist_ok=True)

    # Load PSD
    psd_path = HERE.parent / "psds" / "ET_D_psd.txt"
    psd_freqs, psd_vals = np.loadtxt(str(psd_path), unpack=True)

    # Build frequency grid
    fs = get_freqs(F_L, F_U, F_SAMPLING, T)
    psd_interp = jnp.interp(fs, jnp.array(psd_freqs), jnp.array(psd_vals))
    nyquist_mask = get_nyquist_mask(fs)

    # ── Pre-compile the ripple waveform JIT once ─────────────────────────────
    print("\n  Warming up JAX JIT compilation (takes ~15 s once)...")
    _wf_jit = jax.jit(lambda th: gen_IMRPhenomXAS_NRTidalv3_hphc(fs, th, F_REF)[0])
    _Mc0, _eta0 = ms_to_Mc_eta(jnp.array([1.4, 1.4]))
    _lt0, _dlt0 = lambdas_to_lambda_tildes(jnp.array([0.0, 0.0, 1.4, 1.4]))
    _th0 = jnp.array([float(_Mc0), float(_eta0), 0.0, 0.0,
                       float(_lt0), float(_dlt0), 300.0, 0.0, 0.5, 0.4])
    _wf_jit(_th0).block_until_ready()
    print("  JIT ready.\n")

    # ── Part A: lambda=0 batch ────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"Part A: lambda=0 mismatch decomposition  ({args.n_samples} samples)")
    print(f"{'='*60}")

    r = run_lambda0_batch(args.n_samples, fs, psd_interp, nyquist_mask, _wf_jit)
    n = len(r["M_total"])

    if n == 0:
        print("  No valid samples — check LAL availability.")
        return

    ol_u  = r["ol_unopt"]
    ol_ph = r["ol_phase_opt"]
    t_ms  = r["t_offset_ms"]
    pw    = r["planck_weight"]
    M     = r["M_total"]

    phase_contrib = np.maximum(ol_u - ol_ph, 0.0)

    r_pw_ol  = float(np.corrcoef(pw,  ol_u)[0, 1])
    r_M_ol   = float(np.corrcoef(M,   ol_u)[0, 1])
    r_t_ol   = float(np.corrcoef(t_ms, ol_u)[0, 1])

    print(f"\n  Samples processed: {n}")
    print(f"\n  Overlap loss (unoptimised):")
    print(f"    median = {np.median(ol_u):.2e}   max = {np.max(ol_u):.2e}")
    print(f"\n  Overlap loss (phase-optimised, constant Δφ removed):")
    print(f"    median = {np.median(ol_ph):.2e}   max = {np.max(ol_ph):.2e}")
    print(f"\n  Constant-phase contribution  (ol_unopt - ol_phase_opt):")
    print(f"    median = {np.median(phase_contrib):.2e}   max = {np.max(phase_contrib):.2e}")
    print(f"\n  Time offset |Δt| from linear FD cross-phase (ms):")
    print(f"    median = {np.median(t_ms)*1e3:.2f} μs   max = {np.max(t_ms)*1e3:.2f} μs")
    print(f"\n  Planck taper spectral weight above f_merger:")
    print(f"    median = {np.median(pw)*100:.3f}%   max = {np.max(pw)*100:.3f}%")
    print(f"\n  Pearson correlations (testing hypotheses):")
    print(f"    H1 — corr(|Δt|, ol_unopt)     = {r_t_ol:.3f}  (time-shift driver)")
    print(f"    H2 — corr(planck_weight, ol)   = {r_pw_ol:.3f}  (taper weight)")
    print(f"         corr(M_total,        ol)   = {r_M_ol:.3f}  (mass dependence)")
    print(f"\n  → H1 confirmed if |r_t_ol| is high  (~> 0.7)")
    print(f"    → H2 confirmed if |r_pw_ol| and |r_M_ol| are high")

    path = plot_lambda0(r, out_dir)
    print(f"\n  Saved: {path}")

    # ── Part B: lambda sweep ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Part B: overlap loss vs lambda_tilde  (H3 quantification)")
    print(f"{'='*60}")

    # Lambda grid: 0 → 5000 (covers soft EOS through stiff EOS)
    lambda_values = np.array([0, 100, 300, 500, 1000, 1500, 2000, 3000, 5000],
                             dtype=float)

    # Representative BNS systems: light / canonical / heavy to probe Planck-taper regime
    sweep_configs = [
        (1.4, 1.4, 0.00, 0.00, "1.4+1.4  M=2.8"),
        (1.8, 1.2, 0.00, 0.00, "1.8+1.2  M=3.0"),
        (2.5, 1.5, 0.00, 0.00, "2.5+1.5  M=4.0"),
        (2.8, 2.0, 0.00, 0.00, "2.8+2.0  M=4.8"),
        (1.4, 1.4, 0.05, 0.05, "1.4+1.4  χ=0.05"),
    ]

    sweep_results = []
    for m1, m2, chi1, chi2, label in sweep_configs:
        print(f"\n  {label}  (M_tot={m1+m2:.1f} M☉)")
        lts, losses = run_lambda_sweep(
            m1, m2, chi1, chi2, lambda_values,
            fs, psd_interp, nyquist_mask, _wf_jit, label=label,
        )
        sweep_results.append((label, lts, losses))
        if len(losses) > 0:
            # Find lambda_tilde where mismatch reaches threshold
            above = losses >= 1e-6
            if above.any():
                first_idx = np.argmax(above)
                print(f"    ✗ crosses 1e-6 at λ̃ ≈ {lts[first_idx]:.0f}")
            else:
                lt_max = lts[-1] if len(lts) else float("nan")
                print(f"    ✓ stays below 1e-6 up to λ̃ = {lt_max:.0f}")

    path = plot_lambda_sweep(sweep_results, out_dir)
    print(f"\n  Saved: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
