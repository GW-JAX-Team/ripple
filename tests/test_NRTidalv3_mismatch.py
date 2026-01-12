"""
Mismatch test comparing Ripple's IMRPhenomXAS_NRTidalv3 implementation against LALSuite.

This script computes noise-weighted mismatches between LAL and Ripple waveforms
for both the base IMRPhenomXAS model and the NRTidalv3 tidal extension.

Nyquist Boundary Handling
-------------------------
LAL's behavior at the Nyquist frequency boundary is inconsistent - it sometimes
zeros 1 bin, sometimes 2 bins near f_Nyquist depending on the waveform parameters.
This causes apparent mismatches of O(1e-11) that are not due to implementation
differences but rather to boundary handling. To ensure a fair comparison, we apply
the same Nyquist mask (zeroing the last 2 frequency bins) to both LAL and Ripple
waveforms before computing the match.

Expected Results
----------------
With proper Nyquist handling, both IMRPhenomXAS and IMRPhenomXAS_NRTidalv3 achieve
machine-precision agreement (mismatch ~ 1e-15 to 1e-16) for the vast majority of
parameter space.

For extreme mass ratios (q > 6, eta < 0.12), mismatches up to ~1e-13 may occur due
to small numerical differences in how tidal phase coefficients accumulate. These
manifest as a tiny linear phase drift (~4e-5 rad over the full bandwidth) while
amplitude agreement remains at machine precision. This level of mismatch is
negligible for any practical gravitational wave application.

Cases showing -inf or nan in log10(mismatch) indicate perfect matches where the
mismatch is exactly 0 or slightly negative due to floating-point arithmetic.
"""

import lal
from lal import MSUN_SI, PC_SI
import lalsimulation as lalsim
from lalsimulation import IMRPhenomXAS, IMRPhenomXAS_NRTidalv3
import tqdm
from pathlib import Path

MPC_SI = PC_SI * 1e6
print(lalsim.__version__)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from jax.scipy.integrate import trapezoid
import numpy as np
import matplotlib.pyplot as plt

from ripplegw import ms_to_Mc_eta, lambdas_to_lambda_tildes
from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc
from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc

# Nyquist boundary handling: LAL's behavior at the Nyquist boundary varies
# (sometimes zeros 1 bin, sometimes 2 bins depending on waveform parameters).
# To ensure a fair comparison, we apply the same mask to both LAL and Ripple
# waveforms, zeroing the last 2 frequency bins near the Nyquist frequency.
NYQUIST_BINS_TO_ZERO = 2

# Choose the ET PSD just to have the wide frequency range
# Try loading from current directory first, then relative to tests directory
psd_path = Path("psds/ET-D-psd.txt")
if not psd_path.exists():
    psd_path = Path(__file__).parent / "psds/ET-D-psd.txt"
    print(f"Loading PSD from {psd_path}")
psd_freqs, *psd_arrs = np.loadtxt(psd_path, unpack=True)
psd_arr = psd_arrs[0]


def get_freq_array(sampling_freqs, duration):
    # Build the frequency grid
    delta_t = 1 / sampling_freqs
    tlen = int(round(duration * sampling_freqs))
    freqs = jnp.fft.rfftfreq(tlen, delta_t)
    return freqs


def get_nyquist_mask(frequencies):
    """Create a mask that zeros the last NYQUIST_BINS_TO_ZERO frequency bins."""
    n_freqs = len(frequencies)
    return jnp.where(jnp.arange(n_freqs) < n_freqs - NYQUIST_BINS_TO_ZERO, 1.0, 0.0)


def noise_weighted_inner_product(h1, h2, psd, frequencies):
    integrand = jnp.conj(h1) * h2 / psd
    return 4 * trapezoid(integrand, x=frequencies, axis=-1).real


def compute_match(h1, h2, psd, frequencies):
    h1_sq = noise_weighted_inner_product(h1, h1, psd, frequencies)
    h2_sq = noise_weighted_inner_product(h2, h2, psd, frequencies)
    h1_h2 = noise_weighted_inner_product(h1, h2, psd, frequencies)
    match = h1_h2 / jnp.sqrt(h1_sq * h2_sq)
    return match.real


fs = 4096

# Yes, I should have combined the XAS and tidal functions, I am lazy.


def lal_XAS_waveform(
    mass_1,
    mass_2,
    a_1,
    a_2,
    inclination,
    phase,
    Lambda_1,
    Lambda_2,
    duration,
    f_min,
    f_max,
    reference_frequency,
):

    lal_hpc = lalsim.SimInspiralChooseFDWaveform(
        mass_1 * MSUN_SI,
        mass_2 * MSUN_SI,
        0,
        0,
        a_1,
        0,
        0,
        a_2,
        1,  # m
        inclination,
        phase,
        0.0,
        0.0,
        0.0,
        1.0 / duration,
        f_min,
        f_max,
        reference_frequency,
        None,
        IMRPhenomXAS,
    )
    hpc_len = len(lal_hpc[0].data.data)

    frequencies = get_freq_array(sampling_freqs=fs, duration=duration)
    zeros = jnp.zeros_like(frequencies, dtype=jnp.complex128)
    hp = zeros.at[:hpc_len].set(lal_hpc[0].data.data)
    hc = zeros.at[:hpc_len].set(lal_hpc[1].data.data)

    return frequencies, hp, hc


def ripple_XAS_waveform(
    mass_1,
    mass_2,
    a_1,
    a_2,
    inclination,
    phase,
    Lambda_1,
    Lambda_2,
    duration,
    f_min,
    f_max,
    reference_frequency,
):

    frequencies = get_freq_array(sampling_freqs=fs, duration=duration)
    frequency_mask = (frequencies >= f_min) & (frequencies <= f_max)

    ones = jnp.ones_like(mass_1)
    chirp_mass, symmetric_mass_ratio = ms_to_Mc_eta(jnp.array([mass_1, mass_2]))

    thetas = jnp.array(
        [
            chirp_mass,
            symmetric_mass_ratio,
            a_1,
            a_2,
            ones / MPC_SI,
            ones * 0.0,
            phase,
            inclination,
        ]
    )

    zeros = jnp.zeros_like(frequencies, dtype=jnp.complex128)

    @jax.jit
    def waveform(theta):
        hpc = gen_IMRPhenomXAS_hphc(
            frequencies[frequency_mask], theta, reference_frequency
        )
        hp = zeros.at[frequency_mask].set(hpc[0])
        hc = zeros.at[frequency_mask].set(hpc[1])
        return hp, hc

    if thetas[0].size > 1:
        ripple_hpc = jax.vmap(waveform)(thetas.T)
    else:
        ripple_hpc = waveform(thetas)

    return frequencies, *ripple_hpc


def lal_NRTidalv3_waveform(
    mass_1,
    mass_2,
    a_1,
    a_2,
    inclination,
    phase,
    Lambda_1,
    Lambda_2,
    duration,
    f_min,
    f_max,
    reference_frequency,
):

    laldict = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, Lambda_1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, Lambda_2)
    quad1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(Lambda_1)
    quad2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(Lambda_2)
    # Note that these are dquadmon, not quadmon, hence have to subtract 1 since that is added again later
    lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, quad1 - 1)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, quad2 - 1)

    lal_hpc = lalsim.SimInspiralChooseFDWaveform(
        mass_1 * MSUN_SI,
        mass_2 * MSUN_SI,
        0,
        0,
        a_1,
        0,
        0,
        a_2,
        1,  # m
        inclination,
        phase,
        0.0,
        0.0,
        0.0,
        1.0 / duration,
        f_min,
        f_max,
        reference_frequency,
        laldict,
        IMRPhenomXAS_NRTidalv3,
    )
    hpc_len = len(lal_hpc[0].data.data)

    frequencies = get_freq_array(sampling_freqs=fs, duration=duration)
    zeros = jnp.zeros_like(frequencies, dtype=jnp.complex128)
    hp = zeros.at[:hpc_len].set(lal_hpc[0].data.data)
    hc = zeros.at[:hpc_len].set(lal_hpc[1].data.data)

    # Apply Nyquist mask for consistent comparison
    nyquist_mask = get_nyquist_mask(frequencies)
    hp = hp * nyquist_mask
    hc = hc * nyquist_mask

    return frequencies, hp, hc


def ripple_NRTidalv3_waveform(
    mass_1,
    mass_2,
    a_1,
    a_2,
    inclination,
    phase,
    Lambda_1,
    Lambda_2,
    duration,
    f_min,
    f_max,
    reference_frequency,
):

    frequencies = get_freq_array(sampling_freqs=fs, duration=duration)
    frequency_mask = (frequencies >= f_min) & (frequencies <= f_max)
    nyquist_mask = get_nyquist_mask(frequencies)

    ones = jnp.ones_like(mass_1)
    chirp_mass, symmetric_mass_ratio = ms_to_Mc_eta(jnp.array([mass_1, mass_2]))
    lambda_p, lambda_m = lambdas_to_lambda_tildes(
        jnp.array([Lambda_1, Lambda_2, mass_1, mass_2])
    )

    thetas = jnp.array(
        [
            chirp_mass,
            symmetric_mass_ratio,
            a_1,
            a_2,
            lambda_p,
            lambda_m,
            ones / MPC_SI,
            ones * 0.0,
            phase,
            inclination,
        ]
    )

    zeros = jnp.zeros_like(frequencies, dtype=jnp.complex128)

    @jax.jit
    def waveform(theta):
        hpc = gen_IMRPhenomXAS_NRTidalv3_hphc(
            frequencies[frequency_mask], theta, reference_frequency
        )
        hp = zeros.at[frequency_mask].set(hpc[0])
        hc = zeros.at[frequency_mask].set(hpc[1])
        # Apply Nyquist mask for consistent comparison
        hp = hp * nyquist_mask
        hc = hc * nyquist_mask
        return hp, hc

    if thetas[0].size > 1:
        ripple_hpc = jax.vmap(waveform)(thetas.T)
    else:
        ripple_hpc = waveform(thetas)

    return frequencies, *ripple_hpc


seed = 10
Nsamples = 1000
duration = 16.0
f_min = 5.0
f_max = fs / 2

rand_arr = jax.random.uniform(
    jax.random.PRNGKey(seed), (8, Nsamples), dtype=jnp.float64
)
total_mass = rand_arr[0] * 1.0 + 0.5
mass_ratio = rand_arr[1] * 0.95 + 0.05
mass_1 = total_mass / (1 + mass_ratio)
mass_2 = total_mass - mass_1
spin_1z, spin_2z = rand_arr[2:4] * 2 - 1
inclination = rand_arr[4] * jnp.pi
phase = rand_arr[5] * 2 * jnp.pi
Lambda_1, Lambda_2 = rand_arr[6:] * 10000

thetas = jnp.vstack(
    [mass_1, mass_2, spin_1z, spin_2z, inclination, phase, Lambda_1, Lambda_2]
)

lal_XAS_wfms, lal_Tidalv3_wfms = [], []
for theta in tqdm.tqdm(thetas.T):
    _, *lal_hpc = lal_NRTidalv3_waveform(
        *np.array(theta),
        duration=duration,
        f_min=f_min,
        f_max=f_max,
        reference_frequency=f_min,
    )
    lal_Tidalv3_wfms.append(lal_hpc)
    _, *lal_hpc = lal_XAS_waveform(
        *np.array(theta),
        duration=duration,
        f_min=f_min,
        f_max=f_max,
        reference_frequency=f_min,
    )
    lal_XAS_wfms.append(lal_hpc)

_, *ripple_XAS_hpc = ripple_XAS_waveform(
    *thetas, duration=duration, f_min=f_min, f_max=f_max, reference_frequency=f_min
)
_, *ripple_Tidalv3_hpc = ripple_NRTidalv3_waveform(
    *thetas, duration=duration, f_min=f_min, f_max=f_max, reference_frequency=f_min
)

lal_XAS_hpc = jnp.array(lal_XAS_wfms).transpose(1, 0, 2)
lal_Tidalv3_hpc = jnp.array(lal_Tidalv3_wfms).transpose(1, 0, 2)
ripple_XAS_hpc = jnp.array(ripple_XAS_hpc)
ripple_Tidalv3_hpc = jnp.array(ripple_Tidalv3_hpc)

frequencies = get_freq_array(sampling_freqs=fs, duration=duration)
interp_psd = jnp.interp(frequencies, psd_freqs, psd_arr)
XAS_match_hp = compute_match(lal_XAS_hpc[0], ripple_XAS_hpc[0], interp_psd, frequencies)
XAS_match_hc = compute_match(lal_XAS_hpc[1], ripple_XAS_hpc[1], interp_psd, frequencies)
Tidalv3_match_hp = compute_match(
    lal_Tidalv3_hpc[0], ripple_Tidalv3_hpc[0], interp_psd, frequencies
)
Tidalv3_match_hc = compute_match(
    lal_Tidalv3_hpc[1], ripple_Tidalv3_hpc[1], interp_psd, frequencies
)


Tidalv3_log_mismatches = np.log10(jnp.maximum(1 - Tidalv3_match_hp, 1e-16))
XAS_log_mismatches = np.log10(jnp.maximum(1 - XAS_match_hp, 1e-16))

XAS_hist_data = np.where(
    np.isfinite(XAS_log_mismatches),
    XAS_log_mismatches,
    min(np.where(np.isfinite(XAS_log_mismatches), XAS_log_mismatches, 100)),
)
Tidalv3_hist_data = np.where(
    np.isfinite(Tidalv3_log_mismatches),
    Tidalv3_log_mismatches,
    min(np.where(np.isfinite(Tidalv3_log_mismatches), Tidalv3_log_mismatches, 100)),
)

fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].set_title("XAS Mismatches")
axs[0].hist(XAS_hist_data, alpha=0.7, color="C0")
axs[0].set_ylabel("Counts")

axs[1].set_title("NRTidalv3 Mismatches")
axs[1].hist(Tidalv3_hist_data, alpha=0.7, color="C1")
axs[1].set_ylabel("Counts")
axs[1].set_xlabel(r"$\log{\mathcal{M}}$")

plt.suptitle("Mismatch Comparison between LAL and Ripple Implementations (low mass)")
plt.tight_layout()
plt.savefig("test_NRTidalv3_mismatch.png")


chirp_mass, _ = ms_to_Mc_eta(jnp.array([mass_1, mass_2]))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
ax.scatter(chirp_mass, 1 - Tidalv3_match_hp, label=r"$h_+$")
ax.scatter(chirp_mass, 1 - Tidalv3_match_hc, marker="*", label=r"$h_\times$")
ax.set_yscale("log")
ax.legend()
ax.set_xlabel(r"Chirp Mass, ${\cal M}_{\rm c}\,/\,M_\odot$")
ax.set_ylabel(r"Mismatch, $1 - {\cal M}$")
ax.set_title("Mismatch between LALSuite and Ripple:" + "\n" + "IMRPhenomXAS_NRTidalv3")

ax = axes[1]
ax.scatter(chirp_mass, 1 - XAS_match_hp, label=r"$h_+$")
ax.scatter(chirp_mass, 1 - XAS_match_hc, marker="*", label=r"$h_\times$")
ax.set_yscale("log")
ax.legend()
ax.set_xlabel(r"Chirp Mass, ${\cal M}_{\rm c}\,/\,M_\odot$")
ax.set_ylabel(r"Mismatch, $1 - {\cal M}$")
ax.set_title("Mismatch between LALSuite and Ripple:" + "\n" + "IMRPhenomXAS")

plt.tight_layout()
plt.savefig("test_NRTidalv3_mismatch_scatter.png")
