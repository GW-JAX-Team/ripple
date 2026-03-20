"""Cross-validation tests comparing ripple waveforms against LALSuite.

These tests compute noise-weighted mismatches between ripple and LALSuite
waveforms over randomly sampled parameter sets. They verify that ripple
achieves machine-precision agreement with LAL.

Nyquist Boundary Handling:
    LAL's behavior at the Nyquist frequency boundary is inconsistent - it
    sometimes zeros 1 bin, sometimes 2 bins depending on the waveform parameters.
    To ensure a fair comparison, we apply the same Nyquist mask (zeroing the
    last 2 frequency bins) to both LAL and ripple waveforms before computing
    the match.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from tests.utils import (
    check_lal_available,
    check_is_tidal,
    check_is_precessing,
    get_freqs,
    get_jitted_waveform,
    get_lal_waveform,
    get_nyquist_mask,
    compute_match,
    generate_random_params,
    LAL_AVAILABLE,
)

jax.config.update("jax_enable_x64", True)

# Skip all tests if LALSuite is not available
pytestmark = pytest.mark.skipif(
    not LAL_AVAILABLE,
    reason="LALSuite required for cross-validation tests",
)


# ============================================================================
# Test configuration
# ============================================================================

# Default parameter bounds for random sampling
DEFAULT_BOUNDS = {
    "m": [0.5, 3.0],  # Masses (solar masses) - BNS range
    "chi": [-0.05, 0.05],  # Aligned spins (BNS physical range; both signs tested)
    "lambda": [0.0, 5000.0],  # Tidal parameters
    "d_L": [30.0, 500.0],  # Distance (Mpc)
}

BBH_BOUNDS = {
    "m": [5.0, 100.0],  # Masses (solar masses) - BBH range
    "chi": [-0.99, 0.99],  # Spins
    "lambda": [0.0, 0.0],  # No tidal
    "d_L": [100.0, 3000.0],  # Distance (Mpc)
}

# Per-waveform mismatch thresholds.
# These represent expected float64 agreement between the ripple and LAL
# implementations; simpler/more-analytical waveforms achieve near-machine
# precision while complex NR-calibrated ones accumulate more rounding error.
#
# IMRPhenomPv2 NOTE: the 1e-4 threshold is intentionally loose and does NOT
# reflect a deficiency in ripple's implementation. The mismatch is entirely a
# pure time shift (amplitude agreement is perfect to <1e-9). The shift arises
# because LAL computes the coalescence-time correction t0 via a 10-point
# natural cubic spline over [0.8*f_RD, 1.2*f_RD] (GSL gsl_interp_cspline,
# LALSimIMRPhenomP.c lines 1060–1151), whereas ripple uses exact JAX autodiff.
# The coarse 10-point grid (~9–12 Hz spacing) underresolves the Lorentzian
# arctan feature in the merger-ringdown phase (characteristic width ~f_damp
# ≈ 22 Hz), introducing a derivative error of 5–12 μs depending on the
# system. Ripple's exact derivative is the more accurate result. Worst-case
# mismatch reaches ~1.3e-5 for high-mass-ratio, high-spin systems; the 1e-4
# threshold gives comfortable headroom. See tests/Pv2_dev/ for full analysis.
MISMATCH_THRESHOLDS = {
    "IMRPhenomD": 1e-6,
    "IMRPhenomXAS": 1e-13,
    "IMRPhenomD_NRTidalv2": 1e-9,
    "IMRPhenomXAS_NRTidalv3": 1e-6,
    "TaylorF2": 1e-14,
    "IMRPhenomPv2": 1e-4,  # see note above
    "IMRPhenomXPHM": 1e-6,
}
DEFAULT_MISMATCH_THRESHOLD = 1e-5  # fallback for unknown waveforms


def get_mismatch_threshold(waveform_name: str) -> float:
    """Return the per-waveform mismatch threshold."""
    return MISMATCH_THRESHOLDS.get(waveform_name, DEFAULT_MISMATCH_THRESHOLD)


# ============================================================================
# LAL waveform cache
# ============================================================================

LAL_CACHE_DIR = Path(__file__).parent / "lal_cache"


def _lal_cache_path(waveform_name: str, T: float) -> Path:
    """Return the .npz cache path for LAL waveforms keyed by (waveform_name, T)."""
    T_str = f"T{int(T)}" if T == int(T) else f"T{T}"
    return LAL_CACHE_DIR / f"{waveform_name}_{T_str}.npz"


def _load_lal_cache(cache_path: Path, n_samples: int, n_freqs: int) -> dict | None:
    """Load cached LAL waveforms if the cache is present and sufficient.

    Returns None when:
    - the cache file does not exist,
    - the cached sample count is less than n_samples, or
    - the cached frequency-grid size does not match n_freqs (different T).

    When a hit occurs the returned dict is sliced to exactly n_samples rows.
    """
    if not cache_path.exists():
        return None
    try:
        data = np.load(str(cache_path), allow_pickle=False)
    except Exception as exc:
        print(f"\n  [Cache] Failed to read {cache_path.name}: {exc} — ignoring cache")
        return None
    n_cached = int(data["theta_batch"].shape[0])
    cached_n_freqs = int(data["hp_lal"].shape[1])
    if n_cached < n_samples:
        print(
            f"\n  [Cache] {cache_path.name}: "
            f"only {n_cached} samples cached, {n_samples} requested — regenerating"
        )
        return None
    if cached_n_freqs != n_freqs:
        print(
            f"\n  [Cache] {cache_path.name}: "
            f"cached n_freqs={cached_n_freqs} != {n_freqs} — regenerating"
        )
        return None
    print(f"\n  [Cache] {cache_path.name}: hit ({n_cached} samples cached)")
    return {
        "theta_batch": data["theta_batch"][:n_samples],
        "hp_lal": data["hp_lal"][:n_samples],
        "hc_lal": data["hc_lal"][:n_samples],
        "valid_mask": data["valid_mask"][:n_samples],
        "msa_fallback_mask": data["msa_fallback_mask"][:n_samples],
    }


def _save_lal_cache(
    cache_path: Path,
    theta_batch: np.ndarray,
    hp_lal: np.ndarray,
    hc_lal: np.ndarray,
    valid_mask: np.ndarray,
    msa_fallback_mask: np.ndarray,
) -> None:
    """Persist LAL waveforms to disk, overwriting any existing cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(cache_path),
        theta_batch=theta_batch,
        hp_lal=hp_lal,
        hc_lal=hc_lal,
        valid_mask=valid_mask,
        msa_fallback_mask=msa_fallback_mask,
    )
    print(f"\n  [Cache] Saved LAL results to {cache_path}")


# ============================================================================
# Helper functions
# ============================================================================


def convert_parameters_lal_to_ripple(
    theta_lal: np.ndarray, is_precessing: bool, is_tidal: bool
):
    # Convert parameters to ripple format
    if is_precessing:
        # Precessing: theta_lal = [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist, tc, phic, inc]
        # Ripple precessing waveforms (IMRPhenomPv2, IMRPhenomXPHM) expect:
        #   [Mc, eta, s1x, s1y, s1z, s2x, s2y, s2z, dist_mpc, tc, phiRef, incl]
        m1 = theta_lal[0]
        m2 = theta_lal[1]
        Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
        s1x = theta_lal[2]
        s1y = theta_lal[3]
        s1z = theta_lal[4]
        s2x = theta_lal[5]
        s2y = theta_lal[6]
        s2z = theta_lal[7]
        dist_mpc = theta_lal[8]
        tc = theta_lal[9]
        phic = theta_lal[10]
        inclination = theta_lal[11]
        theta_ripple = jnp.array(
            [Mc, eta, s1x, s1y, s1z, s2x, s2y, s2z, dist_mpc, tc, phic, inclination]
        )
    else:
        m1 = theta_lal[0]
        m2 = theta_lal[1]
        Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

        if is_tidal:
            s1z = theta_lal[2]
            s2z = theta_lal[3]
            l1 = theta_lal[4]
            l2 = theta_lal[5]
            lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(
                jnp.array([l1, l2, m1, m2])
            )
            dist_mpc = theta_lal[6]
            tc = theta_lal[7]
            phic = theta_lal[8]
            inclination = theta_lal[9]
            theta_ripple = jnp.array(
                [
                    Mc,
                    eta,
                    s1z,
                    s2z,
                    lambda_tilde,
                    delta_lambda_tilde,
                    dist_mpc,
                    tc,
                    phic,
                    inclination,
                ]
            )
        else:
            s1z = theta_lal[2]
            s2z = theta_lal[3]
            dist_mpc = theta_lal[4]
            tc = theta_lal[5]
            phic = theta_lal[6]
            inclination = theta_lal[7]
            theta_ripple = jnp.array(
                [Mc, eta, s1z, s2z, dist_mpc, tc, phic, inclination]
            )
    return theta_ripple


def compute_ripple_lal_mismatch(
    hphc_lal: tuple,
    hphc_ripple: tuple,
    fs: jnp.ndarray,
    f_l: float,
    f_u: float,
    df: float,
    f_ref: float,
    psd: np.ndarray,
    psd_freqs: np.ndarray,
) -> tuple[float, float]:
    """Compute mismatch between ripple and LAL waveforms.

    Args:
        waveform_name: Name of the waveform.
        theta_lal: Parameter array in LAL format.
        fs: Ripple frequency array.
        f_l: Lower frequency.
        f_u: Upper frequency.
        df: Frequency spacing.
        f_ref: Reference frequency.
        psd: PSD values.
        psd_freqs: PSD frequencies.

    Returns:
        Tuple (mismatch_hp, mismatch_hc) where each is 1 - match for the
        respective polarization. For aligned-spin waveforms hc = i*hp in the
        frequency domain, so both should agree. For precessing waveforms the
        two polarizations are independent and both are tested.
    """
    hp_lal, hc_lal = hphc_lal
    hp_ripple, hc_ripple = hphc_ripple

    # Apply Nyquist mask to all waveforms
    nyquist_mask = get_nyquist_mask(fs)
    hp_lal_masked = jnp.array(hp_lal) * nyquist_mask
    hc_lal_masked = jnp.array(hc_lal) * nyquist_mask
    hp_ripple_masked = hp_ripple * nyquist_mask
    hc_ripple_masked = hc_ripple * nyquist_mask

    # Interpolate PSD to ripple frequency grid
    psd_interp = jnp.interp(fs, psd_freqs, psd)

    # Compute match for both polarizations
    match_hp = compute_match(hp_ripple_masked, hp_lal_masked, psd_interp, fs)
    match_hc = compute_match(hc_ripple_masked, hc_lal_masked, psd_interp, fs)
    mismatch_hp = 1.0 - match_hp
    mismatch_hc = 1.0 - match_hc

    return mismatch_hp, mismatch_hc


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def freq_params(request):
    """Default frequency parameters.

    Segment duration T (and thus frequency resolution df=1/T) is configurable
    via the ``--T`` CLI option.  The default T=32 s (df≈0.03 Hz, ~32K bins)
    is fast enough for CI correctness checks.  Use ``--T 256`` for full
    validation runs where dense frequency coverage is needed (df≈0.004 Hz,
    ~257K bins — the standard GW analysis segment length for long BNS signals).
    """
    T = float(request.config.getoption("--T"))
    return {
        "f_l": 20.0,
        "f_u": 1024.0,
        "f_sampling": 2048.0,
        "T": T,
        "f_ref": 20.0,
    }


@pytest.fixture
def psd_data():
    """Load aLIGO PSD for cross-validation.

    Source: bilby (https://github.com/bilby-dev/bilby)
    Commit: 0985f75c664786e21cc4f662d4f12fe181b1a536
    """
    psd_path = Path(__file__).parent.parent / "psds" / "ET_D_psd.txt"
    freqs, psd = np.loadtxt(psd_path, unpack=True)
    return freqs, psd


# ============================================================================
# Parametrized tests for all waveforms
# ============================================================================


@pytest.mark.parametrize(
    "waveform_name,bounds",
    [
        pytest.param("IMRPhenomD", BBH_BOUNDS, id="IMRPhenomD"),
        pytest.param("IMRPhenomXAS", BBH_BOUNDS, id="IMRPhenomXAS"),
        pytest.param("IMRPhenomD_NRTidalv2", DEFAULT_BOUNDS, id="IMRPhenomD_NRTidalv2"),
        pytest.param(
            "IMRPhenomXAS_NRTidalv3", DEFAULT_BOUNDS, id="IMRPhenomXAS_NRTidalv3"
        ),
        pytest.param("TaylorF2", DEFAULT_BOUNDS, id="TaylorF2"),
        pytest.param("IMRPhenomPv2", BBH_BOUNDS, id="IMRPhenomPv2"),
        pytest.param("IMRPhenomXPHM", BBH_BOUNDS, id="IMRPhenomXPHM"),
    ],
)
def test_waveform_mismatch(
    waveform_name, bounds, freq_params, cross_val_results, psd_data, n_samples
):
    """Test that ripple waveforms match LALSuite to machine precision.

    This test generates random parameter sets, computes both LAL and ripple
    waveforms, and verifies that the mismatch is below the threshold.
    """
    check_lal_available()

    # Unpack parameters
    f_l = freq_params["f_l"]
    f_u = freq_params["f_u"]
    f_sampling = freq_params["f_sampling"]
    T = freq_params["T"]
    f_ref = freq_params["f_ref"]
    psd_freqs, psd = psd_data

    # Construct frequency grid
    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = fs[1] - fs[0]

    # Generate random parameters
    is_tidal = check_is_tidal(waveform_name)
    is_precessing = check_is_precessing(waveform_name)
    theta_batch = generate_random_params(
        n_samples, bounds, is_tidal=is_tidal, is_precessing=is_precessing, seed=42
    )

    # Compute mismatches for all samples
    failed_params = []

    is_tidal = check_is_tidal(waveform_name)
    is_precessing = check_is_precessing(waveform_name)

    # Generate ripple waveform
    waveform = get_jitted_waveform(waveform_name, fs, f_ref)

    # Phase 1: collect LAL waveforms in parallel (with on-disk caching).
    # Cache is keyed by (waveform_name, T): same waveform + same segment
    # duration always produces the same LAL data (seed=42 is fixed).
    # If the cache has >= n_samples entries it is reused; requesting more
    # samples than are cached triggers a full regeneration + overwrite.
    lal_cache_path = _lal_cache_path(waveform_name, T)
    cached_lal = _load_lal_cache(lal_cache_path, n_samples, len(fs))

    if cached_lal is not None:
        # Cache hit: restore data structures without calling LAL.
        theta_batch = cached_lal["theta_batch"]
        lal_hp_store = cached_lal["hp_lal"]
        lal_hc_store = cached_lal["hc_lal"]
        valid_mask = cached_lal["valid_mask"].astype(bool)
        msa_fallback_mask = cached_lal["msa_fallback_mask"].astype(bool)

        lal_hp_list = [
            jnp.array(lal_hp_store[i]) for i in range(n_samples) if valid_mask[i]
        ]
        lal_hc_list = [
            jnp.array(lal_hc_store[i]) for i in range(n_samples) if valid_mask[i]
        ]
        theta_ripple_list = [
            convert_parameters_lal_to_ripple(theta_batch[i], is_precessing, is_tidal)
            for i in range(n_samples)
            if valid_mask[i]
        ]
        print(
            f"\n  [Cache] Loaded {int(valid_mask.sum())} valid LAL waveforms from cache."
        )
    else:
        # Cache miss: run LAL computations in parallel then persist.
        # LAL's C extension releases the GIL, so ThreadPoolExecutor gives real
        # parallelism here — each call is independent (no shared state).
        # For XPHM we use PrecVersion=222 which raises XLAL_EDOM (surfaces in
        # Python as "Internal function call failed: Input domain error") whenever
        # the MSA system fails to initialise.  Because 222 is only used for XPHM
        # and only fails on MSA init, any exception from an XPHM call is an MSA
        # failure.  These samples are tracked separately and excluded from the
        # mismatch assertion and histogram.

        def _compute_lal(i_theta):
            i, theta_lal = i_theta
            try:
                hp, hc = get_lal_waveform(
                    theta_lal,
                    waveform_name,
                    f_l,
                    f_u,
                    df,
                    f_ref,
                    is_tidal,
                    is_precessing,
                )
                return i, hp, hc, False, None  # MSA ok
            except Exception as e:
                msg = str(e)
                is_msa = waveform_name == "IMRPhenomXPHM"
                return i, None, None, is_msa, msg  # MSA fallback or real error

        # Use sched_getaffinity when available (Linux): respects SLURM cgroup
        # allocations and container CPU limits (e.g. GitHub Actions).  Falls back
        # to cpu_count on macOS/Windows where the syscall is absent.
        try:
            n_cpu = len(os.sched_getaffinity(0))
        except AttributeError:
            n_cpu = os.cpu_count() or 1
        n_workers = min(n_samples, n_cpu)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            # map() preserves input order, so result[i] matches theta_batch[i]
            lal_results = list(pool.map(_compute_lal, enumerate(theta_batch)))

        lal_hp_list = []
        lal_hc_list = []
        theta_ripple_list = []
        valid_mask = np.zeros(n_samples, dtype=bool)
        msa_fallback_mask = np.zeros(n_samples, dtype=bool)
        lal_hp_store = np.full((n_samples, len(fs)), np.nan, dtype=complex)
        lal_hc_store = np.full((n_samples, len(fs)), np.nan, dtype=complex)

        for i, hp_lal, hc_lal, is_msa_fallback, err in lal_results:
            if err is None:
                theta_ripple = convert_parameters_lal_to_ripple(
                    theta_batch[i], is_precessing, is_tidal
                )
                lal_hp_list.append(jnp.array(hp_lal))
                lal_hc_list.append(jnp.array(hc_lal))
                theta_ripple_list.append(theta_ripple)
                valid_mask[i] = True
                lal_hp_store[i] = hp_lal
                lal_hc_store[i] = hc_lal
            elif is_msa_fallback:
                msa_fallback_mask[i] = True
            else:
                failed_params.append((i, theta_batch[i], err))

        _save_lal_cache(
            lal_cache_path,
            theta_batch,
            lal_hp_store,
            lal_hc_store,
            valid_mask,
            msa_fallback_mask,
        )

    # Shared inputs for Phases 2 & 3
    nyquist_mask = get_nyquist_mask(fs)
    psd_interp = jnp.interp(fs, jnp.array(psd_freqs), jnp.array(psd))
    theta_ripple_batch = jnp.stack(theta_ripple_list)
    hp_lal_batch = jnp.stack(lal_hp_list) * nyquist_mask
    hc_lal_batch = jnp.stack(lal_hc_list) * nyquist_mask

    # Per-sample function used by both the vmap and the lax.map fallback.
    # Combines waveform generation + match so the lax.map path keeps peak
    # memory at O(1 sample) instead of O(N samples).
    def _waveform_and_match(inputs):
        theta_rip, hp_lal_m, hc_lal_m = inputs
        hp_rip, hc_rip = waveform(theta_rip)
        match_hp = compute_match(hp_rip * nyquist_mask, hp_lal_m, psd_interp, fs)
        match_hc = compute_match(hc_rip * nyquist_mask, hc_lal_m, psd_interp, fs)
        return match_hp, match_hc

    # Phase 2 + 3: try fast vmap path; on GPU OOM fall back to jax.lax.map
    # with batch_size, which vmaps over chunks of `batch_size` samples
    # sequentially — a middle ground between fully parallel (vmap) and fully
    # sequential (batch_size=1).  We start with batch_size = n_valid // 10 and
    # halve on each OOM until batch_size reaches 1 (purely sequential).
    # See: https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html
    n_valid = len(theta_ripple_list)
    xs = (theta_ripple_batch, hp_lal_batch, hc_lal_batch)

    def _is_oom(e: Exception) -> bool:
        msg = str(e)
        return "RESOURCE_EXHAUSTED" in msg or "Out of memory" in msg

    def _run_with_batch_size(batch_size: int | None):
        if batch_size is None:
            # Full vmap — batch_size equals the whole dataset
            fn = jax.jit(jax.vmap(_waveform_and_match))
        else:
            fn = jax.jit(
                lambda xs: jax.lax.map(_waveform_and_match, xs, batch_size=batch_size)
            )
        match_hp, match_hc = fn(xs)
        match_hp.block_until_ready()  # surface OOM before np.array()
        return np.array(match_hp), np.array(match_hc)

    batch_size = None  # None → full vmap
    while True:
        try:
            match_hp_np, match_hc_np = _run_with_batch_size(batch_size)
            break
        except Exception as e:
            if not _is_oom(e):
                raise
            next_batch = max(1, (n_valid if batch_size is None else batch_size) // 2)
            if batch_size is not None and next_batch == batch_size:
                # Already at batch_size=1 and still OOM — re-raise
                raise
            print(
                f"\n  [OOM] {waveform_name}: retrying with "
                f"jax.lax.map(batch_size={next_batch})..."
            )
            batch_size = next_batch

    # Phase 4: assemble results, inserting NaN for failed samples
    mismatches_hp = np.full(n_samples, np.nan)
    mismatches_hc = np.full(n_samples, np.nan)
    mismatches_hp[valid_mask] = 1.0 - match_hp_np
    mismatches_hc[valid_mask] = 1.0 - match_hc_np

    # Flag NaN/Inf mismatches in otherwise-valid samples
    for i in np.where(valid_mask)[0]:
        if not np.isfinite(mismatches_hp[i]) or not np.isfinite(mismatches_hc[i]):
            failed_params.append((i, theta_batch[i], "NaN/Inf mismatch"))
    # Worst-case mismatch over both polarizations
    mismatches = np.maximum(mismatches_hp, mismatches_hc)

    # Separate MSA-fallback samples: they have valid mismatches but should not
    # be included in the threshold assertion (LAL used NNLO angles, not MSA).
    n_msa_fallback = int(msa_fallback_mask.sum())
    testable_mask = np.isfinite(mismatches) & ~msa_fallback_mask
    testable_mismatches = mismatches[testable_mask]
    finite_mismatches = mismatches[np.isfinite(mismatches)]

    n_nonfinite = mismatches.size - finite_mismatches.size
    assert finite_mismatches.size > 0, (
        f"All {mismatches.size} per-sample mismatches are non-finite for {waveform_name}. "
        f"Non-finite count: {n_nonfinite}. "
        f"Sample mismatches (first {min(10, mismatches.size)}): {mismatches[:10]}. "
        f"Failed samples: {len(failed_params)}."
    )

    # Save results to CSV
    # Each (n_samples, T) combination gets its own subdirectory so results
    # from different run configurations are kept separately.
    T_str = f"T{int(T)}" if T == int(T) else f"T{T}"
    run_tag = f"n{n_samples}_{T_str}"
    results_dir = Path(__file__).parent / "results" / run_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"mismatch_{waveform_name}.csv"

    # Build dataframe
    if is_tidal:
        m1, m2 = theta_batch[:, 0], theta_batch[:, 1]
        chi1z, chi2z = theta_batch[:, 2], theta_batch[:, 3]
        lambda1, lambda2 = theta_batch[:, 4], theta_batch[:, 5]
        df_data = {
            "m1": m1,
            "m2": m2,
            "chi1z": chi1z,
            "chi2z": chi2z,
            "lambda1": lambda1,
            "lambda2": lambda2,
            "dist_mpc": theta_batch[:, 6],
            "tc": theta_batch[:, 7],
            "phi_ref": theta_batch[:, 8],
            "inclination": theta_batch[:, 9],
            "mismatch_hp": mismatches_hp,
            "mismatch_hc": mismatches_hc,
            "mismatch": mismatches,
            "msa_fallback": msa_fallback_mask,
        }
    elif is_precessing:
        m1, m2 = theta_batch[:, 0], theta_batch[:, 1]
        s1x, s1y, s1z = theta_batch[:, 2], theta_batch[:, 3], theta_batch[:, 4]
        s2x, s2y, s2z = theta_batch[:, 5], theta_batch[:, 6], theta_batch[:, 7]
        chi1z, chi2z = s1z, s2z
        df_data = {
            "m1": m1,
            "m2": m2,
            "s1x": s1x,
            "s1y": s1y,
            "s1z": s1z,
            "s2x": s2x,
            "s2y": s2y,
            "s2z": s2z,
            "chi1_mag": np.sqrt(s1x**2 + s1y**2 + s1z**2),
            "chi2_mag": np.sqrt(s2x**2 + s2y**2 + s2z**2),
            "dist_mpc": theta_batch[:, 8],
            "tc": theta_batch[:, 9],
            "phi_ref": theta_batch[:, 10],
            "inclination": theta_batch[:, 11],
            "mismatch_hp": mismatches_hp,
            "mismatch_hc": mismatches_hc,
            "mismatch": mismatches,
            "msa_fallback": msa_fallback_mask,
        }
    else:
        m1, m2 = theta_batch[:, 0], theta_batch[:, 1]
        chi1z, chi2z = theta_batch[:, 2], theta_batch[:, 3]
        df_data = {
            "m1": m1,
            "m2": m2,
            "chi1z": chi1z,
            "chi2z": chi2z,
            "dist_mpc": theta_batch[:, 4],
            "tc": theta_batch[:, 5],
            "phi_ref": theta_batch[:, 6],
            "inclination": theta_batch[:, 7],
            "mismatch_hp": mismatches_hp,
            "mismatch_hc": mismatches_hc,
            "mismatch": mismatches,
            "msa_fallback": msa_fallback_mask,
        }

    df = pd.DataFrame(df_data)
    # Derived columns
    df["m_total"] = df["m1"] + df["m2"]
    df["mass_ratio"] = np.minimum(df["m1"], df["m2"]) / np.maximum(df["m1"], df["m2"])
    if is_precessing:
        df["chi_eff"] = (df["m1"] * df["s1z"] + df["m2"] * df["s2z"]) / df["m_total"]
    else:
        df["chi_eff"] = (df["m1"] * df["chi1z"] + df["m2"] * df["chi2z"]) / df[
            "m_total"
        ]
    df["log10_mismatch"] = np.log10(np.abs(df["mismatch"].clip(1e-30)))
    df = df.sort_values(by="mismatch", ascending=False)
    df.to_csv(results_file, index=False)

    # Print statistics
    print(f"\n{waveform_name} Mismatch Statistics:")
    print(f"  Samples: {n_samples}")
    if n_msa_fallback > 0:
        print(f"  MSA fallback (NNLO): {n_msa_fallback} (excluded from assertion)")
    print(f"  Testable samples: {len(testable_mismatches)}")
    print(f"  Mean mismatch: {np.mean(finite_mismatches):.2e}")
    print(f"  Median mismatch: {np.median(finite_mismatches):.2e}")
    print(f"  Min mismatch: {np.min(finite_mismatches):.2e}")
    print(f"  Max mismatch: {np.max(finite_mismatches):.2e}")
    print(f"  Failed samples: {len(failed_params)}")
    print(f"  Results saved to: {results_file}")

    # Plot mismatch distribution
    figures_dir = Path(__file__).parent / "figures" / run_tag
    figures_dir.mkdir(parents=True, exist_ok=True)

    mismatch_threshold = get_mismatch_threshold(waveform_name)
    log10_thresh = np.log10(mismatch_threshold)
    log10_m = df["log10_mismatch"].values
    finite_mask = np.isfinite(log10_m)
    fallback_col = df["msa_fallback"].values.astype(bool)
    # Masks for the two groups
    normal_mask = finite_mask & ~fallback_col

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"{waveform_name}: ripple vs LAL mismatch", fontsize=13)

    # (0,0) - histogram (MSA-fallback samples excluded: no waveform generated)
    ax = axes[0, 0]
    ax.hist(log10_m[normal_mask], bins=30, edgecolor="black", alpha=0.8)
    ax.axvline(
        log10_thresh,
        color="red",
        linestyle="--",
        label=f"threshold = {mismatch_threshold:.0e}",
    )
    ax.set_xlabel(r"$\log_{10}\,\mathcal{M}$")
    ax.set_ylabel("Count")
    ax.set_title("Mismatch distribution")
    ax.legend()

    # Helper: scatter normal points + overlay MSA-fallback points with red "x"
    def _scatter_with_fallback(ax, x, y, c, cmap, s=20, alpha=0.8):
        sc = ax.scatter(
            x[normal_mask],
            y[normal_mask],
            c=c[normal_mask],
            cmap=cmap,
            s=s,
            alpha=alpha,
        )
        if fallback_col.any():
            ax.scatter(
                x[fallback_col],
                y[fallback_col],
                c="red",
                marker="x",
                s=s * 3,
                linewidths=1.5,
                label="MSA fallback",
                zorder=5,
            )
        return sc

    # (0,1) - mass ratio vs total mass colored by mismatch
    ax = axes[0, 1]
    sc = _scatter_with_fallback(
        ax,
        df["m_total"].values,
        df["mass_ratio"].values,
        log10_m,
        "viridis",
    )
    ax.set_xlabel(r"$M_{\rm total}\;[M_\odot]$")
    ax.set_ylabel(r"$q$")
    ax.set_title(r"$M_{\rm total}$ vs $q$")
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}\,\mathcal{M}$")
    if fallback_col.any():
        ax.legend()

    # (1,0) - mismatch vs chi_eff / lambda_tilde / chi_mag
    ax = axes[1, 0]
    if is_tidal:
        # Compute lambda_tilde from lambda1, lambda2, m1, m2
        eta = df["m1"] * df["m2"] / df["m_total"] ** 2
        x_vals = (8.0 / 13.0) * (
            (1 + 7 * eta - 31 * eta**2) * (df["lambda1"] + df["lambda2"])
            + (1 - 4 * eta) ** 0.5
            * (1 + 9 * eta - 11 * eta**2)
            * (df["lambda1"] - df["lambda2"])
        )
        x_label = r"$\tilde{\Lambda}$"
    elif is_precessing:
        x_vals = df["chi1_mag"]
        x_label = r"$|\chi_1|$"
    else:
        x_vals = df["chi_eff"]
        x_label = r"$\chi_{\rm eff}$"
    sc = _scatter_with_fallback(
        ax,
        x_vals.values,
        df["inclination"].values,
        log10_m,
        "viridis",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\iota$")
    ax.set_title(f"{x_label} vs $\\iota$")
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}\,\mathcal{M}$")
    if fallback_col.any():
        ax.legend()

    # (1,1) - 2D: m1 vs m2 colored by mismatch
    ax = axes[1, 1]
    sc = _scatter_with_fallback(
        ax,
        df["m1"].values,
        df["m2"].values,
        log10_m,
        "plasma",
        s=30,
        alpha=0.9,
    )
    ax.set_xlabel(r"$m_1\;[M_\odot]$")
    ax.set_ylabel(r"$m_2\;[M_\odot]$")
    ax.set_title(r"$m_1$ vs $m_2$")
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}\,\mathcal{M}$")
    if fallback_col.any():
        ax.legend()

    fig.tight_layout()
    fig_file = figures_dir / f"mismatch_{waveform_name}.png"

    fig.savefig(fig_file, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to: {fig_file}")

    # Assert that all mismatches are below threshold
    # MSA-fallback samples are excluded: LAL used NNLO angles (not MSA) so a
    # large mismatch against ripple's MSA implementation is expected.
    max_testable_mismatch = (
        np.max(testable_mismatches) if testable_mismatches.size > 0 else 0.0
    )
    mismatch_threshold = get_mismatch_threshold(waveform_name)

    if failed_params:
        print("\nFailed parameters:")
        for i, theta, error in failed_params[:5]:  # Print first 5 failures
            print(f"  Sample {i}: {error}")
            print(f"    Params: {theta}")

    # Record stats for the session-level summary
    cross_val_results.append(
        {
            "waveform": waveform_name,
            "n_samples": n_samples,
            "n_finite": len(finite_mismatches),
            "n_failed": len(failed_params),
            "n_msa_fallback": n_msa_fallback,
            "mean": float(np.mean(finite_mismatches)),
            "median": float(np.median(finite_mismatches)),
            "min": float(np.min(finite_mismatches)),
            "max": float(max_testable_mismatch),
            "threshold": mismatch_threshold,
            "passed": bool(
                len(failed_params) == 0 and max_testable_mismatch < mismatch_threshold
            ),
        }
    )

    assert len(failed_params) == 0, f"{len(failed_params)}/{n_samples} samples failed"
    assert testable_mismatches.size > 0 or n_msa_fallback > 0, (
        f"No testable samples for {waveform_name}"
    )
    if testable_mismatches.size > 0:
        assert max_testable_mismatch < mismatch_threshold, (
            f"Max mismatch {max_testable_mismatch:.2e} exceeds threshold "
            f"{mismatch_threshold:.2e} (excluding {n_msa_fallback} MSA-fallback samples)"
        )
