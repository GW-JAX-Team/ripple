#!/usr/bin/env python3
"""Runtime benchmarking script for ripple waveforms.

This script benchmarks the runtime performance of ripple's waveform models
and optionally compares them against LALSuite. It records hardware information
and saves results to a JSON file for tracking performance over time.

Usage:
    python tests/benchmarks/benchmark_runtime.py
    python tests/benchmarks/benchmark_runtime.py --waveforms IMRPhenomD IMRPhenomXAS
    python tests/benchmarks/benchmark_runtime.py --with-lal --n-params 10000
    python tests/benchmarks/benchmark_runtime.py --output custom_results.json
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add parent directory to path to import from tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from tests.utils import (
    check_is_tidal,
    check_is_precessing,
    get_freqs,
    get_jitted_waveform,
    generate_random_params,
    LAL_AVAILABLE,
)

if LAL_AVAILABLE:
    import lal
    import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)


# ============================================================================
# Hardware information collection
# ============================================================================


def get_hardware_info():
    """Collect hardware and software version information.

    Returns:
        Dictionary with hardware and software information.
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }

    # JAX information
    info["jax_version"] = jax.__version__
    devices = jax.devices()
    info["jax_devices"] = [
        {"platform": str(d.platform), "device_kind": str(d.device_kind)} for d in devices
    ]
    info["jax_default_backend"] = jax.default_backend()

    # Ripple version (from package)
    try:
        import ripplegw

        info["ripple_version"] = getattr(ripplegw, "__version__", "unknown")
    except Exception:
        info["ripple_version"] = "unknown"

    # LAL version (if available)
    if LAL_AVAILABLE:
        try:
            info["lal_version"] = getattr(lal.version, "verbose", getattr(lal, "__version__", "unknown"))
        except Exception:
            info["lal_version"] = "unknown"
        try:
            info["lalsimulation_version"] = lalsim.__version__
        except Exception:
            info["lalsimulation_version"] = "unknown"
    else:
        info["lal_version"] = None
        info["lalsimulation_version"] = None

    return info


# ============================================================================
# Waveform benchmarking
# ============================================================================

# Default parameter bounds for each waveform type
BBH_BOUNDS = {
    "m": [5.0, 50.0],
    "chi": [-0.99, 0.99],
    "lambda": [0.0, 0.0],
    "d_L": [100.0, 1000.0],
}

BNS_BOUNDS = {
    "m": [0.5, 3.0],
    "chi": [0.0, 0.05],
    "lambda": [0.0, 5000.0],
    "d_L": [30.0, 300.0],
}


def benchmark_waveform_ripple(
    waveform_name: str,
    n_params: int = 1000,
    n_repeats: int = 10,
    n_warmup: int = 3,
):
    """Benchmark a ripple waveform.

    Args:
        waveform_name: Name of the waveform.
        n_params: Number of parameter sets to generate.
        n_repeats: Number of timing repeats.
        n_warmup: Number of warmup iterations.

    Returns:
        Dictionary with benchmark results.
    """
    # Frequency grid
    f_l = 20.0
    f_u = 1024.0
    f_sampling = 2048.0
    T = 16.0
    f_ref = f_l
    fs = get_freqs(f_l, f_u, f_sampling, T)

    # Determine parameter bounds
    is_tidal = check_is_tidal(waveform_name)
    is_precessing = check_is_precessing(waveform_name)

    if is_precessing:
        bounds = BBH_BOUNDS
    elif is_tidal:
        bounds = BNS_BOUNDS
    else:
        bounds = BBH_BOUNDS

    # Generate random parameters (LAL format)
    theta_batch_lal = generate_random_params(
        n_params, bounds, is_tidal=is_tidal, is_precessing=is_precessing, seed=42
    )

    # Convert to ripple format
    theta_batch_ripple = []
    for theta_lal in theta_batch_lal:
        m1 = theta_lal[0]
        m2 = theta_lal[1]
        Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

        if is_precessing:
            # theta_lal = [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist, tc, phic, inc]
            # Ripple IMRPhenomPv2 expects [Mc, eta, s1x, s1y, s1z, s2x, s2y, s2z, dist, tc, phic, inc]
            s1x, s1y, s1z = theta_lal[2], theta_lal[3], theta_lal[4]
            s2x, s2y, s2z = theta_lal[5], theta_lal[6], theta_lal[7]
            dist_mpc = theta_lal[8]
            tc = theta_lal[9]
            phic = theta_lal[10]
            inclination = theta_lal[11]
            theta_ripple = jnp.array(
                [Mc, eta, s1x, s1y, s1z, s2x, s2y, s2z, dist_mpc, tc, phic, inclination]
            )
            theta_batch_ripple.append(theta_ripple)
        else:
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
                theta_ripple = jnp.array([Mc, eta, s1z, s2z, dist_mpc, tc, phic, inclination])

            theta_batch_ripple.append(theta_ripple)

    theta_batch_ripple = jnp.array(theta_batch_ripple)  # type: ignore[assignment]

    # Get waveform function
    waveform = get_jitted_waveform(waveform_name, fs, f_ref)

    # Warmup: JIT compile and run a few times
    print(f"  Warming up {waveform_name}...")
    for _ in range(n_warmup):
        hp, hc = waveform(theta_batch_ripple[0])
        hp.block_until_ready()

    # Benchmark single calls
    print("  Benchmarking single calls...")
    single_times = []
    for i in range(n_repeats):
        start = time.time()
        for theta in theta_batch_ripple:
            hp, hc = waveform(theta)
            hp.block_until_ready()
        end = time.time()
        single_times.append((end - start) / n_params * 1000)  # ms per call

    # Benchmark vmapped calls
    print("  Benchmarking vmapped calls...")
    waveform_vmapped = jax.jit(jax.vmap(waveform))

    # Warmup vmap
    for _ in range(n_warmup):
        hp_batch, hc_batch = waveform_vmapped(theta_batch_ripple)
        hp_batch.block_until_ready()

    vmap_times = []
    for i in range(n_repeats):
        start = time.time()
        hp_batch, hc_batch = waveform_vmapped(theta_batch_ripple)
        hp_batch.block_until_ready()
        end = time.time()
        vmap_times.append((end - start) / n_params * 1000)  # ms per call

    return {
        "waveform": waveform_name,
        "n_params": n_params,
        "n_repeats": n_repeats,
        "single_call_ms_mean": float(np.mean(single_times)),
        "single_call_ms_std": float(np.std(single_times)),
        "vmapped_call_ms_mean": float(np.mean(vmap_times)),
        "vmapped_call_ms_std": float(np.std(vmap_times)),
        "speedup_vmap": float(np.mean(single_times) / np.mean(vmap_times)),
    }


def benchmark_waveform_lal(
    waveform_name: str,
    n_params: int = 1000,
):
    """Benchmark a LALSuite waveform.

    Args:
        waveform_name: Name of the waveform.
        n_params: Number of parameter sets to generate.

    Returns:
        Dictionary with LAL benchmark results.
    """
    if not LAL_AVAILABLE:
        return None

    # Frequency grid
    f_l = 20.0
    f_u = 1024.0
    f_sampling = 2048.0
    T = 16.0
    f_ref = f_l
    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = float(fs[1] - fs[0])

    # Determine parameter bounds
    is_tidal = check_is_tidal(waveform_name)
    is_precessing = check_is_precessing(waveform_name)

    if is_precessing:
        bounds = BBH_BOUNDS
    elif is_tidal:
        bounds = BNS_BOUNDS
    else:
        bounds = BBH_BOUNDS

    # Generate random parameters
    theta_batch = generate_random_params(
        n_params, bounds, is_tidal=is_tidal, is_precessing=is_precessing, seed=42
    )

    # LAL waveform generation
    approximant = lalsim.SimInspiralGetApproximantFromString(waveform_name)

    def generate_lal_waveform(theta):
        """Generate a single LAL waveform."""
        m1_kg = theta[0] * lal.MSUN_SI
        m2_kg = theta[1] * lal.MSUN_SI

        if is_precessing:
            # theta = [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist, tc, phic, inc]
            s1x, s1y, s1z = theta[2], theta[3], theta[4]
            s2x, s2y, s2z = theta[5], theta[6], theta[7]
            distance = theta[8] * 1e6 * lal.PC_SI
            phi_ref = theta[10]
            inclination = theta[11]

            lalsim.SimInspiralChooseFDWaveform(
                m1_kg,
                m2_kg,
                s1x,
                s1y,
                s1z,
                s2x,
                s2y,
                s2z,
                distance,
                inclination,
                phi_ref,
                0,
                0,
                0,
                df,
                f_l,
                f_u,
                f_ref,
                None,
                approximant,
            )
        else:
            if is_tidal:
                s1z = theta[2]
                s2z = theta[3]
                l1 = theta[4]
                l2 = theta[5]
                distance = theta[6] * 1e6 * lal.PC_SI
                phi_ref = theta[8]
                inclination = theta[9]

                laldict = lal.CreateDict()
                lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, l1)
                lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, l2)
                quad1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l1)
                quad2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l2)
                lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, quad1 - 1)
                lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, quad2 - 1)
            else:
                s1z = theta[2]
                s2z = theta[3]
                distance = theta[4] * 1e6 * lal.PC_SI
                phi_ref = theta[6]
                inclination = theta[7]
                laldict = None

            hp, _ = lalsim.SimInspiralChooseFDWaveform(
                m1_kg,
                m2_kg,
                0.0,
                0.0,
                s1z,
                0.0,
                0.0,
                s2z,
                distance,
                inclination,
                phi_ref,
                0,
                0,
                0,
                df,
                f_l,
                f_u,
                f_ref,
                laldict,
                approximant,
            )

    print(f"  Benchmarking LAL {waveform_name}...")
    start = time.time()
    for theta in theta_batch:
        generate_lal_waveform(theta)
    end = time.time()

    lal_time_ms = (end - start) / n_params * 1000

    return {
        "waveform": waveform_name,
        "lal_call_ms": float(lal_time_ms),
    }


# ============================================================================
# Main benchmarking routine
# ============================================================================

ALL_WAVEFORMS = [
    "IMRPhenomD",
    "IMRPhenomXAS",
    "IMRPhenomD_NRTidalv2",
    "IMRPhenomXAS_NRTidalv3",
    "TaylorF2",
    "IMRPhenomPv2",
]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ripple waveform runtime performance"
    )
    parser.add_argument(
        "--waveforms",
        nargs="+",
        choices=ALL_WAVEFORMS,
        default=ALL_WAVEFORMS,
        help="Waveforms to benchmark (default: all)",
    )
    parser.add_argument(
        "--n-params",
        type=int,
        default=1000,
        help="Number of parameter sets to generate (default: 1000)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of timing repeats (default: 10)",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--with-lal",
        action="store_true",
        help="Also benchmark LALSuite for comparison",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: tests/benchmarks/results/benchmark_results.json)",
    )

    args = parser.parse_args()

    # Default output location
    if args.output is None:
        output_file = Path(__file__).parent / "results" / "benchmark_results.json"
    else:
        output_file = Path(args.output)

    print("=" * 80)
    print("Ripple Waveform Benchmark")
    print("=" * 80)

    # Collect hardware info
    print("\nCollecting hardware information...")
    hardware_info = get_hardware_info()
    print(f"  Platform: {hardware_info['platform']}")
    print(f"  Processor: {hardware_info['processor']}")
    print(f"  JAX backend: {hardware_info['jax_default_backend']}")
    print(f"  JAX devices: {hardware_info['jax_devices']}")
    print(f"  Ripple version: {hardware_info['ripple_version']}")
    if args.with_lal:
        print(f"  LAL version: {hardware_info['lal_version']}")

    # Benchmark waveforms
    results = []

    for waveform_name in args.waveforms:
        print(f"\n{'=' * 80}")
        print(f"Benchmarking {waveform_name}")
        print(f"{'=' * 80}")

        # Ripple benchmark
        result = None
        try:
            result = benchmark_waveform_ripple(
                waveform_name,
                n_params=args.n_params,
                n_repeats=args.n_repeats,
                n_warmup=args.n_warmup,
            )
            results.append(result)

            print("\nRipple Results:")
            print(f"  Single call: {result['single_call_ms_mean']:.4f} ± {result['single_call_ms_std']:.4f} ms")
            print(f"  Vmapped call: {result['vmapped_call_ms_mean']:.4f} ± {result['vmapped_call_ms_std']:.4f} ms")
            print(f"  Speedup (vmap): {result['speedup_vmap']:.2f}x")

        except Exception as e:
            print(f"ERROR benchmarking {waveform_name}: {e}")
            results.append({"waveform": waveform_name, "error": str(e)})

        # LAL benchmark
        if args.with_lal:
            try:
                lal_result = benchmark_waveform_lal(waveform_name, n_params=args.n_params)
                if lal_result:
                    # Merge LAL results into the ripple results
                    ripple_entry = None
                    for r in results:
                        if r.get("waveform") == waveform_name:
                            r.update(lal_result)
                            ripple_entry = r
                            break

                    print("\nLAL Results:")
                    print(f"  Single call: {lal_result['lal_call_ms']:.4f} ms")
                    if ripple_entry is not None and "single_call_ms_mean" in ripple_entry:
                        speedup_lal = lal_result["lal_call_ms"] / ripple_entry["single_call_ms_mean"]
                        print(f"  Speedup (ripple single vs LAL): {speedup_lal:.2f}x")

            except Exception as e:
                print(f"ERROR benchmarking LAL {waveform_name}: {e}")

    # Save results
    print(f"\n{'=' * 80}")
    print("Saving results")
    print(f"{'=' * 80}")

    output_data = {
        "hardware": hardware_info,
        "benchmark_config": {
            "n_params": args.n_params,
            "n_repeats": args.n_repeats,
            "n_warmup": args.n_warmup,
            "with_lal": args.with_lal,
        },
        "results": results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")

    print(f"\n{'=' * 80}")
    print("Benchmark complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
