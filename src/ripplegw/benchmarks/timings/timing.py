"""
Command-line interface for timing gravitational waveform generation in ripple.

This script provides a flexible CLI for benchmarking different waveform approximants
with various configurations including hardware selection and precision.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from ripplegw.benchmarks.utils import (
    generate_bbh_parameters,
    generate_bns_parameters,
    get_device_name,
    get_git_hash,
)


def setup_jax_config(use_float64, device):
    """Configure JAX settings for precision and device."""
    jax.config.update("jax_enable_x64", use_float64)

    if device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    print(f"\n{'=' * 60}")
    print("JAX Configuration")
    print(f"{'=' * 60}")
    print(f"Precision: {'float64' if use_float64 else 'float32'}")
    print(f"Requested device: {device}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    for d in jax.devices():
        print(f"  Device: {d.device_kind}, Platform: {d.platform}")
    print(f"{'=' * 60}\n")

    return get_device_name()


def time_imrphenomxphm(params, config):
    """
    Time IMRPhenomXPHM waveform generation.
    # TODO: need to see why the way it is called is different from other waveform models, and whether we want to keep this or not
    """
    from ripplegw.waveforms import IMRPhenomXPHM

    # Build frequency array from config
    f = jnp.arange(
        config["minimum_frequency"],
        config["maximum_frequency"],
        1.0 / config["duration"],
    )

    # Stack parameters into a single array for vmap
    params_stacked = jnp.stack(
        [
            params["mass_1"],
            params["mass_2"],
            params["spin_1x"],
            params["spin_1y"],
            params["spin_1z"],
            params["spin_2x"],
            params["spin_2y"],
            params["spin_2z"],
            params["luminosity_distance"],
            params["theta_jn"],
            params["phase"],
        ],
        axis=1,
    )

    # Create JIT-compiled vmapped version
    @jax.jit
    def generate_xphm_batched(xs):
        return jax.vmap(
            lambda p: IMRPhenomXPHM.generate_xphm(
                p[0],
                p[1],
                p[2],
                p[3],
                p[4],
                p[5],
                p[6],
                p[7],
                p[8],
                p[9],
                p[10],
                f,
                config["reference_frequency"],
            )
        )(xs)

    n_runs = config.get("n_runs", 5)

    # First run (includes JIT compilation)
    print(f"\n{'=' * 60}")
    print("First run (includes JIT compilation)")
    print(f"{'=' * 60}")
    start = time.time()
    hp, hc = generate_xphm_batched(params_stacked)
    hp.block_until_ready()
    hc.block_until_ready()
    first_run_time = time.time() - start
    print(f"First run time (includes JIT compilation): {first_run_time:.3f} s")

    # Timed runs
    print(f"\n{'=' * 60}")
    print(f"Timed runs ({n_runs} repetitions)")
    print(f"{'=' * 60}")
    exec_times = []
    for i in range(n_runs):
        start = time.time()
        hp, hc = generate_xphm_batched(params_stacked)
        hp.block_until_ready()
        hc.block_until_ready()
        t = time.time() - start
        exec_times.append(t)
        print(f"  Run {i + 1}: {t:.6f} s")

    return first_run_time, exec_times


def _prepare_aligned_params(params):
    """Build a batched param dict for aligned-spin BBH waveforms (IMRPhenomXAS, IMRPhenomD)."""
    from ripplegw.conversions import ms_to_Mc_eta

    Mc, eta = ms_to_Mc_eta(jnp.array([params["mass_1"], params["mass_2"]]))
    return {
        "M_c": Mc,
        "eta": eta,
        "s1_z": params["a_1"],
        "s2_z": params["a_2"],
        "d_L": params["luminosity_distance"],
        "phase_c": params["phase"],
        "iota": params["theta_jn"],
    }


def _prepare_precessing_params(params):
    """Build a batched param dict for precessing BBH waveforms (IMRPhenomPv2)."""
    from ripplegw.conversions import ms_to_Mc_eta

    Mc, eta = ms_to_Mc_eta(jnp.array([params["mass_1"], params["mass_2"]]))
    return {
        "M_c": Mc,
        "eta": eta,
        "s1_x": params["spin_1x"],
        "s1_y": params["spin_1y"],
        "s1_z": params["spin_1z"],
        "s2_x": params["spin_2x"],
        "s2_y": params["spin_2y"],
        "s2_z": params["spin_2z"],
        "d_L": params["luminosity_distance"],
        "phase_c": params["phase"],
        "iota": params["theta_jn"],
    }


def _prepare_bns_params(params):
    """Build a batched param dict for BNS waveforms (TaylorF2_BNS, IMRPhenomD_NRTidalv2, IMRPhenomXAS_NRTidalv3)."""
    from ripplegw.conversions import ms_to_Mc_eta

    Mc, eta = ms_to_Mc_eta(jnp.array([params["mass_1"], params["mass_2"]]))
    return {
        "M_c": Mc,
        "eta": eta,
        "s1_z": params["a_1"],
        "s2_z": params["a_2"],
        "lambda_1": params["lambda_1"],
        "lambda_2": params["lambda_2"],
        "d_L": params["luminosity_distance"],
        "phase_c": params["phase"],
        "iota": params["theta_jn"],
    }


def time_waveform(waveform, batched_params, config):
    """Time waveform generation using the class-based waveform interface.

    Args:
        waveform: An instantiated waveform object from ``ripplegw.waveform_preset``.
        batched_params: Dict of JAX arrays, each of shape ``(n_waveforms,)``.
        config: Benchmark configuration dictionary.
    """
    f = jnp.arange(
        config["minimum_frequency"],
        config["maximum_frequency"],
        1.0 / config["duration"],
    )

    waveform_batched = jax.jit(jax.vmap(lambda p: waveform(f, p)))
    n_runs = config.get("n_runs", 5)

    # First run (includes JIT compilation)
    print(f"\n{'=' * 60}")
    print("First run (includes JIT compilation)")
    print(f"{'=' * 60}")
    start = time.time()
    result = waveform_batched(batched_params)
    result["p"].block_until_ready()
    result["c"].block_until_ready()
    first_run_time = time.time() - start
    print(f"First run time (includes JIT compilation): {first_run_time:.3f} s")

    # Timed runs
    print(f"\n{'=' * 60}")
    print(f"Timed runs ({n_runs} repetitions)")
    print(f"{'=' * 60}")
    exec_times = []
    for i in range(n_runs):
        start = time.time()
        result = waveform_batched(batched_params)
        result["p"].block_until_ready()
        result["c"].block_until_ready()
        t = time.time() - start
        exec_times.append(t)
        print(f"  Run {i + 1}: {t:.6f} s")

    return first_run_time, exec_times


def run_timing(args):
    """Main timing function that orchestrates the benchmark."""
    # Setup configuration
    config = {
        "waveform": args.waveform,
        "device": args.device,
        "n_waveforms": args.n_waveforms,
        "n_runs": args.n_runs,
        "precision": args.precision,
        "duration": args.duration,
        "minimum_frequency": args.f_min,
        "maximum_frequency": args.f_max,
        "reference_frequency": args.f_ref,
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
    }

    # Setup JAX and get actual device name
    use_float64 = args.precision == "float64"
    device_name = setup_jax_config(use_float64, args.device)
    config["device_name"] = device_name

    # Print configuration
    print(f"{'=' * 60}")
    print("Timing Configuration")
    print(f"{'=' * 60}")
    print(f"Waveform: {args.waveform}")
    print(f"Number of waveforms: {args.n_waveforms}")
    print(f"Duration: {args.duration} s")
    print(f"Frequency range: {args.f_min} - {args.f_max} Hz")
    print(f"Reference frequency: {args.f_ref} Hz")
    print(f"Git hash: {config['git_hash']}")
    print(f"{'=' * 60}\n")

    # Generate parameters based on waveform type
    waveform_type = get_waveform_type(args.waveform)

    if waveform_type == "bns":
        params = generate_bns_parameters(args.n_waveforms)
    else:
        params = generate_bbh_parameters(args.n_waveforms)

    print(f"Generated {args.n_waveforms} parameter sets")
    print(f"Parameter keys: {list(params.keys())}\n")

    # Run timing based on waveform
    if args.waveform == "IMRPhenomXPHM":
        print(" Running XPHM timing benchmark...")
        first_run_time, exec_times = time_imrphenomxphm(params, config)

    else:
        import ripplegw

        if args.waveform == "IMRPhenomPv2":
            print(
                "Running precessing waveform timing benchmark (note: XPHM is separated)..."
            )
            waveform = ripplegw.waveform_preset["IMRPhenomPv2"](
                f_ref=config["reference_frequency"]  # type: ignore
            )
            batched_params = _prepare_precessing_params(params)
        elif waveform_type == "bns":
            print(f"Running BNS waveform timing benchmark ({args.waveform})...")
            waveform = ripplegw.waveform_preset[args.waveform](
                f_ref=config["reference_frequency"]  # type: ignore
            )
            batched_params = _prepare_bns_params(params)
        else:
            print(
                f"Running aligned-spin waveform timing benchmark ({args.waveform})..."
            )
            waveform = ripplegw.waveform_preset[args.waveform](
                f_ref=config["reference_frequency"]  # type: ignore
            )
            batched_params = _prepare_aligned_params(params)

        first_run_time, exec_times = time_waveform(waveform, batched_params, config)

    # Compute statistics over timed runs
    mean_exec = sum(exec_times) / len(exec_times)
    std_exec = (
        (sum((t - mean_exec) ** 2 for t in exec_times) / (len(exec_times) - 1)) ** 0.5
        if len(exec_times) > 1
        else 0.0
    )
    min_exec = min(exec_times)
    max_exec = max(exec_times)
    mean_tpw_ms = mean_exec / args.n_waveforms * 1000
    std_tpw_ms = std_exec / args.n_waveforms * 1000
    mean_wps = args.n_waveforms / mean_exec
    std_wps = args.n_waveforms * std_exec / (mean_exec**2)

    # Print results
    print(f"\n{'=' * 60}")
    print("Timing Results")
    print(f"{'=' * 60}")
    print(f"First run time (includes JIT compilation): {first_run_time:.6f} s")
    print(f"Timed runs ({args.n_runs} repetitions):")
    print(f"  Mean execution time: {mean_exec:.6f} s")
    print(f"  Std  execution time: {std_exec:.6f} s")
    print(f"  Min  execution time: {min_exec:.6f} s")
    print(f"  Max  execution time: {max_exec:.6f} s")
    print(f"Mean time per waveform: {mean_tpw_ms:.3f} ms  (+/- {std_tpw_ms:.3f} ms)")
    print(f"Mean waveforms per second: {mean_wps:.1f}  (+/- {std_wps:.1f})")
    print(f"{'=' * 60}\n")

    # Save results
    results = {
        **config,
        "first_run_time_s": float(first_run_time),
        "timed_run_times_s": [float(t) for t in exec_times],
        "mean_execution_time_s": float(mean_exec),
        "std_execution_time_s": float(std_exec),
        "min_execution_time_s": float(min_exec),
        "max_execution_time_s": float(max_exec),
        "time_per_waveform_ms": float(mean_tpw_ms),
        "time_per_waveform_std_ms": float(std_tpw_ms),
        "waveforms_per_second": float(mean_wps),
        "waveforms_per_second_std": float(std_wps),
    }

    # Save to JSON
    if args.output:
        output_path = Path(args.output)
    else:
        outdir = (
            Path(__file__).parent.parent.parent.parent.parent / "timings" / "outdir"
        )
        outdir.mkdir(exist_ok=True)
        filename = f"{args.waveform}_{device_name}_{args.precision}.json"
        output_path = outdir / filename

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


def get_waveform_type(waveform):
    """Determine if waveform is BBH or BNS."""
    bns_waveforms = ["TaylorF2", "IMRPhenomD_NRTidalv2", "IMRPhenomXAS_NRTidalv3"]
    return "bns" if waveform in bns_waveforms else "bbh"


def main():
    """Parse arguments and run timing benchmark."""
    parser = argparse.ArgumentParser(
        description="Time gravitational waveform generation in ripple",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "waveform",
        type=str,
        choices=[
            "IMRPhenomXPHM",
            "IMRPhenomXAS",
            "IMRPhenomD",
            "IMRPhenomPv2",
            "TaylorF2",
            "IMRPhenomD_NRTidalv2",
            "IMRPhenomXAS_NRTidalv3",
        ],
        help="Waveform approximant to time",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Hardware device to use",
    )

    parser.add_argument(
        "--n-waveforms",
        type=int,
        default=int(2e4),
        help="Number of waveforms to generate",
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of timed runs to perform after the first (JIT) run",
    )

    parser.add_argument(
        "--precision",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="Floating point precision to use",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Duration of the waveform in seconds",
    )

    parser.add_argument(
        "--f-min", type=float, default=20.0, help="Minimum frequency in Hz"
    )

    parser.add_argument(
        "--f-max", type=float, default=2048.0, help="Maximum frequency in Hz"
    )

    parser.add_argument(
        "--f-ref", type=float, default=50.0, help="Reference frequency in Hz"
    )

    parser.add_argument(
        "--output", type=str, help="Output JSON file path (default: auto-generated)"
    )

    args = parser.parse_args()

    run_timing(args)


if __name__ == "__main__":
    main()
