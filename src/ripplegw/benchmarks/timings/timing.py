"""
Command-line interface for timing gravitational waveform generation in ripple.

This script provides a flexible CLI for benchmarking different waveform approximants
with various configurations including hardware selection and precision.
"""

import argparse
import json
import logging
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

logger = logging.getLogger(__name__)


def setup_jax_config(use_float64, device):
    """Configure JAX settings for precision and device."""
    jax.config.update("jax_enable_x64", use_float64)

    if device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    logger.info("\n%s", "=" * 60)
    logger.info("JAX Configuration")
    logger.info("=" * 60)
    logger.info("Precision: %s", "float64" if use_float64 else "float32")
    logger.info("Requested device: %s", device)
    logger.info("JAX devices: %s", jax.devices())
    logger.info("Default backend: %s", jax.default_backend())
    for d in jax.devices():
        logger.info("  Device: %s, Platform: %s", d.device_kind, d.platform)
    logger.info("=" * 60)

    return get_device_name()


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
    logger.info("\n%s", "=" * 60)
    logger.info("First run (includes JIT compilation)")
    logger.info("=" * 60)
    start = time.time()
    result = waveform_batched(batched_params)
    result["p"].block_until_ready()
    result["c"].block_until_ready()
    first_run_time = time.time() - start
    logger.info("First run time (includes JIT compilation): %.3f s", first_run_time)

    # Timed runs
    logger.info("\n%s", "=" * 60)
    logger.info("Timed runs (%d repetitions)", n_runs)
    logger.info("=" * 60)
    exec_times = []
    for i in range(n_runs):
        start = time.time()
        result = waveform_batched(batched_params)
        result["p"].block_until_ready()
        result["c"].block_until_ready()
        t = time.time() - start
        exec_times.append(t)
        logger.info("  Run %d: %.6f s", i + 1, t)

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
    logger.info("=" * 60)
    logger.info("Timing Configuration")
    logger.info("=" * 60)
    logger.info("Waveform: %s", args.waveform)
    logger.info("Number of waveforms: %d", args.n_waveforms)
    logger.info("Duration: %s s", args.duration)
    logger.info("Frequency range: %s - %s Hz", args.f_min, args.f_max)
    logger.info("Reference frequency: %s Hz", args.f_ref)
    logger.info("Git hash: %s", config["git_hash"])
    logger.info("=" * 60)

    # Generate parameters based on waveform type
    waveform_type = get_waveform_type(args.waveform)

    if waveform_type == "bns":
        params = generate_bns_parameters(args.n_waveforms)
    else:
        params = generate_bbh_parameters(args.n_waveforms)

    logger.info("Generated %d parameter sets", args.n_waveforms)
    logger.info("Parameter keys: %s", list(params.keys()))

    # Run timing based on waveform
    import ripplegw

    precessing_waveforms = ["IMRPhenomXPHM", "IMRPhenomXP", "IMRPhenomPv2"]
    if args.waveform in precessing_waveforms:
        logger.info(
            "Running precessing waveform timing benchmark (%s)...", args.waveform
        )
        waveform = ripplegw.waveform_preset[args.waveform](
            f_ref=config["reference_frequency"]  # type: ignore
        )
        batched_params = _prepare_precessing_params(params)
    elif waveform_type == "bns":
        logger.info("Running BNS waveform timing benchmark (%s)...", args.waveform)
        waveform = ripplegw.waveform_preset[args.waveform](
            f_ref=config["reference_frequency"]  # type: ignore
        )
        batched_params = _prepare_bns_params(params)
    else:
        logger.info(
            "Running aligned-spin waveform timing benchmark (%s)...", args.waveform
        )
        waveform = ripplegw.waveform_preset[args.waveform](
            f_ref=config["reference_frequency"]  # type: ignore
        )
        batched_params = _prepare_aligned_params(params)

    first_run_time, exec_times = time_waveform(waveform, batched_params, config)

    # Compute statistics over timed runs
    exec_times_arr = jnp.array(exec_times)
    mean_exec = float(jnp.mean(exec_times_arr))
    std_exec = float(jnp.std(exec_times_arr, ddof=1)) if len(exec_times) > 1 else 0.0
    min_exec = float(jnp.min(exec_times_arr))
    max_exec = float(jnp.max(exec_times_arr))
    mean_tpw_ms = mean_exec / args.n_waveforms * 1000
    std_tpw_ms = std_exec / args.n_waveforms * 1000
    mean_wps = args.n_waveforms / mean_exec
    std_wps = args.n_waveforms * std_exec / (mean_exec**2)

    # Print results
    logger.info("\n%s", "=" * 60)
    logger.info("Timing Results")
    logger.info("=" * 60)
    logger.info("First run time (includes JIT compilation): %.6f s", first_run_time)
    logger.info("Timed runs (%d repetitions):", args.n_runs)
    logger.info("  Mean execution time: %.6f s", mean_exec)
    logger.info("  Std  execution time: %.6f s", std_exec)
    logger.info("  Min  execution time: %.6f s", min_exec)
    logger.info("  Max  execution time: %.6f s", max_exec)
    logger.info(
        "Mean time per waveform: %.3f ms  (+/- %.3f ms)", mean_tpw_ms, std_tpw_ms
    )
    logger.info("Mean waveforms per second: %.1f  (+/- %.1f)", mean_wps, std_wps)
    logger.info("=" * 60)

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

    logger.info("Results saved to: %s", output_path)


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
            "IMRPhenomXP",
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
