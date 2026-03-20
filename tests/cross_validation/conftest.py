"""Cross-validation test configuration and session summary.

This conftest collects per-waveform mismatch statistics from all
test_waveform_mismatch runs and prints a formatted summary at the end of the
session, including hardware information and pass/fail status per waveform.
"""

import json
import platform
from datetime import datetime
from pathlib import Path

import jax
import pytest


# ---------------------------------------------------------------------------
# Session-level results store
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Attach a shared results store to the pytest config object."""
    config._cross_val_results = []


def pytest_addoption(parser):
    """Register CLI options for cross-validation tests."""
    parser.addoption(
        "--n-samples",
        type=int,
        default=10,
        help="Number of random parameter sets per waveform (default: 10)",
    )
    parser.addoption(
        "--cache-lal",
        action="store_true",
        default=False,
        help="Cache LAL waveforms to disk and reuse on subsequent runs (default: off)",
    )
    parser.addoption(
        "--T",
        type=float,
        default=32.0,
        help="Segment duration in seconds, sets frequency resolution df=1/T (default: 32)",
    )


@pytest.fixture(scope="session")
def n_samples(request):
    """Number of random samples to test per waveform (set via --n-samples)."""
    return request.config.getoption("--n-samples")


@pytest.fixture(scope="session")
def cache_lal(request):
    """Whether to cache/reuse LAL waveforms on disk (set via --cache-lal)."""
    return request.config.getoption("--cache-lal")


@pytest.fixture(scope="session")
def cross_val_results(request):
    """Session-scoped list that accumulates per-waveform result dicts.

    Each entry has the shape::

        {
            "waveform": str,
            "n_samples": int,
            "n_finite": int,
            "n_failed": int,
            "mean": float,
            "median": float,
            "min": float,
            "max": float,
            "threshold": float,
            "passed": bool,
        }
    """
    return request.config._cross_val_results


# ---------------------------------------------------------------------------
# Terminal summary hook
# ---------------------------------------------------------------------------


def _hardware_info() -> dict:
    """Collect hardware / runtime information."""
    import jax.numpy as jnp

    devices = jax.devices()
    device_strs = [str(d) for d in devices]

    x64_enabled = bool(jax.config.jax_enable_x64)
    # Confirm the actual floating-point dtype in use
    float_dtype = str(jnp.zeros(1).dtype)  # "float64" or "float32"

    return {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "jax_devices": device_strs,
        "jax_version": jax.__version__,
        "float_dtype": float_dtype,
        "x64_enabled": x64_enabled,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a cross-validation summary table after all tests finish."""
    results = getattr(config, "_cross_val_results", [])
    if not results:
        return

    hw = _hardware_info()

    # ---- header -----------------------------------------------------------
    terminalreporter.write_sep("=", "Cross-Validation Summary")

    # ---- hardware block ---------------------------------------------------
    terminalreporter.write_line("Hardware / Runtime")
    terminalreporter.write_line(f"  Host       : {hw['host']}")
    terminalreporter.write_line(f"  OS         : {hw['os']}")
    terminalreporter.write_line(f"  CPU        : {hw['cpu']}")
    terminalreporter.write_line(f"  Python     : {hw['python']}")
    terminalreporter.write_line(f"  JAX        : {hw['jax_version']}")
    terminalreporter.write_line(f"  Devices    : {', '.join(hw['jax_devices'])}")
    terminalreporter.write_line(
        f"  Precision  : {hw['float_dtype']} (x64_enabled={hw['x64_enabled']})"
    )
    terminalreporter.write_line(f"  Timestamp  : {hw['timestamp']}")
    terminalreporter.write_line("")

    # ---- results table ----------------------------------------------------
    col_w = 26
    num_w = 12
    header = (
        f"{'Waveform':<{col_w}}"
        f"{'Samples':>{num_w}}"
        f"{'Failed':>{num_w}}"
        f"{'Mean':>{num_w}}"
        f"{'Median':>{num_w}}"
        f"{'Max':>{num_w}}"
        f"{'Threshold':>{num_w}}"
        f"{'Status':>{10}}"
    )
    terminalreporter.write_line(header)
    terminalreporter.write_line("-" * (col_w + num_w * 6 + 10))

    all_passed = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_passed = False

        row = (
            f"{r['waveform']:<{col_w}}"
            f"{r['n_samples']:>{num_w}}"
            f"{r['n_failed']:>{num_w}}"
            f"{r['mean']:>{num_w}.2e}"
            f"{r['median']:>{num_w}.2e}"
            f"{r['max']:>{num_w}.2e}"
            f"{r['threshold']:>{num_w}.2e}"
            f"{status:>{10}}"
        )
        terminalreporter.write_line(row)

    terminalreporter.write_line("-" * (col_w + num_w * 6 + 10))
    overall = "ALL PASSED" if all_passed else "SOME FAILED"
    terminalreporter.write_line(f"Overall: {overall}")
    terminalreporter.write_sep("=", "")

    # ---- persist metadata to disk ----------------------------------------
    # Mirror the run-tag subdirectory scheme used by the test (n{N}_T{T}).
    n_samples_opt = config.getoption("--n-samples", default=10)
    T_opt = config.getoption("--T", default=32.0)
    T_str = f"T{int(T_opt)}" if T_opt == int(T_opt) else f"T{T_opt}"
    run_tag = f"n{n_samples_opt}_{T_str}"
    results_dir = Path(__file__).parent / "results" / run_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "hardware": hw,
        "waveforms": results,
    }
    metadata_file = results_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    terminalreporter.write_line(f"Metadata saved to: {metadata_file}")
