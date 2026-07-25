"""Cross-validation session summary.

Collects one summary dict per completed campaign (from ``cross_val_results``)
and prints a hardware block + results table at the end of the session.
Per-sample data and run metadata are written by ``campaign.write_results``
under ``--outdir``; this hook is display-only.
"""

import platform
from datetime import datetime

import jax
import pytest

from tests.helpers.reference import get_backend


def pytest_configure(config):
    """Attach a shared results store to the pytest config object."""
    config._cross_val_results = []


@pytest.fixture(scope="session")
def reference(reference_name):
    """The reference backend selected by ``--reference`` (default: lal).

    Skips every test that depends on this fixture -- i.e. the whole module --
    when the backend's dependency (e.g. LALSuite) is not installed.
    """
    backend = get_backend(reference_name)
    if not backend.available():
        pytest.skip(
            f"Reference backend {reference_name!r} is not available (missing dependency)"
        )
    return backend


@pytest.fixture(scope="session")
def cross_val_results(request):
    """Session-scoped list of summary dicts, one per completed campaign::

    {"waveform": str, "reference": str, "n_samples": int, "n_failed": int,
     "mean": float, "median": float, "max": float, "threshold": float,
     "passed": bool}
    """
    return request.config._cross_val_results


def _hardware_info() -> dict:
    import jax.numpy as jnp

    return {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_version": jax.__version__,
        "float_dtype": str(jnp.zeros(1).dtype),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # noqa: DTZ005 - local display timestamp only
    }


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a cross-validation summary table after all tests finish."""
    results = getattr(config, "_cross_val_results", [])
    if not results:
        return

    hw = _hardware_info()
    terminalreporter.write_sep("=", "Cross-Validation Summary")
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

    col_w, num_w = 26, 12
    header = (
        f"{'Waveform':<{col_w}}{'Reference':<{12}}{'Samples':>{num_w}}{'Failed':>{num_w}}"
        f"{'Mean':>{num_w}}{'Median':>{num_w}}{'Max':>{num_w}}{'Threshold':>{num_w}}{'Status':>10}"
    )
    terminalreporter.write_line(header)
    terminalreporter.write_line("-" * (col_w + 12 + num_w * 6 + 10))

    all_passed = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        all_passed &= r["passed"]
        terminalreporter.write_line(
            f"{r['waveform']:<{col_w}}{r['reference']:<{12}}{r['n_samples']:>{num_w}}"
            f"{r['n_failed']:>{num_w}}{r['mean']:>{num_w}.2e}{r['median']:>{num_w}.2e}"
            f"{r['max']:>{num_w}.2e}{r['threshold']:>{num_w}.2e}{status:>10}"
        )

    terminalreporter.write_line("-" * (col_w + 12 + num_w * 6 + 10))
    terminalreporter.write_line(
        f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}"
    )
    terminalreporter.write_sep("=", "")
