"""Shared pytest configuration for the whole ripple test suite.

``pytest_addoption`` lives here (not in a subdirectory conftest) because
pytest only registers CLI options declared in a rootdir conftest or a
plugin -- a subdirectory conftest's options are invisible once the run is
anchored at ``testpaths = ["tests"]``.
"""

from functools import cache
from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import ripplegw
from tests.helpers.config import default_config

_DEFAULT_ACCURACY_OUTDIR = (
    Path(__file__).resolve().parent / "cross_validation" / "results"
)


def pytest_addoption(parser):
    parser.addoption(
        "--reference",
        default="lal",
        help="Reference backend for accuracy tests (default: lal).",
    )
    parser.addoption(
        "--n-samples",
        type=int,
        default=10,
        help="Number of random parameter sets per waveform for accuracy tests (default: 10).",
    )
    parser.addoption(
        "--T",
        type=float,
        default=None,
        help=(
            "Segment duration in seconds for accuracy tests. Overrides the "
            "per-regime defaults (BBH: 32 s, BNS: 128 s) when provided."
        ),
    )
    parser.addoption(
        "--outdir",
        default=str(_DEFAULT_ACCURACY_OUTDIR),
        help="Directory for large-scale test results/figures/cache "
        "(default: tests/cross_validation/results).",
    )
    parser.addoption(
        "--cache-reference",
        action="store_true",
        default=False,
        help="Cache reference-backend waveforms to disk and reuse on subsequent runs.",
    )
    parser.addoption(
        "--plots",
        action="store_true",
        default=False,
        help="Generate per-waveform overlap-loss figures (requires matplotlib).",
    )
    parser.addoption(
        "--cw-waveform",
        choices=("PulsarSignal", "BinaryPulsarSignal"),
        default=None,
        help=(
            "Run the CW MakeFakeData test for one selected model. Omitting this "
            "keeps its direct-pytest mixed isolated/binary sweep."
        ),
    )


@pytest.fixture(scope="session")
def reference_name(request) -> str:
    return request.config.getoption("--reference")


@pytest.fixture(scope="session")
def n_samples(request) -> int:
    return request.config.getoption("--n-samples")


@pytest.fixture(scope="session")
def segment_duration_override(request) -> float | None:
    return request.config.getoption("--T")


@pytest.fixture(scope="session")
def accuracy_outdir(request) -> str:
    return request.config.getoption("--outdir")


@pytest.fixture(scope="session")
def cache_reference(request) -> bool:
    return request.config.getoption("--cache-reference")


@pytest.fixture(scope="session")
def make_plots(request) -> bool:
    return request.config.getoption("--plots")


@pytest.fixture(scope="session")
def cw_waveform(request) -> str | None:
    """Optional selected CW model for the large MakeFakeData test."""

    return request.config.getoption("--cw-waveform")


@cache
def _waveform_instance(name: str, config: tuple):
    return ripplegw.waveform(name, **{**default_config(name), **dict(config)})


@cache
def _jit_method(name: str, config: tuple, method: str):
    """Return the session-cached JIT-compiled waveform method and instance."""
    wf = _waveform_instance(name, config)
    return jax.jit(getattr(wf, method)), wf


@pytest.fixture(scope="session")
def compiled_model():
    """``compiled_model(name, method="__call__", **config) -> (jitted_callable, waveform_instance)``.

    ``method`` selects which bound method to jit -- ``"__call__"`` (default),
    ``"amplitude"``, ``"phase"``, ``"strain"``, or ``"at_unit_distance"``. All
    calls for the same ``(name, config)`` share one underlying waveform
    instance, whichever ``method`` is requested.
    """

    def _get(name: str, method: str = "__call__", **config):
        return _jit_method(name, tuple(sorted(config.items())), method)

    return _get
