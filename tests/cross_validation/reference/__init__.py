"""Reference-backend abstraction for the accuracy test.

A reference backend wraps an independent, CPU-based implementation (LAL
today; future waveform families may need others) behind one interface, so
``cross_validation/`` never imports a specific package directly. Adding a
backend is one file here plus a matching ``[<name>.<model>]`` block in
``tests/cross_validation/tolerances.toml`` -- no test file changes.
"""

from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from tests.helpers.grids import Grid


@runtime_checkable
class ReferenceBackend(Protocol):
    name: ClassVar[str]

    @classmethod
    def available(cls) -> bool:
        """Whether the backend's dependency is importable in this environment."""
        ...

    def supports(self, waveform: str) -> bool:
        """Whether this backend can generate ``waveform``."""
        ...

    def constants(self) -> dict[str, float]:
        """Physical constants this backend defines, for ``cross_validation/test_reference_constants.py``."""
        ...

    def generate(
        self, waveform: str, params: dict[str, float], grid: Grid
    ) -> dict[str, np.ndarray]:
        """Generate ``{"p": hp, "c": hc}`` on ``grid.axis``, from a ripple-native ``params`` dict."""
        ...

    def expected_failure(self, waveform: str, error: Exception) -> bool:
        """Whether ``error`` from ``generate`` is an expected, excludable failure
        mode of the reference implementation itself (e.g. a documented
        domain error) rather than a bug to fail the test on. Optional --
        backends without a genuinely expected failure mode need not implement
        it; callers should treat a missing implementation as always ``False``.
        """
        ...


REFERENCE_BACKENDS: dict[str, type] = {}


def register_backend(cls):
    """Class decorator that registers a ``ReferenceBackend`` under ``cls.name``."""
    REFERENCE_BACKENDS[cls.name] = cls
    return cls


def get_backend(name: str) -> ReferenceBackend:
    """Instantiate the named backend."""
    try:
        cls = REFERENCE_BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown reference backend {name!r}. Available: {sorted(REFERENCE_BACKENDS)}"
        ) from None
    return cls()


# Import built-in backends for their registration side effect.
from tests.cross_validation.reference import lal  # noqa: F401
