"""Waveforms must not depend on whether ``ripplegw`` is imported before x64 is enabled.

Any array that ripple builds with ``jax.numpy`` at import time takes the default dtype
in force at that moment, float32 unless x64 is already enabled, and enabling x64
afterwards cannot restore the lost digits.
"""

import functools
import inspect
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import ripplegw  # noqa: F401  (imports every ripplegw module the scan checks)

# The three models cover the three families of import-time tables: the
# IMRPhenomX collocation coefficients, the IMRPhenomD coefficients and QNM
# tables, and IMRPhenomHM's mode phase shifts.
_MODELS = ("IMRPhenomD", "IMRPhenomXAS", "IMRPhenomHM")

_WORKER = Path(__file__).parents[1] / "helpers" / "import_order.py"


def _find_float_jax_arrays(value, path, found, seen, depth=0):
    """Append the path of every float jax array reachable from ``value`` to ``found``."""
    if isinstance(value, jax.Array):
        if jnp.issubdtype(value.dtype, jnp.inexact):
            found.append(f"{path} ({value.dtype})")
        return
    if depth >= 6 or id(value) in seen:
        return
    seen.add(id(value))

    def own(obj):
        return getattr(obj, "__module__", "").startswith("ripplegw")

    def walk(item, item_path):
        _find_float_jax_arrays(item, item_path, found, seen, depth + 1)

    if isinstance(value, (list, tuple, set, frozenset)):
        for i, item in enumerate(value):
            walk(item, f"{path}[{i}]")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            walk(item, f"{path}[{key!r}]")
    elif isinstance(value, type) and own(value):
        for attr, item in vars(value).items():
            walk(item, f"{path}.{attr}")
    elif isinstance(value, (staticmethod, classmethod)):
        walk(value.__func__, path)
    elif isinstance(value, property):
        for accessor in (value.fget, value.fset, value.fdel):
            if accessor is not None:
                walk(accessor, path)
    elif isinstance(value, functools.partial):
        walk(value.func, f"{path}.func")
        walk(value.args, f"{path}.args")
        walk(value.keywords, f"{path}.keywords")
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for i, item in enumerate(value.ravel()):
            walk(item, f"{path}.flat[{i}]")
    elif own(type(value)):
        # An instance of a ripple class: walk its __dict__ and __slots__.
        for attr, item in getattr(value, "__dict__", {}).items():
            walk(item, f"{path}.{attr}")
        for klass in type(value).__mro__:
            for slot in getattr(klass, "__slots__", ()):
                if isinstance(slot, str) and hasattr(value, slot):
                    walk(getattr(value, slot), f"{path}.{slot}")
    elif callable(value) and own(value):
        func = inspect.unwrap(value)
        walk(getattr(func, "__defaults__", None) or (), f"{path}.__defaults__")
        walk(getattr(func, "__kwdefaults__", None) or {}, f"{path}.__kwdefaults__")
        for k, cell in enumerate(getattr(func, "__closure__", None) or ()):
            try:
                contents = cell.cell_contents
            except ValueError:  # empty cell
                continue
            walk(contents, f"{path}.__closure__[{k}]")


def test_no_floating_jax_arrays_are_built_at_import():
    """Import-time float arrays are frozen at the default precision; keep them in NumPy.

    Module globals are scanned, including the contents of containers and of ripple's
    own classes, instances and functions (class attributes, instance attributes and
    slots, default arguments, closures). Not covered: values produced lazily, such as
    ``functools.cache`` results, generators or custom descriptors.
    """
    found, seen = [], set()
    for name, module in sorted(sys.modules.items()):
        if name.startswith("ripplegw") and module is not None:
            for attr, value in vars(module).items():
                _find_float_jax_arrays(value, f"{name}.{attr}", found, seen)
    assert found == [], (
        "module-level jax arrays built at import time take the precision in force "
        f"when ripplegw is imported: {sorted(set(found))}"
    )


def test_waveform_does_not_depend_on_import_order(tmp_path):
    """Importing ripplegw before enabling x64 must give bit-identical waveforms."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    env.pop("JAX_ENABLE_X64", None)
    outputs = {}
    for order in ("x64-first", "import-first"):
        path = tmp_path / f"{order}.npz"
        # A fresh interpreter per import order; all models share it, as each start is slow.
        result = subprocess.run(
            [sys.executable, str(_WORKER), order, str(path), *_MODELS],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-4000:]
        outputs[order] = np.load(path)

    first, second = outputs["x64-first"], outputs["import-first"]
    assert sorted(first.files) == sorted(second.files)
    for key in first.files:
        assert first[key].dtype == np.complex128, key
        assert second[key].dtype == np.complex128, key
        np.testing.assert_array_equal(second[key], first[key], err_msg=key)
