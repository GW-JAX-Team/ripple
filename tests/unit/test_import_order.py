"""Waveforms must not depend on whether ``ripplegw`` is imported before x64 is enabled.

Users may import ``ripplegw`` before calling
``jax.config.update("jax_enable_x64", True)``. Any array that ripple builds with
``jax.numpy`` at import time takes the default dtype in force at that moment, which
is float32, and enabling x64 afterwards cannot restore the lost digits.

The test session itself enables x64 before importing ripple (see ``conftest.py``),
so each check runs in a fresh interpreter where the import order can be controlled.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

# The three models cover the three families of import-time tables: the
# IMRPhenomX collocation coefficients, the IMRPhenomD coefficients and QNM
# tables, and IMRPhenomHM's mode phase shifts.
_MODELS = ("IMRPhenomD", "IMRPhenomXAS", "IMRPhenomHM")

_WAVEFORM_SCRIPT = textwrap.dedent(
    """
    import sys

    import jax

    enable_first = sys.argv[1] == "x64-first"
    if enable_first:
        jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import numpy as np

    import ripplegw

    if not enable_first:
        jax.config.update("jax_enable_x64", True)

    frequency = jnp.arange(20.0, 1024.0, 0.25)
    params = {
        "M_c": 28.3,
        "eta": 0.247,
        "s1_z": 0.1,
        "s2_z": -0.2,
        "d_L": 440.0,
        "phase_c": 0.0,
        "iota": 0.3,
    }
    out = {}
    for name in sys.argv[3].split(","):
        strain = ripplegw.waveform(name)(frequency, params)
        for pol, value in strain.items():
            out[f"{name}_{pol}"] = np.asarray(value)
    np.savez(sys.argv[2], **out)
    """
)

_SCAN_SCRIPT = textwrap.dedent(
    """
    import functools
    import inspect
    import json
    import sys
    from collections.abc import Mapping

    import jax
    import jax.numpy as jnp
    import numpy as np

    import ripplegw  # x64 is off; it is never enabled in this scan

    seen = set()

    def own(obj):
        return getattr(obj, "__module__", "").startswith("ripplegw")

    def walk(value, path, found, depth=0):
        # Besides bare arrays, look inside containers, NumPy object arrays and ripple's
        # own classes, instances and functions (class attributes, instance __dict__ and
        # __slots__, default arguments, closures): an array frozen there at import time
        # is just as stuck at float32.
        if isinstance(value, jax.Array):
            if jnp.issubdtype(value.dtype, jnp.inexact):
                found.append(f"{path} ({value.dtype})")
            return
        if depth >= 6 or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (list, tuple, set, frozenset)):
            for i, item in enumerate(value):
                walk(item, f"{path}[{i}]", found, depth + 1)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, f"{path}[{key!r}]", found, depth + 1)
        elif isinstance(value, type) and own(value):
            for attr, item in vars(value).items():
                walk(item, f"{path}.{attr}", found, depth + 1)
        elif isinstance(value, (staticmethod, classmethod)):
            walk(value.__func__, path, found, depth + 1)
        elif isinstance(value, property):
            for accessor in (value.fget, value.fset, value.fdel):
                if accessor is not None:
                    walk(accessor, path, found, depth + 1)
        elif isinstance(value, functools.partial):
            walk(value.func, f"{path}.func", found, depth + 1)
            walk(value.args, f"{path}.args", found, depth + 1)
            walk(value.keywords, f"{path}.keywords", found, depth + 1)
        elif isinstance(value, np.ndarray) and value.dtype == object:
            for i, item in enumerate(value.ravel()):
                walk(item, f"{path}.flat[{i}]", found, depth + 1)
        elif own(type(value)):
            # An instance of a ripple class: walk its __dict__ and __slots__.
            for attr, item in getattr(value, "__dict__", {}).items():
                walk(item, f"{path}.{attr}", found, depth + 1)
            for klass in type(value).__mro__:
                for slot in getattr(klass, "__slots__", ()):
                    if isinstance(slot, str) and hasattr(value, slot):
                        walk(getattr(value, slot), f"{path}.{slot}", found, depth + 1)
        elif callable(value) and own(value):
            func = inspect.unwrap(value)
            walk(getattr(func, "__defaults__", None) or (), f"{path}.__defaults__", found, depth + 1)
            walk(getattr(func, "__kwdefaults__", None) or {}, f"{path}.__kwdefaults__", found, depth + 1)
            for k, cell in enumerate(getattr(func, "__closure__", None) or ()):
                try:
                    contents = cell.cell_contents
                except ValueError:  # empty cell
                    continue
                walk(contents, f"{path}.__closure__[{k}]", found, depth + 1)

    found = []
    for module_name, module in sorted(sys.modules.items()):
        if not module_name.startswith("ripplegw") or module is None:
            continue
        for attr, value in vars(module).items():
            walk(value, f"{module_name}.{attr}", found)
    print(json.dumps(sorted(set(found))))
    """
)


def _run(args, tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env.pop("JAX_ENABLE_X64", None)
    result = subprocess.run(
        [sys.executable, "-c", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return result.stdout


def test_no_floating_jax_arrays_are_built_at_import(tmp_path):
    """Import-time float arrays are frozen at the default precision; keep them in NumPy.

    Module globals are scanned, including the contents of containers and of ripple's
    own classes, instances and functions (class attributes, instance attributes and
    slots, default arguments, closures). Not covered: values produced lazily, such as
    ``functools.cache`` results, generators or custom descriptors.
    """
    stdout = _run([_SCAN_SCRIPT], tmp_path)
    found = json.loads(stdout.strip().splitlines()[-1])
    assert found == [], (
        "module-level jax arrays built at import time take the precision in force "
        f"when ripplegw is imported: {found}"
    )


@pytest.mark.parametrize("model", _MODELS)
def test_waveform_does_not_depend_on_import_order(tmp_path, model):
    """Importing ripplegw before enabling x64 must give bit-identical waveforms."""
    outputs = {}
    for order in ("x64-first", "import-first"):
        path = tmp_path / f"{order}.npz"
        _run([_WAVEFORM_SCRIPT, order, str(path), model], tmp_path)
        outputs[order] = np.load(path)

    first, second = outputs["x64-first"], outputs["import-first"]
    assert sorted(first.files) == sorted(second.files)
    for key in first.files:
        assert first[key].dtype == np.complex128, key
        assert second[key].dtype == np.complex128, key
        np.testing.assert_array_equal(second[key], first[key], err_msg=key)


def test_qnm_grid_is_bit_identical_to_jax_linspace():
    """The QNM grid keeps the exact values ``jnp.linspace`` gave with x64 enabled.

    The splined QNM tables are evaluated on this grid, so any change of rounding in
    the grid moves IMRPhenomD-family waveforms at the 1e-15 level.
    """
    import jax.numpy as jnp

    from ripplegw.waveforms.cbc.IMRPhenomD import IMRPhenomD_QNMdata

    grid = IMRPhenomD_QNMdata.QNMData_a
    assert isinstance(grid, np.ndarray)
    assert grid.dtype == np.float64
    np.testing.assert_array_equal(grid, np.asarray(jnp.linspace(-1, 1, 500_000)))
