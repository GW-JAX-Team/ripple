"""Save waveforms computed with x64 enabled before or after importing ``ripplegw``.

Run in a fresh interpreter: ``python import_order.py {x64-first,import-first} OUT.npz MODEL...``.
"""

import sys

import jax
import numpy as np


def main(order: str, path: str, models: list[str]) -> None:
    """Save each model's polarizations to ``path`` as ``{model}_{pol}`` arrays."""
    if order == "x64-first":
        jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp

    import ripplegw

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
    for name in models:
        for pol, value in ripplegw.waveform(name)(frequency, params).items():
            out[f"{name}_{pol}"] = np.asarray(value)
    np.savez(path, **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
