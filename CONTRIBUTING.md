# Contributing

Contributions of any kind are welcome and appreciated.
See the guidelines below.

## Expectations

ripple is developed and maintained by the GW JAX Team and community contributors.
While we try to be responsive, we don’t always get to every issue immediately.
If it has been more than a week or two, feel free to ping the maintainers on the issue.

## Did you find a bug?

**Ensure the bug was not already reported** by searching on GitHub under [Issues](https://github.com/GW-JAX-Team/ripple/issues).
If you’re unable to find an open issue addressing the problem, [open a new one](https://github.com/GW-JAX-Team/ripple/issues/new).
Be sure to include a **title and clear description**, as much relevant information as possible, and the simplest possible **code sample** demonstrating the expected behaviour that is not occurring.

## Did you write a patch that fixes a bug?

Open a new GitHub pull request with the patch.
Ensure the PR description clearly describes the problem and solution.
Include the relevant issue number if applicable.

## Do you intend to add a new feature or change an existing one?

Open a new GitHub pull request with the feature or change.
Please follow these principles:

1. New features should be able to take advantage of `jax.jit` wherever possible.
2. Modular implementation is preferred.
3. If a waveform is ported from an existing implementation, it should match the original to machine precision. If that is not achievable, a clear explanation of the discrepancy must be provided. New waveforms implemented directly in ripple are not subject to this requirement.

If you are unsure whether a feature fits, open an issue first to discuss it with the maintainers.

### Adding a new waveform

Waveforms are **self-registering**: a model advertises itself to the top-level
API (`ripplegw.waveform`, `ripplegw.list_waveforms`) via the `@register`
decorator, so you do **not** edit any central list, `interfaces.py`, or
`__init__.py`. The `Waveform` contract is family-agnostic — it is not limited to
IMRPhenom/CBC models.

**In ripple (this repository):**

- `src/ripplegw/waveforms/NewWaveform.py` — implement the model:
  - expose the low-level generator (e.g. `gen_NewWaveform(axis, ...)`);
  - define a `Waveform` subclass and decorate it with `@register`, attaching any
    descriptive metadata as keyword arguments:

    ```python
    from ripplegw.interfaces import Waveform
    from ripplegw.registry import register

    @register("NewWaveform", domain="FD", is_tidal=False, is_precessing=False)
    class NewWaveform(Waveform):
        def __init__(self, f_ref: float = 20.0):   # construction-time config
            self.f_ref = f_ref

        @property
        def parameter_names(self) -> tuple[str, ...]:
            return ("M_c", "eta", ...)

        def __call__(self, axis, params):          # -> {"p": ..., "c": ...}
            ...
    ```

    The class name (or the first `@register` argument) is the string users pass
    to `ripplegw.waveform(...)`. Metadata keys are free-form and are used by
    `ripplegw.list_waveforms(**filters)` / `ripplegw.get_waveform_metadata(name)`
    (`domain` is the conventional one; `is_tidal`/`is_precessing` are CBC tags).
- `src/ripplegw/waveforms/__init__.py` — nothing required: it auto-imports every
  non-private submodule at package import, so a new `NewWaveform.py` self-
  registers with no edit here. Because the module is imported at
  `import ripplegw`, keep import-time work cheap — defer heavy data loading or
  optional third-party imports to `__init__`/first use (or ship the family as a
  plugin package, below), so importing ripple stays fast and dependency-light.
- `README.md` and `docs/index.md` — add `NewWaveform` to the list of supported
  approximants.
- `src/ripplegw/benchmarks/timings/timing.py` and `timings/submit_*.sh` — add
  `"NewWaveform"` to the models list (optional, for benchmarking).
- `tests/` — add tests. At minimum evaluate the model and assert it appears via
  `ripplegw.list_waveforms()` / is constructible via `ripplegw.waveform(...)`.
  If a LAL reference exists, add a cross-validation case.

**Distributing a waveform as a separate package (no ripple edits at all):**

A third-party package can expose a waveform *without modifying ripple* by
declaring a `ripplegw.waveforms` entry point pointing at a `Waveform` subclass:

```toml
# your package's pyproject.toml
[project.entry-points."ripplegw.waveforms"]
NewWaveform = "your_pkg.module:NewWaveform"
```

On install it is discovered automatically and usable via
`ripplegw.waveform("NewWaveform", ...)`. The entry-point value must load to a
`Waveform` subclass. This is the recommended path for large or optional-data
families (e.g. NR surrogates).

**In [Jim](https://github.com/GW-JAX-Team/jim) (the inference framework):**

Jim wraps ripple waveforms under `Ripple`-prefixed aliases.
Once your ripple PR is merged (or during development against a local ripple install), open a separate pull request in the Jim repository with the following changes:

- `src/jimgw/core/single_event/waveform.py` — add `NewWaveform` to the `from ripplegw import (...)` block and add `RippleNewWaveform = NewWaveform` plus an entry in `__all__`.
- `src/jimgw/cli/_waveform.py` — import `RippleNewWaveform` and add `"NewWaveform": RippleNewWaveform` to `_REGISTRY`.
- `src/jimgw/cli/_config.py` — add `"NewWaveform"` to the `Approximant` `Literal` type so the CLI accepts it.

## Do you intend to add an example or tutorial?

Open a new GitHub pull request with the example or tutorial.
The example should be self-contained and keep imports from other packages to a minimum.
Leave case-specific analysis details out.
