# Distributing a Waveform Plugin

A waveform can be shipped as a separate, installable Python package — with **zero edits to ripple** — by declaring a `ripplegw.waveforms` entry point that points at a `Waveform` subclass.
This is the recommended path for large or optional-data families (e.g. NR surrogates), or anything with an unusual optional dependency ripple shouldn't require of every user.

The model itself follows exactly the same contract as an in-tree model — see [Adding a Waveform](adding_a_waveform.md) for the `Waveform`/`@register` shape.
Only the packaging differs.

## Package layout

```
my-ripple-plugin/
  pyproject.toml
  src/
    my_ripple_plugin/
      __init__.py
      new_waveform.py
```

`new_waveform.py` implements the model exactly as an in-tree module would — generator function, `_split_params`, registered class:

```python
# src/my_ripple_plugin/new_waveform.py
from ripplegw.interfaces import FrequencyDomainWaveform, DistanceScaledWaveform
from ripplegw.registry import register


@register("NewWaveform", is_tidal=False, is_precessing=False)
class NewWaveform(FrequencyDomainWaveform, DistanceScaledWaveform):
    def __init__(self, f_ref: float = 20.0) -> None:
        self.f_ref = f_ref

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("M_c", "eta", "s1_z", "s2_z", "d_L", "phase_c", "iota")

    def __call__(self, frequency, params):
        ...
        return {"p": hp, "c": hc}
```

Note this module calls `@register(...)` itself, just like an in-tree module — registration happens the same way whether the module is discovered by ripple's own auto-import or loaded via your entry point.

## Declaring the entry point

```toml
# my-ripple-plugin/pyproject.toml
[project]
name = "my-ripple-plugin"
dependencies = ["rippleGW"]

[project.entry-points."ripplegw.waveforms"]
NewWaveform = "my_ripple_plugin.new_waveform:NewWaveform"
```

The entry-point value is `module.path:ClassName` — it must resolve, on load, to a `Waveform` subclass.
The key (`NewWaveform` here) becomes the registry name *if* the target class doesn't already carry an `@register(...)` name of its own — in practice, since the class above already self-registers via its own decorator, the entry-point key mainly needs to be unique and descriptive; what matters is that `ep.load()` returns the class.

## What happens on install

Once `pip install`ed alongside ripple, the plugin is discovered the first time anything triggers `ripplegw.registry.load_plugins()` — which happens automatically inside `ripplegw.waveform(...)`, `list_waveforms(...)`, and `get_waveform_metadata(...)`, and once at `import ripplegw`.
No explicit registration step is needed on the user's side:

```python
import ripplegw

ripplegw.waveform("NewWaveform", f_ref=20.0)   # works immediately after pip install
```

Three things worth knowing about discovery, from [Architecture](architecture.md#registry-mechanics):

- **In-tree registrations always win.**
  If a plugin registers a name ripple already ships, the in-tree version is used and the plugin's registration is silently skipped — don't reuse a built-in name.
- **A broken plugin doesn't break discovery for everyone else.**
  If `ep.load()` raises, or the loaded object isn't a `Waveform` subclass, `load_plugins()` emits a `RuntimeWarning` naming the plugin and continues to the next one.
- **Discovery is idempotent** — `load_plugins()` only does the entry-point scan once per process.

## Testing your plugin locally

Install it editable alongside ripple (`uv pip install -e .` or `pip install -e .` from the plugin's directory) and confirm it's discovered:

```python
import ripplegw

assert "NewWaveform" in ripplegw.list_waveforms()
wf = ripplegw.waveform("NewWaveform", f_ref=20.0)
```

If it doesn't show up, check `python -c "from importlib.metadata import entry_points; print(list(entry_points(group='ripplegw.waveforms')))"` to confirm the entry point was actually installed (a common cause: forgetting to reinstall after editing `pyproject.toml`).
