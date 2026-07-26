# Installation

The simplest way to install ripple is through pip:

```bash
pip install rippleGW
```

This will install the latest stable release and its dependencies.
ripple is built on [JAX](https://github.com/jax-ml/jax).
By default, this installs the CPU version of JAX.
If you have an NVIDIA GPU, install the CUDA-enabled version:

```bash
pip install "rippleGW[cuda]"
```

If you want to install the latest version of ripple, you can clone this repo and install it locally:

```bash
git clone https://github.com/GW-JAX-Team/ripple.git
cd ripple
pip install -e .
```

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment.
After cloning the repository, run `uv sync` to create a virtual environment with all dependencies installed.

## Continuous-wave ephemeris files

The continuous-wave models (`ExactPulsarSignal`, `PulsarSignal`, `BinaryPulsarSignal`) need a JPL solar-system ephemeris to barycenter the signal.
You pass the name of a standard LALPulsar ephemeris file, e.g. `earth00-40-DE405.dat.gz` (and `sun00-40-DE405.dat.gz` for the full and binary models, which include the Shapiro delay).
ripple parses these files itself — it does not import `lal`/`lalpulsar` at runtime.

**You don't need to fetch these files yourself.** If a standard name isn't found locally, ripple downloads it from the [LALSuite repository](https://git.ligo.org/lscsoft/lalsuite/-/tree/master/lalpulsar/lib) the first time a waveform that needs it is constructed, and caches it (under `$XDG_CACHE_HOME/ripplegw/ephemeris`, `~/Library/Caches/ripplegw/ephemeris` on macOS, or `$RIPPLEGW_CACHE_DIR` if set) for reuse on later runs — no repeat downloads, no manual setup.
If you already have a copy (e.g. from an installed LALSuite, `$LALPULSAR_DATADIR`), just pass its path directly and ripple uses it as-is, with no network access.
Either way, make sure your observation span lies within the file's coverage (the standard `earth00-40-*`/`sun00-40-*` files cover GPS years 2000–2040).
