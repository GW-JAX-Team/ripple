# Comparison against the compiled LALPulsar functions

> The metric is the **overlap loss** between ripple's reconstructed detector
> strain and the compiled-LAL strain — not byte-level equality.

`XLALSimulateExactPulsarSignal` and `XLALGeneratePulsarSignal` take a
`PulsarSignalParams` argument that **swiglal does not wrap** — it contains
anonymous nested structs (`struct { … } pulsar;` / `orbit;`), so the type is not
registered as a Python class and the functions cannot be invoked from Python.

To validate ripple against the *actual compiled* functions (not just the
SWIG-exposed building blocks `XLALBarycenter` / `XLALComputeAMCoeffs` /
`XLALGenerateSpinOrbitCW`), `harness.c` declares the needed structs itself
(self-verifying the layout via the H1 detector location), calls the compiled
functions, and dumps each `REAL4TimeSeries` to a small binary file. `compare.py`
then reconstructs the detector strain from ripple's `{p, c}` polarizations and
LAL's own antenna response, and reports the relative difference.

This is a manual, environment-dependent check (it needs `lalsuite` installed and
JPL ephemeris files), so it is **not** collected as a pytest test.

## Recipe

```bash
# 1. A Python env with lalsuite (provides the .so libraries and SWIG lalpulsar)
python -m venv lalenv && . lalenv/bin/activate
pip install lalsuite numpy "jax[cpu]" jaxtyping

# 2. Earth/Sun ephemerides (the pip wheel does NOT bundle them); e.g. from the
#    LALSuite mirror:
base=https://git.ligo.org/lscsoft/lalsuite/-/raw/master/lalpulsar/lib
curl -LO $base/earth00-40-DE405.dat.gz
curl -LO $base/sun00-40-DE405.dat.gz

# 3. Build the harness against the wheel's shared libraries (no LAL headers needed)
LIBDIR=$(python -c "import lalpulsar,os;print(os.path.dirname(os.path.dirname(lalpulsar.__file__)))")/lalsuite.libs
gcc -O2 -Wall harness.c -o harness \
    -L"$LIBDIR" \
    -l:"$(cd $LIBDIR && ls liblalpulsar-*.so*)" \
    -l:"$(cd $LIBDIR && ls liblal-*.so*)" \
    -l:"$(cd $LIBDIR && ls liblalsupport-*.so*)" \
    -Wl,-rpath,"$LIBDIR" -lm

# 4. Run the compiled XLAL functions and dump their output
./harness earth00-40-DE405.dat.gz sun00-40-DE405.dat.gz \
    out_exact.bin out_gen0.bin out_genhet.bin

# 5. Compare against ripple's JAX implementation
JAX_ENABLE_X64=1 PYTHONPATH=/path/to/ripple/src \
    python compare.py earth00-40-DE405.dat.gz sun00-40-DE405.dat.gz \
        out_exact.bin out_gen0.bin out_genhet.bin
```

## Reference result

The figure of merit is the **overlap loss** (mismatch) `1 − ⟨h₁|h₂⟩/√(⟨h₁|h₁⟩⟨h₂|h₂⟩)`
between the reconstructed strain and the compiled-LAL strain — the same metric
(and numerically-stable form) used by `test_lal_overlap.py`, reported as `log10`.

```
EXACT       vs compiled XLALSimulateExactPulsarSignal: overlap loss = 3.4e-13  log10 = -12.46
GENERATE fHet= 0.0 vs compiled XLALGeneratePulsarSignal:  overlap loss = 9.9e-13  log10 = -12.00
GENERATE fHet=12.0 vs compiled XLALGeneratePulsarSignal:  overlap loss = 9.9e-13  log10 = -12.00
```

The exact-signal floor (log10 ≈ −12.5) is set by the float64 GPS-time precision
of LAL's own per-sample phase (`t ≈ 1e9`); ripple's int+frac time split is in
fact more precise. The generate floor is LAL's internal interpolation error (it
shrinks as `sourceDeltaT` / `dtDelayBy2` / `dtPolBy2` are reduced); ripple
computes the phase directly per sample and matches the per-sample
`XLALBarycenter` truth to overlap loss well below this.

## Parameter-sweep figures (`make_figs.py`, `harness_sweep.c`)

`harness_sweep.c` runs the compiled functions over many parameter sets read from
a CSV; `make_figs.py` draws random parameters (isolated exact / isolated
generate / binary), drives the harness, reconstructs the ripple strain, and
plots `log10` overlap loss vs sky position and frequency. Output PNGs are in
`figures/`. `run_macmini.sh` is the macOS (Apple-Silicon) build+run recipe used
to generate them.

Key point for the PR: the overlap loss is bounded by **LAL's own reference
precision**, not ripple's:

* **isolated** (`cw_overlap_exact.png`, `cw_overlap_generate.png`): the loss
  follows the analytic floor `1 − O ≈ (2π f₀ · ulp(t))² / 24` (overlaid in red),
  from LAL evaluating the phase in REAL8 GPS time (`t ≈ 1e9`). It grows ∝ f₀²
  and is sky-position independent. ripple's int+frac time arithmetic sits on or
  below this floor.
* **binary** (`cw_overlap_binary.png`): the floor is LAL `GenerateSpinOrbitCW`'s
  Kepler tolerance `dxMax = 0.01/(f₀·P)` (~0.01 rad → 1−O ~ 5e-5). ripple solves
  Kepler to machine precision.

So across the parameter space ripple reproduces the compiled LAL functions to
LAL's own numerical floor (and is intrinsically more accurate).
