#!/bin/bash
# Build the sweep harness against the lalsuite wheel dylibs and run the figure
# script, all on the macmini (macOS arm64). Accuracy only.
set -e
cd ~/ripple_cw
PY=venv/bin/python

DYL=$("$PY" -c "import lal,os; print(os.path.join(os.path.dirname(lal.__file__),'.dylibs'))")
echo "dylib dir: $DYL"
ls "$DYL" | grep -E 'liblal(pulsar|support)?\.|liblal\.' || ls "$DYL" | head

# Link directly against the dylib files (macOS ld has no -l:); add rpath and
# rely on DYLD_FALLBACK_LIBRARY_PATH at runtime for @rpath/@loader_path deps.
clang -O2 -Wall harness_sweep.c -o harness_sweep \
  "$DYL"/liblalpulsar.*.dylib "$DYL"/liblal.*.dylib "$DYL"/liblalsupport.*.dylib \
  -Wl,-rpath,"$DYL" -lm
echo "compiled harness_sweep"
otool -L harness_sweep | head

export DYLD_FALLBACK_LIBRARY_PATH="$DYL"
JAX_ENABLE_X64=1 PYTHONPATH="$HOME/ripple_cw/src" \
  "$PY" make_figs.py ephem/earth00-40-DE405.dat.gz ephem/sun00-40-DE405.dat.gz
echo "FIGS_DONE"
ls -la cw_overlap_*.png
