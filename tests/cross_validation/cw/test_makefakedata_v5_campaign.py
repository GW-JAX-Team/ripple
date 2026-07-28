"""Large-scale CW-vs-``CWMakeFakeData`` campaign: fits the overlap-loss-vs-f0 scaling law.

``test_makefakedata_v5.py`` checks agreement at one fixed point (f0=12.3 Hz), enough
for a fast CI regression check but not enough to characterize the floor's actual
behavior. The overlap loss vs. ``CWMakeFakeData`` is **not a flat floor -- it scales
as roughly f0**2** (see ``campaign.py``'s module docstring for the analysis: traced
through the LALSuite source plus four differential experiments ruling out sampling
rate, duration, source-table cadence, and a ripple-side bug). This file runs the
comparison over a random draw of ``--n-samples`` trials spanning sky position,
amplitude parameters, f0 (log-uniform, 10-2000 Hz -- the typical CW all-sky search
band), spindown, detector site, and (for roughly half) binary orbital elements, then
fits ``loss(f0) ~= C * (f0/100Hz)**exponent`` per population. ``test_makefakedata_v5.py``'s
frequency-scaled ``_threshold(f0)`` is calibrated from this fit.

Not run by default CI (expensive at any useful ``--n-samples``); intended for
HPC-scale runs via ``submit_slurm.sh``. Uses the same ``--n-samples``/``--outdir``/
``--plots`` CLI options as the frequency-domain campaign in ``cross_validation/campaign.py``
(registered in ``tests/conftest.py``), so e.g.::

    pytest tests/cross_validation/cw/test_makefakedata_v5_campaign.py \\
        --n-samples 500 --outdir accuracy-results/cw --plots

Skipped, like every other file in this directory, unless both ``lalpulsar`` and an
Earth/Sun ephemeris file are available.
"""

from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

lal = pytest.importorskip("lal")
lalpulsar = pytest.importorskip("lalpulsar")

from tests.cross_validation.cw._lal_helpers import find_ephemeris
from tests.cross_validation.cw.campaign import (
    plot_results,
    run_campaign,
    summarize,
    write_results,
)

EARTH_FILE, SUN_FILE = find_ephemeris()
pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.skipif(
        EARTH_FILE is None or SUN_FILE is None,
        reason="LALPulsar Earth and Sun ephemeris files required",
    ),
]

# Generous sanity ceiling, *not* the precision floor itself: this campaign's job is
# to *measure* the f0-dependent floor (see summary/fit output below), not assert a
# fixed bound against it. At f0 up to 2000 Hz the floor itself reaches ~1e-3 to
# ~1e-2 (see campaign.py); this only catches genuine blow-ups (bugs, NaNs), not the
# known, expected scaling.
_SANITY_CEILING = 0.1


def test_makefakedata_v5_campaign(n_samples, accuracy_outdir, make_plots):
    result = run_campaign(n_samples, lal=lal, lalpulsar=lalpulsar)

    outdir = Path(accuracy_outdir)
    results_file = write_results(result, outdir)
    print(f"\n  Results saved to: {results_file}")
    if make_plots:
        fig_file = plot_results(result, outdir)
        print(f"  Figure saved to: {fig_file}")

    summary = summarize(result)
    print(f"\n  CW vs CWMakeFakeData campaign (n={n_samples}):")
    for population in ("isolated", "binary", "overall"):
        s = summary[population]
        if s["n"] == 0:
            print(f"    {population:<10} n=0")
            continue
        fit = s["power_law_fit"]
        print(
            f"    {population:<10} n={s['n']:<6} mean={s['mean']:.2e} "
            f"median={s['median']:.2e} p99={s['p99']:.2e} max={s['max']:.2e}"
        )
        print(
            f"      fit: loss(f0) ~= {fit['coefficient_at_f0_ref']:.2e} * "
            f"(f0/{fit['f0_ref']:.0f}Hz)^{fit['exponent']:.2f}"
        )
    if result.errors:
        print(f"    {len(result.errors)}/{n_samples} trials failed to generate")
        for i, err in list(result.errors.items())[:5]:
            print(f"      trial {i}: {err}")

    if result.errors:
        pytest.fail(
            f"{len(result.errors)}/{n_samples} trials failed to generate: "
            f"{list(result.errors.items())[:5]}"
        )
    assert result.testable.size > 0, "No testable trials"
    assert result.max_loss < _SANITY_CEILING, (
        f"max overlap loss {result.max_loss:.2e} exceeds sanity ceiling "
        f"{_SANITY_CEILING:.2e} -- this is well beyond the expected f0^2 floor, "
        f"likely a genuine bug rather than the known precision limit"
    )
