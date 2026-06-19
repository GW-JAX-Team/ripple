"""log10 overlap-loss figures: ripple CW signals vs compiled LALPulsar.

Two stages, decoupled so figure styling can be iterated cheaply:

  compute  (needs lalsuite + jax + ripple + the compiled ./harness_sweep):
      draw random parameter sets, run the compiled XLAL functions, reconstruct
      the ripple strain, compute the white-inner-product overlap loss, and write
      the per-set results to a CSV cache.

  plot     (needs only numpy + matplotlib):
      read the CSV cache and render the figures.

Usage:
  # full run (compute + cache + plot)
  python make_figs.py <earth> <sun>
  # re-plot only, e.g. after tweaking the plotting code or style
  python make_figs.py --from-cache cw_sweep_results.csv
  # force recompute even if the cache exists
  python make_figs.py <earth> <sun> --recompute

The cache is written to --cache (default cw_sweep_results.csv) and figures to
--outdir (default the current directory).
"""

import argparse
import csv
import math
import struct
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

START_GPS, DURATION, FS = 1_000_000_000, 1800, 16.0
N = int(FS * DURATION)
DT = 1.0 / FS
FIELDS = ["kind", "alpha", "delta", "f0", "f1", "f2", "phi0", "psi", "aplus",
          "across", "asini", "ecc", "period", "argp", "tp", "fhet",
          "overlap_loss", "log10_overlap_loss"]


def _overlap_loss(h1, h2):
    h1 = np.asarray(h1, float)
    h2 = np.asarray(h2, float)
    a, b, c = float(h1 @ h1), float(h2 @ h2), float(h1 @ h2)
    d = math.sqrt(a * b)
    return max((a * b - c * c) / (d * (d + c)), 0.0)


def _draw(rng, n, kind):
    rows = []
    for _ in range(n):
        r = dict(
            kind=kind,
            alpha=rng.uniform(0, 2 * np.pi),
            delta=math.asin(rng.uniform(-1, 1)),  # uniform on the sphere
            f0=rng.uniform(10.0, 500.0),
            f1=-10 ** rng.uniform(-12, -8),
            f2=rng.uniform(-1e-17, 1e-17),
            phi0=rng.uniform(0, 2 * np.pi),
            psi=rng.uniform(0, np.pi),
            aplus=1.0,
            across=rng.uniform(0.2, 1.0),
            asini=0.0, ecc=0.0, period=0.0, argp=0.0, tp=0.0, fhet=0.0,
        )
        r["mode"] = 0 if kind == "exact" else 1
        if kind == "binary":
            r.update(asini=rng.uniform(0.5, 3.0), ecc=rng.uniform(0.0, 0.5),
                     period=rng.uniform(3600.0, 86400.0),
                     argp=rng.uniform(0, 2 * np.pi),
                     tp=START_GPS + rng.uniform(0, DURATION))
        rows.append(r)
    return rows


def compute(earth, sun, n_exact=200, n_generate=100, n_binary=100):
    """Run the compiled XLAL sweep + ripple, return per-set rows with losses."""
    # Heavy deps imported lazily so plot-only mode needs only numpy/matplotlib.
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import lal
    import lalpulsar

    from ripplegw.cw.detectors import get_detector
    from ripplegw.cw.ephemeris import read_ephemeris_file
    from ripplegw.cw.pulsar_signal import (
        exact_pulsar_polarizations,
        generate_binary_pulsar_polarizations,
        generate_pulsar_polarizations,
    )

    rng = np.random.default_rng(20240619)
    sets = (_draw(rng, n_exact, "exact") + _draw(rng, n_generate, "generate")
            + _draw(rng, n_binary, "binary"))

    eph = read_ephemeris_file(earth)
    seph = read_ephemeris_file(sun)
    loc = get_detector("H1").location
    edat = lalpulsar.InitBarycenter(earth, sun)
    det = lal.CachedDetectors[lal.LALDetectorIndexLHODIFF]
    eE = (eph.gps0, eph.dt, jnp.asarray(eph.pos), jnp.asarray(eph.vel),
          jnp.asarray(eph.acc))
    eS = (seph.gps0, seph.dt, jnp.asarray(seph.pos), jnp.asarray(seph.vel),
          jnp.asarray(seph.acc))

    ts = lalpulsar.CreateTimestampVector(N)
    for i in range(N):
        g = START_GPS + i * DT
        ts.data[i] = lal.LIGOTimeGPS(int(g // 1), int(round((g % 1) * 1e9)))
    ts.deltaT = DT
    detstates = lalpulsar.GetDetectorStates(ts, det, edat, 0.0)
    t_rel = jnp.arange(N, dtype=jnp.float64) * DT

    def antenna(alpha, delta, psi):
        sk = lal.SkyPosition()
        sk.system = lal.COORDINATESYSTEM_EQUATORIAL
        sk.longitude, sk.latitude = float(alpha), float(delta)
        am = lalpulsar.ComputeAMCoeffs(detstates, sk)
        a = np.array(am.a.data, float)
        b = np.array(am.b.data, float)
        c2, s2 = math.cos(2 * psi), math.sin(2 * psi)
        return a * c2 + b * s2, b * c2 - a * s2  # F+, Fx (sinZeta=1 for H1)

    # Locate the compiled harness (a build artifact): prefer one in the current
    # directory, else next to this script. Use absolute paths throughout so the
    # script is not tied to a particular working directory; the temp CSV/bin go
    # alongside the chosen harness binary.
    here = Path(__file__).resolve().parent
    candidates = [Path.cwd() / "harness_sweep", here / "harness_sweep"]
    harness = next((c for c in candidates if c.exists()), candidates[0])
    work = harness.parent
    csv_path = work / "sweep_params.csv"
    bin_path = work / "sweep_out.bin"

    # write CSV for the C harness and run it
    with open(csv_path, "w") as f:
        for r in sets:
            f.write(" ".join(f"{r[k]:.17g}" if k != "mode" else f"{r['mode']}"
                             for k in ["mode", "alpha", "delta", "f0", "f1", "f2",
                                       "phi0", "psi", "aplus", "across", "asini",
                                       "ecc", "period", "argp", "tp", "fhet"]) + "\n")
    print("running compiled-LAL sweep harness ...", flush=True)
    subprocess.run([str(harness), earth, sun, str(csv_path), str(bin_path)],
                   check=True)
    with open(bin_path, "rb") as f:
        n_sets, n_samp = struct.unpack("II", f.read(8))
        assert n_sets == len(sets) and n_samp == N
        lal_h = np.frombuffer(f.read(4 * n_sets * n_samp), dtype=np.float32)
        lal_h = lal_h.astype(np.float64).reshape(n_sets, n_samp)

    print("computing ripple strains + overlap losses ...", flush=True)
    for i, r in enumerate(sets):
        fkdot = (r["f1"], r["f2"])
        if r["kind"] == "exact":
            hp, hc = exact_pulsar_polarizations(
                t_rel, START_GPS, r["alpha"], r["delta"], r["f0"], r["phi0"],
                r["aplus"], r["across"], loc, *eE, fkdot=fkdot)
        elif r["kind"] == "generate":
            hp, hc = generate_pulsar_polarizations(
                t_rel, START_GPS, r["alpha"], r["delta"], r["f0"], r["phi0"],
                r["aplus"], r["across"], loc, *eE, *eS, fkdot=fkdot,
                f_heterodyne=r["fhet"])
        else:
            hp, hc = generate_binary_pulsar_polarizations(
                t_rel, START_GPS, r["alpha"], r["delta"], r["f0"], r["phi0"],
                r["aplus"], r["across"], r["asini"], r["ecc"], r["period"],
                r["argp"], r["tp"], loc, *eE, *eS, fkdot=fkdot,
                f_heterodyne=r["fhet"])
        fp, fc = antenna(r["alpha"], r["delta"], r["psi"])
        h_me = fp * np.array(hp) + fc * np.array(hc)
        r["overlap_loss"] = _overlap_loss(h_me, lal_h[i])
        r["log10_overlap_loss"] = (math.log10(r["overlap_loss"])
                                   if r["overlap_loss"] > 0 else -17.0)
    return sets


def save_cache(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"cached {len(rows)} results -> {path}")


def load_cache(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k in FIELDS:
                if k != "kind":
                    r[k] = float(r[k])
            rows.append(r)
    print(f"loaded {len(rows)} results from {path}")
    return rows


# ---------------------------------------------------------------------------
# Plotting (edit freely to restyle — needs only the CSV cache)
# ---------------------------------------------------------------------------
def _fig(rows, kind, title, fname):
    rs = [r for r in rows if r["kind"] == kind]
    log10 = np.array([r["log10_overlap_loss"] for r in rs])
    f0 = np.array([r["f0"] for r in rs])
    alpha = np.array([r["alpha"] for r in rs])
    delta = np.array([r["delta"] for r in rs])
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=13)

    ax[0, 0].hist(log10, bins=25, edgecolor="black", alpha=0.8)
    ax[0, 0].axvline(np.median(log10), color="red", ls="--",
                     label=f"median = {np.median(log10):.2f}")
    ax[0, 0].set_xlabel(r"$\log_{10}(1-\mathcal{O})$")
    ax[0, 0].set_ylabel("count")
    ax[0, 0].set_title("overlap-loss distribution")
    ax[0, 0].legend()

    sc = ax[0, 1].scatter(f0, delta, c=log10, cmap="viridis", s=24)
    ax[0, 1].set_xlabel(r"$f_0$ [Hz]")
    ax[0, 1].set_ylabel(r"$\delta$ [rad]")
    ax[0, 1].set_title(r"$f_0$ vs $\delta$")
    fig.colorbar(sc, ax=ax[0, 1], label=r"$\log_{10}(1-\mathcal{O})$")

    sc = ax[1, 0].scatter(alpha, delta, c=log10, cmap="viridis", s=24)
    ax[1, 0].set_xlabel(r"$\alpha$ [rad]")
    ax[1, 0].set_ylabel(r"$\delta$ [rad]")
    ax[1, 0].set_title("sky position")
    fig.colorbar(sc, ax=ax[1, 0], label=r"$\log_{10}(1-\mathcal{O})$")

    ax[1, 1].scatter(f0, log10, s=24, alpha=0.7)
    ax[1, 1].set_xlabel(r"$f_0$ [Hz]")
    ax[1, 1].set_ylabel(r"$\log_{10}(1-\mathcal{O})$")
    ax[1, 1].set_title(r"overlap loss vs $f_0$")
    if kind == "exact":
        ulp = float(np.spacing(float(START_GPS)))
        ff = np.linspace(max(f0.min(), 1.0), f0.max(), 200)
        ax[1, 1].plot(ff, np.log10((2 * np.pi * ff * ulp) ** 2 / 24.0), "r-",
                      lw=2, label=r"LAL REAL8 GPS floor $\propto f_0^2$")
        ax[1, 1].legend(loc="lower right", fontsize=9)
    else:
        note = ("floor: LAL PulsarSimulateCoherentGW interpolation\n"
                r"(sourceDeltaT, dt$_{\rm delay}$, dt$_{\rm pol}$); ripple per-sample"
                ) if kind == "generate" else (
                "floor: LAL GenerateSpinOrbitCW Kepler tolerance\n"
                r"$\delta x_{\max}=0.01/(f_0 P)$; ripple solves to machine precision")
        ax[1, 1].text(0.03, 0.97, note, transform=ax[1, 1].transAxes, va="top",
                      ha="left", fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                                alpha=0.8))
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def plot(rows, outdir):
    import os

    os.makedirs(outdir, exist_ok=True)
    for kind, title, fname in [
        ("exact", "ripple vs compiled XLALSimulateExactPulsarSignal (H1)",
         "cw_overlap_exact.png"),
        ("generate", "ripple vs compiled XLALGeneratePulsarSignal — isolated (H1)",
         "cw_overlap_generate.png"),
        ("binary", "ripple vs compiled XLALGeneratePulsarSignal — binary orbit (H1)",
         "cw_overlap_binary.png"),
    ]:
        _fig(rows, kind, title, f"{outdir}/{fname}")
    for kind in ("exact", "generate", "binary"):
        ls = [r["log10_overlap_loss"] for r in rows if r["kind"] == kind]
        print(f"  {kind:9s}: n={len(ls)}  log10 overlap loss  "
              f"median={np.median(ls):.2f}  max={np.max(ls):.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("earth", nargs="?", help="Earth ephemeris (compute mode)")
    p.add_argument("sun", nargs="?", help="Sun ephemeris (compute mode)")
    p.add_argument("--cache", default="cw_sweep_results.csv")
    p.add_argument("--from-cache", help="plot only, from this CSV")
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--outdir", default=".")
    args = p.parse_args()

    if args.from_cache:
        rows = load_cache(args.from_cache)
    else:
        import os
        if os.path.exists(args.cache) and not args.recompute:
            rows = load_cache(args.cache)
        else:
            assert args.earth and args.sun, "earth/sun ephemeris required to compute"
            rows = compute(args.earth, args.sun)
            save_cache(rows, args.cache)
    plot(rows, args.outdir)
    print("done.")


if __name__ == "__main__":
    main()
