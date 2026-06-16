#!/usr/bin/env python3
"""
Summarise the shear field as a function of halo shape across a sweep produced
by ``_excalibur_runs/run_shear_shape_sweep.py``.

For each ``.npz`` in the sweep directory it reports the shear MAGNITUDE
|gamma| = sqrt(gamma1^2 + gamma2^2) -- which is invariant under any rotation
of the Sachs screen basis, so it carries no sign/convention ambiguity (unlike
the tangential/cross split, which depends on the per-photon basis alignment).

Statistics are taken in radial annuli on the lens plane so shapes are compared
fairly in the weak / intermediate-shear regime rather than at the (grid-noisy)
strong-lensing cusp:

    core    : r < r_s
    weak    : r_s <= r < R_200      <-- the science-relevant shear estimate
    outer   : r >= R_200

Usage:
    python analyze_shear_shape_sweep.py [--subdir shear_shape_sweep] [--csv out.csv]
"""

import argparse
import csv
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.abspath(os.path.join(HERE, "..", "_data", "output"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subdir", default="shear_shape_sweep")
    p.add_argument("--csv", default=None, help="Optional CSV output path.")
    return p.parse_args()


def shape_tag(d):
    ba = float(d["axis_ratio_ba"]); ca = float(d["axis_ratio_ca"])
    if np.isclose(ba, 1) and np.isclose(ca, 1):
        kind = "sphere"
    elif np.isclose(ba, 1):
        kind = "oblate"
    elif np.isclose(ba, ca):
        kind = "prolate"
    else:
        kind = "triaxial"
    R = np.asarray(d["halo_rotation_matrix"], float)
    orient = "aligned" if np.allclose(R, np.eye(3)) else "rotated"
    return kind, orient


def annulus_stats(r, g, lo, hi):
    m = (r >= lo) & (r < hi) & np.isfinite(g)
    if not m.any():
        return (np.nan, np.nan, np.nan, 0)
    gg = np.abs(g[m])
    return (float(np.median(gg)), float(np.mean(gg)), float(gg.max()), int(m.sum()))


def main():
    args = parse_args()
    sweep_dir = os.path.join(OUTPUT_ROOT, args.subdir)
    files = sorted(glob.glob(os.path.join(sweep_dir, "*.npz")))
    if not files:
        raise SystemExit(f"No .npz found in {sweep_dir}")

    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        label = str(d["display_label"]) or os.path.basename(f)
        ba = float(d["axis_ratio_ba"]); ca = float(d["axis_ratio_ca"])
        rs = float(d["r_s_Mpc"]); R200 = float(d["R_200_Mpc"])
        kind, orient = shape_tag(d)

        b1 = np.asarray(d["b1_map_Mpc"], float)
        b2 = np.asarray(d["b2_map_Mpc"], float)
        r = np.hypot(b1, b2)
        g = np.asarray(d["gamma_map"], float)
        k = np.asarray(d["kappa_map"], float)

        weak = annulus_stats(r, g, rs, R200)
        core = annulus_stats(r, g, 0.0, rs)
        outer = annulus_stats(r, g, R200, np.inf)

        # sanity: lens actually detected (peak kappa well above background)
        k_bg = float(np.asarray(d["kappa_profile"], float)[-1])
        dk_max = float(np.nanmax(np.abs(np.asarray(d["kappa_profile"], float) - k_bg)))

        rows.append(dict(
            label=label, kind=kind, orient=orient, ba=ba, ca=ca,
            rs=rs, R200=R200, k_bg=k_bg, dk_max=dk_max,
            g_weak_med=weak[0], g_weak_mean=weak[1], g_weak_max=weak[2], n_weak=weak[3],
            g_core_med=core[0], g_core_max=core[2],
            g_outer_med=outer[0],
        ))

    # ------- print table -------
    rows.sort(key=lambda x: (x["kind"], -x["ba"], -x["ca"], x["orient"]))
    print(f"\nShear-vs-shape summary  ({len(rows)} runs)   dir={sweep_dir}")
    print("  |gamma| statistics per radial annulus on the lens plane.")
    print("  'weak' annulus (r_s<r<R_200) is the science-relevant shear estimate.\n")
    hdr = (f"  {'label':24s} {'b/a':>4s} {'c/a':>4s} | "
           f"{'<|g|>weak':>10s} {'med weak':>9s} {'max weak':>9s} | "
           f"{'med core':>9s} {'dk_max':>8s} {'sane?':>5s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for x in rows:
        sane = "yes" if x["dk_max"] > 10 * max(x["k_bg"], 1e-9) else "NO!"
        print(f"  {x['label']:24s} {x['ba']:4.2f} {x['ca']:4.2f} | "
              f"{x['g_weak_mean']:10.4e} {x['g_weak_med']:9.3e} {x['g_weak_max']:9.3e} | "
              f"{x['g_core_med']:9.3e} {x['dk_max']:8.2e} {sane:>5s}")

    # reference row = sphere, for relative shear bias
    sph = next((x for x in rows if x["kind"] == "sphere"), None)
    if sph and np.isfinite(sph["g_weak_mean"]) and sph["g_weak_mean"] > 0:
        print(f"\n  Relative <|gamma|>_weak vs sphere ({sph['g_weak_mean']:.3e}):")
        for x in rows:
            if np.isfinite(x["g_weak_mean"]):
                print(f"    {x['label']:24s} {x['g_weak_mean']/sph['g_weak_mean']:+.3f}x")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n  CSV written: {args.csv}")


if __name__ == "__main__":
    main()
