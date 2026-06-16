#!/usr/bin/env python3
"""
Analyse the RADIAL lensing profiles gamma(b), kappa(b), mu(b) across the
shear-vs-shape sweep (``_excalibur_runs/run_shear_shape_sweep.py``).

The profile photons sample a single azimuthal cut on the sky (along the
screen axis e_perp1). For ALIGNED (broadside) oblate/triaxial halos that cut
runs along the projected MAJOR axis; for END-ON prolate the projection is
axisymmetric so the cut is representative; keep this in mind when reading the
elliptical cases -- the cut is one slice, not an azimuthal average.

For each shape it extracts, in the pipeline's (conformal/RT) convention:
  * gamma(b) at reference radii (interpolated): the shear estimate vs radius.
  * the WEAK-regime logarithmic slope  d ln|gamma| / d ln b  on r_s<b<R_200.
  * the tangential critical (Einstein) radius b_E along the cut, from the
    outermost zero of the tangential eigenvalue lambda_t = 1 - kappa - gamma
    (where mu diverges), and the radial critical radius from
    lambda_r = 1 - kappa + gamma.

Outputs a table, a CSV, and PNG figures (gamma & kappa profiles grouped by
shape family, plus the tangential eigenvalue / Einstein radii).

Usage:
    python analyze_shear_profiles.py [--subdir shear_shape_sweep] [--no-plots]
"""

import argparse
import csv
import glob
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.abspath(os.path.join(HERE, "..", "_data", "output"))

REF_RADII = [0.1, 0.2, 0.3, 0.5, 1.0, 1.5]  # Mpc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subdir", default="shear_shape_sweep")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def shape_kind(ba, ca, aligned):
    if np.isclose(ba, 1) and np.isclose(ca, 1):
        return "sphere"
    if np.isclose(ba, 1):
        return "oblate"
    if np.isclose(ba, ca):
        return "prolate"
    return "triaxial"


def clean_profile(b, *arrs):
    """Sort by b and average exact-duplicate b samples; returns (b, *arrs)."""
    order = np.argsort(b)
    b = b[order]
    arrs = [a[order] for a in arrs]
    ub, inv = np.unique(np.round(b, 9), return_inverse=True)
    out = []
    for a in arrs:
        acc = np.zeros_like(ub, float)
        cnt = np.zeros_like(ub, float)
        np.add.at(acc, inv, a)
        np.add.at(cnt, inv, 1.0)
        out.append(acc / cnt)
    return (ub, *out)


def outermost_zero(b, f):
    """Largest b where f changes sign, linearly interpolated. NaN if none."""
    s = np.sign(f)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size == 0:
        return np.nan
    i = idx[-1]
    b0, b1, f0, f1 = b[i], b[i + 1], f[i], f[i + 1]
    return float(b0 - f0 * (b1 - b0) / (f1 - f0))


def loginterp(b, y, x):
    """Interpolate y at radius x using log-b axis (y may be signed)."""
    m = b > 0
    return float(np.interp(np.log(x), np.log(b[m]), y[m]))


def weak_slope(b, g, rs, R200):
    m = (b >= rs) & (b <= R200) & (g > 0)
    if m.sum() < 3:
        return np.nan
    coef = np.polyfit(np.log(b[m]), np.log(g[m]), 1)
    return float(coef[0])


def main():
    args = parse_args()
    sweep_dir = os.path.join(OUTPUT_ROOT, args.subdir)
    files = sorted(glob.glob(os.path.join(sweep_dir, "*.npz")))
    if not files:
        raise SystemExit(f"No .npz found in {sweep_dir}")

    runs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        ba = float(d["axis_ratio_ba"]); ca = float(d["axis_ratio_ca"])
        R = np.asarray(d["halo_rotation_matrix"], float)
        aligned = np.allclose(R, np.eye(3))
        b, k, g, mu = clean_profile(
            np.asarray(d["b_profile_Mpc"], float),
            np.asarray(d["kappa_profile"], float),
            np.asarray(d["gamma_profile"], float),
            np.asarray(d["mu_profile"], float),
        )
        lam_t = 1.0 - k - g
        lam_r = 1.0 - k + g
        rs = float(d["r_s_Mpc"]); R200 = float(d["R_200_Mpc"])
        runs.append(dict(
            label=str(d["display_label"]) or os.path.basename(f),
            kind=shape_kind(ba, ca, aligned), ba=ba, ca=ca,
            aligned=aligned, rs=rs, R200=R200,
            b=b, k=k, g=g, mu=mu, lam_t=lam_t, lam_r=lam_r,
            b_E=outermost_zero(b, lam_t),
            b_rad=outermost_zero(b, lam_r),
            slope=weak_slope(b, g, rs, R200),
            g_at={r: loginterp(b, g, r) for r in REF_RADII},
            g_rs=loginterp(b, g, rs), g_R200=loginterp(b, g, R200),
            k_bg=float(k[-1]),
        ))

    order = {"sphere": 0, "oblate": 1, "prolate": 2, "triaxial": 3}
    runs.sort(key=lambda x: (order[x["kind"]], -x["ba"], -x["ca"], x["aligned"]))

    # ---------------- table ----------------
    print(f"\nRadial shear-profile analysis  ({len(runs)} runs)   dir={sweep_dir}")
    print("  Conformal/RT convention; profile is a single cut along e_perp1.")
    print(f"  r_s={runs[0]['rs']:.3f}*  R_200 varies with shape (M_200 fixed).\n")
    cols = "  ".join(f"g({r})" for r in REF_RADII)
    print(f"  {'label':24s} {'b_E':>6s} {'slope':>6s} | " +
          "  ".join(f"{'g(%.1f)'%r:>8s}" for r in REF_RADII))
    print("  " + "-" * 96)
    for x in runs:
        gline = "  ".join(f"{x['g_at'][r]:8.4f}" for r in REF_RADII)
        bE = f"{x['b_E']:.3f}" if np.isfinite(x["b_E"]) else "  -  "
        print(f"  {x['label']:24s} {bE:>6s} {x['slope']:6.2f} | {gline}")

    sph = next(x for x in runs if x["kind"] == "sphere")
    print(f"\n  Tangential Einstein radius b_E (Mpc) along the cut, and gamma at 0.3 Mpc"
          f" relative to sphere (b_E={sph['b_E']:.3f}, g={sph['g_at'][0.3]:.4f}):")
    for x in runs:
        rb = x["b_E"] / sph["b_E"] if np.isfinite(x["b_E"]) else np.nan
        rg = x["g_at"][0.3] / sph["g_at"][0.3]
        print(f"    {x['label']:24s} b_E={x['b_E']:.3f} ({rb:+.3f}x)   "
              f"g(0.3)={x['g_at'][0.3]:.4f} ({rg:+.3f}x)   slope={x['slope']:+.2f}")

    # ---------------- CSV ----------------
    csv_path = os.path.join(sweep_dir, "shear_profile_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        fld = ["label", "kind", "ba", "ca", "aligned", "rs", "R200",
               "b_E", "b_rad", "slope", "g_rs", "g_R200"] + [f"g_{r}" for r in REF_RADII]
        w = csv.DictWriter(fh, fieldnames=fld)
        w.writeheader()
        for x in runs:
            row = {k: x[k] for k in ["label", "kind", "ba", "ca", "aligned",
                                     "rs", "R200", "b_E", "b_rad", "slope",
                                     "g_rs", "g_R200"]}
            row.update({f"g_{r}": x["g_at"][r] for r in REF_RADII})
            w.writerow(row)
    print(f"\n  CSV: {csv_path}")

    if args.no_plots:
        return

    # ---------------- figures ----------------
    families = [
        ("sphere+oblate", lambda x: x["kind"] in ("sphere", "oblate")),
        ("prolate orientations", lambda x: x["kind"] in ("sphere", "prolate")),
        ("sphere+triaxial", lambda x: x["kind"] in ("sphere", "triaxial")),
    ]

    # gamma(b)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, (name, sel) in zip(axes, families):
        for x in runs:
            if not sel(x):
                continue
            lw = 2.5 if x["kind"] == "sphere" else 1.4
            ls = "--" if x["kind"] == "sphere" else "-"
            ax.loglog(x["b"], np.abs(x["g"]), ls, lw=lw, label=x["label"])
        ax.axvline(sph["rs"], color="grey", ls=":", lw=0.8)
        ax.axvline(sph["R200"], color="grey", ls=":", lw=0.8)
        ax.set_title(name); ax.set_xlabel("b [Mpc]")
        ax.set_xlim(0.05, 15); ax.set_ylim(1e-3, 1.0)
        ax.legend(fontsize=7, ncol=1)
    axes[0].set_ylabel(r"$|\gamma|$ (conformal)")
    fig.suptitle(r"Shear profile $|\gamma|(b)$ vs halo shape "
                 r"(M$_{200}=10^{15}$, c=5, $z_l$=0.5, $z_s$=2). "
                 r"Dotted: $r_s$, $R_{200}$.", fontsize=11)
    fig.tight_layout()
    p1 = os.path.join(sweep_dir, "shear_profiles.png")
    fig.savefig(p1, dpi=130); plt.close(fig)

    # kappa(b)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, (name, sel) in zip(axes, families):
        for x in runs:
            if not sel(x):
                continue
            lw = 2.5 if x["kind"] == "sphere" else 1.4
            ls = "--" if x["kind"] == "sphere" else "-"
            ax.loglog(x["b"], np.abs(x["k"] - x["k_bg"]), ls, lw=lw, label=x["label"])
        ax.axvline(sph["rs"], color="grey", ls=":", lw=0.8)
        ax.set_title(name); ax.set_xlabel("b [Mpc]"); ax.set_xlim(0.05, 15)
        ax.legend(fontsize=7)
    axes[0].set_ylabel(r"$\kappa-\kappa_{bg}$ (conformal)")
    fig.suptitle(r"Convergence profile $\kappa(b)$ vs halo shape", fontsize=11)
    fig.tight_layout()
    p2 = os.path.join(sweep_dir, "kappa_profiles.png")
    fig.savefig(p2, dpi=130); plt.close(fig)

    # Einstein radius bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [x["label"] for x in runs]
    bE = [x["b_E"] if np.isfinite(x["b_E"]) else 0.0 for x in runs]
    colors = {"sphere": "k", "oblate": "tab:blue",
              "prolate": "tab:red", "triaxial": "tab:green"}
    ax.bar(range(len(runs)), bE, color=[colors[x["kind"]] for x in runs])
    ax.axhline(sph["b_E"], color="grey", ls="--", lw=1, label=f"sphere b_E={sph['b_E']:.3f}")
    ax.set_xticks(range(len(runs))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("tangential Einstein radius $b_E$ [Mpc] (along cut)")
    ax.set_title("Einstein radius vs halo shape/orientation (fixed mass)")
    ax.legend()
    fig.tight_layout()
    p3 = os.path.join(sweep_dir, "einstein_radius_vs_shape.png")
    fig.savefig(p3, dpi=130); plt.close(fig)

    print(f"  Figures:\n    {p1}\n    {p2}\n    {p3}")


if __name__ == "__main__":
    main()
