#!/usr/bin/env python3
"""
PART A -- Weak-lensing FLEXION maps beyond shear.

Following Fleury, Larena & Uzan 2019 (PRD 99, 023525; arXiv:1809.03924),
"Weak lensing distortions beyond shear": an infinitesimal beam can only be
converged and sheared, but the *derivatives* of the shear field already encode
the next distortion modes -- the FLEXIONS:

    first  flexion  F  (spin 1)  = d kappa            = d* gamma
    second flexion  G  (spin 3)  = d gamma

with the complex derivative  d = d/dx + i d/dy  (and  d* = d/dx - i d/dy).
In Fleury's notation F = -2 dgamma/dalpha, G = -2 dgamma/dalpha* (their Eq. 52);
the spin-1/spin-3 character and "F from interior matter, G from exterior matter"
follow directly. F is the spin-1 "pointing"/centroid-shift distortion, G the
spin-3 "triangularity / arckiness" of an image.

These come for FREE from Excalibur's ray-traced maps: the npz stores the full
lensing Jacobian  A = d beta / d theta  ('D_flat_map', row-major [A11,A12,A21,A22]):

    kappa = 1 - tr(A)/2 ,   gamma1 = (A22-A11)/2 ,   gamma2 = -A12 .

FLEXION IS A *THIRD* DERIVATIVE OF THE POTENTIAL (d.kappa ~ d^3 psi). On the
coarse 31x31 ray-traced map (0.1 Mpc pixels) a naive pixel finite-difference is
therefore noise-amplified, and -- because d.kappa and d*.gamma amplify the
(noisier) shear vs the (smoother) convergence differently -- the identity
F = d.kappa = d*.gamma, which holds EXACTLY in the continuum (we verified
|kappa'| = |gamma_t' + 2 gamma_t/r| analytically for any circular lens), is only
recovered to O(1) per pixel. This residual is purely NUMERICAL, not a finite-beam
effect, and is reported as a health metric.

Two robust products are therefore built:
  1. 2D morphology maps from LIGHTLY GAUSSIAN-SMOOTHED kappa/gamma (default
     sigma=1 px). Smoothing collapses the spurious m=4 Cartesian-grid pattern
     (sphere |G| m4: 0.41 -> 0.08) while preserving the real m=2 shape signature.
  2. A quantitative RADIAL reference from the finely-sampled 1D ray-traced
     profile (kappa_profile/gamma_profile, ~100 nodes): azimuthal averaging is
     the natural denoiser, so the azimuthally-averaged map flexion is validated
     against |F|=|kappa'| and |G|=|d|gamma|/dr - 2|gamma|/r|.

Conventions / units:
  * RT (conformal/comoving) convention -- same as the stored kappa/gamma; flexion
    carries the same overall (1+z_l) factor, irrelevant for SHAPE comparisons.
  * Derivatives w.r.t. lens-plane impact parameter b [Mpc] -> |F|,|G| in 1/Mpc.
    (Angular flexion = D_A,l * value; CSV also gives 1/arcmin for reference.)
  * |F|,|G| are spin moduli: invariant under screen-basis rotation, hence -- like
    |gamma| in analyze_shear_shape_sweep.py -- free of basis-alignment ambiguity.
  * Stats use a FIXED clean annulus 0.5 < r < 1.2 Mpc (outside the central cusp,
    inside the 1.5 Mpc map with edge margin). NB R_200 (2-4.6 Mpc) exceeds the
    map, so the sweep's r_s<r<R_200 annulus would spill into the grid corners.

Outputs (all under _data/output/beyond_shear/A_flexion_maps/):
  * <label>_flexion.png    per-shape diagnostic (kappa+F quiver, |F|, |G|, spin-3)
  * compare_G_flexion.png  |G| (triangularity) map across the shape family
  * flexion_radial.png     <|F|>(r),<|G|>(r) map (az-avg) vs 1D-profile reference
  * flexion_summary.csv    weak-annulus stats + consistency residual + m2/m4

Usage:
    python analyze_flexion_maps.py [--subdir shear_shape_sweep] [--smooth 1.0]
                                   [--no-plots]
"""

import argparse
import csv
import glob
import os

import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.abspath(os.path.join(HERE, "..", "_data", "output"))
OUT_DIR = os.path.join(OUTPUT_ROOT, "beyond_shear", "A_flexion_maps")

ARCMIN_PER_RAD = 180.0 * 60.0 / np.pi
ANN_LO, ANN_HI = 0.5, 1.2   # fixed clean annulus [Mpc]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subdir", default="shear_shape_sweep")
    p.add_argument("--jacobian", choices=("betamap", "deviation"), default="betamap",
                   help="betamap (default): FD Jacobian of beta(theta) -- integrable, "
                        "self-consistent flexion. deviation: stored D_flat_map (per-ray, "
                        "inconsistent at flexion order).")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="Gaussian sigma (pixels) for the 2D morphology maps. With the "
                        "betamap Jacobian flexion is self-consistent, so 0 (none) is the "
                        "default; raise it only for the noisier 'deviation' Jacobian.")
    p.add_argument("--no-plots", action="store_true")
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


def load_fields(d, jacobian="betamap"):
    """(b1,b2 grids, dx,dy, kappa,g1,g2) all (n,n).  Verified layout:
    reshape(n,n)[i,j] at (x=b1[i], y=b2[j]); axis0=x-dir, axis1=y-dir.

    jacobian:
      "deviation" -- kappa,gamma from the stored per-ray deviation-equation
        Jacobian 'D_flat_map'. This is the accurate LOCAL shear, but the field
        D_flat_map(theta) is NOT the integrable Jacobian of the sampled lens map
        (it is transported along each ray, beyond-Born), so its kappa and gamma
        are mutually INCONSISTENT at the 3rd-derivative (flexion) level: the
        F = d.kappa = d*.gamma identity fails by O(1) and does NOT improve with
        map resolution.
      "betamap" (default) -- kappa,gamma from the finite-difference Jacobian of
        the ray-traced source-plane map beta(theta).  Integrable by construction,
        so F = d.kappa = d*.gamma holds and CONVERGES with resolution (residual
        0.026 at 31x31 -> 0.002 at 121x121). In the science annulus it reproduces
        the deviation-equation shear to ~1e-5, so it is the correct field to
        differentiate for flexion.
    """
    n = int(d["n_map_1d"])
    b1 = np.asarray(d["b1_map_Mpc"], float).reshape(n, n)
    b2 = np.asarray(d["b2_map_Mpc"], float).reshape(n, n)
    dx = float(np.mean(np.diff(b1[:, 0]))); dy = float(np.mean(np.diff(b2[0, :])))
    if jacobian == "deviation":
        A = np.asarray(d["D_flat_map"], float)
        A11, A12, A22 = A[:, 0], A[:, 1], A[:, 3]
        kappa = (1.0 - 0.5 * (A11 + A22)).reshape(n, n)
        g1 = (0.5 * (A22 - A11)).reshape(n, n)
        g2 = (-A12).reshape(n, n)
        return b1, b2, dx, dy, kappa, g1, g2
    # ---- betamap: FD Jacobian A = d beta_ang / d theta_ang (dimensionless) ----
    DCl = float(d["DA_l_Mpc"]) * (1 + float(d["z_l"]))
    DCs = float(d["DA_s_Mpc"]) * (1 + float(d["z_source"]))
    e1 = np.asarray(d["screen_e1"], float); e2 = np.asarray(d["screen_e2"], float)
    spc = np.asarray(d["source_plane_center_Mpc"], float)
    fin = np.asarray(d["final_pos_map_Mpc"], float)
    be1 = (((fin - spc) @ e1) / DCs).reshape(n, n)   # source angle, radians
    be2 = (((fin - spc) @ e2) / DCs).reshape(n, n)
    dthx = dx / DCl; dthy = dy / DCl                  # theta grid spacing (rad)
    A11, A12 = np.gradient(be1, dthx, dthy)           # dbeta1/dth1, dbeta1/dth2
    A21, A22 = np.gradient(be2, dthx, dthy)
    kappa = 1.0 - 0.5 * (A11 + A22)
    g1 = 0.5 * (A22 - A11)
    g2 = -0.5 * (A12 + A21)                            # symmetrised off-diagonal
    return b1, b2, dx, dy, kappa, g1, g2


def compute_flexion(kappa, g1, g2, dx, dy):
    """F = d.kappa (spin 1), G = d.gamma (spin 3), F_shear = d*.gamma (spin-1
    cross-check).  d = d/dx + i d/dy.  Complex, (n,n)."""
    kx, ky = np.gradient(kappa, dx, dy)
    g1x, g1y = np.gradient(g1, dx, dy)
    g2x, g2y = np.gradient(g2, dx, dy)
    F_kappa = kx + 1j * ky
    G = (g1x - g2y) + 1j * (g2x + g1y)
    F_shear = (g1x + g2y) + 1j * (g2x - g1y)
    return F_kappa, G, F_shear


def profile_reference(d):
    """1D ray-traced radial cut -> reference |F|=|kappa'|, |G|=|d|g|/dr-2|g|/r|.
    Deduplicated and sorted in b. Returns (b, |F|_ref, |G|_ref)."""
    bp = np.asarray(d["b_profile_Mpc"], float)
    kp = np.asarray(d["kappa_profile"], float)
    gp = np.asarray(d["gamma_profile"], float)
    o = np.argsort(bp); bp, kp, gp = bp[o], kp[o], gp[o]
    _, idx = np.unique(np.round(bp, 9), return_index=True)
    bp, kp, gp = bp[idx], kp[idx], gp[idx]
    good = bp > 1e-6
    bp, kp, gp = bp[good], kp[good], gp[good]
    Fref = np.abs(np.gradient(kp, bp))
    Gref = np.abs(np.gradient(gp, bp) - 2.0 * gp / bp)
    return bp, Fref, Gref


def azimuthal_modulation(r, theta, field, lo, hi, m, half):
    sel = (r >= lo) & (r < hi) & (np.abs(r) > 0)
    v = np.abs(field[sel]); th = theta[sel]
    if v.size < 8 or v.mean() <= 0:
        return np.nan
    am = 2.0 * np.mean(v * np.cos(m * th))
    bm = 2.0 * np.mean(v * np.sin(m * th))
    return float(np.hypot(am, bm) / v.mean())


def annulus_sel(r, b1, b2, half, lo=ANN_LO, hi=ANN_HI):
    return (r >= lo) & (r < hi) & (np.abs(b1) < half - 1e-9) & (np.abs(b2) < half - 1e-9)


def family_key(rec):
    order = {"sphere": 0, "oblate": 1, "prolate": 2, "triaxial": 3}
    return (order.get(rec["kind"], 9), -rec["ba"], -rec["ca"], rec["orient_deg"])


def main():
    global OUT_DIR
    args = parse_args()
    # route non-default sweeps to their own output folder so figures aren't clobbered
    if args.subdir != "shear_shape_sweep":
        tag = args.subdir.replace("shear_shape_", "")
        OUT_DIR = os.path.join(OUTPUT_ROOT, "beyond_shear", f"A_flexion_maps_{tag}")
    os.makedirs(OUT_DIR, exist_ok=True)
    sweep_dir = os.path.join(OUTPUT_ROOT, args.subdir)
    files = sorted(glob.glob(os.path.join(sweep_dir, "*.npz")))
    if not files:
        raise SystemExit(f"No .npz found in {sweep_dir}")

    recs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        label = str(d["display_label"]) or os.path.basename(f)[:12]
        ba = float(d["axis_ratio_ba"]); ca = float(d["axis_ratio_ca"])
        rs = float(d["r_s_Mpc"]); R200 = float(d["R_200_Mpc"])
        kind, orient = shape_tag(d)
        Rm = np.asarray(d["halo_rotation_matrix"], float)
        orient_deg = 0 if orient == "aligned" else int(round(
            np.degrees(np.arctan2(Rm[0, 2], Rm[0, 0])))) % 360

        b1, b2, dx, dy, kappa, g1, g2 = load_fields(d, jacobian=args.jacobian)
        half = float(d["map_half_Mpc"])
        r = np.hypot(b1, b2); theta = np.arctan2(b2, b1)

        # raw (for consistency residual) and smoothed (for morphology) flexion
        F_raw, G_raw, Fs_raw = compute_flexion(kappa, g1, g2, dx, dy)
        s = args.smooth
        if s > 0:
            ks, a1, a2 = gaussian_filter(kappa, s), gaussian_filter(g1, s), gaussian_filter(g2, s)
        else:
            ks, a1, a2 = kappa, g1, g2
        F, G, Fs = compute_flexion(ks, a1, a2, dx, dy)

        sel = annulus_sel(r, b1, b2, half)
        Fw = float(np.mean(np.abs(F[sel]))) if sel.any() else np.nan
        Gw = float(np.mean(np.abs(G[sel]))) if sel.any() else np.nan
        # consistency residual on the RAW maps (the honest numerical health metric)
        denom = np.mean(np.abs(F_raw[sel]))
        resid = (float(np.mean(np.abs(F_raw[sel] - Fs_raw[sel])) / denom)
                 if sel.any() and denom > 0 else np.nan)
        DA_l = float(d["DA_l_Mpc"])
        Fw_arcmin = Fw * DA_l / ARCMIN_PER_RAD if np.isfinite(Fw) else np.nan

        bref, Fref, Gref = profile_reference(d)

        recs.append(dict(
            file=f, label=label, kind=kind, orient=orient, orient_deg=orient_deg,
            ba=ba, ca=ca, rs=rs, R200=R200, half=half,
            F_weak=Fw, G_weak=Gw, F_weak_arcmin=Fw_arcmin, consistency_resid=resid,
            G_m2=azimuthal_modulation(r, theta, G, ANN_LO, ANN_HI, 2, half),
            G_m4=azimuthal_modulation(r, theta, G, ANN_LO, ANN_HI, 4, half),
            F_m2=azimuthal_modulation(r, theta, F, ANN_LO, ANN_HI, 2, half),
            b1=b1, b2=b2, r=r, theta=theta, kappa=kappa, F=F, G=G,
            bref=bref, Fref=Fref, Gref=Gref,
        ))

    recs.sort(key=family_key)

    # -------------------- table --------------------
    print(f"\nFlexion-vs-shape summary  ({len(recs)} runs)   dir={sweep_dir}")
    print(f"  Jacobian source = {args.jacobian};  |F|,|G| in 1/Mpc, fixed annulus "
          f"[{ANN_LO},{ANN_HI}] Mpc, smoothed sigma={args.smooth}px.")
    print("  G_m2 = cos(2.theta) modulation of |G| (ROBUST shape signature).")
    print("  G_m4 = cos(4.theta) modulation (noise/grid-sensitive; expect ~0 for spheres).")
    print("  resid = mean|d.kappa - d*.gamma|/mean|d.kappa| on RAW maps "
          "(numerical 3rd-deriv health, not physics).\n")
    hdr = (f"  {'label':22s} {'b/a':>4s} {'c/a':>4s} {'orient':>7s} | "
           f"{'<|F|>':>9s} {'<|G|>':>9s} | {'G_m2':>6s} {'G_m4':>6s} | {'resid':>6s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for x in recs:
        print(f"  {x['label']:22s} {x['ba']:4.2f} {x['ca']:4.2f} {x['orient']:>7s} | "
              f"{x['F_weak']:9.3e} {x['G_weak']:9.3e} | "
              f"{x['G_m2']:6.3f} {x['G_m4']:6.3f} | {x['consistency_resid']:6.3f}")

    sph = next((x for x in recs if x["kind"] == "sphere"), None)
    if sph and np.isfinite(sph["G_weak"]) and sph["G_weak"] > 0:
        print(f"\n  Relative <|G|>_weak vs sphere ({sph['G_weak']:.3e} /Mpc):")
        for x in recs:
            if np.isfinite(x["G_weak"]):
                print(f"    {x['label']:22s} {x['G_weak']/sph['G_weak']:+.3f}x"
                      f"   (G_m2={x['G_m2']:.3f})")

    # -------------------- CSV --------------------
    csv_path = os.path.join(OUT_DIR, "flexion_summary.csv")
    cols = ["label", "kind", "orient", "orient_deg", "ba", "ca", "rs", "R200",
            "F_weak", "G_weak", "F_weak_arcmin", "consistency_resid",
            "G_m2", "G_m4", "F_m2"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for x in recs:
            w.writerow(x)
    print(f"\n  CSV written: {csv_path}")

    if args.no_plots:
        return
    for x in recs:
        plot_per_shape(x)
    plot_compare_G(recs)
    plot_radial(recs)
    plot_azimuthal(recs)
    print(f"  Figures written under: {OUT_DIR}")


DISP_RMAX = 1.3   # Mpc: trim corners where 3rd-deriv grid noise dominates


def display_mask(x, rmax=DISP_RMAX):
    """NaN-out the central cusp (r<r_s) and the grid corners (r>rmax) so the
    panels show only the trustworthy science annulus."""
    r = x["r"]
    return (r < x["rs"]) | (r > rmax)


def _imshow(ax, x, field, title, cmap="viridis", clip=99.0):
    half = x["half"]
    field = np.where(display_mask(x), np.nan, field)
    vmax = np.nanpercentile(np.abs(field), clip)
    im = ax.imshow(field.T, origin="lower", extent=[-half, half, -half, half],
                   cmap=cmap, vmin=0, vmax=vmax)
    for rad, ls in [(x["R200"], "--"), (x["rs"], ":")]:
        if rad < half * 1.4:
            ax.add_patch(plt.Circle((0, 0), rad, fill=False, ec="w", lw=0.8, ls=ls))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("b1 [Mpc]", fontsize=8); ax.set_ylabel("b2 [Mpc]", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_per_shape(x):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    half = x["half"]

    ax = axes[0]
    km = x["kappa"]
    im = ax.imshow(km.T, origin="lower", extent=[-half, half, -half, half],
                   cmap="magma", vmin=0, vmax=np.nanpercentile(km, 99))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    step = max(1, km.shape[0] // 16)
    b1 = x["b1"][::step, ::step]; b2 = x["b2"][::step, ::step]; F = x["F"][::step, ::step]
    mask = np.hypot(b1, b2) > x["rs"]
    ax.quiver(b1[mask], b2[mask], np.real(F)[mask], np.imag(F)[mask],
              color="cyan", width=0.004, alpha=0.9)
    ax.set_title(f"{x['label']}: kappa + F flexion (spin 1)", fontsize=9)
    ax.set_xlabel("b1 [Mpc]", fontsize=8); ax.set_ylabel("b2 [Mpc]", fontsize=8)

    _imshow(axes[1], x, np.abs(x["F"]), "|F|  first flexion [1/Mpc]", "cividis")
    _imshow(axes[2], x, np.abs(x["G"]), "|G|  second flexion (triangularity) [1/Mpc]", "inferno")

    ax = axes[3]
    Gmag = np.where(display_mask(x), np.nan, np.abs(x["G"])); phase = np.angle(x["G"]) / 3.0
    im = ax.imshow(Gmag.T, origin="lower", extent=[-half, half, -half, half],
                   cmap="inferno", vmin=0, vmax=np.nanpercentile(Gmag, 99))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    s = max(1, Gmag.shape[0] // 14)
    b1 = x["b1"][::s, ::s]; b2 = x["b2"][::s, ::s]; ph = phase[::s, ::s]
    rr = np.hypot(b1, b2); m = (rr > x["rs"]) & (rr < min(x["R200"], half))
    L = 0.07 * half
    ax.quiver(b1[m], b2[m], (L * np.cos(ph))[m], (L * np.sin(ph))[m], color="cyan",
              headwidth=1, headlength=0, pivot="mid", width=0.004, scale=1, scale_units="xy")
    ax.set_title("G spin-3 orientation", fontsize=9)
    ax.set_xlabel("b1 [Mpc]", fontsize=8); ax.set_ylabel("b2 [Mpc]", fontsize=8)

    fig.suptitle(f"Flexion beyond shear -- {x['label']}  "
                 f"(b/a={x['ba']:.2f}, c/a={x['ca']:.2f}, {x['orient']};  "
                 f"raw d.kappa-vs-d*.gamma resid={x['consistency_resid']:.2f})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, f"{x['label']}_flexion.png"), dpi=120)
    plt.close(fig)


def plot_compare_G(recs):
    n = len(recs)
    ncol = min(5, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.4 * nrow), squeeze=False)
    masked = [np.where(display_mask(x), np.nan, np.abs(x["G"])) for x in recs]
    vmax = np.nanpercentile(np.concatenate([m.ravel() for m in masked]), 99)
    im = None
    for k, x in enumerate(recs):
        ax = axes[k // ncol][k % ncol]; half = x["half"]
        im = ax.imshow(masked[k].T, origin="lower", extent=[-half, half, -half, half],
                       cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"{x['label']}  ({x['ba']:.1f},{x['ca']:.1f},{x['orient'][:4]})\n"
                     f"G_m2={x['G_m2']:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("|G| second-flexion (triangularity) across the shape family "
                 "-- shared colour scale  [G_m2 = m=2 modulation]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    cax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="|G| [1/Mpc]")
    fig.savefig(os.path.join(OUT_DIR, "compare_G_flexion.png"), dpi=120)
    plt.close(fig)


def plot_radial(recs):
    """Azimuthally-averaged map flexion vs 1D-profile reference (validation)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    rbins = np.linspace(0.15, 1.35, 25)
    rc = 0.5 * (rbins[1:] + rbins[:-1])
    cmap = plt.get_cmap("tab10")
    for k, x in enumerate(recs):
        r = x["r"].ravel()
        for ax, fld in [(axes[0], np.abs(x["F"]).ravel()), (axes[1], np.abs(x["G"]).ravel())]:
            prof = np.array([np.nanmean(fld[(r >= rbins[i]) & (r < rbins[i + 1])])
                             for i in range(len(rc))])
            ax.plot(rc, prof, lw=1.2, color=cmap(k % 10),
                    label=f"{x['label']} ({x['ba']:.1f},{x['ca']:.1f},{x['orient'][:4]})")
    # sphere 1D-profile reference (dashed black)
    sph = next((x for x in recs if x["kind"] == "sphere"), None)
    if sph is not None:
        mref = (sph["bref"] > 0.15) & (sph["bref"] < 1.35)
        axes[0].plot(sph["bref"][mref], sph["Fref"][mref], "k--", lw=2,
                     label="sphere 1D-profile ref |kappa'|")
        axes[1].plot(sph["bref"][mref], sph["Gref"][mref], "k--", lw=2,
                     label="sphere 1D-profile ref")
    for ax, ttl in [(axes[0], "<|F|>(r) first flexion (map az-avg vs 1D ref)"),
                    (axes[1], "<|G|>(r) second flexion (map az-avg vs 1D ref)")]:
        ax.set_xlabel("b [Mpc]"); ax.set_ylabel("[1/Mpc]"); ax.set_yscale("log")
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3)
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle("Flexion radial profiles: azimuthally-averaged maps vs smooth "
                 "1D ray-traced reference", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, "flexion_radial.png"), dpi=120)
    plt.close(fig)


def plot_azimuthal(recs):
    """|G|(phi)/<|G|> in the science annulus for representative shapes -- the
    cleanest demonstration that the spin-3 flexion's m=2 modulation tracks the
    projected major axis (sphere/end-on/face-on flat; broadside/triaxial 2-lobe).
    The residual ~m=4 wiggle (peaks at +/-45 deg) is the grid-stencil artifact."""
    pick = ["sphere", "prolate_q0.5_endon", "prolate_q0.5_incl45",
            "prolate_q0.5_broadside", "triaxial_0.6_0.3"]
    by = {x["label"]: x for x in recs}
    fig, ax = plt.subplots(figsize=(8.5, 5))
    phi_edges = np.linspace(0, 360, 25)
    phic = 0.5 * (phi_edges[1:] + phi_edges[:-1])
    cmap = plt.get_cmap("turbo")
    for j, lbl in enumerate(pick):
        x = by.get(lbl)
        if x is None:
            continue
        r = x["r"].ravel(); th = (np.degrees(x["theta"]).ravel()) % 360
        G = np.abs(x["G"]).ravel()
        sel = (r >= 0.55) & (r < 1.1)
        thb, gb = th[sel], G[sel]
        prof = np.array([np.nanmean(gb[(thb >= phi_edges[i]) & (thb < phi_edges[i + 1])])
                         for i in range(len(phic))])
        prof = prof / np.nanmean(prof)
        ax.plot(phic, prof, "-o", ms=3, lw=1.4, color=cmap(j / max(1, len(pick) - 1)),
                label=f"{lbl}  (G_m2={x['G_m2']:.2f})")
    for d in (45, 135, 225, 315):
        ax.axvline(d, color="grey", ls=":", lw=0.7)
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xlabel("azimuth phi [deg]  (dotted = 45 deg grid diagonals)")
    ax.set_ylabel("|G|(phi) / <|G|>")
    ax.set_title("Second-flexion azimuthal modulation in annulus 0.55<r<1.1 Mpc\n"
                 "m=2 lobes track the projected major axis; residual m=4 = grid artifact")
    ax.set_xlim(0, 360); ax.set_xticks(np.arange(0, 361, 45))
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "flexion_azimuthal.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
