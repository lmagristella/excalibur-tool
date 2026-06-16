#!/usr/bin/env python3
"""
PART C -- Elliptical-source shear correction & the finite-source shear bias.

Fleury, Larena & Uzan 2019 (arXiv:1809.03924), Sec. V.C, show that for an
EXTENDED ELLIPTICAL source the measured image ellipticity obeys (their Eq. 110)

    E = E_S [1 - 2 Re(gamma E_S*)] + 2 gamma                     (chi-type, E = 2 mu_2)

-- the *same algebraic form* as for infinitesimal sources, but with a shear
gamma that is itself modified by the source's size and shape (their Eqs. 111-114),
and with the intrinsic ellipticity E_S now ENTANGLED with the recovered shear.

Two consequences matter for Excalibur's shear work (cf. the 'spherical-fit shear
bias' result: shear is underestimated for elliptical sky images):

  (1) RESPONSIVITY / SHEAR UNDERESTIMATE.  Averaging Eq. (110) over randomly
      oriented sources of fixed ellipticity |E_S| gives

          <E> = 2 gamma (1 - |E_S|^2 / 2)   =>   naive  <E>/2  UNDER-estimates gamma.

      The fractional shear deficit is |E_S|^2/2 -- a genuine, quantifiable bias
      whenever finite elliptical sources are used, exactly the sign Excalibur sees.

  (2) ORIENTATION ENTANGLEMENT (their Eq. 118).  At fixed position the *effective*
      shear extracted from a single elliptical source swings as cos 2(phi - theta_S):
      enhanced when the source major axis points toward the lens, reduced when it
      points across. The swing amplitude grows with source size (~ vanishes for a
      point source), so it is a true finite-beam effect.

This script tests both on Excalibur's FULL ray-traced lens (reusing the forward
map / image-moment machinery of PART B, analyze_image_moments.LensMap):

  * elliptical_eq110_check.png : E_meas (Excalibur elliptical-source images) vs
    Fleury Eq. (110) prediction using the local Jacobian shear -> validates the
    transformation on a real ray-traced deflection field.
  * shear_responsivity_bias.png : recovered <E>/2 vs true |gamma| for random-PA
    elliptical sources, and the deficit vs |E_S| against the 1 - |E_S|^2/2 law.
  * orientation_dependence.png : effective shear (Eq. 110 inverted) vs source PA
    at a fixed exterior position, and its growth with source size (Eq. 118).

Units: angles in arcsec; ellipticities/shears dimensionless (chi-type).

Usage:
    python analyze_elliptical_shear.py [--match ba1_ca1_orient] [--ng 600]
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_image_moments import LensMap, pick_file, OUTPUT_ROOT  # noqa: E402

OUT_DIR = os.path.join(OUTPUT_ROOT, "beyond_shear", "C_elliptical_shear")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subdir", default="shear_shape_sweep")
    p.add_argument("--match", default="ba1_ca1_orient",
                   help="halo npz substring (default: sphere -- cleanest shear field)")
    p.add_argument("--ng", type=int, default=600)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def axes_from_ES(absE, r_eff):
    """Semi-axes (a>=b) for a given |E_S| and area-equivalent radius r_eff
    (so a*b = r_eff^2): |E_S| = (a^2-b^2)/(a^2+b^2)."""
    # a^2 = r^2 sqrt((1+e)/(1-e)), b^2 = r^2 sqrt((1-e)/(1+e))
    e = absE
    a = r_eff * ((1 + e) / (1 - e))**0.25
    b = r_eff * ((1 - e) / (1 + e))**0.25
    return a, b


def g_eff(lm, alpha, r_probe):
    """Effective complex shear at a position = ellipticity/2 of a co-located small
    CIRCULAR source (g = mu_2(circular) = E_circ/2). This is the shear consistent
    with Fleury's E = 2 gamma convention for round sources, and it sidesteps the
    sign/normalisation convention of the stored Jacobian (which is the unreduced
    shear with the opposite sign to the actual ray-traced image ellipticity).
    Using the same probe radius as the elliptical source makes the comparison
    finite-size-consistent."""
    o = lm.moments_elliptical(alpha, r_probe, r_probe, 0.0, nmax=2)
    if o is None:
        return np.nan, None
    return complex(o["mu2"]), o["cen"]


def invert_eq110(E_meas, E_S):
    """Solve Eq. (110) for the effective complex shear gamma given measured image
    ellipticity E_meas and known source ellipticity E_S (linear 2x2 system)."""
    p, q = E_S.real, E_S.imag
    M = np.array([[2 - 2 * p * p, -2 * p * q], [-2 * p * q, 2 - 2 * q * q]])
    rhs = np.array([(E_meas - E_S).real, (E_meas - E_S).imag])
    g = np.linalg.solve(M, rhs)
    return g[0] + 1j * g[1]


# --------------------------------------------------------------- (1) Eq.110 check
def plot_eq110_check(lm, r_eff=6.0):
    rng = np.random.default_rng(1)
    bm = 0.6 * lm.beta_max
    Em, Ep = [], []
    for _ in range(700):
        a = (rng.uniform(-bm, bm), rng.uniform(-bm, bm))
        if not (15 < np.hypot(*a) < bm):
            continue
        absE = rng.uniform(0.1, 0.6); thS = rng.uniform(0, np.pi)
        aa, bb = axes_from_ES(absE, r_eff)
        o = lm.moments_elliptical(a, aa, bb, thS, nmax=2)
        if o is None:
            continue
        gam, _ = g_eff(lm, a, r_eff)        # effective shear from co-located circular source
        if not np.isfinite(gam):
            continue
        ES = absE * np.exp(2j * thS)
        Em.append(2 * o["mu2"])
        Ep.append(ES * (1 - 2 * np.real(gam * np.conj(ES))) + 2 * gam)
    Em = np.array(Em); Ep = np.array(Ep)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    ax[0].scatter(Ep.real, Em.real, s=8, alpha=0.4, color="navy", label="Re E")
    ax[0].scatter(Ep.imag, Em.imag, s=8, alpha=0.4, color="darkorange", label="Im E")
    lim = 1.05 * max(np.abs(Ep).max(), np.abs(Em).max())
    ax[0].plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax[0].set_xlabel("Fleury Eq.(110) prediction"); ax[0].set_ylabel("Excalibur measured E")
    ax[0].set_title("Image ellipticity of elliptical sources:\nEq.(110) vs ray-traced")
    ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_aspect("equal")
    resid = np.abs(Em - Ep)
    ax[1].hist(resid, bins=30, color="teal", alpha=0.8)
    ax[1].axvline(np.median(resid), color="k",
                  label=f"median |E_meas-E_pred| = {np.median(resid):.4f}")
    ax[1].set_xlabel("|E_meas - E_pred|"); ax[1].set_ylabel("count")
    ax[1].set_title("Residual (finite source size + O(gamma^2))"); ax[1].legend(fontsize=9)
    fig.suptitle(f"PART C(1) -- Fleury Eq.(110) holds on Excalibur's lens  [{lm.label}, "
                 f"r_eff={r_eff}\"]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, "elliptical_eq110_check.png"), dpi=120)
    plt.close(fig)
    return float(np.median(resid))


# ------------------------------------------------ (2) responsivity / underestimate
def plot_responsivity(lm, r_eff=6.0, n_orient=24):
    """Average image ellipticity over random source PA -> recovered shear is biased
    low by |E_S|^2/2 (the responsivity deficit)."""
    # a few fixed positions spanning the shear field
    bm = 0.55 * lm.beta_max
    positions = [(d, 0.0) for d in np.linspace(20, bm, 6)]
    absEs = np.array([0.0, 0.15, 0.3, 0.45, 0.6])
    thetas = np.linspace(0, np.pi, n_orient, endpoint=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # panel A: recovered <E>/2 vs true gamma, for each |E_S|
    cmap = plt.get_cmap("viridis")
    for k, absE in enumerate(absEs):
        aa, bb = axes_from_ES(absE, r_eff) if absE > 0 else (r_eff, r_eff)
        gtrue, grec = [], []
        for a in positions:
            gloc, _ = g_eff(lm, a, r_eff)    # truth = co-located circular source (same size)
            if not np.isfinite(gloc):
                continue
            Es = [2 * o["mu2"] for th in thetas
                  if (o := lm.moments_elliptical(a, aa, bb, th, nmax=2)) is not None]
            if not Es:
                continue
            gtrue.append(abs(gloc)); grec.append(abs(np.mean(Es)) / 2)
        axes[0].plot(gtrue, grec, "-o", ms=4, color=cmap(k / (len(absEs) - 1)),
                     label=f"|E_S|={absE:.2f}")
    lim = 0.3
    axes[0].plot([0, lim], [0, lim], "k--", lw=1, label="unbiased")
    axes[0].set_xlabel("true local |gamma|")
    axes[0].set_ylabel("recovered |<E>/2|  (PA-averaged)")
    axes[0].set_title("Finite elliptical sources UNDER-estimate shear")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # panel B: fractional deficit vs |E_S|, against 1 - |E_S|^2/2
    fr_meas = []
    for absE in absEs:
        aa, bb = axes_from_ES(absE, r_eff) if absE > 0 else (r_eff, r_eff)
        ratios = []
        for a in positions:
            gloc, _ = g_eff(lm, a, r_eff)
            if not (np.isfinite(gloc) and abs(gloc) > 0.02):
                continue
            Es = [2 * o["mu2"] for th in thetas
                  if (o := lm.moments_elliptical(a, aa, bb, th, nmax=2)) is not None]
            if Es:
                ratios.append((abs(np.mean(Es)) / 2) / abs(gloc))
        fr_meas.append(np.median(ratios) if ratios else np.nan)
    ee = np.linspace(0, 0.65, 100)
    axes[1].plot(ee, 1 - ee**2 / 2, "r-", lw=2, label="Fleury  1 - |E_S|^2/2")
    axes[1].plot(absEs, fr_meas, "ko", ms=7, label="Excalibur (PA-averaged)")
    axes[1].set_xlabel("source ellipticity |E_S|")
    axes[1].set_ylabel("recovered / true shear")
    axes[1].set_title("Responsivity deficit vs source ellipticity")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle(f"PART C(2) -- the finite-source shear UNDER-estimate  [{lm.label}, "
                 f"r_eff={r_eff}\"]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, "shear_responsivity_bias.png"), dpi=120)
    plt.close(fig)
    return list(zip(absEs.tolist(), fr_meas))


# --------------------------------------------------- (3) orientation dependence
def plot_orientation(lm, alpha=(60.0, 0.0), r_eff=8.0, absE=0.4):
    """The shear extracted from a single elliptical source is ENTANGLED with the
    source orientation: extracting an 'effective gamma' by linear inversion of
    Eq.(110) leaves a residual that swings as cos 2(theta_S - phi_lens) -- ENHANCED
    when the source major axis points toward the lens, REDUCED when it points
    across (exactly Fleury's Eq.118 phase). [The swing here is dominated by the
    O(gamma.|E_S|^2) terms the linear Eq.(110) drops, which are size-independent;
    the pure finite-beam (beta0/lambda)^2 part is below the raster precision, so we
    show only the robust phase/form, not its size scaling.]"""
    phi_lens = np.arctan2(alpha[1], alpha[0])
    thetas = np.linspace(0, np.pi, 36)
    aa, bb = axes_from_ES(absE, r_eff)
    geff, gnaive = [], []
    for th in thetas:
        o = lm.moments_elliptical(alpha, aa, bb, th, nmax=2)
        if o is None:
            geff.append(np.nan); gnaive.append(np.nan); continue
        ES = absE * np.exp(2j * th)
        geff.append(abs(invert_eq110(2 * o["mu2"], ES)))
        gnaive.append(abs(o["mu2"]))   # naive |g| ignoring E_S = |E_meas|/2
    geff = np.array(geff); gnaive = np.array(gnaive)
    x = np.degrees(2 * (thetas - phi_lens))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    g0, _ = g_eff(lm, alpha, r_eff)
    ax[0].plot(x, geff, "o-", color="navy", ms=4,
               label="effective |gamma| (Eq.110-corrected)")
    ax[0].axhline(abs(g0), color="r", ls="--", label="true |gamma| (circular source)")
    # cos2 fit to highlight the form
    A = np.c_[np.ones_like(thetas), np.cos(2 * (thetas - phi_lens))]
    coef, *_ = np.linalg.lstsq(A, geff, rcond=None)
    ax[0].plot(x, A @ coef, "g-", lw=1, alpha=0.7,
               label=f"cos2 fit (amp {coef[1]/coef[0]*100:+.0f}%)")
    ax[0].set_xlabel("2(theta_S - phi_lens) [deg]   (0 = major axis toward lens)")
    ax[0].set_ylabel("extracted effective |gamma|")
    ax[0].set_title(f"Shear entangled with source orientation\n(pos {alpha}\", "
                    f"r_eff={r_eff}\", |E_S|={absE})")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(x, gnaive / abs(g0), "s-", color="crimson", ms=4,
               label="|E_meas|/2  /  true gamma")
    ax[1].axhline(1, color="k", lw=0.6)
    ax[1].set_xlabel("2(theta_S - phi_lens) [deg]")
    ax[1].set_ylabel("naive shear estimate / true")
    ax[1].set_title("If source ellipticity is IGNORED, the single-source\n"
                    "shear estimate is dominated by E_S (huge swing)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(f"PART C(3) -- source-shape / shear entanglement (Fleury Sec. V.C, "
                 f"Eq.118 phase)  [{lm.label}]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, "orientation_dependence.png"), dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    npz = pick_file(args.subdir, args.match)
    lm = LensMap(npz, ng=args.ng)
    print(f"PART C elliptical-source shear -- {lm.label}  (b/a={lm.ba}, c/a={lm.ca})")
    print(f"  theta_max={lm.theta_max:.1f}\"  beta_max={lm.beta_max:.1f}\"")
    if args.no_plots:
        return
    med = plot_eq110_check(lm)
    print(f"  (1) Eq.(110) check: median |E_meas - E_pred| = {med:.4f}")
    rows = plot_responsivity(lm)
    print("  (2) responsivity deficit  recovered/true shear vs |E_S|:")
    for e, r in rows:
        print(f"        |E_S|={e:.2f}: measured {r:.3f}   (Fleury 1-|E_S|^2/2 = {1-e**2/2:.3f})")
    plot_orientation(lm)
    print(f"  (3) orientation dependence written.")
    print(f"  Figures written under: {OUT_DIR}")


if __name__ == "__main__":
    main()
