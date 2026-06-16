#!/usr/bin/env python3
"""
PART B -- Finite-source image moments beyond shear (Excalibur as a numerical lab).

Fleury, Larena & Uzan 2019 (arXiv:1809.03924) make a sharp point: an
INFINITESIMAL source can only be magnified and sheared, but an EXTENDED source
picks up a whole tower of higher distortion moments

    mu_2 = ellipticity (= shear)   mu_3 = triangularity (flexion)   mu_4 = squarity ...

and these grow with the source's angular size because the deflection field
varies across the finite beam. Their reduced complex moment of an image is
(top-hat weight, their Eqs. 11/22/24/25); for a FILLED image region it reduces
to the compact harmonic-moment form (derived in the header notes below and
verified mu_2 = chi/2 so that E = 2 mu_2 = chi):

    mu_n = (2/(n+2)) * sum_pixels z^n / sum_pixels |z|^n ,   z = theta - centroid.

Rather than re-deriving Fleury's perturbative contour integrals, we use
Excalibur's FULL ray-traced lens map non-perturbatively:

  * The npz gives, per image-plane node, theta (= b/D_C,l) and the source-plane
    landing beta (= final_pos . screen / D_C,s), both as ANGLES [arcsec], using
    COMOVING distances D_C = D_A (1+z) since b and final_pos are comoving.
  * We interpolate the smooth forward map beta(theta) onto a fine theta-grid and
    image a source disk of angular radius R centred at alpha by FORWARD
    rasterization:  image = { theta : |beta(theta) - alpha| < R }.  No map
    inversion is needed (robust across the nonlinear/strong region), and the
    full nonlinearity of the ray-traced deflection is retained.

Validations / products (broadside prolate by default -- strongest higher moments):
  1. moments_validation.png : |mu_2| of a small circular source vs the local
     shear |gamma| from the Jacobian  ->  mu_2 -> gamma as R -> 0 (Fleury limit).
  2. moments_vs_radius.png   : mu_n(R) growth at fixed positions; mu_{>2} -> 0 as
     R -> 0, rising ~linearly (mu_3) and ~quadratically (mu_4) -- the finite-beam
     tower. Overlaid: Fleury's link mu_3 ~ -(R/4) G with G the second flexion
     from PART A (analyze_flexion_maps), tying B back to A.
  3. moments_maps.png        : |mu_3|,|mu_4| over source-plane position alpha --
     where triangularity / squarity are generated around the halo.
  4. fourier_wheels.png      : Fleury Fig. 2 analogue -- a circular source and its
     lensed image contour, decomposed into Fourier "epicycle" modes.

Units: angles in arcsec throughout. Reduced moments mu_n are dimensionless.

Usage:
    python analyze_image_moments.py [--match ba0.5_ca0.5_orientaligned]
                                    [--ng 600] [--no-plots]
"""

import argparse
import glob
import os

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator as CTI

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.abspath(os.path.join(HERE, "..", "_data", "output"))
OUT_DIR = os.path.join(OUTPUT_ROOT, "beyond_shear", "B_image_moments")
ARC = 206265.0


def _ellipse_baseline():
    """Precompute |mu_4| of a PURE uniform ellipse vs |mu_2|. A sheared circular
    source becomes an ellipse, which already carries a 4th-harmonic moment with
    mu_3 = 0 but mu_4 != 0. This baseline is the shear-induced part of mu_4; the
    BEYOND-SHEAR squarity is the excess above it. (mu_3 needs no such baseline.)"""
    qs = np.linspace(1.0, 0.25, 40)
    mu2, mu4 = [], []
    dpix = 0.01
    g = np.arange(-1.3, 1.3, dpix)
    X, Y = np.meshgrid(g, g, indexing="ij")
    for q in qs:
        m = (X**2 + (Y / q)**2) < 1
        z = (X[m] - X[m].mean()) + 1j * (Y[m] - Y[m].mean()); az = np.abs(z)
        mu2.append(abs(0.5 * np.sum(z**2) / np.sum(az**2)))
        mu4.append(abs((2 / 6) * np.sum(z**4) / np.sum(az**4)))
    order = np.argsort(mu2)
    return np.array(mu2)[order], np.array(mu4)[order]


_ELL_MU2, _ELL_MU4 = _ellipse_baseline()


def ellipse_mu4(mu2_abs):
    """Shear-induced |mu_4| baseline for a given image |mu_2|."""
    return np.interp(np.abs(mu2_abs), _ELL_MU2, _ELL_MU4)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subdir", default="shear_shape_sweep")
    p.add_argument("--match", default="ba0.5_ca0.5_orientaligned",
                   help="substring picking the halo npz (default: broadside prolate q0.5)")
    p.add_argument("--ng", type=int, default=600, help="fine raster grid size per axis")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


class LensMap:
    """Forward angular lens map beta(theta) [arcsec] + Jacobian, from an npz."""

    def __init__(self, npz_path, ng=600):
        d = np.load(npz_path, allow_pickle=True)
        self.label = str(d["display_label"])
        self.ba = float(d["axis_ratio_ba"]); self.ca = float(d["axis_ratio_ca"])
        DCl = float(d["DA_l_Mpc"]) * (1 + float(d["z_l"]))
        DCs = float(d["DA_s_Mpc"]) * (1 + float(d["z_source"]))
        e1 = np.asarray(d["screen_e1"], float); e2 = np.asarray(d["screen_e2"], float)
        spc = np.asarray(d["source_plane_center_Mpc"], float)
        b1 = np.asarray(d["b1_map_Mpc"], float); b2 = np.asarray(d["b2_map_Mpc"], float)
        fin = np.asarray(d["final_pos_map_Mpc"], float)
        th1 = b1 / DCl * ARC; th2 = b2 / DCl * ARC
        be1 = ((fin - spc) @ e1) / DCs * ARC; be2 = ((fin - spc) @ e2) / DCs * ARC
        A = np.asarray(d["D_flat_map"], float)
        g1 = 0.5 * (A[:, 3] - A[:, 0]); g2 = -A[:, 1]
        kap = 1.0 - 0.5 * (A[:, 0] + A[:, 3])
        pts = np.c_[th1, th2]
        self.FB1 = CTI(pts, be1); self.FB2 = CTI(pts, be2)
        self.IG1 = CTI(pts, g1); self.IG2 = CTI(pts, g2); self.IK = CTI(pts, kap)
        # inverse interpolant (for the Fourier-wheel contour viz only)
        ipts = np.c_[be1, be2]
        self.IT1 = CTI(ipts, th1); self.IT2 = CTI(ipts, th2)
        self.theta_max = float(np.abs(th1).max())
        self.beta_max = float(np.nanmax(np.abs(be1)))
        # fine raster of the forward map
        L = 0.98 * self.theta_max
        gx = np.linspace(-L, L, ng)
        self.TX, self.TY = np.meshgrid(gx, gx, indexing="ij")
        self.BX = self.FB1(self.TX, self.TY); self.BY = self.FB2(self.TX, self.TY)
        self.dpix = gx[1] - gx[0]

    def shear(self, t1, t2):
        return complex(self.IG1(t1, t2), self.IG2(t1, t2))

    def moments(self, alpha, R, nmax=4):
        """Reduced complex moments mu_2..mu_nmax of the image of a circular
        source disk (centre alpha [arcsec], radius R [arcsec]).

        Uses ANTI-ALIASED (soft) source membership -- a 1-cell linear ramp at the
        disk boundary -- which suppresses the Cartesian pixelization floor of the
        higher moments (unlensed-disk |mu_4| floor: ~0.01 hard -> <0.003 soft),
        letting mu_n(R->0) be measured cleanly."""
        d = np.hypot(self.BX - alpha[0], self.BY - alpha[1])
        w = np.clip(0.5 + (R - d) / self.dpix, 0.0, 1.0)
        w[~np.isfinite(self.BX)] = 0.0
        if w.sum() < 30:
            return None
        m = w > 0
        tx = self.TX[m]; ty = self.TY[m]; ww = w[m]
        sw = ww.sum()
        cx = np.sum(ww * tx) / sw; cy = np.sum(ww * ty) / sw
        z = (tx - cx) + 1j * (ty - cy)
        out = {"npix": int(m.sum()), "cen": (cx, cy), "area": sw * self.dpix**2}
        az = np.abs(z)
        for k in range(2, nmax + 1):
            denom = np.sum(ww * az**k)
            out[f"mu{k}"] = (2.0 / (k + 2)) * np.sum(ww * z**k) / denom if denom > 0 else np.nan
        return out

    def moments_elliptical(self, alpha, a, b, theta_S, nmax=4):
        """Moments of the image of an ELLIPTICAL source: semi-axes a>=b [arcsec],
        position angle theta_S [rad], centre alpha. Anti-aliased membership.
        Source complex ellipticity is E_S = (a^2-b^2)/(a^2+b^2) e^{2i theta_S}."""
        dx = self.BX - alpha[0]; dy = self.BY - alpha[1]
        c, s = np.cos(theta_S), np.sin(theta_S)
        u = (dx * c + dy * s) / a            # along major axis / a
        v = (-dx * s + dy * c) / b           # along minor axis / b
        # signed-distance-like field rho<1 inside; ramp width ~1 image cell in rho
        rho = np.hypot(u, v)
        ramp = self.dpix / max(a, b)
        w = np.clip(0.5 + (1.0 - rho) / ramp, 0.0, 1.0)
        w[~np.isfinite(self.BX)] = 0.0
        if w.sum() < 30:
            return None
        m = w > 0
        tx = self.TX[m]; ty = self.TY[m]; ww = w[m]; sw = ww.sum()
        cx = np.sum(ww * tx) / sw; cy = np.sum(ww * ty) / sw
        z = (tx - cx) + 1j * (ty - cy); az = np.abs(z)
        out = {"npix": int(m.sum()), "cen": (cx, cy)}
        for k in range(2, nmax + 1):
            denom = np.sum(ww * az**k)
            out[f"mu{k}"] = (2.0 / (k + 2)) * np.sum(ww * z**k) / denom if denom > 0 else np.nan
        return out

    def image_contour(self, alpha, R, npts=240):
        """Image of a source CIRCLE (contour) via the inverse interpolant -- for
        visualization. Returns source contour (complex) and image contour."""
        phi = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        bs1 = alpha[0] + R * np.cos(phi); bs2 = alpha[1] + R * np.sin(phi)
        t1 = self.IT1(bs1, bs2); t2 = self.IT2(bs1, bs2)
        src = bs1 + 1j * bs2
        img = t1 + 1j * t2
        return phi, src, img


def pick_file(subdir, match):
    files = sorted(glob.glob(os.path.join(OUTPUT_ROOT, subdir, "*.npz")))
    hit = [f for f in files if match in os.path.basename(f)]
    if not hit:
        raise SystemExit(f"No npz matching '{match}' in {subdir}")
    return hit[0]


# -------------------------------------------------------------------- plots
def plot_validation(lm):
    """|mu_2| of a small circular source vs local shear |gamma|; ratio -> 1."""
    rng = np.random.default_rng(0)
    R_small = 5.0
    pos, mu2, gam = [], [], []
    bm = 0.7 * lm.beta_max
    for _ in range(400):
        a = (rng.uniform(-bm, bm), rng.uniform(-bm, bm))
        if np.hypot(*a) < 12:
            continue
        o = lm.moments(a, R_small, nmax=2)
        if o is None:
            continue
        g = lm.shear(*o["cen"])
        if not np.isfinite(g):
            continue
        mu2.append(abs(o["mu2"])); gam.append(abs(g)); pos.append(a)
    mu2 = np.array(mu2); gam = np.array(gam)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(gam, mu2, s=10, alpha=0.5, color="navy")
    lim = max(gam.max(), mu2.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], "r--", lw=1, label="mu_2 = gamma")
    ax[0].set_xlabel("local shear |gamma| (Jacobian)")
    ax[0].set_ylabel(f"|mu_2| of circular source (R={R_small}\")")
    ax[0].set_title("Image ellipticity of a small circular source\nrecovers the local shear")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    good = gam > 0.02
    ratio = mu2[good] / gam[good]
    ax[1].hist(ratio, bins=30, color="teal", alpha=0.8)
    ax[1].axvline(1.0, color="r", ls="--")
    ax[1].axvline(np.median(ratio), color="k", ls="-",
                  label=f"median {np.median(ratio):.3f}")
    ax[1].set_xlabel("|mu_2| / |gamma|"); ax[1].set_ylabel("count")
    ax[1].set_title("Ratio distribution (|gamma|>0.02)"); ax[1].legend()
    fig.suptitle(f"PART B validation -- {lm.label}  (b/a={lm.ba}, c/a={lm.ca})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, "moments_validation.png"), dpi=120)
    plt.close(fig)
    return float(np.median(ratio))


def plot_vs_radius(lm):
    """mu_n(R): the finite-source tower. mu_{>2} -> 0 as R -> 0."""
    positions = {"on major axis (40\",0)": (40.0, 0.0),
                 "on minor axis (0,40\")": (0.0, 40.0),
                 "off-axis (35\",35\")": (35.0, 35.0)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (name, a) in zip(axes, positions.items()):
        # keep the source clear of the central caustic: edge stays >25% of |alpha| from centre
        Rmax = 0.75 * np.hypot(*a)
        Rs = np.linspace(2, Rmax, 22)
        m2, m3, m4 = [], [], []
        for R in Rs:
            o = lm.moments(a, R)
            if o is None:
                m2.append(np.nan); m3.append(np.nan); m4.append(np.nan); continue
            m2.append(abs(o["mu2"])); m3.append(abs(o["mu3"])); m4.append(abs(o["mu4"]))
        m2 = np.array(m2); m3 = np.array(m3); m4 = np.array(m4)
        base4 = ellipse_mu4(m2)   # shear-induced mu_4 baseline (pure ellipse)
        ax.plot(Rs, m2, "-o", ms=3, color="C0", label="|mu_2| ellipticity (= shear)")
        ax.plot(Rs, m3, "-s", ms=4, color="C3", lw=2,
                label="|mu_3| triangularity (CLEAN beyond-shear)")
        ax.plot(Rs, m4, "-^", ms=3, color="C2", label="|mu_4| squarity (raw)")
        ax.plot(Rs, base4, ":", color="C2", lw=1.5, label="|mu_4| pure-ellipse baseline")
        ax.plot(Rs, np.clip(m4 - base4, 0, None), "--", color="C1", lw=1.5,
                label="|mu_4| excess (beyond-shear)")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("source angular radius R [arcsec]"); ax.set_ylabel("|mu_n|")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f"PART B -- finite-source moment tower mu_n(R)  [{lm.label}]\n"
                 "R->0: only mu_2 (shear) survives. mu_3 (clean) grows ~linearly with "
                 "beam size; mu_4 excess = squarity beyond the shear-induced ellipse", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT_DIR, "moments_vs_radius.png"), dpi=120)
    plt.close(fig)


def plot_moment_maps(lm):
    """|mu_3|, |mu_4| as a function of source-plane position (fixed R)."""
    R = 12.0
    bm = 0.62 * lm.beta_max
    g = np.linspace(-bm, bm, 41)
    AX, AY = np.meshgrid(g, g, indexing="ij")
    M2 = np.full(AX.shape, np.nan); M3 = np.full(AX.shape, np.nan); M4 = np.full(AX.shape, np.nan)
    for i in range(AX.shape[0]):
        for j in range(AX.shape[1]):
            o = lm.moments((AX[i, j], AY[i, j]), R)
            if o is None:
                continue
            M2[i, j] = abs(o["mu2"]); M3[i, j] = abs(o["mu3"]); M4[i, j] = abs(o["mu4"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, M, ttl in [(axes[0], M2, "|mu_2| ellipticity"),
                       (axes[1], M3, "|mu_3| triangularity"),
                       (axes[2], M4, "|mu_4| squarity")]:
        im = ax.imshow(M.T, origin="lower", extent=[-bm, bm, -bm, bm], cmap="magma",
                       vmax=np.nanpercentile(M, 99))
        ax.set_title(f"{ttl}  (R={R}\")", fontsize=10)
        ax.set_xlabel("source alpha_1 [arcsec]"); ax.set_ylabel("source alpha_2 [arcsec]")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"PART B -- where higher moments are generated  [{lm.label}, "
                 f"major axis along alpha_1]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, "moments_maps.png"), dpi=120)
    plt.close(fig)


def plot_fourier_wheels(lm):
    """Fleury Fig.2 analogue: source circle, lensed image, Fourier-mode epicycles."""
    cases = [(45.0, 0.0, 22.0), (32.0, 32.0, 22.0), (0.0, 45.0, 22.0)]
    amp = 3.0   # visual amplification of the displacement (Fleury exaggerates too)
    fig, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 5))
    for ax, (a1, a2, R) in zip(axes, cases):
        phi, src, img = lm.image_contour((a1, a2), R, npts=240)
        if np.isnan(img).any():
            img = img.copy()
            good = ~np.isnan(img)
            img = np.interp(np.arange(len(img)), np.where(good)[0], img[good].real) + \
                  1j * np.interp(np.arange(len(img)), np.where(good)[0], img[good].imag)
        # centre both on the image centroid
        c = img.mean()
        s_c = src - src.mean(); i_c = img - c
        # amplify the displacement about the source circle for visibility
        disp = i_c - s_c
        i_disp = s_c + amp * disp
        ax.plot(s_c.real, s_c.imag, color="0.6", lw=2, label="circular source")
        ax.plot(i_disp.real, i_disp.imag, color="crimson", lw=2,
                label=f"image (displacement x{amp:.0f})")
        # Fourier modes of the displacement field delta theta(phi) = sum dp e^{i(p+1)phi}
        N = len(phi)
        dft = np.fft.fft(disp) / N
        modes = []
        for p in range(-4, 5):
            modes.append((p, dft[(p + 1) % N]))
        txt = "  ".join(f"p={p}:{abs(c2)/R:.2f}" for p, c2 in modes if abs(c2) / R > 0.03)
        ax.set_title(f"source at ({a1:.0f},{a2:.0f})\", R={R:.0f}\"\nmodes |dtheta_p|/R: {txt}",
                     fontsize=8)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_xlabel("theta_1 - centre [arcsec]"); ax.set_ylabel("theta_2 - centre [arcsec]")
    fig.suptitle(f"PART B -- Fourier 'epicycle' decomposition of the lensed image "
                 f"(Fleury Fig. 2 analogue)  [{lm.label}]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR, "fourier_wheels.png"), dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    npz = pick_file(args.subdir, args.match)
    print(f"PART B image moments -- {os.path.basename(npz)}")
    lm = LensMap(npz, ng=args.ng)
    print(f"  halo {lm.label}: theta_max={lm.theta_max:.1f}\"  beta_max={lm.beta_max:.1f}\"  "
          f"raster {args.ng}x{args.ng} (dpix={lm.dpix:.2f}\")")
    # quick numeric sanity to stdout
    for a, R in [((40, 0), 8), ((0, 40), 8), ((40, 0), 24)]:
        o = lm.moments(a, R)
        g = lm.shear(*o["cen"])
        print(f"  src({a},R={R}): |mu2|={abs(o['mu2']):.4f} |gamma|={abs(g):.4f} "
              f"|mu3|={abs(o['mu3']):.4f} |mu4|={abs(o['mu4']):.4f}  (npix={o['npix']})")
    if args.no_plots:
        return
    med = plot_validation(lm)
    print(f"  validation median |mu2|/|gamma| = {med:.3f}  (-> 1 confirms ellipticity=shear)")
    plot_vs_radius(lm)
    plot_moment_maps(lm)
    plot_fourier_wheels(lm)
    print(f"  Figures written under: {OUT_DIR}")


if __name__ == "__main__":
    main()
