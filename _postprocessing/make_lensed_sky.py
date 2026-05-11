#!/usr/bin/env python3
"""
Produce lensed galaxy images from an EXCALIBUR simulation .npz.

Primary mode: reconstruct the deflection field from the Jacobi matrix D
obtained by integrating the geodesic + Sachs optical equations.  The D
matrix encodes convergence, shear (with direction), and magnification
self-consistently.

Optional analytic NFW overlay for comparison (Wright & Brainerd 2000).

Usage:
    python make_lensed_sky.py  path/to/results.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.interpolate import RectBivariateSpline

import matplotlib
if os.environ.get("EXCALIBUR_MPL_BACKEND"):
    matplotlib.use(os.environ["EXCALIBUR_MPL_BACKEND"])
elif __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from excalibur.io.filename_utils import RunNamer, latest_run


# ==================================================================
#  Physical constants
# ==================================================================
_G    = 6.67430e-11     # m^3 kg^-1 s^-2
_c    = 2.99792e8       # m/s
_Msun = 1.98892e30      # kg
_Mpc  = 3.08568e22      # m


# ==================================================================
#  Extract lensing from D matrix  (simulation data)
# ==================================================================

def lensing_from_D_map(d):
    """
    Extract kappa, gamma1, gamma2, mu maps from D_flat_map in the .npz.

    D_flat_map is the *normalised* Jacobi map (D / lambda_S) so that
    the unlensed background gives the identity matrix.

    IMPORTANT: the Sachs screen basis (e1, e2) chosen by init_sachs_basis
    depends on the photon direction.  For photons travelling primarily
    along z with impact parameters (b1, b2):
      - |b1| <= |b2|  =>  e1 ~ e_b1,  e2 ~ e_b2   (global frame)
      - |b1| >  |b2|  =>  e1 ~ e_b2,  e2 ~ -e_b1   (rotated 90 deg)
    We must rotate the D matrix to the global (b1, b2) frame before
    extracting gamma1, gamma2.

    Convention (Bartelmann & Schneider):
        A = (1 - kappa) I  -  Gamma
        Gamma = [[gamma1, gamma2], [gamma2, -gamma1]]

        kappa  = 1 - (A11 + A22) / 2
        gamma1 = -(A11 - A22) / 2
        gamma2 = -(A12 + A21) / 2

    Tangential shear:
        gamma_t = -(gamma1 cos(2 phi) + gamma2 sin(2 phi))
    """
    n1d  = int(d["n_map_1d"])
    half = float(d["map_half_Mpc"])
    Df   = d["D_flat_map"]                        # (n1d*n1d, 4)

    D11s = Df[:, 0].reshape(n1d, n1d)
    D12s = Df[:, 1].reshape(n1d, n1d)
    D21s = Df[:, 2].reshape(n1d, n1d)
    D22s = Df[:, 3].reshape(n1d, n1d)

    b1 = d["b1_map_Mpc"].reshape(n1d, n1d)
    b2 = d["b2_map_Mpc"].reshape(n1d, n1d)

    # Rotate D from ray-dependent Sachs frame to global (b1, b2) frame.
    # For |b1| > |b2|, the Sachs basis was (e_b2, -e_b1), so
    #   A_global = [[D22, -D21], [-D12, D11]]
    # Use tolerance to handle |b1| == |b2| boundary correctly:
    # when equal, init_sachs_basis picks seed (1,0,0) -> e1=e_b1 -> no swap.
    swap = np.abs(b1) > np.abs(b2) + 1e-10

    A11 = np.where(swap, D22s, D11s)
    A12 = np.where(swap, -D21s, D12s)
    A21 = np.where(swap, -D12s, D21s)
    A22 = np.where(swap, D11s, D22s)

    kappa  = 1.0 - 0.5 * (A11 + A22)
    gamma1 = -0.5 * (A11 - A22)
    gamma2 = -0.5 * (A12 + A21)

    det_A  = A11 * A22 - A12 * A21
    mu     = np.where(np.abs(det_A) > 1e-30, 1.0 / det_A, 0.0)

    return dict(kappa=kappa, gamma1=gamma1, gamma2=gamma2, mu=mu,
                b1=b1, b2=b2, n1d=n1d, half_Mpc=half,
                D11=A11, D12=A12, D21=A21, D22=A22)


# ==================================================================
#  FFT lensing potential -> alpha, gamma  (all self-consistent)
# ==================================================================

def lensing_from_kappa_fft(kappa_2d, dx):
    """
    From kappa via FFT, derive alpha1, alpha2, gamma1, gamma2.

    All fields are derived from the same lensing potential psi:
        nabla^2 psi = 2 kappa
        alpha  = grad(psi)
        gamma1 = 0.5 * (psi_11 - psi_22)
        gamma2 = psi_12

    This ensures kappa, alpha and gamma are self-consistent.
    """
    N = kappa_2d.shape[0]
    freq = np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0

    khat = np.fft.fft2(kappa_2d)
    psi_hat = -2.0 * khat / ((2.0 * np.pi)**2 * k2)
    psi_hat[0, 0] = 0.0

    twopi = 2.0 * np.pi

    # alpha = grad(psi)
    a1 = np.fft.ifft2(1j * twopi * kx * psi_hat).real
    a2 = np.fft.ifft2(1j * twopi * ky * psi_hat).real

    # Second derivatives of psi
    psi_11_hat = -(twopi**2) * kx**2 * psi_hat
    psi_22_hat = -(twopi**2) * ky**2 * psi_hat
    psi_12_hat = -(twopi**2) * kx * ky * psi_hat

    g1 = 0.5 * np.fft.ifft2(psi_11_hat - psi_22_hat).real
    g2 = np.fft.ifft2(psi_12_hat).real

    return a1, a2, g1, g2


def _coarse_x(n1d, half):
    """Coordinate array matching the node-centred grid in the .npz."""
    return np.linspace(-half, half, n1d)


def upsample(field, x_coarse, x_fine, log=False):
    """Bicubic upsample a 2D field.

    If *log=True*, interpolate in log-space so that strictly positive
    fields (like convergence kappa) stay positive.  This avoids the
    ringing / undershoot that a cubic spline produces when the field
    drops by an order of magnitude in a single coarse cell.
    """
    if log:
        f = np.log(np.clip(field, 1e-30, None))
        return np.exp(RectBivariateSpline(x_coarse, x_coarse, f,
                                          kx=3, ky=3)(x_fine, x_fine))
    return RectBivariateSpline(x_coarse, x_coarse, field, kx=3, ky=3)(
        x_fine, x_fine)


def _interp2d(field, x_from, x_to):
    """Interpolate a 2D field from one regular grid to another."""
    return RectBivariateSpline(x_from, x_from, field, kx=3, ky=3)(x_to, x_to)


# ==================================================================
#  Analytic NFW (Wright & Brainerd 2000) -- optional comparison
# ==================================================================

def compute_kappa_s(d):
    """kappa_s = rho_s * rs / Sigma_cr."""
    c_NFW = float(d["c_NFW"])
    M200  = float(d["M_200_Msun"]) * _Msun
    R200  = float(d["R_200_Mpc"])  * _Mpc
    rs    = float(d["r_s_Mpc"])    * _Mpc
    Scr   = float(d["Sigma_cr"])
    delta_c = (200.0 / 3.0) * c_NFW**3 / (
        np.log(1 + c_NFW) - c_NFW / (1 + c_NFW))
    rho_crit = M200 / (4.0 / 3.0 * np.pi * R200**3 * 200.0)
    rho_s = delta_c * rho_crit
    return rho_s * rs / Scr


def _nfw_kappa_analytic(b, rs, kappa_s):
    """NFW convergence kappa(b)."""
    x = np.asarray(b / rs, dtype=np.float64)
    kap = np.zeros_like(x)
    lo = x < 1.0 - 1e-6
    hi = x > 1.0 + 1e-6
    eq = ~lo & ~hi
    if lo.any():
        xl = x[lo]
        kap[lo] = 2.0 * kappa_s / (xl**2 - 1.0) * (
            1.0 - 1.0 / np.sqrt(1.0 - xl**2) * np.arccosh(1.0 / xl))
    if hi.any():
        xh = x[hi]
        kap[hi] = 2.0 * kappa_s / (xh**2 - 1.0) * (
            1.0 - 1.0 / np.sqrt(xh**2 - 1.0) * np.arccos(1.0 / xh))
    if eq.any():
        kap[eq] = 2.0 * kappa_s / 3.0
    return kap


def _nfw_mean_kappa(b, rs, kappa_s):
    """Mean convergence inside projected radius b."""
    x = np.asarray(b / rs, dtype=np.float64)
    g = np.zeros_like(x)
    lo = x < 1.0 - 1e-6
    hi = x > 1.0 + 1e-6
    eq = ~lo & ~hi
    if lo.any():
        xl = x[lo]
        g[lo] = (np.log(xl / 2.0)
                 + 1.0 / np.sqrt(1.0 - xl**2) * np.arccosh(1.0 / xl))
    if hi.any():
        xh = x[hi]
        g[hi] = (np.log(xh / 2.0)
                 + 1.0 / np.sqrt(xh**2 - 1.0) * np.arccos(1.0 / xh))
    if eq.any():
        g[eq] = 1.0 + np.log(0.5)
    return 4.0 * kappa_s * g / x**2


def _nfw_alpha_analytic(b, rs, kappa_s):
    """NFW deflection amplitude |alpha|(b) in same units as b."""
    return _nfw_mean_kappa(b, rs, kappa_s) * b


def _nfw_gamma_analytic(b, rs, kappa_s):
    """NFW tangential shear gamma_t(b)."""
    return _nfw_mean_kappa(b, rs, kappa_s) - _nfw_kappa_analytic(b, rs, kappa_s)


def analytic_deflection_field(t1, t2, rs, kappa_s):
    """Full 2D deflection from analytic NFW centred at origin [Mpc]."""
    r = np.sqrt(t1**2 + t2**2)
    r_safe = np.clip(r, 1e-10, None)
    alpha_r = _nfw_alpha_analytic(r_safe, rs, kappa_s)
    return alpha_r * t1 / r_safe, alpha_r * t2 / r_safe


# ==================================================================
#  Source galaxy model (Sersic)
# ==================================================================

def sersic_source(b1, b2, center=(0.0, 0.0), R_e=0.05, n=1.0,
                  ellip=0.3, pa_deg=30.0, I0=1.0):
    """Sersic surface brightness (optionally elliptical)."""
    b_n = 1.9992 * n - 0.3271
    db1 = b1 - center[0]
    db2 = b2 - center[1]
    pa = np.radians(pa_deg)
    c, s = np.cos(pa), np.sin(pa)
    u =  c * db1 + s * db2
    v = -s * db1 + c * db2
    q = max(1.0 - ellip, 0.01)
    r = np.sqrt(u**2 + (v / q)**2)
    return I0 * np.exp(-b_n * ((r / R_e) ** (1.0 / n) - 1.0))


# ==================================================================
#  Einstein radius
# ==================================================================

def einstein_radius_Mpc(M_Msun, DA_l, DA_s, DA_ls):
    """Einstein radius for a *point mass* on the lens plane [Mpc]."""
    M_kg = M_Msun * _Msun
    theta_E = np.sqrt(4 * _G * M_kg / _c**2
                      * (DA_ls * _Mpc) / (DA_l * _Mpc * DA_s * _Mpc))
    return theta_E * DA_l


def einstein_radius_nfw_Mpc(rs, kappa_s):
    """True Einstein radius of an NFW lens [Mpc].

    Solves bar_kappa(<b) = 1 numerically.  Returns 0 if the lens is
    subcritical at all radii resolvable above 1e-8 rs.
    """
    b_test = np.logspace(-8, np.log10(30.0 * rs), 20000)
    bk = _nfw_mean_kappa(b_test, rs, kappa_s)
    above = np.where(bk >= 1.0)[0]
    if len(above) == 0:
        return 0.0
    return float(b_test[above[-1]])


# ==================================================================
#  Build lensed image -- core engine
# ==================================================================

def make_lensed_image(d, half_view, source_kw, Nfine=1024,
                      mode="simulation"):
    """
    Build source and lensed images on [-half_view, +half_view].

    mode = "simulation"  uses the D matrix from geodesic integration.
    mode = "analytic"    uses Wright & Brainerd NFW (for comparison).

    For simulation mode:
      1) Upsample kappa from the coarse grid to a fine grid covering the
         FULL simulation domain (so kappa -> 0 at boundaries = periodic).
      2) FFT on the full domain -> alpha, gamma1, gamma2 (all self-
         consistent from the same lensing potential).
      3) Interpolate alpha, gamma to the viewing sub-grid.
    """
    xf = np.linspace(-half_view, half_view, Nfine)
    t1, t2 = np.meshgrid(xf, xf, indexing="ij")

    if mode == "analytic":
        rs = float(d["r_s_Mpc"])
        kappa_s = compute_kappa_s(d)
        a1, a2 = analytic_deflection_field(t1, t2, rs, kappa_s)
        r_grid = np.sqrt(t1**2 + t2**2)
        r_safe = np.clip(r_grid, 1e-10, None)
        kap = _nfw_kappa_analytic(r_safe, rs, kappa_s)
        gam_t = _nfw_gamma_analytic(r_safe, rs, kappa_s)
        phi = np.arctan2(t2, t1)
        g1 = -gam_t * np.cos(2 * phi)
        g2 = -gam_t * np.sin(2 * phi)
        mu = 1.0 / ((1.0 - kap)**2 - gam_t**2)
    else:
        # --- Simulation mode: D matrix ---
        lens = lensing_from_D_map(d)
        n1d  = lens["n1d"]
        half = lens["half_Mpc"]
        xc   = _coarse_x(n1d, half)
        rs   = float(d["r_s_Mpc"])
        kappa_s = compute_kappa_s(d)

        # Step 1: upsample kappa to a fine grid over the FULL domain.
        # Use log-space interpolation so the steep NFW cusp does not
        # produce negative undershoot in the cubic spline.
        Nfull = 512
        x_full = np.linspace(-half, half, Nfull)
        dx_full = 2.0 * half / Nfull
        kap_full = upsample(lens["kappa"], xc, x_full, log=True)

        # Subtract FLRW background convergence before FFT.
        # The simulation D matrix includes the homogeneous Ricci
        # focusing from the FLRW background, but the FFT should only
        # see the NFW *perturbation* (a uniform kappa sheet adds a
        # spurious linear deflection alpha_bg = kappa_bg * b).
        # Estimate the background from the grid corners (maximum r)
        # by removing the known analytical NFW contribution there.
        r_corner = np.sqrt(2.0) * half
        kap_nfw_corner = _nfw_kappa_analytic(
            np.array([r_corner]), rs, kappa_s)[0]
        kap_corner = np.mean([lens["kappa"][0, 0], lens["kappa"][0, -1],
                              lens["kappa"][-1, 0], lens["kappa"][-1, -1]])
        kappa_bg = kap_corner - kap_nfw_corner
        kap_full = kap_full - kappa_bg

        # Step 2: Zero-pad before FFT to reduce periodic-boundary
        # artifacts (the NFW kappa does not vanish at the box edge).
        Npad = 2 * Nfull
        kap_padded = np.zeros((Npad, Npad))
        offset = (Npad - Nfull) // 2
        kap_padded[offset:offset + Nfull,
                   offset:offset + Nfull] = kap_full
        dx_pad = dx_full                       # same pixel size
        a1_pad, a2_pad, g1_pad, g2_pad = lensing_from_kappa_fft(
            kap_padded, dx_pad)

        # Extract the central region that corresponds to the original domain
        a1_full = a1_pad[offset:offset + Nfull, offset:offset + Nfull]
        a2_full = a2_pad[offset:offset + Nfull, offset:offset + Nfull]
        g1_full = g1_pad[offset:offset + Nfull, offset:offset + Nfull]
        g2_full = g2_pad[offset:offset + Nfull, offset:offset + Nfull]

        # Step 3: interpolate all fields to the viewing sub-grid
        a1  = _interp2d(a1_full, x_full, xf)
        a2  = _interp2d(a2_full, x_full, xf)
        kap = _interp2d(kap_full, x_full, xf)
        g1  = _interp2d(g1_full, x_full, xf)
        g2  = _interp2d(g2_full, x_full, xf)

        gamma2 = g1**2 + g2**2
        det = (1.0 - kap)**2 - gamma2
        mu = np.where(np.abs(det) > 1e-30, 1.0 / det, 0.0)

    beta1 = t1 - a1
    beta2 = t2 - a2

    lensed = sersic_source(beta1, beta2, **source_kw)
    source = sersic_source(t1, t2, **source_kw)

    extent = [-half_view, half_view, -half_view, half_view]
    return dict(lensed=lensed, source=source,
                alpha1=a1, alpha2=a2,
                kappa=kap, gamma1=g1, gamma2=g2, mu=mu,
                xf=xf, extent=extent)


# ==================================================================
#  Hero plot
# ==================================================================

def plot_hero(d, namer, source_kw, tag, Nfine=1024, mode="simulation",
              fov_override=None):
    """Source + lensed side by side."""
    rs   = float(d["r_s_Mpc"])
    kappa_s = compute_kappa_s(d)
    r_E_nfw = einstein_radius_nfw_Mpc(rs, kappa_s)
    fov = fov_override if fov_override else max(4.0 * rs, 0.5)

    out = make_lensed_image(d, fov, source_kw, Nfine=Nfine, mode=mode)
    ext = out["extent"]
    th = np.linspace(0, 2 * np.pi, 300)

    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(16, 7.5))
    vmax = max(out["source"].max(), out["lensed"].max(), 1e-10)

    # --- Left: unlensed source ---
    img_s = np.sqrt(np.clip(out["source"].T / vmax, 0, 1))
    ax_s.imshow(img_s, origin="lower", extent=ext,
                cmap="magma", vmin=0, vmax=1, aspect="equal")
    if r_E_nfw > fov * 0.005:
        ax_s.plot(r_E_nfw * np.cos(th), r_E_nfw * np.sin(th), "lime",
                  ls="-", lw=1.5, alpha=0.9,
                  label=r"$r_E^{\rm NFW}$ = %.4f Mpc" % r_E_nfw)
    ax_s.plot(rs * np.cos(th), rs * np.sin(th), "cyan", ls=":",
              lw=0.7, alpha=0.5, label=r"$r_s$ = %.3f Mpc" % rs)
    sc = source_kw.get("center", (0, 0))
    ax_s.plot(sc[0], sc[1], "w+", ms=14, mew=2, alpha=0.9)
    ax_s.set_title("Unlensed source", fontsize=12, color="white", pad=8)
    ax_s.set_xlabel(r"$b_1$ [Mpc]", fontsize=11, color="white")
    ax_s.set_ylabel(r"$b_2$ [Mpc]", fontsize=11, color="white")
    ax_s.legend(fontsize=9, loc="upper right",
                facecolor="black", edgecolor="gray",
                labelcolor="white", framealpha=0.7)

    # --- Right: lensed image ---
    img_l = np.sqrt(np.clip(out["lensed"].T / vmax, 0, 1))
    ax_l.imshow(img_l, origin="lower", extent=ext,
                cmap="magma", vmin=0, vmax=1, aspect="equal")
    if r_E_nfw > fov * 0.005:
        ax_l.plot(r_E_nfw * np.cos(th), r_E_nfw * np.sin(th), "lime",
                  ls="-", lw=1.5, alpha=0.9,
                  label=r"$r_E^{\rm NFW}$ = %.4f Mpc" % r_E_nfw)
    ax_l.plot(rs * np.cos(th), rs * np.sin(th), "cyan", ls=":",
              lw=0.7, alpha=0.4, label=r"$r_s$ = %.3f Mpc" % rs)
    ax_l.plot(sc[0], sc[1], "w+", ms=14, mew=2, alpha=0.9)

    label_mode = "geodesic D" if mode == "simulation" else "analytic NFW"
    ax_l.set_title("Lensed (%s)" % label_mode,
                   fontsize=12, color="white", pad=8)
    ax_l.set_xlabel(r"$b_1$ [Mpc]", fontsize=11, color="white")
    ax_l.legend(fontsize=9, loc="upper right",
                facecolor="black", edgecolor="gray",
                labelcolor="white", framealpha=0.7)

    for ax in (ax_s, ax_l):
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("gray")

    zl = float(d["z_l"]); zs = float(d["z_source"])
    M15 = float(d["M_200_Msun"]) / 1e15
    fig.suptitle(
        r"NFW $M_{200}$=%.1f$\times10^{15}\,M_\odot$, $c$=%.0f"
        r"  |  $z_l$=%.3f, $z_s$=%.3f  |  $\kappa_s$=%.3f"
        % (M15, float(d["c_NFW"]), zl, zs, kappa_s),
        fontsize=12, color="white", y=0.98,
    )
    fig.set_facecolor("black")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = namer.plot(tag)
    fig.savefig(fname, dpi=200, bbox_inches="tight", facecolor="black")
    print("  [ok] %s" % fname)
    plt.close(fig)


# ==================================================================
#  Diagnostic plot (6 panels)
# ==================================================================

def plot_diagnostic(d, namer, source_kw, tag, Nfine=1024, mode="simulation",
                    fov_override=None):
    """Source | Lensed | |alpha| // kappa | gamma whisker | mu."""
    rs   = float(d["r_s_Mpc"])
    kappa_s = compute_kappa_s(d)
    r_E_nfw = einstein_radius_nfw_Mpc(rs, kappa_s)
    fov = fov_override if fov_override else max(4.0 * rs, 0.5)
    out = make_lensed_image(d, fov, source_kw, Nfine=Nfine, mode=mode)
    ext = out["extent"]
    th = np.linspace(0, 2 * np.pi, 200)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    vmax = max(out["source"].max(), out["lensed"].max(), 1e-10)

    # -- (0,0) Unlensed source --
    ax = axes[0, 0]
    ax.imshow(np.sqrt(np.clip(out["source"].T / vmax, 0, 1)),
              origin="lower", extent=ext, cmap="magma", vmin=0, vmax=1)
    sc = source_kw.get("center", (0, 0))
    ax.plot(sc[0], sc[1], "w+", ms=10, mew=1.5)
    ax.set_title("Unlensed source", fontsize=11)
    ax.set_xlabel(r"$b_1$ [Mpc]"); ax.set_ylabel(r"$b_2$ [Mpc]")

    # -- (0,1) Lensed image --
    ax = axes[0, 1]
    ax.imshow(np.sqrt(np.clip(out["lensed"].T / vmax, 0, 1)),
              origin="lower", extent=ext, cmap="magma", vmin=0, vmax=1)
    if r_E_nfw > fov * 0.005:
        ax.plot(r_E_nfw * np.cos(th), r_E_nfw * np.sin(th), "lime", ls="-",
                lw=1.0, alpha=0.8, label=r"$r_E^{\rm NFW}$=%.4f" % r_E_nfw)
    ax.plot(rs * np.cos(th), rs * np.sin(th), "cyan", ls=":",
            lw=0.7, alpha=0.5, label=r"$r_s$=%.3f" % rs)
    ax.legend(fontsize=8, loc="upper right")
    src_label = "D matrix" if mode == "simulation" else "analytic"
    ax.set_title("Lensed (%s)" % src_label, fontsize=11)
    ax.set_xlabel(r"$b_1$ [Mpc]")

    # -- (0,2) Deflection magnitude --
    ax = axes[0, 2]
    alpha_mag = np.sqrt(out["alpha1"]**2 + out["alpha2"]**2)
    im = ax.imshow(alpha_mag.T, origin="lower", extent=ext, cmap="viridis")
    ax.plot(rs * np.cos(th), rs * np.sin(th), "w--", lw=0.6, alpha=0.7)
    ax.set_title(r"$|\alpha|$ (deflection) [Mpc]", fontsize=11)
    ax.set_xlabel(r"$b_1$ [Mpc]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # -- (1,0) kappa map --
    ax = axes[1, 0]
    kap = out["kappa"].T
    kap_pos = np.clip(kap, 1e-5, None)
    vmin_k = max(1e-3, kap_pos[kap_pos > 0].min() * 0.5)
    vmax_k = kap_pos.max()
    im = ax.imshow(kap_pos, origin="lower", extent=ext,
                   cmap="inferno", norm=LogNorm(vmin=vmin_k, vmax=vmax_k))
    ax.plot(rs * np.cos(th), rs * np.sin(th), "w--", lw=0.6)
    ax.set_title(r"$\kappa$ (convergence)", fontsize=11)
    ax.set_xlabel(r"$b_1$ [Mpc]"); ax.set_ylabel(r"$b_2$ [Mpc]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # -- (1,1) Shear whisker plot --
    ax = axes[1, 1]
    gamma_mag = np.sqrt(out["gamma1"]**2 + out["gamma2"]**2)
    im = ax.imshow(gamma_mag.T, origin="lower", extent=ext, cmap="viridis")

    # Whisker grid -- use the FFT-derived gamma (self-consistent)
    Nq = 24
    xq = np.linspace(-fov * 0.85, fov * 0.85, Nq)
    xq1, xq2 = np.meshgrid(xq, xq, indexing="ij")
    g1_q = _interp2d(out["gamma1"], out["xf"], xq)
    g2_q = _interp2d(out["gamma2"], out["xf"], xq)
    gm_q = np.sqrt(g1_q**2 + g2_q**2)

    # Shear stick angle: phi_gamma = 0.5 * arctan2(gamma2, gamma1)
    # For tangential shear around a mass this gives phi + pi/2
    phi_g = 0.5 * np.arctan2(g2_q, g1_q)
    scale = fov / Nq * 1.8
    gm_max = max(gm_q.max(), 1e-10)
    dx_w = gm_q / gm_max * np.cos(phi_g) * scale
    dy_w = gm_q / gm_max * np.sin(phi_g) * scale

    # quiver: x=b1 (axis 0), y=b2 (axis 1)
    # with ij indexing, xq1[i,j]=xq[i], xq2[i,j]=xq[j]
    # imshow .T makes axis0->col(x), axis1->row(y) -- so quiver needs .T too
    ax.quiver(xq1.T, xq2.T, dx_w.T, dy_w.T,
              color="white", alpha=0.85, headaxislength=0,
              headlength=0, pivot="middle", scale_units="xy",
              scale=1.0, linewidth=0.8)
    ax.plot(rs * np.cos(th), rs * np.sin(th), "lime", ls="--",
            lw=0.6, alpha=0.6)
    ax.set_title(r"$|\gamma|$ + shear whiskers", fontsize=11)
    ax.set_xlabel(r"$b_1$ [Mpc]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # -- (1,2) Magnification --
    ax = axes[1, 2]
    mu_data = out["mu"].T
    mu_clip = np.clip(mu_data, -20, 50)
    im = ax.imshow(mu_clip, origin="lower", extent=ext,
                   cmap="RdYlBu_r", vmin=-5, vmax=15)
    ax.plot(rs * np.cos(th), rs * np.sin(th), "k--", lw=0.6)
    ax.set_title(r"$\mu$ (magnification)", fontsize=11)
    ax.set_xlabel(r"$b_1$ [Mpc]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    zl = float(d["z_l"]); zs = float(d["z_source"])
    M15 = float(d["M_200_Msun"]) / 1e15
    fig.suptitle(
        r"Lensing diagnostic (%s)  |  NFW $M_{200}$=%.1f"
        r"$\times10^{15}\,M_\odot$,  $c$=%.0f  |  "
        r"$z_l$=%.3f,  $z_s$=%.3f  |  $\kappa_s$=%.3f"
        % ("D matrix" if mode == "simulation" else "analytic",
           M15, float(d["c_NFW"]), zl, zs, kappa_s),
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fname = namer.plot(tag + "_diag")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    print("  [ok] %s" % fname)
    plt.close(fig)


# ==================================================================
#  Deflection profile: simulation vs analytic
# ==================================================================

def plot_deflection_profile(d, namer):
    """Radial deflection & kappa: 1D profile from .npz vs analytic NFW."""
    rs   = float(d["r_s_Mpc"])
    R200 = float(d["R_200_Mpc"])
    kappa_s = compute_kappa_s(d)

    # --- Use the finely-sampled 1D profile stored in the .npz ---
    b_prof = d["b_profile_Mpc"]
    kap_prof = d["kappa_profile"].copy()

    # Subtract FLRW background (estimated from profile at large b)
    ka_large = d["kappa_analytic"][-3:]
    kn_large = kap_prof[-3:]
    kappa_bg = np.mean(kn_large - ka_large)
    kap_prof -= kappa_bg

    # Remove duplicate / b=0 entries for clean integration
    keep = b_prof > 0
    b_clean = b_prof[keep]
    k_clean = kap_prof[keep]
    # Sort by radius and remove duplicates
    order = np.argsort(b_clean)
    b_clean = b_clean[order]
    k_clean = k_clean[order]
    _, uniq = np.unique(b_clean, return_index=True)
    b_clean = b_clean[uniq]
    k_clean = k_clean[uniq]

    # Compute deflection by direct radial integration:
    # alpha(b) = bar_kappa(<b) * b = (2/b) * int_0^b kappa(b') b' db'
    # Use trapezoidal integration on the fine 1D profile.
    cum = np.zeros_like(b_clean)
    for i in range(1, len(b_clean)):
        db = b_clean[i] - b_clean[i - 1]
        cum[i] = cum[i - 1] + 0.5 * (
            k_clean[i] * b_clean[i] + k_clean[i - 1] * b_clean[i - 1]) * db
    a_prof = np.where(b_clean > 0, 2.0 * cum / b_clean, 0.0)

    r_max = min(3 * R200, float(d["map_half_Mpc"]))
    r_an = np.linspace(0.01, r_max, 500)
    a_an = _nfw_alpha_analytic(r_an, rs, kappa_s)
    k_an = _nfw_kappa_analytic(np.clip(r_an, 1e-10, None), rs, kappa_s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    mask_a = b_clean <= r_max
    ax1.plot(b_clean[mask_a], a_prof[mask_a], "C0o-", ms=3, lw=1.5,
             alpha=0.8, label=r"$|\alpha|$ from 1D $\kappa$ profile")
    ax1.plot(r_an, a_an, "C3-", lw=2, label=r"$|\alpha|$ analytic NFW")
    ax1.axvline(rs, color="C1", ls=":", alpha=0.6,
                label=r"$r_s$=%.3f Mpc" % rs)
    ax1.axvline(R200, color="C2", ls="--", alpha=0.6,
                label=r"$R_{200}$=%.2f Mpc" % R200)
    ax1.set_xlabel(r"$b$ [Mpc]", fontsize=11)
    ax1.set_ylabel(r"$|\alpha|$ [Mpc]", fontsize=11)
    ax1.set_title("Deflection: simulation D vs analytic", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, r_max)
    ax1.grid(alpha=0.3)

    ax2.plot(b_clean[mask_a], k_clean[mask_a], "C0o", ms=3, alpha=0.6,
             label=r"$\kappa$ from D matrix (bg-subtracted)")
    ax2.plot(r_an, k_an, "C3-", lw=2,
             label=r"$\kappa$ analytic NFW")
    ax2.axvline(rs, color="C1", ls=":", alpha=0.6)
    ax2.axvline(R200, color="C2", ls="--", alpha=0.6)
    ax2.set_xlabel(r"$b$ [Mpc]", fontsize=11)
    ax2.set_ylabel(r"$\kappa$", fontsize=11)
    ax2.set_title(r"Convergence: simulation D vs analytic", fontsize=12)
    ax2.set_yscale("log")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, r_max)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fname = namer.plot("deflection_profile")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print("  [ok] %s" % fname)
    plt.close(fig)


# ==================================================================
#  Shear comparison: simulation D vs analytic
# ==================================================================

def plot_shear_comparison(d, namer):
    """Compare tangential shear from D matrix vs analytic NFW."""
    lens = lensing_from_D_map(d)
    rs   = float(d["r_s_Mpc"])
    R200 = float(d["R_200_Mpc"])
    kappa_s = compute_kappa_s(d)

    b1 = lens["b1"]; b2 = lens["b2"]
    r = np.sqrt(b1**2 + b2**2)
    phi = np.arctan2(b2, b1)

    g1_sim = lens["gamma1"]; g2_sim = lens["gamma2"]
    # Tangential shear: gamma_t = -(gamma1 cos(2phi) + gamma2 sin(2phi))
    gamma_t_sim = -(g1_sim * np.cos(2 * phi) + g2_sim * np.sin(2 * phi))

    r_flat = r.ravel()
    order = np.argsort(r_flat)
    r_sorted = r_flat[order]
    gt_sorted = gamma_t_sim.ravel()[order]

    r_an = np.linspace(0.01, float(d["map_half_Mpc"]), 500)
    gt_an = _nfw_gamma_analytic(r_an, rs, kappa_s)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(r_sorted, gt_sorted, "C0o", ms=4, alpha=0.5,
            label=r"$\gamma_t$ from D matrix")
    ax.plot(r_an, gt_an, "C3-", lw=2,
            label=r"$\gamma_t$ analytic NFW")
    ax.axvline(rs, color="C1", ls=":", alpha=0.6,
               label=r"$r_s$=%.3f Mpc" % rs)
    ax.axvline(R200, color="C2", ls="--", alpha=0.6,
               label=r"$R_{200}$=%.2f Mpc" % R200)
    ax.set_xlabel(r"$b$ [Mpc]", fontsize=11)
    ax.set_ylabel(r"$\gamma_t$", fontsize=11)
    ax.set_title("Tangential shear: simulation D vs analytic NFW", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, min(3 * R200, float(d["map_half_Mpc"])))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = namer.plot("shear_comparison")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print("  [ok] %s" % fname)
    plt.close(fig)


# ==================================================================
#  Main
# ==================================================================

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = latest_run("lensing_nfw")

    if not os.path.isfile(path):
        sys.exit("File not found: %s" % path)

    d = np.load(path, allow_pickle=True)
    namer = RunNamer.from_npz(path)

    n1d = int(d["n_map_1d"])
    zl  = float(d["z_l"])
    zs  = float(d["z_source"])
    M15 = float(d["M_200_Msun"]) / 1e15
    rs  = float(d["r_s_Mpc"])

    kappa_s = compute_kappa_s(d)
    r_E_nfw = einstein_radius_nfw_Mpc(rs, kappa_s)

    # Sanity check: D-matrix kappa vs stored kappa_map
    lens = lensing_from_D_map(d)
    k_D = lens["kappa"].ravel()
    k_stored = d["kappa_map"]
    k_diff = np.max(np.abs(k_D - k_stored))

    print("Loaded: %s" % os.path.basename(path))
    print("  z_l=%.4f  z_s=%.4f" % (zl, zs))
    print("  M_200=%.1fe15 Msun  c=%.0f  rs=%.4f Mpc"
          % (M15, float(d["c_NFW"]), rs))
    print("  kappa_s = %.4f" % kappa_s)
    print("  r_E (NFW, bar_kappa=1) = %.6f Mpc" % r_E_nfw)
    dx_grid = 2 * float(d["map_half_Mpc"]) / n1d
    if r_E_nfw > 0 and r_E_nfw < dx_grid:
        print("  WARNING: NFW Einstein ring (%.2e Mpc) is below grid"
              " resolution (%.3f Mpc) -- no strong lensing possible"
              % (r_E_nfw, dx_grid))
    print("  Map: %dx%d  half=%.1f Mpc" % (n1d, n1d, float(d["map_half_Mpc"])))
    print("  kappa(D) vs kappa(stored): max|diff| = %.2e" % k_diff)
    print("  gamma range from D: |gamma| in [%.4e, %.4e]"
          % (np.sqrt(lens["gamma1"]**2 + lens["gamma2"]**2).min(),
             np.sqrt(lens["gamma1"]**2 + lens["gamma2"]**2).max()))
    print()

    # --- Layout: scale everything on rs (the physical halo scale) ---
    Nfine = 1024
    fov = max(4.0 * rs, 0.5)       # see ~10 rs across the field
    R_src = 0.1 * rs                # compact source relative to halo

    print("  rs = %.4f Mpc" % rs)
    print("  Source R_e = %.4f Mpc" % R_src)
    print("  FoV = +/- %.4f Mpc" % fov)
    print()

    # ---- Simulation D-matrix mode ----
    print("=== Simulation D-matrix mode ===")

    print("1) Einstein ring (source at centre)")
    src1 = dict(center=(0.0, 0.0), R_e=R_src, n=1.0, ellip=0.0, pa_deg=0.0)
    plot_hero(d, namer, src1, "lensed_ring", Nfine=Nfine,
              mode="simulation", fov_override=fov)
    plot_diagnostic(d, namer, src1, "lensed_ring", Nfine=Nfine,
                    mode="simulation", fov_override=fov)

    offset = rs * 0.5
    print("2) Arc (source offset = %.4f Mpc)" % offset)
    src2 = dict(center=(offset, 0.05 * rs), R_e=R_src * 0.8, n=1.0,
                ellip=0.4, pa_deg=30.0)
    plot_hero(d, namer, src2, "lensed_arc", Nfine=Nfine,
              mode="simulation", fov_override=fov)
    plot_diagnostic(d, namer, src2, "lensed_arc", Nfine=Nfine,
                    mode="simulation", fov_override=fov)

    print("3) Tangential arc (source at 2 rs)")
    src3 = dict(center=(rs * 2.0, 0.0), R_e=R_src, n=1.0,
                ellip=0.5, pa_deg=0.0)
    plot_hero(d, namer, src3, "lensed_tangential", Nfine=Nfine,
              mode="simulation", fov_override=fov)

    print("4) Weak lensing (source at 5 rs)")
    src4 = dict(center=(5.0 * rs, 0.0), R_e=R_src * 2, n=4.0,
                ellip=0.15, pa_deg=45.0)
    plot_hero(d, namer, src4, "lensed_weak", Nfine=Nfine,
              mode="simulation", fov_override=fov * 2)

    # ---- Analytic NFW comparison ----
    print()
    print("=== Analytic NFW comparison ===")

    print("5) Einstein ring (analytic)")
    plot_hero(d, namer, src1, "lensed_ring_analytic", Nfine=Nfine,
              mode="analytic", fov_override=fov)
    plot_diagnostic(d, namer, src1, "lensed_ring_analytic", Nfine=Nfine,
                    mode="analytic", fov_override=fov)

    print("6) Giant arc (analytic)")
    plot_hero(d, namer, src2, "lensed_arc_analytic", Nfine=Nfine,
              mode="analytic", fov_override=fov)

    # ---- Profile comparisons ----
    print()
    print("=== Profile comparisons ===")

    print("7) Deflection profile")
    plot_deflection_profile(d, namer)

    print("8) Shear comparison")
    plot_shear_comparison(d, namer)

    print("\nDone -- %d plots generated." % 10)


if __name__ == "__main__":
    main()
