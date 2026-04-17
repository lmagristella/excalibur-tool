#!/usr/bin/env python3
r"""
Post-processing for the NFW multi-photon lensing simulation.

Reads ``lensing_nfw_results.npz`` and produces:
    1.  kappa(b) radial profile  vs  NFW analytic  (linear + log-log)
    2.  |gamma|(b) radial profile vs  NFW analytic  (linear + log-log)
    3.  2D convergence map   Deltakappa(b1, b2)
    4.  2D shear map         |gamma|(b1, b2)
    5.  2D magnification map mu-1

Usage::
    python analyze_lensing_nfw.py                     # default path
    python analyze_lensing_nfw.py  path/to/results.npz
"""

import numpy as np
import os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from excalibur.io.filename_utils import RunNamer, latest_run


# ------------------------------------------------------------------
#  Load
# ------------------------------------------------------------------
def load_data(path=None):
    if path is None:
        # Auto-discover the most recent lensing_nfw result file
        path = latest_run("lensing_nfw")
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__), "..", "_data", "output",
                "lensing_nfw_amr_rk4_FLRWP1_nfw_M2.1e15_c10_Rvir2.642_rs0.2642_zl0.53282_zs1_Dl1995_Ds3303.8_obs2000_2000_5_box4000_N256_AMR6_Nph5108_results.npz",
            )
    d = np.load(path, allow_pickle=True)
    return d, path


# ------------------------------------------------------------------
#  Radial profiles  (4-panel : kappa raw, Deltakappa lin, Deltakappa log, |gamma| log)
# ------------------------------------------------------------------
def plot_radial_profiles(d, namer):
    b   = d["b_profile_Mpc"]
    k   = d["kappa_profile"]
    g   = d["gamma_profile"]
    ka  = d["kappa_analytic"]
    ga  = d["gamma_analytic"]

    R200 = float(d["R_200_Mpc"])
    rs   = float(d["r_s_Mpc"])
    c_NFW = float(d["c_NFW"])

    k_bg = k[-1]
    dk   = np.abs(k - k_bg)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ---------- Panel 1 : raw kappa(b) ----------
    ax = axes[0, 0]
    ax.plot(b, k, "o-", ms=3, lw=1.2, color="C2", label=r"Numerical $\kappa$")
    ax.axhline(k_bg, color="C3", ls="--", lw=1, alpha=0.7,
               label=rf"$\kappa_\mathrm{{bg}} = {k_bg:.3e}$")
    ax.axvline(R200, color="grey", ls=":", lw=1, label=rf"$R_{{200}} = {R200:.2f}$ Mpc")
    ax.axvline(rs, color="orange", ls=":", lw=1, label=rf"$r_s = {rs:.2f}$ Mpc")
    ax.set_xlabel("Impact parameter  $b$  [Mpc]")
    ax.set_ylabel(r"$\kappa$")
    ax.set_title(r"Raw convergence $\kappa(b)$")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    # ---------- Panel 2 : Deltakappa linear ----------
    ax = axes[0, 1]
    mask = b > 0
    ax.plot(b[mask], dk[mask], "o-", ms=3, lw=1.2,
            label=r"Numerical $\Delta\kappa$")
    ax.plot(b[mask], ka[mask], "k--", lw=1.5,
            label=rf"NFW analytic ($c={c_NFW:.0f}$)")
    ax.axvline(R200, color="grey", ls=":", lw=1)
    ax.axvline(rs, color="orange", ls=":", lw=1)
    ax.set_xlabel("Impact parameter  $b$  [Mpc]")
    ax.set_ylabel(r"$\Delta\kappa = |\kappa - \kappa_\mathrm{bg}|$")
    ax.set_title(r"Convergence profile (linear)")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    # ---------- Panel 3 : Deltakappa log-log ----------
    ax = axes[1, 0]
    pos = mask & (dk > 0) & (ka > 0)
    ax.loglog(b[pos], dk[pos], "o", ms=4, color="C0",
              label=r"Numerical $\Delta\kappa$")
    ax.loglog(b[pos], ka[pos], "k-", lw=1.5,
              label=rf"NFW analytic ($c={c_NFW:.0f}$)")
    ax.axvline(R200, color="grey", ls=":", lw=1)
    ax.axvline(rs, color="orange", ls=":", lw=1, label=rf"$r_s$")
    ax.set_xlabel("$b$  [Mpc]")
    ax.set_ylabel(r"$\Delta\kappa$")
    ax.set_title(r"Convergence profile (log-log)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # ---------- Panel 4 : |gamma| log-log ----------
    ax = axes[1, 1]
    pos_g = mask & (g > 0)
    ax.loglog(b[pos_g], g[pos_g], "s", ms=4, color="C1",
              label=r"Numerical $|\gamma|$")
    pos_ga = mask & (ga > 0)
    ax.loglog(b[pos_ga], ga[pos_ga], "k-", lw=1.5,
              label=rf"NFW analytic $\gamma_t$ ($c={c_NFW:.0f}$)")
    ax.axvline(R200, color="grey", ls=":", lw=1)
    ax.axvline(rs, color="orange", ls=":", lw=1, label=rf"$r_s$")
    ax.set_xlabel("$b$  [Mpc]")
    ax.set_ylabel(r"$|\gamma|$")
    ax.set_title(r"Tangential shear profile (log-log)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"NFW Lensing Profiles  --  {namer.title_line()}",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fname = namer.plot("profiles")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"   [ok] {fname}")
    plt.close(fig)


# ------------------------------------------------------------------
#  Ratio plot  Deltakappa_num / kappa_analytic
# ------------------------------------------------------------------
def plot_ratio(d, namer):
    b  = d["b_profile_Mpc"]
    k  = d["kappa_profile"]
    ka = d["kappa_analytic"]
    ga = d["gamma_analytic"]
    g  = d["gamma_profile"]
    R200 = float(d["R_200_Mpc"])
    rs   = float(d["r_s_Mpc"])

    k_bg = k[-1]
    dk   = np.abs(k - k_bg)

    # Background-subtraction correction: the reference photon at b_max
    # still has NFW signal kappa_NFW(b_max).  The fair analytic comparison
    # is  kappa_NFW(b) - kappa_NFW(b_max)  rather than  kappa_NFW(b).
    ka_bg = ka[-1]                          # kappa_NFW at the outermost photon
    ka_corrected = np.maximum(ka - ka_bg, 0.0)  # background-subtracted analytic

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # kappa ratio  (corrected for background subtraction)
    ax = axes[0]
    ok = (b > 0.03) & (ka_corrected > 0) & np.isfinite(dk / ka_corrected)
    ratio_k = dk[ok] / ka_corrected[ok]
    ax.semilogx(b[ok], ratio_k, "o-", ms=4, color="C0",
                label=r"$\Delta\kappa_\mathrm{num} / (\kappa_\mathrm{NFW} - \kappa_\mathrm{NFW}^{bg})$")
    # Also show the uncorrected ratio in light grey for comparison
    ok_raw = (b > 0.03) & (ka > 0) & np.isfinite(dk / ka)
    ratio_raw = dk[ok_raw] / ka[ok_raw]
    ax.semilogx(b[ok_raw], ratio_raw, "x--", ms=3, color="0.6", lw=0.8, alpha=0.7,
                label=r"uncorrected $\Delta\kappa_\mathrm{num}/\kappa_\mathrm{NFW}$")
    ax.axhline(1.0, color="k", ls="-", lw=0.8, alpha=0.5)
    ax.axvline(R200, color="grey", ls=":", lw=1, label=rf"$R_{{200}}$")
    ax.axvline(rs, color="orange", ls=":", lw=1, label=rf"$r_s$")
    ax.set_xlabel("$b$  [Mpc]")
    ax.set_ylabel(r"$\Delta\kappa_\mathrm{num}\;/\;\kappa_\mathrm{NFW}^\mathrm{eff}$")
    ax.set_title(r"Convergence ratio (corrected for bg subtraction)")
    ax.set_ylim(0.5, 1.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # gamma ratio
    ax = axes[1]
    ok_g = (b > 0.03) & (ga > 0) & (g > 0) & np.isfinite(g / ga)
    ratio_g = g[ok_g] / ga[ok_g]
    ax.semilogx(b[ok_g], ratio_g, "s-", ms=4, color="C1")
    ax.axhline(1.0, color="k", ls="-", lw=0.8, alpha=0.5)
    ax.axvline(R200, color="grey", ls=":", lw=1, label=rf"$R_{{200}}$")
    ax.axvline(rs, color="orange", ls=":", lw=1, label=rf"$r_s$")
    ax.set_xlabel("$b$  [Mpc]")
    ax.set_ylabel(r"$|\gamma|_\mathrm{num}\;/\;\gamma_{t,\mathrm{NFW}}$")
    ax.set_title(r"Shear ratio")
    ax.set_ylim(0.5, 1.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Numerical / Analytic ratio", fontsize=14, y=1.02)
    fig.tight_layout()
    fname = namer.plot("ratio")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"   [ok] {fname}")
    plt.close(fig)


# ------------------------------------------------------------------
#  2D maps  (kappa, gamma, mu)
# ------------------------------------------------------------------
def plot_maps(d, namer):
    n1d   = int(d["n_map_1d"])
    half  = float(d["map_half_Mpc"])
    kmap_raw = d["kappa_map"].reshape(n1d, n1d)
    gmap  = d["gamma_map"].reshape(n1d, n1d)
    mmap  = d["mu_map"].reshape(n1d, n1d)
    R200  = float(d["R_200_Mpc"])
    rs    = float(d["r_s_Mpc"])

    k_bg  = kmap_raw[0, 0]
    kmap  = np.abs(kmap_raw - k_bg)

    extent = [-half, half, -half, half]
    th = np.linspace(0, 2 * np.pi, 200)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Convergence ---
    ax = axes[0]
    vmax = max(kmap.max(), 1e-8)
    im = ax.imshow(
        kmap.T, origin="lower", extent=extent,
        cmap="inferno", aspect="equal",
        norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax),
    )
    ax.plot(R200 * np.cos(th), R200 * np.sin(th), "w--", lw=0.8, label="$R_{200}$")
    ax.plot(rs * np.cos(th), rs * np.sin(th), "cyan", ls=":", lw=0.8, label="$r_s$")
    ax.set_xlabel(r"$b_1$ [Mpc]"); ax.set_ylabel(r"$b_2$ [Mpc]")
    ax.set_title(r"$\Delta\kappa$  (log scale)")
    div = make_axes_locatable(ax)
    plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))
    ax.legend(loc="upper right", fontsize=8)

    # --- Shear ---
    ax = axes[1]
    vmax_g = max(gmap.max(), 1e-8)
    im = ax.imshow(
        gmap.T, origin="lower", extent=extent,
        cmap="magma", aspect="equal",
        norm=LogNorm(vmin=vmax_g * 1e-4, vmax=vmax_g),
    )
    ax.plot(R200 * np.cos(th), R200 * np.sin(th), "w--", lw=0.8)
    ax.plot(rs * np.cos(th), rs * np.sin(th), "cyan", ls=":", lw=0.8)
    ax.set_xlabel(r"$b_1$ [Mpc]"); ax.set_ylabel(r"$b_2$ [Mpc]")
    ax.set_title(r"$|\gamma|$  (log scale)")
    div = make_axes_locatable(ax)
    plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))

    # --- Magnification ---
    ax = axes[2]
    mu_dev = mmap - 1.0
    vmax_m = max(abs(mu_dev).max(), 1e-8)
    im = ax.imshow(
        mu_dev.T, origin="lower", extent=extent,
        cmap="PRGn", aspect="equal",
        norm=SymLogNorm(linthresh=vmax_m * 1e-3, vmin=-vmax_m, vmax=vmax_m),
    )
    ax.plot(R200 * np.cos(th), R200 * np.sin(th), "k--", lw=0.8)
    ax.plot(rs * np.cos(th), rs * np.sin(th), "orange", ls=":", lw=0.8)
    ax.set_xlabel(r"$b_1$ [Mpc]"); ax.set_ylabel(r"$b_2$ [Mpc]")
    ax.set_title(r"$\mu - 1$")
    div = make_axes_locatable(ax)
    plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))

    fig.suptitle(
        f"NFW 2D Lensing Maps  --  {namer.title_line()}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fname = namer.plot("maps")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"   [ok] {fname}")
    plt.close(fig)


# ------------------------------------------------------------------
#  Angular-diameter distance comparison  (ray-traced vs FLRW)
# ------------------------------------------------------------------
def plot_distance_comparison(d, namer):
    """
    Compare the angular-diameter distance from the Jacobi map (ray-traced)
    with the FLRW background prediction.

    Works with both old npz files (without lambda_S/H0) and new ones.
    """
    from excalibur.observables.optical_tidal_matrix import angular_diameter_distance_from_jacobi
    from excalibur.core.constants import one_Mpc, c as c_light

    # -- Recover lambda_S  (affine distance to source) -----------------------
    # lambda is in seconds (affine parameterised so that k^0 = deta/dlambda ~ 1).
    # The comoving distance is chi = c * lambda_S, so the physical-units Jacobi
    # map is  D_phys = D_flat_norm * lambda_S * c   (metres per radian).
    if "lambda_S" in d:
        lambda_S = float(d["lambda_S"])
    elif "n_steps" in d and "dt_init" in d:
        lambda_S = float(d["n_steps"]) * float(d["dt_init"])
    else:
        print("   WARNING:  Cannot compute lambda_S  --  skipping distance comparison.")
        return

    chi_S = c_light * lambda_S       # comoving distance to source (metres)

    # -- Reconstruct cosmology -------------------------------------------
    H0  = float(d["H0_kms_Mpc"]) if "H0_kms_Mpc" in d else 70.0
    Om  = float(d["Omega_m"])     if "Omega_m"     in d else 0.3
    Ol  = float(d["Omega_lambda"])if "Omega_lambda" in d else 0.7

    from excalibur.core.cosmology import LCDM_Cosmology
    cosmo = LCDM_Cosmology(H0, Omega_m=Om, Omega_r=0, Omega_lambda=Ol)

    # -- FLRW background distance ----------------------------------------
    # D_s (comoving) is stored in the file; for the background D_A we
    # need a redshift.  We estimate z_s from the comoving distance.
    D_s_Mpc = float(d["D_s_Mpc"])
    D_s_m   = D_s_Mpc * one_Mpc

    # Invert chi(z)  -> z numerically
    from scipy.optimize import brentq
    try:
        z_source = brentq(lambda z: cosmo.comoving_distance(z) - D_s_m, 0.0, 20.0)
    except ValueError:
        z_source = 0.5   # safe fallback
    DA_FLRW = cosmo.angular_diameter_distance(z_source)
    DA_FLRW_Mpc = DA_FLRW / one_Mpc

    # -- Ray-traced D_A for each photon (profile) -----------------------
    D_flat_prof = d["D_flat_profile"]   # shape (n_profile, 4), normalised by lambda_actual
    b_prof = d["b_profile_Mpc"]

    # Use per-photon lambda_actual if available (more precise), else fall back to lambda_S
    has_lam_actual = "lambda_actual_profile" in d
    lam_actual_prof = d["lambda_actual_profile"] if has_lam_actual else None

    DA_ray_Mpc = np.empty(len(b_prof))
    for i in range(len(b_prof)):
        if has_lam_actual:
            # D_flat_prof was normalized by lambda_actual[i];
            # un-normalize to get D_flat in seconds, then convert to metres
            D_raw = D_flat_prof[i] * lam_actual_prof[i] * c_light
        else:
            D_raw = D_flat_prof[i] * chi_S      # old fallback
        # angular_diameter_distance_from_jacobi returns COMOVING D_A
        # (= sqrt|det D|); divide by (1+z_s) to get physical D_A.
        DA_ray_Mpc[i] = angular_diameter_distance_from_jacobi(D_raw) / one_Mpc / (1.0 + z_source)

    delta_DA = (DA_ray_Mpc - DA_FLRW_Mpc) / DA_FLRW_Mpc

    R200 = float(d["R_200_Mpc"])
    rs   = float(d["r_s_Mpc"])

    # -- Ray-traced D_A for 2D map ---------------------------------------
    has_map = "D_flat_map" in d
    if has_map:
        D_flat_map = d["D_flat_map"]
        n1d  = int(d["n_map_1d"])
        half = float(d["map_half_Mpc"])
        has_lam_map = "lambda_actual_map" in d
        lam_actual_map = d["lambda_actual_map"] if has_lam_map else None

        DA_map = np.empty(D_flat_map.shape[0])
        for i in range(len(DA_map)):
            if has_lam_map:
                D_raw = D_flat_map[i] * lam_actual_map[i] * c_light
            else:
                D_raw = D_flat_map[i] * chi_S
            # Comoving  -> physical D_A
            DA_map[i] = angular_diameter_distance_from_jacobi(D_raw) / one_Mpc / (1.0 + z_source)
        delta_map = ((DA_map - DA_FLRW_Mpc) / DA_FLRW_Mpc).reshape(n1d, n1d)

    # =====================  FIGURE  ====================================
    ncols = 3 if has_map else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5.5))

    # --- Panel 1: D_A(b) profile (absolute) ---
    ax = axes[0]
    ax.plot(b_prof, DA_ray_Mpc, "o-", ms=3, lw=1.2, color="C0",
            label=r"$D_A^{\mathrm{ray}}(b)$  (Jacobi)")
    ax.axhline(DA_FLRW_Mpc, color="C3", ls="--", lw=1.5,
               label=rf"$D_A^\mathrm{{FLRW}}(z_s={z_source:.3f}) = {DA_FLRW_Mpc:.1f}$ Mpc")
    ax.axvline(R200, color="grey", ls=":", lw=1, label=rf"$R_{{200}}$")
    ax.axvline(rs, color="orange", ls=":", lw=1, label=rf"$r_s$")
    ax.set_xlabel("Impact parameter  $b$  [Mpc]")
    ax.set_ylabel(r"$D_A$  [Mpc]")
    ax.set_title(r"Angular-diameter distance profile")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: deltaD_A / D_A profile ---
    ax = axes[1]
    ax.plot(b_prof, delta_DA * 100, "o-", ms=3, lw=1.2, color="C4")
    ax.axhline(0, color="k", ls="-", lw=0.5, alpha=0.4)
    ax.axvline(R200, color="grey", ls=":", lw=1, label=rf"$R_{{200}}$")
    ax.axvline(rs, color="orange", ls=":", lw=1, label=rf"$r_s$")
    ax.set_xlabel("Impact parameter  $b$  [Mpc]")
    ax.set_ylabel(r"$\delta D_A / D_A^\mathrm{FLRW}$  [%]")
    ax.set_title(r"Relative distance deviation")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: 2D deltaD_A map ---
    if has_map:
        ax = axes[2]
        extent = [-half, half, -half, half]
        vmax = max(abs(delta_map).max(), 1e-6)
        im = ax.imshow(
            delta_map.T * 100, origin="lower", extent=extent,
            cmap="RdBu_r", aspect="equal",
            norm=SymLogNorm(linthresh=vmax * 1e-2 * 100, vmin=-vmax * 100, vmax=vmax * 100),
        )
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R200 * np.cos(th), R200 * np.sin(th), "k--", lw=0.8, label="$R_{200}$")
        ax.plot(rs * np.cos(th), rs * np.sin(th), "orange", ls=":", lw=0.8, label="$r_s$")
        ax.set_xlabel(r"$b_1$ [Mpc]"); ax.set_ylabel(r"$b_2$ [Mpc]")
        ax.set_title(r"$\delta D_A / D_A^\mathrm{FLRW}$  [%]")
        div = make_axes_locatable(ax)
        plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        rf"Angular-diameter distance  --  {namer.title_line()}"
        rf"  --  $D_A^\mathrm{{FLRW}} = {DA_FLRW_Mpc:.1f}$ Mpc  ($z_s = {z_source:.3f}$)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fname = namer.plot("distance")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"   [ok] {fname}")
    plt.close(fig)

    # -- Summary to console ---
    print(f"\n   Distance comparison (z_s ~ {z_source:.4f}):")
    print(f"     D_A^FLRW        = {DA_FLRW_Mpc:.2f} Mpc")
    print(f"     D_A^ray (b ->inf)   = {DA_ray_Mpc[-1]:.2f} Mpc")
    print(f"     deltaD_A/D_A (b ->inf)  = {delta_DA[-1]*100:.4f} %")
    inner = (b_prof > 0.03) & (b_prof < R200)
    if inner.any():
        print(f"     deltaD_A/D_A (b<R200): [{delta_DA[inner].min()*100:.4f}, "
              f"{delta_DA[inner].max()*100:.4f}] %")


# ------------------------------------------------------------------
#  Statistics
# ------------------------------------------------------------------
def print_statistics(d):
    b  = d["b_profile_Mpc"]
    k  = d["kappa_profile"]
    g  = d["gamma_profile"]
    ka = d["kappa_analytic"]
    ga = d["gamma_analytic"]
    R200 = float(d["R_200_Mpc"])
    rs   = float(d["r_s_Mpc"])

    k_bg = k[-1]
    dk = np.abs(k - k_bg)

    print("\n" + "=" * 65)
    print("  NFW LENSING ANALYSIS SUMMARY")
    print("=" * 65)
    print(f"  Halo  : NFW, M_200 = {float(d['M_200_Msun']):.0e} Msun, "
          f"c = {float(d['c_NFW']):.0f}")
    print(f"          R_200 = {R200:.3f} Mpc,  r_s = {rs*1000:.0f} kpc")
    print(f"  Grid  : {int(d['N_grid'])}^3, {float(d['box_Mpc']):.0f} Mpc")
    if 'D_l_Mpc' in d:
        print(f"  D_l = {float(d['D_l_Mpc']):.2f},  D_s = {float(d['D_s_Mpc']):.2f},  "
              f"D_ls = {float(d['D_ls_Mpc']):.2f}  Mpc")
    if 'Sigma_cr' in d:
        from excalibur.core.constants import one_Mpc, one_Msun
        Scr = float(d['Sigma_cr']) * one_Mpc**2 / one_Msun
        print(f"  Sigma_cr = {Scr:.3e} Msun/Mpc^2")
    print()
    print(f"  kappa_bg (far field)    : {k_bg:.6e}")
    print(f"  Deltakappa max  (numerical) : {dk.max():.4e}")
    print(f"  kappa  max  (analytic)  : {ka.max():.4e}")
    print(f"  |gamma| max (numerical) : {g.max():.4e}")
    print(f"  gamma_t max (analytic)  : {ga.max():.4e}")

    # Ratio inside R_200  (corrected for bg subtraction)
    ka_bg = ka[-1]
    ka_corr = np.maximum(ka - ka_bg, 0.0)

    mask = (b > 0.03) & (b < R200) & (ka_corr > 0)
    if mask.any():
        ratio_k = dk[mask] / ka_corr[mask]
        ratio_k = ratio_k[np.isfinite(ratio_k)]
        if len(ratio_k) > 0:
            print(f"\n  Deltakappa_num / (kappa_NFW - kappa_NFW^bg)  (0.03 < b < R_200, bg-corrected):")
            print(f"     mean  = {ratio_k.mean():.4f}")
            print(f"     std   = {ratio_k.std():.4f}")
            print(f"     range = [{ratio_k.min():.4f}, {ratio_k.max():.4f}]")

    # Uncorrected ratio for reference
    mask_raw = (b > 0.03) & (b < R200) & (ka > 0)
    if mask_raw.any():
        ratio_raw = dk[mask_raw] / ka[mask_raw]
        ratio_raw = ratio_raw[np.isfinite(ratio_raw)]
        if len(ratio_raw) > 0:
            print(f"  (uncorrected Deltakappa_num/kappa_NFW: mean = {ratio_raw.mean():.4f} +/- {ratio_raw.std():.4f})")

    mask_g = (b > 0.03) & (b < R200) & (ga > 0) & (g > 0)
    if mask_g.any():
        ratio_g = g[mask_g] / ga[mask_g]
        ratio_g = ratio_g[np.isfinite(ratio_g)]
        if len(ratio_g) > 0:
            print(f"\n  |gamma|_num / gamma_NFW  (0.03 < b < R_200):")
            print(f"     mean  = {ratio_g.mean():.4f}")
            print(f"     std   = {ratio_g.std():.4f}")
            print(f"     range = [{ratio_g.min():.4f}, {ratio_g.max():.4f}]")

    print("=" * 65)


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = None

    print("Loading NFW lensing results ...")
    d, npz_path = load_data(path)
    print(f"   [ok] Loaded {len(d.files)} arrays from {os.path.basename(npz_path)}")

    # Build a RunNamer from the npz path  --  all plot names auto-match
    namer = RunNamer.from_npz(npz_path)
    print(f"   RunNamer: {namer.stem}")

    print("\nGenerating radial profiles ...")
    plot_radial_profiles(d, namer)

    print("Generating ratio plot ...")
    plot_ratio(d, namer)

    print("Generating 2D maps ...")
    plot_maps(d, namer)

    print("Generating distance comparison ...")
    plot_distance_comparison(d, namer)

    print_statistics(d)
    print(f"\n   All plots saved to {namer.outdir}")


if __name__ == "__main__":
    main()
