#!/usr/bin/env python3
r"""
Post-processing for the multi-photon lensing cone simulation.

Reads ``lensing_cone_results.npz`` and produces:
    1.  kappa(b) radial profile  vs  analytic prediction
    2.  |gamma|(b) radial profile
    3.  2D convergence map   kappa(b1, b2)
    4.  2D shear map         |gamma|(b1, b2)  + shear sticks
    5.  2D magnification map mu(b1, b2)
"""

import numpy as np
import os, sys

# -- plotting setup ------------------------------------------------
import matplotlib
matplotlib.use("Agg")            # non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from excalibur.io import RunNamer, latest_run


def load_data(path=None):
    if path is None:
        found = latest_run("lensing_cone")
        if found is not None:
            path = found
        else:
            path = os.path.join(
                os.path.dirname(__file__), "..", "_data", "output",
                "lensing_cone_results.npz",
            )
    d = np.load(path, allow_pickle=True)
    return d, path


def plot_radial_profiles(d, namer):
    """Plots 1-3: raw kappa(b), Deltakappa(b) vs analytic, and |gamma|(b) radial profiles."""
    b   = d["b_profile_Mpc"]
    k   = d["kappa_profile"]
    g   = d["gamma_profile"]
    ka  = d["kappa_analytic"]
    Rv  = float(d["R_vir_Mpc"])

    # Background subtraction: use outermost photon as reference
    k_bg = k[-1]
    dk = k - k_bg    # halo-specific convergence
    # For backward ray-tracing (k^0 < 0), the convergence sign is flipped
    # relative to the forward convention.  Take |Deltakappa| for the physical signal.
    dk = np.abs(dk)

    mask = b >= 0

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # --- Panel 1: raw kappa(b) (no background subtraction) ---
    ax = axes[0]
    ax.plot(b[mask], k[mask], "o-", ms=4, lw=1.5, color="C2",
            label=r"Numerical $\kappa$ (raw)")
    ax.axvline(Rv, color="grey", ls=":", lw=1, label=f"$R_{{vir}}$ = {Rv:.1f} Mpc")
    ax.axhline(k_bg, color="C3", ls="--", lw=1, alpha=0.7,
               label=rf"$\kappa_\mathrm{{bg}} = {k_bg:.3e}$")
    ax.set_xlabel("Impact parameter  $b$  [Mpc]", fontsize=12)
    ax.set_ylabel(r"$\kappa(b)$", fontsize=12)
    ax.set_title(r"Raw convergence $\kappa(b)$", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Deltakappa(b) vs analytic ---
    ax = axes[1]
    ax.plot(b[mask], dk[mask], "o-", ms=4, lw=1.5, label=r"Numerical $\Delta\kappa$")
    if ka.max() > 0:
        ax.plot(b[mask], ka[mask], "k--", lw=1.2, label=r"Analytic (uniform sphere)")
    ax.axvline(Rv, color="grey", ls=":", lw=1, label=f"$R_{{vir}}$")
    ax.axhline(0, color="k", ls="-", lw=0.5, alpha=0.3)
    ax.set_xlabel("Impact parameter  $b$  [Mpc]", fontsize=12)
    ax.set_ylabel(r"$\Delta\kappa = |\kappa(b) - \kappa_\mathrm{bg}|$", fontsize=12)
    ax.set_title(r"Convergence profile $\Delta\kappa(b)$", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: |gamma|(b) ---
    ax = axes[2]
    ax.plot(b[mask], g[mask], "s-", ms=4, lw=1.5, color="C1",
            label=r"Numerical $|\gamma|$")
    ax.axvline(Rv, color="grey", ls=":", lw=1, label=f"$R_{{vir}}$")
    ax.set_xlabel("Impact parameter  $b$  [Mpc]", fontsize=12)
    ax.set_ylabel(r"Shear  $|\gamma|$", fontsize=12)
    ax.set_title(r"Shear profile $|\gamma|(b)$", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Lensing profiles  --  {namer.title_line()}"
        rf"  --  $\kappa_\mathrm{{bg}} = {k_bg:.3e}$",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fname = namer.plot("profiles")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"   [ok] {fname}")
    plt.close(fig)


def plot_maps(d, namer):
    """Plots 3-5: 2D convergence, shear, and magnification maps."""
    n1d   = int(d["n_map_1d"])
    half  = float(d["map_half_Mpc"])
    b1    = d["b1_map_Mpc"]
    b2    = d["b2_map_Mpc"]
    kmap_raw = d["kappa_map"].reshape(n1d, n1d)
    gmap  = d["gamma_map"].reshape(n1d, n1d)
    mmap  = d["mu_map"].reshape(n1d, n1d)
    Rv    = float(d["R_vir_Mpc"])

    # Background subtraction: use the corner value (furthest from center)
    k_bg = kmap_raw[0, 0]  # corner pixel ~ far field
    kmap = np.abs(kmap_raw - k_bg)  # |Deltakappa| for backward ray-tracing

    extent = [-half, half, -half, half]
    circle_theta = np.linspace(0, 2 * np.pi, 100)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # === Convergence map ===
    ax = axes[0]
    vmax_k = max(abs(kmap.max()), abs(kmap.min()))
    if vmax_k == 0:
        vmax_k = 1e-6
    im = ax.imshow(
        kmap.T, origin="lower", extent=extent,
        cmap="RdBu_r", aspect="equal",
        norm=SymLogNorm(linthresh=vmax_k * 1e-3, vmin=-vmax_k, vmax=vmax_k),
    )
    ax.plot(Rv * np.cos(circle_theta), Rv * np.sin(circle_theta),
            "k--", lw=0.8, label=f"$R_{{vir}}$")
    ax.set_xlabel(r"$b_1$ [Mpc]")
    ax.set_ylabel(r"$b_2$ [Mpc]")
    ax.set_title(r"Convergence $\Delta\kappa$")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.legend(loc="upper right", fontsize=8)

    # === Shear map ===
    ax = axes[1]
    vmax_g = gmap.max()
    if vmax_g == 0:
        vmax_g = 1e-6
    im = ax.imshow(
        gmap.T, origin="lower", extent=extent,
        cmap="magma", aspect="equal",
        vmin=0, vmax=vmax_g,
    )
    ax.plot(Rv * np.cos(circle_theta), Rv * np.sin(circle_theta),
            "w--", lw=0.8)
    ax.set_xlabel(r"$b_1$ [Mpc]")
    ax.set_ylabel(r"$b_2$ [Mpc]")
    ax.set_title(r"Shear $|\gamma|$")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # === Magnification map ===
    ax = axes[2]
    # mu ~ 1 + 2kappa for weak lensing; show deviation from unity
    mu_dev = mmap - 1.0
    vmax_m = max(abs(mu_dev.max()), abs(mu_dev.min()))
    if vmax_m == 0:
        vmax_m = 1e-6
    im = ax.imshow(
        mu_dev.T, origin="lower", extent=extent,
        cmap="PRGn", aspect="equal",
        norm=SymLogNorm(linthresh=vmax_m * 1e-3, vmin=-vmax_m, vmax=vmax_m),
    )
    ax.plot(Rv * np.cos(circle_theta), Rv * np.sin(circle_theta),
            "k--", lw=0.8)
    ax.set_xlabel(r"$b_1$ [Mpc]")
    ax.set_ylabel(r"$b_2$ [Mpc]")
    ax.set_title(r"Magnification $\mu - 1$")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    fig.suptitle(
        f"2D Lensing Maps  --  {namer.title_line()}",
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

    # -- Recover lambda_S -------------------------------------------------
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

    # -- Reconstruct cosmology ----------------------------------------
    H0  = float(d["H0_kms_Mpc"]) if "H0_kms_Mpc" in d else 70.0
    Om  = float(d["Omega_m"])     if "Omega_m"     in d else 0.3
    Ol  = float(d["Omega_lambda"])if "Omega_lambda" in d else 0.7

    from excalibur.core.cosmology import LCDM_Cosmology
    cosmo = LCDM_Cosmology(H0, Omega_m=Om, Omega_r=0, Omega_lambda=Ol)

    # -- FLRW background distance ------------------------------------
    D_s_Mpc = float(d["D_s_Mpc"])
    D_s_m   = D_s_Mpc * one_Mpc

    from scipy.optimize import brentq
    try:
        z_source = brentq(lambda z: cosmo.comoving_distance(z) - D_s_m, 0.0, 20.0)
    except ValueError:
        z_source = 0.5
    DA_FLRW = cosmo.angular_diameter_distance(z_source)
    DA_FLRW_Mpc = DA_FLRW / one_Mpc

    # -- Ray-traced D_A for each photon (profile) --------------------
    D_flat_prof = d["D_flat_profile"]
    b_prof = d["b_profile_Mpc"]

    DA_ray_Mpc = np.empty(len(b_prof))
    for i in range(len(b_prof)):
        D_raw = D_flat_prof[i] * chi_S
        # Comoving  -> physical D_A: divide by (1+z_s)
        DA_ray_Mpc[i] = angular_diameter_distance_from_jacobi(D_raw) / one_Mpc / (1.0 + z_source)

    delta_DA = (DA_ray_Mpc - DA_FLRW_Mpc) / DA_FLRW_Mpc

    Rv = float(d["R_vir_Mpc"])

    # -- Ray-traced D_A for 2D map -----------------------------------
    has_map = "D_flat_map" in d
    if has_map:
        D_flat_map = d["D_flat_map"]
        n1d  = int(d["n_map_1d"])
        half = float(d["map_half_Mpc"])

        DA_map = np.empty(D_flat_map.shape[0])
        for i in range(len(DA_map)):
            D_raw = D_flat_map[i] * chi_S
            # Comoving  -> physical D_A: divide by (1+z_s)
            DA_map[i] = angular_diameter_distance_from_jacobi(D_raw) / one_Mpc / (1.0 + z_source)
        delta_map = ((DA_map - DA_FLRW_Mpc) / DA_FLRW_Mpc).reshape(n1d, n1d)

    # =====================  FIGURE  =================================
    ncols = 3 if has_map else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5.5))

    # --- Panel 1: D_A(b) profile ---
    ax = axes[0]
    ax.plot(b_prof, DA_ray_Mpc, "o-", ms=3, lw=1.2, color="C0",
            label=r"$D_A^{\mathrm{ray}}(b)$  (Jacobi)")
    ax.axhline(DA_FLRW_Mpc, color="C3", ls="--", lw=1.5,
               label=rf"$D_A^\mathrm{{FLRW}}(z_s={z_source:.3f}) = {DA_FLRW_Mpc:.1f}$ Mpc")
    ax.axvline(Rv, color="grey", ls=":", lw=1, label=rf"$R_{{vir}}$")
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
    ax.axvline(Rv, color="grey", ls=":", lw=1, label=rf"$R_{{vir}}$")
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
        ax.plot(Rv * np.cos(th), Rv * np.sin(th), "k--", lw=0.8, label="$R_{vir}$")
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

    # -- Console summary ---
    print(f"\n   Distance comparison (z_s ~ {z_source:.4f}):")
    print(f"     D_A^FLRW        = {DA_FLRW_Mpc:.2f} Mpc")
    print(f"     D_A^ray (b ->inf)   = {DA_ray_Mpc[-1]:.2f} Mpc")
    print(f"     deltaD_A/D_A (b ->inf)  = {delta_DA[-1]*100:.4f} %")
    inner = (b_prof > 0) & (b_prof < Rv)
    if inner.any():
        print(f"     deltaD_A/D_A (b<Rv): [{delta_DA[inner].min()*100:.4f}, "
              f"{delta_DA[inner].max()*100:.4f}] %")


def print_statistics(d):
    """Print summary statistics."""
    b   = d["b_profile_Mpc"]
    k   = d["kappa_profile"]
    g   = d["gamma_profile"]
    ka  = d["kappa_analytic"]

    # Background subtraction
    k_bg = k[-1]
    dk = np.abs(k - k_bg)  # |Deltakappa| for backward ray-tracing

    print("\n" + "=" * 60)
    print("  LENSING ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Grid             : {int(d['N_grid'])}^3, {float(d['box_Mpc']):.0f} Mpc")
    print(f"  Halo             : {float(d['M_Msun']):.0e} Msun, "
          f"R_vir = {float(d['R_vir_Mpc']):.1f} Mpc")
    print(f"  sigma                : {float(d['sigma_kms']):.0f} km/s")
    print(f"  n_steps          : {int(d['n_steps'])}")
    # Display lens geometry if available
    if 'D_l_Mpc' in d:
        print(f"  D_l              : {float(d['D_l_Mpc']):.1f} Mpc")
        print(f"  D_s              : {float(d['D_s_Mpc']):.1f} Mpc")
        print(f"  D_ls             : {float(d['D_ls_Mpc']):.1f} Mpc")
    print()
    print(f"  kappa_background     : {k_bg:.6e}")
    print(f"  Deltakappa range (halo)  : [{dk.min():.3e}, {dk.max():.3e}]")

    mask = b > 0
    if mask.any():
        # Compare numerical vs analytic
        mask_inside = (b > 0) & (b < float(d["R_vir_Mpc"]))
        if mask_inside.any() and ka[mask_inside].max() > 0:
            ratio = dk[mask_inside] / ka[mask_inside]
            ratio = ratio[np.isfinite(ratio) & (ka[mask_inside] > 0)]
            if len(ratio) > 0:
                print(f"\n  Deltakappa_num / kappa_analytic (b < R_vir):")
                print(f"     mean  = {ratio.mean():.4f}")
                print(f"     std   = {ratio.std():.4f}")
                print(f"     range = [{ratio.min():.4f}, {ratio.max():.4f}]")

    print()
    print(f"  Profile Deltakappa range : [{dk.min():.3e}, {dk.max():.3e}]")
    print(f"  Profile |gamma| range: [{g.min():.3e}, {g.max():.3e}]")

    kmap = d["kappa_map"]
    gmap = d["gamma_map"]
    k_map_bg = kmap.min()  # approximate background from map
    print(f"  Map Deltakappa range     : [{(kmap - k_map_bg).min():.3e}, {(kmap - k_map_bg).max():.3e}]")
    print(f"  Map |gamma| range    : [{gmap.min():.3e}, {gmap.max():.3e}]")
    print("=" * 60)


# =====================================================================
#  MAIN
# =====================================================================
def main():
    print("Loading lensing results ...")
    d, npz_path = load_data()
    print(f"   [ok] Loaded {len(d.files)} arrays  ({npz_path})")

    namer = RunNamer.from_npz(npz_path)

    print("\nGenerating radial profiles ...")
    plot_radial_profiles(d, namer)

    print("Generating 2D maps ...")
    plot_maps(d, namer)

    print("Generating distance comparison ...")
    plot_distance_comparison(d, namer)

    print_statistics(d)
    print(f"\n   All plots saved to {namer.outdir}")


if __name__ == "__main__":
    main()
