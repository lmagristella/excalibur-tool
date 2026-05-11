#!/usr/bin/env python3
"""
Produce lensed galaxy images from an EXCALIBUR simulation .npz using
classical inverse ray tracing.

Unlike make_lensed_sky.py, this script does not use the Jacobi map D to
reconstruct the lensing field. Instead it:

1. reads the recorded photon trajectories,
2. intersects each ray with a source plane orthogonal to the mean optical
   axis,
3. projects each hit onto reduced source-plane coordinates,
4. interpolates that image-plane -> source-plane mapping to render the
   lensed source.

The reduced coordinates are the physical source-plane hit coordinates
rescaled by D_l / D_s so that the unlensed limit is beta ~= theta. This
keeps the source sizes and FoVs comparable to make_lensed_sky.py while the
mapping itself still comes from explicit ray/plane intersections.

This requires trajectory-enabled outputs, e.g. files containing:
    traj_x4_Mpc, traj_n_pts, obs_pos_Mpc

Usage:
    python make_lensed_sky_raytracing.py path/to/results.npz
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
from matplotlib.colors import SymLogNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from excalibur.io.filename_utils import RunNamer, latest_run


# ==================================================================
#  Physical constants
# ==================================================================
_G = 6.67430e-11      # m^3 kg^-1 s^-2
_c = 2.99792e8       # m/s
_Msun = 1.98892e30   # kg
_Mpc = 3.08568e22    # m


# ==================================================================
#  Analytic NFW helpers (overlay / reference scale only)
# ==================================================================

def compute_kappa_s(d):
    """kappa_s = rho_s * rs / Sigma_cr."""
    c_NFW = float(d["c_NFW"])
    M200 = float(d["M_200_Msun"]) * _Msun
    R200 = float(d["R_200_Mpc"]) * _Mpc
    rs = float(d["r_s_Mpc"]) * _Mpc
    Scr = float(d["Sigma_cr"])
    delta_c = (200.0 / 3.0) * c_NFW**3 / (
        np.log(1.0 + c_NFW) - c_NFW / (1.0 + c_NFW))
    rho_crit = M200 / (4.0 / 3.0 * np.pi * R200**3 * 200.0)
    rho_s = delta_c * rho_crit
    return rho_s * rs / Scr


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


def einstein_radius_nfw_Mpc(rs, kappa_s):
    """True Einstein radius of an NFW lens [Mpc]."""
    b_test = np.logspace(-8, np.log10(30.0 * rs), 20000)
    bk = _nfw_mean_kappa(b_test, rs, kappa_s)
    above = np.where(bk >= 1.0)[0]
    if len(above) == 0:
        return 0.0
    return float(b_test[above[-1]])


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
    u = c * db1 + s * db2
    v = -s * db1 + c * db2
    q = max(1.0 - ellip, 0.01)
    r = np.sqrt(u**2 + (v / q)**2)
    return I0 * np.exp(-b_n * ((r / R_e) ** (1.0 / n) - 1.0))


# ==================================================================
#  Trajectory-based ray tracing
# ==================================================================

def _normalize(vec):
    norm = np.linalg.norm(vec)
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero-length vector")
    return vec / norm


def _validate_raytrace_inputs(d):
    required = [
        "traj_x4_Mpc",
        "traj_n_pts",
        "obs_pos_Mpc",
        "b_profile_Mpc",
        "b1_map_Mpc",
        "b2_map_Mpc",
        "n_map_1d",
        "D_l_Mpc",
        "D_s_Mpc",
        "map_half_Mpc",
    ]
    missing = [key for key in required if key not in d]
    if missing:
        msg = [
            "This .npz does not contain the trajectory information required ",
            "for classical plane-intersection ray tracing.",
            "Missing keys: %s" % ", ".join(missing),
        ]
        if "D_flat_map" in d:
            msg.append(
                "This file does contain D_flat_map, so it supports "
                "make_lensed_sky.py but not explicit trajectory ray tracing."
            )
        raise KeyError("\n".join(msg))


def _initial_direction(traj_x4, n_pts):
    """Best-effort initial spatial direction from the first non-zero step."""
    pts = np.asarray(traj_x4[:n_pts, 1:4], dtype=np.float64)
    if pts.shape[0] < 2:
        raise ValueError("Trajectory has fewer than 2 recorded points")
    p0 = pts[0]
    for i in range(1, pts.shape[0]):
        dp = pts[i] - p0
        norm = np.linalg.norm(dp)
        if norm > 0.0:
            return dp / norm
    raise ValueError("Trajectory does not contain any non-zero displacement")


def _fit_raytrace_basis(traj_map_x4, traj_map_n_pts, b1_map, b2_map):
    """Recover the optical axis and transverse basis from the map rays."""
    dirs = np.array([
        _initial_direction(traj_map_x4[i], int(traj_map_n_pts[i]))
        for i in range(len(traj_map_n_pts))
    ])

    dir_hat = _normalize(np.mean(dirs, axis=0))
    dirs_perp = dirs - np.outer(dirs @ dir_hat, dir_hat)

    e1_vec = np.sum(b1_map[:, None] * dirs_perp, axis=0)
    e1_vec = e1_vec - np.dot(e1_vec, dir_hat) * dir_hat
    e1 = _normalize(e1_vec)

    e2_vec = np.sum(b2_map[:, None] * dirs_perp, axis=0)
    e2_vec = e2_vec - np.dot(e2_vec, dir_hat) * dir_hat
    e2_vec = e2_vec - np.dot(e2_vec, e1) * e1
    e2 = _normalize(e2_vec)

    if np.dot(np.cross(dir_hat, e1), e2) < 0.0:
        e2 = -e2

    return dir_hat, e1, e2


def _intersect_line_with_plane(p0, p1, plane_point, plane_normal):
    """Linear segment/plane intersection, allowing mild extrapolation."""
    s0 = np.dot(p0 - plane_point, plane_normal)
    s1 = np.dot(p1 - plane_point, plane_normal)
    denom = s1 - s0
    if abs(denom) < 1e-15:
        return p1.copy(), s1
    t = -s0 / denom
    p = p0 + t * (p1 - p0)
    resid = np.dot(p - plane_point, plane_normal)
    return p, resid


def _intersect_trajectory_with_plane(traj_x4, n_pts, plane_point, plane_normal):
    """Intersect one recorded trajectory with the source plane."""
    pts = np.asarray(traj_x4[:n_pts, 1:4], dtype=np.float64)
    signed = (pts - plane_point) @ plane_normal

    cross = np.where(signed[:-1] * signed[1:] <= 0.0)[0]
    if len(cross) > 0:
        idx = int(cross[0])
        return _intersect_line_with_plane(
            pts[idx], pts[idx + 1], plane_point, plane_normal)

    idx_min = int(np.argmin(np.abs(signed)))
    if idx_min == 0:
        return pts[0].copy(), signed[0]
    if idx_min == len(pts) - 1:
        return _intersect_line_with_plane(
            pts[-2], pts[-1], plane_point, plane_normal)

    if abs(signed[idx_min - 1]) <= abs(signed[idx_min + 1]):
        return _intersect_line_with_plane(
            pts[idx_min - 1], pts[idx_min], plane_point, plane_normal)
    return _intersect_line_with_plane(
        pts[idx_min], pts[idx_min + 1], plane_point, plane_normal)


def build_raytrace_map(d):
    """Construct the image-plane -> reduced source-plane mapping."""
    _validate_raytrace_inputs(d)

    n1d = int(d["n_map_1d"])
    n_profile = len(d["b_profile_Mpc"])
    n_map = n1d * n1d

    traj_x4 = np.asarray(d["traj_x4_Mpc"], dtype=np.float64)
    traj_n_pts = np.asarray(d["traj_n_pts"], dtype=np.int32)
    traj_map_x4 = traj_x4[n_profile:n_profile + n_map]
    traj_map_n_pts = traj_n_pts[n_profile:n_profile + n_map]

    b1_flat = np.asarray(d["b1_map_Mpc"], dtype=np.float64)
    b2_flat = np.asarray(d["b2_map_Mpc"], dtype=np.float64)
    obs_pos = np.asarray(d["obs_pos_Mpc"], dtype=np.float64)

    dir_hat, e1, e2 = _fit_raytrace_basis(
        traj_map_x4, traj_map_n_pts, b1_flat, b2_flat)

    D_l = float(d["D_l_Mpc"])
    D_s = float(d["D_s_Mpc"])
    source_center = obs_pos + D_s * dir_hat
    reduce_factor = D_l / D_s

    hits = np.empty((n_map, 3), dtype=np.float64)
    residuals = np.empty(n_map, dtype=np.float64)
    beta_phys = np.empty((n_map, 2), dtype=np.float64)
    beta_red = np.empty((n_map, 2), dtype=np.float64)

    for i in range(n_map):
        hit, resid = _intersect_trajectory_with_plane(
            traj_map_x4[i], int(traj_map_n_pts[i]), source_center, dir_hat)
        rel = hit - source_center
        beta1_phys = np.dot(rel, e1)
        beta2_phys = np.dot(rel, e2)

        hits[i] = hit
        residuals[i] = resid
        beta_phys[i, 0] = beta1_phys
        beta_phys[i, 1] = beta2_phys
        beta_red[i, 0] = beta1_phys * reduce_factor
        beta_red[i, 1] = beta2_phys * reduce_factor

    b1 = b1_flat.reshape(n1d, n1d)
    b2 = b2_flat.reshape(n1d, n1d)
    beta1 = beta_red[:, 0].reshape(n1d, n1d)
    beta2 = beta_red[:, 1].reshape(n1d, n1d)
    beta1_phys = beta_phys[:, 0].reshape(n1d, n1d)
    beta2_phys = beta_phys[:, 1].reshape(n1d, n1d)
    hit_resid = residuals.reshape(n1d, n1d)

    return dict(
        n1d=n1d,
        b1=b1,
        b2=b2,
        x_img=b1[:, 0].copy(),
        y_img=b2[0, :].copy(),
        beta1=beta1,
        beta2=beta2,
        beta1_phys=beta1_phys,
        beta2_phys=beta2_phys,
        dir_hat=dir_hat,
        e1=e1,
        e2=e2,
        obs_pos=obs_pos,
        source_center=source_center,
        D_l_Mpc=D_l,
        D_s_Mpc=D_s,
        reduce_factor=reduce_factor,
        hit_residual_Mpc=hit_resid,
        hits_Mpc=hits.reshape(n1d, n1d, 3),
    )


def _interp2d(field, x_from, y_from, x_to, y_to):
    """Interpolate a 2D field on a regular tensor-product grid."""
    kx = min(3, len(x_from) - 1)
    ky = min(3, len(y_from) - 1)
    spline = RectBivariateSpline(x_from, y_from, field, kx=kx, ky=ky)
    return spline(x_to, y_to)


def _jacobian_fields(beta1, beta2, dx):
    """Numerical Jacobian of the ray-traced mapping beta(theta)."""
    d11, d12 = np.gradient(beta1, dx, dx, edge_order=2)
    d21, d22 = np.gradient(beta2, dx, dx, edge_order=2)

    det_A = d11 * d22 - d12 * d21
    mu = np.where(np.abs(det_A) > 1e-30, 1.0 / det_A, 0.0)
    kappa = 1.0 - 0.5 * (d11 + d22)
    gamma1 = -0.5 * (d11 - d22)
    gamma2 = -0.5 * (d12 + d21)

    return d11, d12, d21, d22, det_A, mu, kappa, gamma1, gamma2


def make_lensed_image(d, raytrace, half_view, source_kw, Nfine=1024):
    """Render source and lensed image from explicit source-plane hits."""
    xf = np.linspace(-half_view, half_view, Nfine)
    yf = xf.copy()
    t1, t2 = np.meshgrid(xf, yf, indexing="ij")

    beta1 = _interp2d(raytrace["beta1"], raytrace["x_img"], raytrace["y_img"],
                      xf, yf)
    beta2 = _interp2d(raytrace["beta2"], raytrace["x_img"], raytrace["y_img"],
                      xf, yf)
    alpha1 = t1 - beta1
    alpha2 = t2 - beta2
    shift = np.sqrt(alpha1**2 + alpha2**2)

    dx = xf[1] - xf[0]
    _, _, _, _, _, mu, kappa, gamma1, gamma2 = _jacobian_fields(beta1, beta2, dx)

    lensed = sersic_source(beta1, beta2, **source_kw)
    source = sersic_source(t1, t2, **source_kw)
    extent = [float(xf[0]), float(xf[-1]), float(yf[0]), float(yf[-1])]

    return dict(
        lensed=lensed,
        source=source,
        beta1=beta1,
        beta2=beta2,
        alpha1=alpha1,
        alpha2=alpha2,
        shift=shift,
        kappa=kappa,
        gamma1=gamma1,
        gamma2=gamma2,
        mu=mu,
        xf=xf,
        yf=yf,
        extent=extent,
    )


# ==================================================================
#  Plotting
# ==================================================================

def _view_fov(d, requested=None):
    rs = float(d["r_s_Mpc"])
    default = max(4.0 * rs, 0.5)
    target = default if requested is None else requested
    return min(target, 0.98 * float(d["map_half_Mpc"]))


def plot_hero(d, raytrace, namer, source_kw, tag, Nfine=1024, fov_override=None):
    """Source + ray-traced lensed image side by side."""
    rs = float(d["r_s_Mpc"])
    kappa_s = compute_kappa_s(d)
    r_E_nfw = einstein_radius_nfw_Mpc(rs, kappa_s)
    fov = _view_fov(d, fov_override)

    out = make_lensed_image(d, raytrace, fov, source_kw, Nfine=Nfine)
    ext = out["extent"]
    th = np.linspace(0.0, 2.0 * np.pi, 300)

    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(16, 7.5))
    vmax = max(out["source"].max(), out["lensed"].max(), 1e-10)

    img_s = np.sqrt(np.clip(out["source"].T / vmax, 0.0, 1.0))
    ax_s.imshow(img_s, origin="lower", extent=ext,
                cmap="magma", vmin=0.0, vmax=1.0, aspect="equal")
    if r_E_nfw > fov * 0.005:
        ax_s.plot(r_E_nfw * np.cos(th), r_E_nfw * np.sin(th), "lime",
                  ls="-", lw=1.5, alpha=0.9,
                  label=r"$r_E^{\rm NFW}$ = %.4f Mpc" % r_E_nfw)
    ax_s.plot(rs * np.cos(th), rs * np.sin(th), "cyan", ls=":",
              lw=0.7, alpha=0.5, label=r"$r_s$ = %.3f Mpc" % rs)
    sc = source_kw.get("center", (0.0, 0.0))
    ax_s.plot(sc[0], sc[1], "w+", ms=14, mew=2, alpha=0.9)
    ax_s.set_title("Unlensed source", fontsize=12, color="white", pad=8)
    ax_s.set_xlabel(r"$b_1$ [Mpc]", fontsize=11, color="white")
    ax_s.set_ylabel(r"$b_2$ [Mpc]", fontsize=11, color="white")
    ax_s.legend(fontsize=9, loc="upper right",
                facecolor="black", edgecolor="gray",
                labelcolor="white", framealpha=0.7)

    img_l = np.sqrt(np.clip(out["lensed"].T / vmax, 0.0, 1.0))
    ax_l.imshow(img_l, origin="lower", extent=ext,
                cmap="magma", vmin=0.0, vmax=1.0, aspect="equal")
    if r_E_nfw > fov * 0.005:
        ax_l.plot(r_E_nfw * np.cos(th), r_E_nfw * np.sin(th), "lime",
                  ls="-", lw=1.5, alpha=0.9,
                  label=r"$r_E^{\rm NFW}$ = %.4f Mpc" % r_E_nfw)
    ax_l.plot(rs * np.cos(th), rs * np.sin(th), "cyan", ls=":",
              lw=0.7, alpha=0.4, label=r"$r_s$ = %.3f Mpc" % rs)
    ax_l.plot(sc[0], sc[1], "w+", ms=14, mew=2, alpha=0.9)
    ax_l.set_title("Lensed (ray/plane intersection)",
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

    zl = float(d["z_l"])
    zs = float(d["z_source"])
    M15 = float(d["M_200_Msun"]) / 1e15
    fig.suptitle(
        r"NFW $M_{200}$=%.1f$\times10^{15}\,M_\odot$, $c$=%.0f"
        r"  |  $z_l$=%.3f, $z_s$=%.3f  |  explicit ray/source-plane tracing"
        % (M15, float(d["c_NFW"]), zl, zs),
        fontsize=12, color="white", y=0.98,
    )
    fig.set_facecolor("black")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = namer.plot(tag)
    fig.savefig(fname, dpi=200, bbox_inches="tight", facecolor="black")
    print("  [ok] %s" % fname)
    plt.close(fig)


def plot_diagnostic(d, raytrace, namer, source_kw, tag, Nfine=1024, fov_override=None):
    """Diagnostics centered on the explicit ray hits on the source plane."""
    rs = float(d["r_s_Mpc"])
    fov = _view_fov(d, fov_override)
    out = make_lensed_image(d, raytrace, fov, source_kw, Nfine=Nfine)
    ext = out["extent"]
    vmax = max(out["source"].max(), out["lensed"].max(), 1e-10)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.imshow(np.sqrt(np.clip(out["lensed"].T / vmax, 0.0, 1.0)),
              origin="lower", extent=ext, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title("Lensed image (ray traced)", fontsize=11)
    ax.set_xlabel(r"$b_1$ [Mpc]")
    ax.set_ylabel(r"$b_2$ [Mpc]")

    ax = axes[0, 1]
    hit_b1 = raytrace["beta1"].ravel()
    hit_b2 = raytrace["beta2"].ravel()
    image_radius = np.sqrt(raytrace["b1"].ravel()**2 + raytrace["b2"].ravel()**2)
    sc = ax.scatter(hit_b1, hit_b2, c=image_radius, s=22,
                    cmap="viridis", edgecolors="none", alpha=0.9)
    src_center = source_kw.get("center", (0.0, 0.0))
    ax.plot(src_center[0], src_center[1], "r+", ms=12, mew=2)
    ax.set_title("Coarse ray hits on reduced source plane", fontsize=11)
    ax.set_xlabel(r"$\beta_1$ [Mpc]")
    ax.set_ylabel(r"$\beta_2$ [Mpc]")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                 label=r"$|b|$ on image plane [Mpc]")

    ax = axes[1, 0]
    im = ax.imshow(out["shift"].T, origin="lower", extent=ext, cmap="viridis")
    ax.plot(rs * np.cos(np.linspace(0.0, 2.0 * np.pi, 300)),
            rs * np.sin(np.linspace(0.0, 2.0 * np.pi, 300)),
            "w--", lw=0.6, alpha=0.7)
    ax.set_title(r"$|\theta - \beta|$ from ray intersections [Mpc]", fontsize=11)
    ax.set_xlabel(r"$b_1$ [Mpc]")
    ax.set_ylabel(r"$b_2$ [Mpc]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    mu_clip = np.clip(out["mu"].T, -20.0, 50.0)
    im = ax.imshow(mu_clip, origin="lower", extent=ext,
                   cmap="RdYlBu_r", vmin=-5.0, vmax=15.0)
    ax.set_title(r"$\mu$ from numerical Jacobian of $\beta(\theta)$", fontsize=11)
    ax.set_xlabel(r"$b_1$ [Mpc]")
    ax.set_ylabel(r"$b_2$ [Mpc]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fname = namer.plot(tag + "_diag")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    print("  [ok] %s" % fname)
    plt.close(fig)


def plot_mapping_fields(d, raytrace, namer):
    """Visualize beta1/beta2 and intersection residuals on the coarse map."""
    ext = [
        float(raytrace["x_img"][0]),
        float(raytrace["x_img"][-1]),
        float(raytrace["y_img"][0]),
        float(raytrace["y_img"][-1]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    vmax_b = max(np.max(np.abs(raytrace["beta1"])), np.max(np.abs(raytrace["beta2"])), 1e-8)

    im = axes[0].imshow(raytrace["beta1"].T, origin="lower", extent=ext,
                        cmap="coolwarm",
                        norm=SymLogNorm(linthresh=max(1e-4, 0.02 * vmax_b),
                                        vmin=-vmax_b, vmax=vmax_b))
    axes[0].set_title(r"$\beta_1$ (reduced source plane)")
    axes[0].set_xlabel(r"$b_1$ [Mpc]")
    axes[0].set_ylabel(r"$b_2$ [Mpc]")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    im = axes[1].imshow(raytrace["beta2"].T, origin="lower", extent=ext,
                        cmap="coolwarm",
                        norm=SymLogNorm(linthresh=max(1e-4, 0.02 * vmax_b),
                                        vmin=-vmax_b, vmax=vmax_b))
    axes[1].set_title(r"$\beta_2$ (reduced source plane)")
    axes[1].set_xlabel(r"$b_1$ [Mpc]")
    axes[1].set_ylabel(r"$b_2$ [Mpc]")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    resid_kpc = 1e3 * raytrace["hit_residual_Mpc"]
    vmax_r = max(np.max(np.abs(resid_kpc)), 1e-9)
    im = axes[2].imshow(resid_kpc.T, origin="lower", extent=ext,
                        cmap="RdBu_r",
                        norm=SymLogNorm(linthresh=max(1e-6, 0.02 * vmax_r),
                                        vmin=-vmax_r, vmax=vmax_r))
    axes[2].set_title("Source-plane intersection residual [kpc]")
    axes[2].set_xlabel(r"$b_1$ [Mpc]")
    axes[2].set_ylabel(r"$b_2$ [Mpc]")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fname = namer.plot("raytrace_mapping")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    print("  [ok] %s" % fname)
    plt.close(fig)


# ==================================================================
#  Main
# ==================================================================

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = latest_run("lensing_nfw_analytic") or latest_run("lensing_nfw")

    if not path:
        sys.exit("No .npz result file found")
    if not os.path.isfile(path):
        sys.exit("File not found: %s" % path)

    d = np.load(path, allow_pickle=True)
    namer = RunNamer.from_npz(path)

    try:
        raytrace = build_raytrace_map(d)
    except KeyError as exc:
        sys.exit(str(exc))

    zl = float(d["z_l"])
    zs = float(d["z_source"])
    rs = float(d["r_s_Mpc"])
    M15 = float(d["M_200_Msun"]) / 1e15
    max_resid_kpc = 1e3 * np.max(np.abs(raytrace["hit_residual_Mpc"]))
    rms_resid_kpc = 1e3 * np.sqrt(np.mean(raytrace["hit_residual_Mpc"]**2))

    print("Loaded: %s" % os.path.basename(path))
    print("  z_l=%.4f  z_s=%.4f" % (zl, zs))
    print("  M_200=%.1fe15 Msun  c=%.0f  rs=%.4f Mpc"
          % (M15, float(d["c_NFW"]), rs))
    print("  Map: %dx%d  half=%.3f Mpc"
          % (int(d["n_map_1d"]), int(d["n_map_1d"]), float(d["map_half_Mpc"])))
    print("  Optical axis = [%+.6f, %+.6f, %+.6f]"
          % tuple(raytrace["dir_hat"]))
    print("  Reduced source-plane scaling D_l/D_s = %.6f"
          % raytrace["reduce_factor"])
    print("  Plane-intersection residuals: rms=%.3e kpc  max=%.3e kpc"
          % (rms_resid_kpc, max_resid_kpc))
    print()

    Nfine = 1024
    fov = _view_fov(d)
    R_src = 0.1 * rs

    print("  Source R_e = %.4f Mpc" % R_src)
    print("  FoV = +/- %.4f Mpc" % fov)
    print()

    print("=== Ray-tracing on explicit source plane ===")

    print("1) Einstein ring (source at centre)")
    src1 = dict(center=(0.0, 0.0), R_e=R_src, n=1.0, ellip=0.0, pa_deg=0.0)
    plot_hero(d, raytrace, namer, src1, "lensed_ring_raytracing",
              Nfine=Nfine, fov_override=fov)
    plot_diagnostic(d, raytrace, namer, src1, "lensed_ring_raytracing",
                    Nfine=Nfine, fov_override=fov)

    offset = rs * 0.5
    print("2) Arc (source offset = %.4f Mpc)" % offset)
    src2 = dict(center=(offset, 0.05 * rs), R_e=R_src * 0.8, n=1.0,
                ellip=0.4, pa_deg=30.0)
    plot_hero(d, raytrace, namer, src2, "lensed_arc_raytracing",
              Nfine=Nfine, fov_override=fov)

    print("3) Tangential arc (source at 2 rs)")
    src3 = dict(center=(rs * 2.0, 0.0), R_e=R_src, n=1.0,
                ellip=0.5, pa_deg=0.0)
    plot_hero(d, raytrace, namer, src3, "lensed_tangential_raytracing",
              Nfine=Nfine, fov_override=fov)

    print("4) Weak lensing (source at 5 rs)")
    src4 = dict(center=(5.0 * rs, 0.0), R_e=R_src * 2.0, n=4.0,
                ellip=0.15, pa_deg=45.0)
    plot_hero(d, raytrace, namer, src4, "lensed_weak_raytracing",
              Nfine=Nfine, fov_override=_view_fov(d, 2.0 * fov))

    print("5) Mapping diagnostics")
    plot_mapping_fields(d, raytrace, namer)

    print("\nDone -- 5 ray-tracing plot groups generated.")


if __name__ == "__main__":
    main()