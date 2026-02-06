#!/usr/bin/env python3
"""Analyze and compare trajectory outputs from the Schwarzschild-vs-FLRW runner.

Robust version:
- Determines spherical vs cartesian from filename/meta (NOT a numeric heuristic)
- Reconstructs Schwarzschild xyz in lens-centered frame and shifts by lens_center if available
- Uses cartesian spatial velocity for bending and speed in both cases
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np

from excalibur.core.constants import one_Mpc, c, G
from excalibur.io.filename_utils import parse_trajectory_filename


def _load_trajectories(path: str):
    """Return (trajs, attrs). trajs shape: (n_photons, n_records, n_cols)."""
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        trajs = data["trajectories"]
        attrs = {k: data[k].item() if data[k].shape == () else data[k] for k in data.files if k != "trajectories"}
        return trajs, attrs

    try:
        import h5py  # type: ignore

        with h5py.File(path, "r") as f:
            trajs = f["trajectories"][...]
            attrs = {k: f.attrs[k] for k in f.attrs.keys()}
        return trajs, attrs
    except Exception as e:
        raise RuntimeError(f"Could not read {path} as HDF5. Try the .npz fallback. Error: {e}")


def _unit(v, eps=1e-30):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def _infer_is_spherical(path: str, attrs: dict) -> bool:
    """Decide whether trajectory stores (r,theta,phi) or (x,y,z)."""
    # 1) Try meta attr "metric" (written by your runner)
    m = attrs.get("metric", None)
    if isinstance(m, (bytes, np.bytes_)):
        m = m.decode("utf-8", errors="ignore")
    if isinstance(m, str):
        ml = m.lower()
        if "schwarz" in ml:
            return True
        if "flrw" in ml:
            return False

    # 2) Try filename parser
    parsed = parse_trajectory_filename(path)
    if parsed and parsed.get("metric_type"):
        mt = str(parsed["metric_type"]).lower()
        if "schwarz" in mt:
            return True
        if "flrw" in mt:
            return False

    # 3) Last resort: assume cartesian (safer for FLRW-like outputs)
    return False


@dataclass
class TrajStats:
    n_records: int
    finite_fraction: float
    path_length: float
    min_r: float
    final_r: float
    bending_angle_rad: float
    speed_mean: float
    speed_max: float


def _sph_to_xyz_rel(r, th, ph):
    """(r,theta,phi) -> xyz in the lens-centered frame."""
    sth = np.sin(th)
    return np.stack(
        [
            r * sth * np.cos(ph),
            r * sth * np.sin(ph),
            r * np.cos(th),
        ],
        axis=1,
    )


def _sph_vel_to_cart(r, th, ph, dr, dth, dph):
    """Convert (dr, dtheta, dphi) into cartesian velocity components in lens-centered frame."""
    sth = np.sin(th)
    cth = np.cos(th)
    sph = np.sin(ph)
    cph = np.cos(ph)

    # e_r     = (sinθ cosφ, sinθ sinφ, cosθ)
    # e_theta = (cosθ cosφ, cosθ sinφ, -sinθ)
    # e_phi   = (-sinφ, cosφ, 0)
    vx = dr * (sth * cph) + (r * dth) * (cth * cph) + (r * sth * dph) * (-sph)
    vy = dr * (sth * sph) + (r * dth) * (cth * sph) + (r * sth * dph) * (cph)
    vz = dr * (cth) + (r * dth) * (-sth)
    return np.stack([vx, vy, vz], axis=1)


def _trajectory_stats(traj: np.ndarray, lens_center: np.ndarray | None, *, is_spherical: bool) -> TrajStats:
    n_records = int(traj.shape[0])
    u = traj[:, 4:8]

    # Build absolute xyz positions + cartesian spatial velocities
    if is_spherical:
        r = traj[:, 1]
        th = traj[:, 2]
        ph = traj[:, 3]
        xyz_rel = _sph_to_xyz_rel(r, th, ph)

        # In your runner, Schwarzschild coordinates are lens-centered by construction.
        if lens_center is None:
            xyz = xyz_rel
        else:
            xyz = xyz_rel + lens_center[None, :]

        dr = u[:, 1]
        dth = u[:, 2]
        dph = u[:, 3]
        v_cart = _sph_vel_to_cart(r, th, ph, dr, dth, dph)
    else:
        xyz = traj[:, 1:4]
        v_cart = u[:, 1:4]

    finite = np.isfinite(traj).all(axis=1)
    finite_fraction = float(finite.mean()) if n_records else 0.0

    # Path length in lens-centered coordinates when available
    if lens_center is None:
        xyz_path = xyz
    else:
        xyz_path = xyz - lens_center[None, :]

    if n_records >= 2:
        xyz_finite = np.isfinite(xyz_path).all(axis=1)
        seg_ok = xyz_finite[1:] & xyz_finite[:-1]
        dxyz = xyz_path[1:] - xyz_path[:-1]
        seg_len = np.linalg.norm(dxyz, axis=1)
        path_length = float(np.sum(np.where(seg_ok, seg_len, 0.0)))
    else:
        path_length = float("nan")

    # Distance to lens center
    rr = np.linalg.norm(xyz_path, axis=1)
    rr = np.where(np.isfinite(rr), rr, np.nan)

    min_r = float(np.nanmin(rr)) if n_records else float("nan")
    final_r = float(rr[-1]) if n_records else float("nan")

    # Bending angle: compare first finite and last finite cartesian spatial directions
    if n_records >= 2:
        v_finite = np.isfinite(v_cart).all(axis=1)
        if v_finite.any():
            i0 = int(np.argmax(v_finite))
            i1 = int(len(v_finite) - 1 - np.argmax(v_finite[::-1]))
            if i1 > i0:
                v0 = _unit(v_cart[i0])
                v1 = _unit(v_cart[i1])
                dot = float(np.clip(np.dot(v0, v1), -1.0, 1.0))
                bending = float(np.arccos(dot))
            else:
                bending = 0.0
        else:
            bending = 0.0
    else:
        bending = 0.0

    # Coordinate speed proxy: |v_cart| / |deta/dλ|
    detadl = u[:, 0]
    vmag = np.linalg.norm(v_cart, axis=1)
    denom = np.where(np.abs(detadl) > 0, np.abs(detadl), np.nan)
    speed = vmag / denom
    speed_mean = float(np.nanmean(speed))
    speed_max = float(np.nanmax(speed))

    return TrajStats(
        n_records=n_records,
        finite_fraction=finite_fraction,
        path_length=path_length,
        min_r=min_r,
        final_r=final_r,
        bending_angle_rad=bending,
        speed_mean=speed_mean,
        speed_max=speed_max,
    )


def _summarize(stats: list[TrajStats]) -> dict:
    arr_n = np.array([s.n_records for s in stats], dtype=float)
    arr_f = np.array([s.finite_fraction for s in stats], dtype=float)
    arr_L = np.array([s.path_length for s in stats], dtype=float)
    arr_rmin = np.array([s.min_r for s in stats], dtype=float)
    arr_b = np.array([s.bending_angle_rad for s in stats], dtype=float)

    def _basic(x):
        x = np.asarray(x, dtype=float)
        return {
            "mean": float(np.nanmean(x)),
            "std": float(np.nanstd(x)),
            "min": float(np.nanmin(x)),
            "max": float(np.nanmax(x)),
            "p50": float(np.nanquantile(x, 0.50)),
            "p90": float(np.nanquantile(x, 0.90)),
        }

    return {
        "n_records": _basic(arr_n),
        "finite_fraction": _basic(arr_f),
        "path_length": _basic(arr_L),
        "min_r": _basic(arr_rmin),
        "bending_angle_rad": _basic(arr_b),
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze and compare two trajectory files")
    ap.add_argument("--schwarzschild", required=True, help="Path to Schwarzschild trajectory output (.h5 or .npz)")
    ap.add_argument("--flrw", required=True, help="Path to FLRW trajectory output (.h5 or .npz)")
    ap.add_argument("--csv", default=None, help="Optional output CSV path")
    args = ap.parse_args()

    s_trajs, s_meta = _load_trajectories(args.schwarzschild)
    f_trajs, f_meta = _load_trajectories(args.flrw)

    # Lens center inference (from filename)
    lens_center = None
    parsed_s = parse_trajectory_filename(args.schwarzschild)
    parsed_f = parse_trajectory_filename(args.flrw)
    parsed = parsed_s or parsed_f
    if parsed and parsed.get("mass_position_m") is not None:
        lens_center = np.asarray(parsed["mass_position_m"], dtype=float)

    s_is_sph = _infer_is_spherical(args.schwarzschild, s_meta)
    f_is_sph = _infer_is_spherical(args.flrw, f_meta)

    s_stats = [_trajectory_stats(s_trajs[i], lens_center, is_spherical=s_is_sph) for i in range(s_trajs.shape[0])]
    f_stats = [_trajectory_stats(f_trajs[i], lens_center, is_spherical=f_is_sph) for i in range(f_trajs.shape[0])]

    s_sum = _summarize(s_stats)
    f_sum = _summarize(f_stats)

    def _fmt_basic(d):
        return f"mean={d['mean']:.4g} std={d['std']:.3g} min={d['min']:.4g} p50={d['p50']:.4g} p90={d['p90']:.4g} max={d['max']:.4g}"

    print("=== Files ===")
    print("Schwarzschild:", args.schwarzschild, f"(spherical={int(s_is_sph)})")
    print("FLRW:         ", args.flrw,          f"(spherical={int(f_is_sph)})")
    if lens_center is not None:
        lc_mpc = lens_center / one_Mpc
        print(f"Lens center (from filename): [{lc_mpc[0]:.3g}, {lc_mpc[1]:.3g}, {lc_mpc[2]:.3g}] Mpc")

    print("\n=== Meta (FLRW) ===")
    for k in sorted(f_meta.keys()):
        print(f"  {k}: {f_meta[k]}")

    print("\n=== Trajectory stats (per photon distribution) ===")
    for name, summ in [("Schwarzschild", s_sum), ("FLRW", f_sum)]:
        print(f"\n-- {name} --")
        print("n_records        ", _fmt_basic(summ["n_records"]))
        print("finite_fraction  ", _fmt_basic(summ["finite_fraction"]))
        print("path_length [m]  ", _fmt_basic(summ["path_length"]))
        print("min_r [m]        ", _fmt_basic(summ["min_r"]))
        print("bend angle [rad] ", _fmt_basic(summ["bending_angle_rad"]))

    # Direct comparison
    s_b = np.array([st.bending_angle_rad for st in s_stats], dtype=float)
    f_b = np.array([st.bending_angle_rad for st in f_stats], dtype=float)
    delta_b = f_b - s_b

    arcsec = (180.0 / np.pi) * 3600.0
    print("\n=== Direct comparison (FLRW - Schwarzschild) ===")
    print(f"bend angle delta [rad]: mean={np.nanmean(delta_b):.3e} std={np.nanstd(delta_b):.3e} min={np.nanmin(delta_b):.3e} max={np.nanmax(delta_b):.3e}")
    print(f"bend angle delta [arcsec]: mean={(np.nanmean(delta_b)*arcsec):.3e} std={(np.nanstd(delta_b)*arcsec):.3e}")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(
                [
                    "photon",
                    "S_n_records",
                    "S_path_length_m",
                    "S_min_r_m",
                    "S_bend_rad",
                    "F_n_records",
                    "F_path_length_m",
                    "F_min_r_m",
                    "F_bend_rad",
                    "delta_bend_rad",
                ]
            )
            n = min(len(s_stats), len(f_stats))
            for i in range(n):
                w.writerow(
                    [
                        i,
                        s_stats[i].n_records,
                        s_stats[i].path_length,
                        s_stats[i].min_r,
                        s_stats[i].bending_angle_rad,
                        f_stats[i].n_records,
                        f_stats[i].path_length,
                        f_stats[i].min_r,
                        f_stats[i].bending_angle_rad,
                        delta_b[i],
                    ]
                )
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
