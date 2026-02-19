"""Comparable runs: Schwarzschild vs (perturbed) FLRW.

Goal
----
Run two scenarios with the *same lens mass* and the *same observer–lens distance*:

1. Schwarzschild (exact, static, spherical coordinates).
2. Perturbed FLRW (Newtonian-gauge scalar perturbation from the same mass profile).

This script is meant for "physics runs": it focuses on reproducibility and
precise output naming.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np

from excalibur.core.constants import c, one_Gpc, one_Mpc, one_Msun
from excalibur.core.coordinates import cartesian_to_spherical
from excalibur.core.cosmology import LCDM_Cosmology, StaticCosmology
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator import Interpolator
#from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.grid.interpolator_4d_fast import InterpolatorFast
from excalibur.integration.integrator import Integrator
from excalibur.io.filename_utils import generate_trajectory_filename
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.metrics.schwarzschild_metric import SchwarzschildMetric
from excalibur.metrics.schwarzschild_metric_fast import SchwarzschildMetricFast
from excalibur.objects.spherical_mass import spherical_mass
from excalibur.photon.photons import Photons


class _ShiftedInterpolator:
    """Wrapper that shifts query points before interpolating.

    This lets us integrate in a numerically stable, local coordinate system (e.g.
    relative to the observer) without changing the underlying grid or metric code.
    """

    def __init__(self, base, shift: np.ndarray):
        self._base = base
        self._shift = np.asarray(shift, dtype=float)

    def value_gradient_and_time_derivative(self, x, field, t=None):
        x = np.asarray(x, dtype=float) + self._shift
        return self._base.value_gradient_and_time_derivative(x, field, t)

    def interpolate(self, x, field, t=None):
        x = np.asarray(x, dtype=float) + self._shift
        return self._base.interpolate(x, field, t)


@dataclass(frozen=True)
class RunConfig:
    mass: float
    radius: float
    obs_to_lens_distance: float
    n_photons: int
    cone_angle_deg: float
    n_steps: int
    integrator: str
    mode: str
    output_dir: str
    static: bool
    flrw_grid_N: int
    flrw_grid_size: float
    record_every: int
    flrw_fast: bool
    progress_every: int
    steps_per_cell: float
    flrw_local_coords: bool
    trace_norm: bool
    renormalize_every: int
    sampling: str
    b_min: float
    b_max: float
    b_nbins: int
    b_nperbin: int


def _parse_args() -> RunConfig:
    p = argparse.ArgumentParser(
        description="Run comparable Schwarzschild vs FLRW lensing simulations",
    )

    p.add_argument("--mass-msun", type=float, default=1e15, help="Lens mass in Msun")
    p.add_argument("--radius-mpc", type=float, default=3.0, help="Lens radius in Mpc")
    p.add_argument(
        "--distance-mpc",
        type=float,
        default=500.0,
        help="Observer-to-lens distance in Mpc",
    )

    p.add_argument("--n-photons", type=int, default=200)
    p.add_argument("--cone-angle-deg", type=float, default=10.0)
    p.add_argument("--n-steps", type=int, default=5000)
    p.add_argument("--integrator", type=str, default="rk4", choices=["rk4", "rk45", "leapfrog4"])
    p.add_argument("--mode", type=str, default="sequential", choices=["sequential", "parallel"])

    p.add_argument("--sampling", type=str, default="cone", choices=["cone", "impact"])

    p.add_argument("--b-min-mpc", type=float, default=0.5, help="Impact parameter min (Mpc). Used if sampling=impact.")
    p.add_argument("--b-max-mpc", type=float, default=10.0, help="Impact parameter max (Mpc). Used if sampling=impact.")
    p.add_argument("--b-nbins", type=int, default=10, help="Number of impact-parameter bins. Used if sampling=impact.")
    p.add_argument("--b-nperbin", type=int, default=50, help="Photons per bin. Used if sampling=impact.")


    p.add_argument(
        "--static",
        action="store_true",
        help="Use a static cosmology for FLRW: a(eta)=1 and adot=0",
    )

    p.add_argument(
        "--flrw-grid-N",
        type=int,
        default=256,
        help="FLRW potential grid resolution per axis",
    )
    p.add_argument(
        "--flrw-grid-size-mpc",
        type=float,
        default=1000.0,
        help="FLRW grid physical size in Mpc (box size)",
    )

    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("_data", "output"),
        help="Directory to write trajectory HDF5 outputs",
    )

    p.add_argument(
        "--record-every",
        type=int,
        default=50,
        help="Record one state every N steps (0 = only final state).",
    )

    p.add_argument(
        "--no-flrw-fast",
        action="store_true",
        help="Disable the fast FLRW metric/interpolator (use the reference implementation instead).",
    )

    p.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N photons (0 disables).",
    )

    p.add_argument(
        "--steps-per-cell",
        type=float,
        default=3.0,
        help=(
            "Heuristic for setting dlambda: target ~N integration steps per FLRW grid cell "
            "based on cell_size/c (larger dlambda -> fewer, bigger steps)."
        ),
    )

    p.add_argument(
        "--flrw-local-coords",
        action="store_true",
        help=(
            "Integrate FLRW using a numerically stable local coordinate system: "
            "x_local = x_abs - observer_pos. Queries into the FLRW grid are shifted back "
            "to absolute coordinates internally. Output files still store absolute x,y,z."
        ),
    )

    p.add_argument(
        "--trace-norm",
        action="store_true",
        help=(
            "Record per-step relative null-condition error on each Photon as `photon.norm_history`. "
            "(Useful for debugging drift; not yet written into the HDF5 output by default.)"
        ),
    )

    p.add_argument(
        "--renormalize-every",
        type=int,
        default=0,
        help=(
            "If >0, project the 4-velocity back onto the null cone every N accepted steps by "
            "recomputing u0 from the metric and fixed u_spatial."
        ),
    )

    args = p.parse_args()

    return RunConfig(
        mass=args.mass_msun * one_Msun,
        radius=args.radius_mpc * one_Mpc,
        obs_to_lens_distance=args.distance_mpc * one_Mpc,
        n_photons=args.n_photons,
        cone_angle_deg=args.cone_angle_deg,
        n_steps=args.n_steps,
        integrator=args.integrator,
        mode=args.mode,
        output_dir=args.output_dir,
        static=args.static,
        flrw_grid_N=args.flrw_grid_N,
        flrw_grid_size=args.flrw_grid_size_mpc * one_Mpc,
        record_every=args.record_every,
        # Prefer the fast implementation by default unless explicitly disabled.
        # (Most of the validation/bench tooling in this repo targets the fast path.)
        flrw_fast=not bool(args.no_flrw_fast),
        progress_every=args.progress_every,
        steps_per_cell=args.steps_per_cell,
        flrw_local_coords=bool(args.flrw_local_coords),
        trace_norm=bool(args.trace_norm),
        renormalize_every=int(args.renormalize_every),
        sampling=args.sampling,
        b_min=args.b_min_mpc * one_Mpc,
        b_max=args.b_max_mpc * one_Mpc,
        b_nbins=args.b_nbins,
        b_nperbin=args.b_nperbin,
    )


def _integrate_all(
    photons: Photons,
    integrator: Integrator,
    n_steps: int,
    *,
    record_every: int,
    progress_every: int,
    label: str,
    x_offset: np.ndarray | None = None,
    trace_norm: bool = False,
    renormalize_every: int = 0,
):
    trajectories = []
    n = len(photons.photons)
    x_offset_arr = None if x_offset is None else np.asarray(x_offset, dtype=float)
    for idx, photon in enumerate(photons.photons):
        if progress_every and (idx % progress_every) == 0:
            print(f"[{label}] photon {idx+1}/{n}")

        try:
            integrator.integrate_single(
                photon,
                stop_mode="steps",
                stop_value=n_steps,
                record_every=record_every,
                trace_norm=trace_norm,
                renormalize_every=renormalize_every,
            )
        except ValueError as exc:
            msg = str(exc)
            if "inside the massive object" in msg:
                print(f"[{label}] photon {idx+1}/{n} stopped: {msg}")
            else:
                raise

        # Photon.history is a PhotonHistory wrapper; states may occasionally have
        # different lengths (e.g., if some quantities are appended). Pad to a
        # rectangular array to keep HDF5-friendly output.
        states = list(getattr(photon.history, "states", []))
        if not states:
            hist = np.zeros((0, 0), dtype=float)
        else:
            max_len = max(len(s) for s in states)
            hist = np.full((len(states), max_len), np.nan, dtype=float)
            for ii, s in enumerate(states):
                s_arr = np.asarray(s, dtype=float)
                hist[ii, : s_arr.shape[0]] = s_arr

            # If requested, shift position columns (x,y,z) for output.
            # State layout is assumed to be [eta, x, y, z, ...] for cartesian metrics.
            if x_offset_arr is not None and max_len >= 4:
                hist[:, 1:4] = hist[:, 1:4] + x_offset_arr[None, :]

        trajectories.append(hist)
    return trajectories


def main():
    cfg = _parse_args()

    # Geometry convention:
    # The FLRW potential lives on a finite periodic box [0, L)^3.
    # To avoid leaving the box during integration, we place the lens at the
    # *box center* and the observer at a distance D along -x.
    # Schwarzschild uses the same relative geometry.
    lens_center_cart = np.array(
        [0.5 * cfg.flrw_grid_size, 0.5 * cfg.flrw_grid_size, 0.5 * cfg.flrw_grid_size],
        dtype=float,
    )
    # If the requested observer-to-lens distance doesn't fit in the box geometry, clamp.
    # Keep the observer at least ~one cell away from the x=0 boundary.
    # Also require y/z stay within the box: since we place the observer only along -x,
    # y,z coordinates equal the box center and are always valid. The main constraint is x.
    # Note: distances are in meters. To ensure the observer stays within the box,
    # we need D <= L/2 - one_cell.
    max_D = 0.5 * cfg.flrw_grid_size - (cfg.flrw_grid_size / cfg.flrw_grid_N)
    if cfg.obs_to_lens_distance > max_D:
        print(
            f"[runner] distance_mpc too large for box: clamping D from {cfg.obs_to_lens_distance/one_Mpc:.3g} "
            f"to {max_D/one_Mpc:.3g} Mpc so observer stays inside grid"
        )
        obs_to_lens_distance = float(max_D)
    else:
        obs_to_lens_distance = float(cfg.obs_to_lens_distance)

    observer_cart = lens_center_cart + np.array([-obs_to_lens_distance, 0.0, 0.0], dtype=float)
    # Safety clamp all coordinates into the strict interior required by InterpolatorFast.
    # Its fused interpolation+gradient routine uses central differences and therefore
    # requires 1 <= index <= N-3 (i.e. position in [origin+dx, origin+(N-2)*dx)).
    cell = cfg.flrw_grid_size / cfg.flrw_grid_N
    inner_min = 1.0 * cell
    inner_max = (cfg.flrw_grid_N - 2.0) * cell  # exclusive upper bound in practice
    observer_cart = np.clip(observer_cart, inner_min, inner_max)
    lens_center_cart = np.clip(lens_center_cart, inner_min, inner_max)
    central_dir_cart = lens_center_cart - observer_cart

    # We'll integrate to a travel distance of 2D by default (observer -> beyond lens).
    # IMPORTANT NUMERICS:
    # Positions are O(1e24..1e25) meters in typical setups. If dx per step is too small,
    # float64 rounding can swallow the update entirely, yielding frozen coordinates.
    # To avoid this, choose dlambda from the FLRW grid cell size: aim for ~N steps per cell.
    travel_distance = 2.0 * obs_to_lens_distance
    cell_size = cfg.flrw_grid_size / float(cfg.flrw_grid_N)
    dx_target = cell_size / float(cfg.steps_per_cell)
    dlambda = dx_target / c

    # Keep total distance roughly consistent with n_steps by adjusting stop steps instead
    # of shrinking dlambda into the rounding-noise regime.
    n_steps_effective = int(np.ceil(travel_distance / dx_target))
    if n_steps_effective < 1:
        n_steps_effective = 1
    if cfg.n_steps != n_steps_effective:
        print(
            f"[runner] Overriding n_steps {cfg.n_steps} -> {n_steps_effective} "
            f"to match steps_per_cell={cfg.steps_per_cell} (cell={cell_size:.3e} m)"
        )
    n_steps = n_steps_effective

    cone_angle = np.deg2rad(cfg.cone_angle_deg)

    if cfg.sampling == "impact":
        b_edges = np.linspace(cfg.b_min, cfg.b_max, cfg.b_nbins + 1)
        n_photons_effective = cfg.b_nbins * cfg.b_nperbin
    else:
        b_edges = None
        n_photons_effective = cfg.n_photons


    # -----------------------------
    # Schwarzschild run
    # -----------------------------

    if cfg.integrator == "rk45":
        dt_min = abs(dlambda) / 1e6
        dt_max = abs(dlambda)
    else:
        dt_min = dt_max = abs(dlambda)

    schwarz_metric = SchwarzschildMetricFast(
        mass=cfg.mass,
        radius=cfg.radius,
        center=lens_center_cart,
        coords="spherical",
        analytical_geodesics=True
    )

    schwarz_integrator = Integrator(
        metric=schwarz_metric,
        dt=dlambda,
        mode=cfg.mode,
        integrator=cfg.integrator,
        dt_min=dt_min,
        dt_max=dt_max,
    )

    schwarz_photons = Photons(metric=schwarz_metric)
    origin_sph = cartesian_to_spherical(*(observer_cart - lens_center_cart))
    origin_sph_4d = np.array([0.0, *origin_sph], dtype=float)
    origin_cart_4d = np.array([0.0, *observer_cart], dtype=float)

    if cfg.sampling == "cone":
        schwarz_photons.generate_cone_grid(
            n_photons=n_photons_effective,
            origin=origin_sph_4d,
            central_direction=central_dir_cart,
            cone_angle=cone_angle,
            direction_basis="cartesian",
            direction_coords=None,
        )
    else:
        schwarz_photons.generate_impact_parameter_bins(
            origin=origin_sph_4d,                 # sphérique (t,r,theta,phi)
            central_direction=central_dir_cart,   # cartésien
            b_edges=b_edges,
            n_per_bin=cfg.b_nperbin,
            direction_basis="cartesian",
            seed=0,
            u0_sign=-1.0,                         # backward ray tracing
        )



    schwarz_before = np.array([p.photon_norm(schwarz_metric) for p in schwarz_photons.photons])
    schwarz_trajs = _integrate_all(
        schwarz_photons,
        schwarz_integrator,
        n_steps,
        record_every=cfg.record_every,
        progress_every=cfg.progress_every,
        label="Schwarzschild",
        trace_norm=cfg.trace_norm,
        renormalize_every=cfg.renormalize_every,
    )
    schwarz_after = np.array([p.photon_norm(schwarz_metric) for p in schwarz_photons.photons])

    schwarz_fname = generate_trajectory_filename(
        mass_kg=cfg.mass,
        radius_m=cfg.radius,
        mass_position_m=lens_center_cart,
        observer_position_m=observer_cart,
        metric_type="schwarzschild",
        n_photons=n_photons_effective,
        integrator=cfg.integrator,
        stop_mode="steps",
    stop_value=n_steps,
        output_dir=cfg.output_dir,
        extra_tags={
            "D": cfg.obs_to_lens_distance / one_Mpc,
            "cone": f"{cfg.cone_angle_deg:.3g}deg",
        },
    )

    # -----------------------------
    # FLRW run
    # -----------------------------
    if cfg.static:
        cosmo = StaticCosmology()
    else:
        cosmo = LCDM_Cosmology(H0=70, Omega_m=0.3, Omega_r=0.0, Omega_lambda=0.7)

    N = cfg.flrw_grid_N
    grid_size = cfg.flrw_grid_size
    dx = dy = dz = grid_size / N
    # IMPORTANT: `spherical_mass.potential` is evaluated on coordinates x,y,z in [0, L)
    # (absolute meters). Therefore the Grid must use origin=(0,0,0) to match.
    # If Grid defaults to origin=0.5 (normalized coords), interpolation bounds checks
    # will incorrectly reject valid physical points.
    grid = Grid(shape=(N, N, N), spacing=(dx, dy, dz), origin=(0.0, 0.0, 0.0))
    interp_base = InterpolatorFast(grid) if cfg.flrw_fast else Interpolator(grid)

    x = y = z = np.linspace(0, grid_size, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    mass_object = spherical_mass(mass=cfg.mass, radius=cfg.radius, center=lens_center_cart)
    phi = mass_object.potential(X, Y, Z)
    grid.add_field("Phi", phi)

    # Optional local coordinates for FLRW integration (reversible): integrate positions
    # relative to the observer to avoid float64 cancellation when |x|~1e25.
    flrw_shift = observer_cart if cfg.flrw_local_coords else None
    interp = _ShiftedInterpolator(interp_base, flrw_shift) if flrw_shift is not None else interp_base

    if cfg.flrw_fast:
        flrw_metric = PerturbedFLRWMetricFast(
            a_of_eta=cosmo.a_of_eta,
            adot_of_eta=cosmo.adot_of_eta,
            grid=grid,
            interpolator=interp,
        )
    else:
        flrw_metric = PerturbedFLRWMetric(cosmo, grid, interp)
    flrw_integrator = Integrator(
        metric=flrw_metric,
        dt=dlambda,
        mode=cfg.mode,
        integrator=cfg.integrator,
        dt_min=abs(dlambda),
        dt_max=abs(dlambda),
    )

    flrw_photons = Photons(metric=flrw_metric)
    eta0 = float(getattr(cosmo, "_eta_at_a1", 0.0))
    observer_for_state = (observer_cart - flrw_shift) if flrw_shift is not None else observer_cart
    lens_for_state = (lens_center_cart - flrw_shift) if flrw_shift is not None else lens_center_cart
    central_dir_for_state = lens_for_state - observer_for_state
    
    if cfg.sampling == "cone":
        flrw_photons.generate_cone_grid(
            n_photons=n_photons_effective,
            origin=np.array([eta0, *observer_for_state], dtype=float),
            central_direction=central_dir_for_state,
            cone_angle=cone_angle,
            direction_basis="cartesian",
        )
    else:
        flrw_photons.generate_impact_parameter_bins(
            origin=np.array([eta0, *observer_for_state], dtype=float),
            central_direction=central_dir_for_state,
            b_edges=b_edges,
            n_per_bin=cfg.b_nperbin,
            direction_basis="cartesian",
            seed=0,
            u0_sign=-1.0,
        )


    u = np.array([p.u for p in flrw_photons.photons])
    speed = np.linalg.norm(u[:,1:4], axis=1)
    print("FLRW |u_spatial| mean/min/max =", speed.mean(), speed.min(), speed.max())
    print("expected ~c =", c)
    print("dlambda [s] =", dlambda, "dx per step [m] mean =", speed.mean()*dlambda)


    flrw_before = np.array([p.photon_norm(flrw_metric) for p in flrw_photons.photons])
    flrw_trajs = _integrate_all(
        flrw_photons,
        flrw_integrator,
        n_steps,
        record_every=cfg.record_every,
        progress_every=cfg.progress_every,
        label="FLRW",
        x_offset=flrw_shift,
        trace_norm=cfg.trace_norm,
        renormalize_every=cfg.renormalize_every,
    )
    flrw_after = np.array([p.photon_norm(flrw_metric) for p in flrw_photons.photons])

    metric_type = "perturbed_flrw_static" if cfg.static else "perturbed_flrw"
    flrw_fname = generate_trajectory_filename(
        mass_kg=cfg.mass,
        radius_m=cfg.radius,
        mass_position_m=lens_center_cart,
        observer_position_m=observer_cart,
        metric_type=metric_type,
        n_photons=n_photons_effective,
        integrator=cfg.integrator,
        stop_mode="steps",
        stop_value=n_steps,
        output_dir=cfg.output_dir,
        extra_tags={
            "static": cfg.static,
            "D": cfg.obs_to_lens_distance / one_Mpc,
            "cone": f"{cfg.cone_angle_deg:.3g}deg",
            "local": int(cfg.flrw_local_coords),
            "sampling": cfg.sampling, 
            "b": f"{cfg.b_min/one_Mpc:.3g}-{cfg.b_max/one_Mpc:.3g}Mpc_{cfg.b_nbins}bins"
        },
    )

    # --- write outputs ---
    # We write a minimal HDF5 file with the trajectories and a compressed npz fallback.
    def _write_output(path: str, trajectories, meta: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Pad trajectories to a common shape before stacking.
        if len(trajectories) == 0:
            stacked = np.zeros((0, 0, 0), dtype=float)
        else:
            max_t = max(t.shape[0] for t in trajectories)
            max_d = max((t.shape[1] if t.ndim > 1 else 0) for t in trajectories)
            padded = np.full((len(trajectories), max_t, max_d), np.nan, dtype=float)
            for i, t in enumerate(trajectories):
                if t.size == 0:
                    continue
                tt = np.asarray(t, dtype=float)
                if tt.ndim == 1:
                    tt = tt[:, None]
                padded[i, : tt.shape[0], : tt.shape[1]] = tt
            stacked = padded
        try:
            import h5py  # type: ignore

            with h5py.File(path, "w") as f:
                f.create_dataset(
                    "trajectories",
                    data=stacked,
                    compression="gzip",
                    compression_opts=4,
                )
                for k, v in meta.items():
                    f.attrs[str(k)] = v
        except Exception as e:
            # Fallback: npz (portable, no binary deps).
            npz_path = path.replace(".h5", ".npz")
            np.savez_compressed(npz_path, trajectories=stacked, **meta)
            print(f"[warn] Could not write HDF5 to {path}: {e}. Wrote {npz_path} instead.")

    common_meta = {
        "mass_kg": float(cfg.mass),
        "radius_m": float(cfg.radius),
        "obs_to_lens_distance_m": float(cfg.obs_to_lens_distance),
        "n_photons": int(cfg.n_photons),
        "cone_angle_deg": float(cfg.cone_angle_deg),
        "n_steps": int(n_steps),
        "dlambda": float(dlambda),
        "integrator": str(cfg.integrator),
        "mode": str(cfg.mode),
        "trace_norm": bool(cfg.trace_norm),
        "renormalize_every": int(cfg.renormalize_every),
        "sampling": str(cfg.sampling),
        "b_min_m": float(cfg.b_min),
        "b_max_m": float(cfg.b_max),
        "b_nbins": int(cfg.b_nbins),
        "b_nperbin": int(cfg.b_nperbin),
        "n_photons_effective": int(n_photons_effective),

    }

    _write_output(
        schwarz_fname,
        schwarz_trajs,
        {**common_meta, "metric": "schwarzschild"},
    )
    _write_output(
        flrw_fname,
        flrw_trajs,
        {**common_meta, "metric": "perturbed_flrw", "static": bool(cfg.static)},
    )

    # Compact run summary
    def _stats(arr):
        return float(np.mean(arr)), float(np.max(arr)), float(np.std(arr))

    s0, smax0, sstd0 = _stats(schwarz_before)
    s1, smax1, sstd1 = _stats(schwarz_after)
    f0, fmax0, fstd0 = _stats(flrw_before)
    f1, fmax1, fstd1 = _stats(flrw_after)

    print("=== Outputs ===")
    print("Schwarzschild ->", schwarz_fname)
    print("FLRW          ->", flrw_fname)
    print("=== Null-condition rel error (mean / max / std) ===")
    print(f"Schwarzschild init:  {s0:.3e} / {smax0:.3e} / {sstd0:.3e}")
    print(f"Schwarzschild final: {s1:.3e} / {smax1:.3e} / {sstd1:.3e}")
    print(f"FLRW init:           {f0:.3e} / {fmax0:.3e} / {fstd0:.3e}")
    print(f"FLRW final:          {f1:.3e} / {fmax1:.3e} / {fstd1:.3e}")


if __name__ == "__main__":
    main()
