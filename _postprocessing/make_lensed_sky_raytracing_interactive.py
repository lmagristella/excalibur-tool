#!/usr/bin/env python3
"""Interactive viewer for explicit ray/plane-intersection lensing."""
from __future__ import annotations

import argparse
import os

import numpy as np

from lensed_sky_interactive_common import (
    InteractiveLensingFigure,
    LensProfile,
    PrecomputedMapping,
    default_half_view,
    default_source_params,
    load_dataset,
    precompute_raytrace_mapping,
    summarize_dataset,
)


def _is_cache(d) -> bool:
    return "viewer_cache" in getattr(d, "files", [])


def _mapping_from_cache(d) -> PrecomputedMapping:
    """Rebuild the precomputed mapping directly from a compact viewer cache."""
    return PrecomputedMapping(
        label="Ray tracing",
        beta1=np.asarray(d["beta1"], dtype=np.float64),
        beta2=np.asarray(d["beta2"], dtype=np.float64),
        t1=np.asarray(d["t1"], dtype=np.float64),
        t2=np.asarray(d["t2"], dtype=np.float64),
        xf=np.asarray(d["xf"], dtype=np.float64),
        yf=np.asarray(d["yf"], dtype=np.float64),
        extent=list(np.asarray(d["extent"], dtype=np.float64)),
        half_view=float(d["half_view"]),
    )


def profile_label(d) -> str:
    """Human label for the lens-profile radio button, inferred from the dataset.

    Distinguishes three geometries:
      * Spherical            - round NFW.
      * Elliptical (b/a=..)  - triaxial seen with an elliptical on-sky projection.
      * Cigar || LOS (b/a=..) - prolate halo whose long axis points along the line
        of sight: the on-sky projection is *circular* (like the sphere) but the
        convergence is enhanced. Detected from the saved rotation matrix, whose
        first column (the major axis) lands on the z/LOS direction.
    """
    files = getattr(d, "files", [])
    # An explicit label saved by the run script (or copied into the cache) wins.
    if "display_label" in files:
        explicit = str(d["display_label"]).strip()
        if explicit:
            return explicit

    halo_type = str(d["halo_type"]) if "halo_type" in files else "NFW"
    if "Triax" not in halo_type and "axis_ratio_ba" not in files:
        return "Spherical"

    ba = float(d["axis_ratio_ba"]) if "axis_ratio_ba" in files else 1.0
    ca = float(d["axis_ratio_ca"]) if "axis_ratio_ca" in files else 1.0
    if ba >= 0.999 and ca >= 0.999:
        return "Spherical"

    los_aligned = False
    if "halo_rotation_matrix" in files:
        R = np.asarray(d["halo_rotation_matrix"], dtype=float)
        if R.shape == (3, 3):
            los_aligned = abs(R[2, 0]) > 0.9  # major axis (col 0) along z = LOS
    if los_aligned and abs(ba - ca) < 1e-3:
        return f"Cigar ∥ LOS (b/a={ba:.2f})"
    return f"Elliptical (b/a={ba:.2f})"


_DEFAULT_DEMO_NPZ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "_data",
        "output",
        "lensing_nfw_analytic_rk4_FLRWP1_nfw_M2.0e15_c7_Rvir2.599_rs0.3713_zl1_zs2_Dl3303.8_Ds5179.9_obs2602.4_2602.4_5_box5204.86_Nph1000094_results.npz",
    )
)

_DEFAULT_DEMO_SOURCE_SCALE = 1.65
_DEFAULT_DEMO_SOURCE_ELLIP = 0.48
_DEFAULT_DEMO_SOURCE_PA_DEG = 42.0
_DEFAULT_DEMO_FPS = 12
_DEFAULT_DEMO_PATH_SCALE = 0.68
_DEFAULT_CINEMATIC_PATH_SCALE = 1.0


def resolve_default_demo_path() -> str | None:
    if os.path.isfile(_DEFAULT_DEMO_NPZ):
        return _DEFAULT_DEMO_NPZ
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive explicit-raytracing viewer for EXCALIBUR lensing outputs."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a trajectory-enabled .npz file. Defaults to the analytic NFW demo dataset if present.",
    )
    parser.add_argument(
        "--add-profile",
        action="append",
        default=[],
        metavar="NPZ",
        help="Extra .npz dataset(s) added as selectable lens profiles "
             "(e.g. an elliptical/triaxial run). Repeatable. A radio-button "
             "switcher appears when more than one profile is loaded.",
    )
    parser.add_argument("--n-fine", type=int, default=768)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--source-x", type=float, default=0.0)
    parser.add_argument("--source-y", type=float, default=0.0)
    parser.add_argument("--source-re", type=float, default=None)
    parser.add_argument("--source-n", type=float, default=1.0)
    parser.add_argument("--source-ellip", type=float, default=None)
    parser.add_argument("--source-pa", type=float, default=None)
    parser.add_argument("--source-i0", type=float, default=1.0)
    parser.add_argument("--animate", action="store_true", help="Automatically move the source on a smooth demo path.")
    parser.add_argument("--animation-out", type=str, default=None, help="Save the moving-source demo as .gif/.webp/.mp4.")
    parser.add_argument("--mp4-out", type=str, default=None, help="Save the moving-source demo explicitly as an .mp4 file.")
    parser.add_argument("--animation-frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=_DEFAULT_DEMO_FPS)
    parser.add_argument("--path-scale", type=float, default=_DEFAULT_DEMO_PATH_SCALE)
    parser.add_argument(
        "--track-mode",
        choices=["loop", "strong-to-weak"],
        default="loop",
        help="Animation path: looping demo orbit or a one-way strong-to-weak transition.",
    )
    parser.add_argument(
        "--panels-only",
        action="store_true",
        help="Hide sliders and reset UI, leaving only the source and lensed panels.",
    )
    parser.add_argument(
        "--hide-source-marker",
        action="store_true",
        help="Hide the cross marker in the source panel.",
    )
    parser.add_argument("--no-repeat", action="store_true", help="Play the interactive demo path only once.")
    parser.add_argument("--snapshot-out", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.animation_out and args.mp4_out:
        raise ValueError("Use either --animation-out or --mp4-out, not both.")
    movie_out = args.mp4_out or args.animation_out
    if args.snapshot_out and movie_out:
        raise ValueError("Use either --snapshot-out or an animation export option, not both.")

    requested_path = args.path or resolve_default_demo_path()

    # Load the primary dataset plus any extra lens profiles. All profiles share a
    # common half_view (the smallest, so the fine grid stays within every map's
    # coverage) so switching profiles only swaps the beta field, not the axes.
    datasets = [load_dataset(requested_path)]
    for extra in args.add_profile:
        datasets.append(load_dataset(extra))

    # Caches carry a baked half_view; full datasets derive theirs. Use the
    # smallest as the common one so every fine grid stays within map coverage.
    def _ds_half_view(dd):
        return float(dd["half_view"]) if _is_cache(dd) else default_half_view(dd, args.fov)

    half_view = min(_ds_half_view(dd) for dd, _ in datasets)

    d, npz_path = datasets[0]
    using_default_demo = os.path.abspath(npz_path) == os.path.abspath(_DEFAULT_DEMO_NPZ)
    summarize_dataset(d, half_view, prefix=f"Loaded {npz_path}")

    profiles = []
    for dd, pth in datasets:
        label = profile_label(dd)
        if _is_cache(dd):
            mp = _mapping_from_cache(dd)
            print(f"  [{label}] loaded from precomputed cache")
        else:
            mp, rt = precompute_raytrace_mapping(dd, half_view, args.n_fine)
            max_resid_kpc = 1e3 * float(abs(rt["hit_residual_Mpc"]).max())
            print(f"  [{label}] raytrace max residual = {max_resid_kpc:.3e} kpc")
        profiles.append(LensProfile(label=label, d=dd, mapping=mp))

    animate_demo = args.animate or movie_out is not None or (args.path is None and using_default_demo)
    if args.track_mode == "strong-to-weak" and args.path_scale == _DEFAULT_DEMO_PATH_SCALE:
        args.path_scale = _DEFAULT_CINEMATIC_PATH_SCALE

    source_kw = default_source_params(d, half_view)
    source_kw["center"] = (args.source_x, args.source_y)
    if animate_demo:
        source_kw["R_e"] *= _DEFAULT_DEMO_SOURCE_SCALE
        source_kw["ellip"] = _DEFAULT_DEMO_SOURCE_ELLIP
        source_kw["pa_deg"] = _DEFAULT_DEMO_SOURCE_PA_DEG

    source_kw["R_e"] = args.source_re if args.source_re is not None else source_kw["R_e"]
    source_kw["n"] = args.source_n
    if args.source_ellip is not None:
        source_kw["ellip"] = args.source_ellip
    if args.source_pa is not None:
        source_kw["pa_deg"] = args.source_pa
    source_kw["I0"] = args.source_i0

    title = None if args.panels_only else (
        "Interactive lensing | explicit ray/source-plane mapping"
        f" | z_l={float(d['z_l']):.3f}, z_s={float(d['z_source']):.3f}"
    )
    if animate_demo:
        note = None if args.panels_only else (
            "Demo mode: the source follows a smooth path across the source plane. "
            "Drag it in the left panel or use the sliders to inspect a specific position."
        )
    else:
        note = None if args.panels_only else "Drag the source in the left panel or use the sliders below."
    app = InteractiveLensingFigure(
        profiles,
        title=title,
        source_kw=source_kw,
        note=note,
        show_controls=not args.panels_only,
        show_source_marker=not args.hide_source_marker,
    )

    if args.snapshot_out:
        app.save_snapshot(args.snapshot_out)
        print(f"  [ok] Snapshot saved to {args.snapshot_out}")
        return

    if movie_out:
        app.save_demo_animation(
            movie_out,
            frames=args.animation_frames,
            fps=args.fps,
            orbit_scale=args.path_scale,
            track_mode=args.track_mode,
        )
        print(f"  [ok] Animation saved to {movie_out}")
        return

    if animate_demo:
        app.start_demo_animation(
            frames=args.animation_frames,
            fps=args.fps,
            orbit_scale=args.path_scale,
            track_mode=args.track_mode,
            repeat=not args.no_repeat,
        )

    app.show()


if __name__ == "__main__":
    main()