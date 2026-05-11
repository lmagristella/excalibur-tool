#!/usr/bin/env python3
"""Interactive viewer for explicit ray/plane-intersection lensing."""
from __future__ import annotations

import argparse

from lensed_sky_interactive_common import (
    InteractiveLensingFigure,
    default_half_view,
    default_source_params,
    load_dataset,
    precompute_raytrace_mapping,
    summarize_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive explicit-raytracing viewer for EXCALIBUR lensing outputs."
    )
    parser.add_argument("path", nargs="?", help="Path to a trajectory-enabled .npz file")
    parser.add_argument("--n-fine", type=int, default=768)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--source-x", type=float, default=0.0)
    parser.add_argument("--source-y", type=float, default=0.0)
    parser.add_argument("--source-re", type=float, default=None)
    parser.add_argument("--source-n", type=float, default=1.0)
    parser.add_argument("--source-ellip", type=float, default=0.3)
    parser.add_argument("--source-pa", type=float, default=30.0)
    parser.add_argument("--source-i0", type=float, default=1.0)
    parser.add_argument("--snapshot-out", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    d, npz_path = load_dataset(args.path)
    half_view = default_half_view(d, args.fov)
    summarize_dataset(d, half_view, prefix=f"Loaded {npz_path}")

    mapping, raytrace = precompute_raytrace_mapping(d, half_view, args.n_fine)
    max_resid_kpc = 1e3 * float(abs(raytrace["hit_residual_Mpc"]).max())
    print(f"  Raytrace intersection max residual = {max_resid_kpc:.3e} kpc")

    source_kw = default_source_params(d, half_view)
    source_kw["center"] = (args.source_x, args.source_y)
    source_kw["R_e"] = args.source_re if args.source_re is not None else source_kw["R_e"]
    source_kw["n"] = args.source_n
    source_kw["ellip"] = args.source_ellip
    source_kw["pa_deg"] = args.source_pa
    source_kw["I0"] = args.source_i0

    title = (
        "Interactive lensing | explicit ray/source-plane mapping"
        f" | z_l={float(d['z_l']):.3f}, z_s={float(d['z_source']):.3f}"
    )
    note = "Drag the source in the left panel or use the sliders below."
    app = InteractiveLensingFigure(d, [mapping], title=title, source_kw=source_kw, note=note)

    if args.snapshot_out:
        app.save_snapshot(args.snapshot_out)
        print(f"  [ok] Snapshot saved to {args.snapshot_out}")
        return

    app.show()


if __name__ == "__main__":
    main()