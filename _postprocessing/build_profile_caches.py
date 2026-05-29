#!/usr/bin/env python3
"""Precompute compact viewer caches for the interactive lensing app.

A full simulation .npz (hundreds of MB) carries per-photon trajectories that the
viewer only uses once, to build the image->source mapping. This script does that
build *once* per profile and writes a small cache (~15 MB: the fine-grid beta
mapping + the scalar halo params), dropping the trajectories. The packaged app
then bundles these caches instead of the giant .npz files, so it stays small and
starts instantly.

All caches are built on a *common* half_view (the smallest across the inputs) so
the in-app profile switcher can swap them without rescaling the axes.

Usage:
    python _postprocessing/build_profile_caches.py OUT_DIR FULL1.npz [FULL2.npz ...]
    python _postprocessing/build_profile_caches.py --half-view 1.47 OUT_DIR FULL.npz ...

Pass --half-view to force the common half_view (instead of deriving it from the
inputs). Use this when adding caches that must match an existing batch built at a
known half_view, so all profiles share the exact same grid in the viewer.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lensed_sky_interactive_common import (
    default_half_view,
    load_dataset,
    precompute_raytrace_mapping,
)
from make_lensed_sky_raytracing_interactive import profile_label

_N_FINE = 768
_SCALAR_MAX_BYTES = 200_000  # copy everything but the big map/trajectory arrays


def _cache_name(label: str) -> str:
    slug = (
        label.lower()
        .replace("∥", "par")
        .replace("/", "")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace(".", "p")
        .replace(" ", "_")
    )
    return f"lensing_profile_cache_{slug}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact viewer caches.")
    parser.add_argument("out_dir")
    parser.add_argument("inputs", nargs="+", help="Full simulation .npz files.")
    parser.add_argument("--half-view", type=float, default=None,
                        help="Force the common half_view [Mpc] instead of deriving "
                             "it from the inputs (use to match an existing batch).")
    args = parser.parse_args()
    out_dir, inputs = args.out_dir, args.inputs
    os.makedirs(out_dir, exist_ok=True)

    datasets = [load_dataset(p) for p in inputs]
    if args.half_view is not None:
        half_view = args.half_view
    else:
        half_view = min(default_half_view(d, None) for d, _ in datasets)
    print(f"Common half_view = {half_view:.4f} Mpc, n_fine = {_N_FINE}")

    for d, path in datasets:
        label = profile_label(d)
        mapping, _ = precompute_raytrace_mapping(d, half_view, _N_FINE)

        out = {}
        # Carry over every small scalar/vector the viewer reads (halo params,
        # cosmology, screen vectors, profile curves); skip the big arrays.
        for key in d.files:
            arr = d[key]
            if np.asarray(arr).nbytes <= _SCALAR_MAX_BYTES:
                out[key] = arr
        # The precomputed fine-grid mapping (float32 keeps the cache small).
        out.update(
            viewer_cache=True,
            profile_label=label,
            half_view=np.float64(half_view),
            extent=np.asarray(mapping.extent, dtype=np.float64),
            beta1=mapping.beta1.astype(np.float32),
            beta2=mapping.beta2.astype(np.float32),
            t1=mapping.t1.astype(np.float32),
            t2=mapping.t2.astype(np.float32),
            xf=mapping.xf.astype(np.float64),
            yf=mapping.yf.astype(np.float64),
        )

        out_path = os.path.join(out_dir, _cache_name(label))
        np.savez_compressed(out_path, **out)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  [{label}] -> {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
