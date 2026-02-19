#!/usr/bin/env python3
"""
Generate a manifest CSV for a large parameter sweep.

This manifest encodes *one impact-parameter bin per run*.
That way, if you want 10 "bins" and you run one simulation per bin, the bins
become the 4th sweep axis and you indeed get 10 (mass) * 10 (distance) * 10 (N) * 10 (bin) = 10,000 runs.

Usage example:
  python make_manifest.py \
    --out results/manifest.csv \
    --masses "1e13,3e13,1e14,3e14,1e15,3e15" \
    --distances "100,200,300,400,500,600,700,800,900,1000" \
    --Ns "64,80,96,112,128,160,192,224,256,320" \
    --b-min 0.5 --b-max 10.0 --b-nbins 10 \
    --b-nperbin 200 \
    --static 1 --local 1

Notes:
- masses in Msun
- distances in Mpc
- b_min/b_max in Mpc (impact parameter range). We split into b_nbins equal-width bins.
  Each job uses exactly ONE bin, encoded as b_min_mpc/b_max_mpc in the manifest row.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import List

import numpy as np


def _parse_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="Output manifest CSV path")
    p.add_argument("--masses", type=str, required=True, help="Comma-separated masses in Msun (10 values recommended)")
    p.add_argument("--distances", type=str, required=True, help="Comma-separated distances in Mpc (10 values recommended)")
    p.add_argument("--Ns", type=str, required=True, help="Comma-separated FLRW grid N values (10 values recommended)")

    # Lens / runner core params.
    p.add_argument("--radius-mpc", type=float, default=3.0, help="Lens radius in Mpc")
    p.add_argument(
        "--integrator",
        type=str,
        default="rk4",
        choices=["rk4", "rk45", "leapfrog4"],
        help="Integrator used by the runner",
    )

    # Plural sweep variants (comma-separated). If provided, they define the sweep axes and
    # take precedence over the singular flags above.
    p.add_argument(
        "--radii-mpc",
        type=str,
        default=None,
        help="Comma-separated lens radii in Mpc to sweep (overrides --radius-mpc)",
    )
    p.add_argument(
        "--integrators",
        type=str,
        default=None,
        help="Comma-separated integrators to sweep, e.g. 'rk4,leapfrog4' (overrides --integrator)",
    )

    p.add_argument("--sampling", type=str, default="impact", choices=["impact", "cone"], help="Sampling method for photons (impact or cone)")
    p.add_argument("--b-min", type=float, required=False, help="Impact parameter min in Mpc")
    p.add_argument("--b-max", type=float, required=False, help="Impact parameter max in Mpc")
    p.add_argument("--b-nbins", type=int, default=10, help="Number of impact-parameter bins")

    p.add_argument("--cone-angle-deg", type=float, default=10.0, help="Cone angle in degrees (only if sampling=cone)")
    p.add_argument("--cone-nphotons", type=int, default=200, help="Photons per run (only if sampling=cone)")

    p.add_argument("--b-nperbin", type=int, default=200, help="Photons per run (since each run uses one bin)")
    p.add_argument("--static", type=int, default=1, choices=[0,1], help="Use static cosmology (1) or LCDM (0)")
    p.add_argument("--local", type=int, default=1, choices=[0,1], help="Use local coords (1) or absolute (0)")
    p.add_argument("--seed0", type=int, default=0, help="Base RNG seed; actual seed = seed0 + jobid")

    # Runner behavior overrides (optional): stored in manifest and forwarded by run_one_job.
    # Keep defaults aligned with the runner defaults, but allow sweeps to override.
    p.add_argument("--record-every", type=int, default=4, help="Runner: record every N steps (0 disables recording)")
    p.add_argument("--progress-every", type=int, default=0, help="Runner: print progress every N photons (0 disables)")
    p.add_argument("--renormalize-every", type=int, default=0, help="Runner: renormalize 4-velocity every N steps (0 disables)")

    # Step control overrides.
    p.add_argument("--n-steps", type=int, default=5000, help="Runner: max number of integration steps")
    p.add_argument(
        "--steps-per-cell",
        type=float,
        default=3.0,
        help="Runner: heuristic steps per FLRW grid cell (affects dlambda)",
    )

    # FLRW grid geometry overrides.
    p.add_argument(
        "--flrw-grid-size-mpc",
        type=float,
        default=1000.0,
        help="Runner: FLRW comoving box size in Mpc",
    )

    args = p.parse_args()

    masses = _parse_list(args.masses)
    distances = _parse_list(args.distances)
    Ns = [int(x) for x in _parse_list(args.Ns)]

    radii_mpc = _parse_list(args.radii_mpc) if args.radii_mpc else [float(args.radius_mpc)]
    integrators = _parse_str_list(args.integrators) if args.integrators else [str(args.integrator)]

    


   
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    
    if args.sampling == "impact":
        b_edges = np.linspace(args.b_min, args.b_max, args.b_nbins + 1)

        if args.b_max <= args.b_min:
            raise SystemExit("b-max must be > b-min")
    
        header = [
            "jobid",
            "mass_msun",
            "radius_mpc",
            "distance_mpc",
            "integrator",
            "flrw_grid_N",
            "bin_id",
            "b_min_mpc",
            "b_max_mpc",
            "b_nbins",
            "b_nperbin",
            "static",
            "local",
            "record_every",
            "progress_every",
            "renormalize_every",
            "n_steps",
            "steps_per_cell",
            "flrw_grid_size_mpc",
            "seed",
        ]

        jobid = 0
        with open(args.out, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(header)
            for m in masses:
                for d in distances:
                    for N in Ns:
                        for r_mpc in radii_mpc:
                            for integ in integrators:
                                for bin_id in range(args.b_nbins):
                                    bmin = float(b_edges[bin_id])
                                    bmax = float(b_edges[bin_id + 1])
                                    seed = int(args.seed0 + jobid)
                                    w.writerow(
                                        [
                                            jobid,
                                            m,
                                            r_mpc,
                                            d,
                                            integ,
                                            N,
                                            bin_id,
                                            bmin,
                                            bmax,
                                            1,
                                            args.b_nperbin,
                                            args.static,
                                            args.local,
                                            args.record_every,
                                            args.progress_every,
                                            args.renormalize_every,
                                            args.n_steps,
                                            args.steps_per_cell,
                                            args.flrw_grid_size_mpc,
                                            seed,
                                        ]
                                    )
                                    jobid += 1

    elif args.sampling == "cone":

        header = [
            "jobid",
            "mass_msun",
            "radius_mpc",
            "distance_mpc",
            "integrator",
            "flrw_grid_N",
            "cone_angle_deg",
            "cone_nphotons",
            "static",
            "local",
            "record_every",
            "progress_every",
            "renormalize_every",
            "n_steps",
            "steps_per_cell",
            "flrw_grid_size_mpc",
            "seed",
        ]

        jobid = 0
        with open(args.out, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(header)
            for m in masses:
                for d in distances:
                    for N in Ns:
                        for r_mpc in radii_mpc:
                            for integ in integrators:
                                seed = int(args.seed0 + jobid)
                                w.writerow(
                                    [
                                        jobid,
                                        m,
                                        r_mpc,
                                        d,
                                        integ,
                                        N,
                                        args.cone_angle_deg,
                                        args.cone_nphotons,
                                        args.static,
                                        args.local,
                                        args.record_every,
                                        args.progress_every,
                                        args.renormalize_every,
                                        args.n_steps,
                                        args.steps_per_cell,
                                        args.flrw_grid_size_mpc,
                                        seed,
                                    ]
                                )
                                jobid += 1

    print(f"Wrote {jobid} jobs to {args.out}")


if __name__ == "__main__":
    main()
