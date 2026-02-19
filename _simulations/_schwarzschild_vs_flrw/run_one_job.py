#!/usr/bin/env python3
"""
Run ONE job from a manifest row:
1) call runner (Schwarzschild + FLRW) for that parameter point
2) parse output file paths from runner stdout
3) call analyzer to produce a per-run CSV

Assumptions:
- Your runner has been updated to accept:
    --sampling impact
    --b-min-mpc, --b-max-mpc, --b-nbins, --b-nperbin
  and uses Photons.generate_impact_parameter_bins(...)

Usage:
  python run_one_job.py \
    --manifest results/manifest.csv \
    --jobid 1234 \
    --runner /path/to/excalibur_run_compare_schwarzschild_vs_flrw.py \
    --analyzer /path/to/analyze_compare_trajectories.py \
    --outdir _data/output \
    --results results
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Job:
    jobid: int
    mass_msun: float
    distance_mpc: float
    flrw_grid_N: int
    radius_mpc: float = 3.0
    integrator: str = "rk4"
    sampling: str = "cone"
    cone_nphotons: int = 200
    cone_angle_deg: float = 10.0
    bin_id: int = 0
    b_min_mpc: float = 0.0
    b_max_mpc: float = 0.0
    b_nbins: int = 0
    b_nperbin: int = 0
    static: int = 1
    local: int = 1
    record_every: int = 4
    progress_every: int = 0
    renormalize_every: int = 0
    n_steps: int = 5000
    steps_per_cell: float = 3.0
    flrw_grid_size_mpc: float = 1000.0
    seed: int = 0


def read_job(manifest: str, jobid: int) -> Job:
    with open(manifest, newline="") as fp:
        r = csv.DictReader(fp)
        for row in r:
            if int(row["jobid"]) == int(jobid):
                record_every = int(row.get("record_every", 4) or 4)
                progress_every = int(row.get("progress_every", 0) or 0)
                renormalize_every = int(row.get("renormalize_every", 0) or 0)
                n_steps = int(row.get("n_steps", 5000) or 5000)
                steps_per_cell = float(row.get("steps_per_cell", 3.0) or 3.0)
                flrw_grid_size_mpc = float(row.get("flrw_grid_size_mpc", 1000.0) or 1000.0)
                radius_mpc = float(row.get("radius_mpc", 3.0) or 3.0)
                integrator = str(row.get("integrator", "rk4") or "rk4")
                if "b_min_mpc" not in row or "b_max_mpc" not in row or "b_nbins" not in row or "b_nperbin" not in row:
                    return Job(
                        jobid=int(row["jobid"]),
                        mass_msun=float(row["mass_msun"]),
                        radius_mpc=radius_mpc,
                        distance_mpc=float(row["distance_mpc"]),
                        flrw_grid_N=int(row["flrw_grid_N"]),
                        integrator=integrator,
                        sampling=row.get("sampling", "cone"),
                        cone_nphotons=int(row.get("cone_nphotons", 200)),
                        cone_angle_deg=float(row.get("cone_angle_deg", 10.0)),
                        static=int(row.get("static", 1)),
                        local=int(row.get("local", 1)),
                        record_every=record_every,
                        progress_every=progress_every,
                        renormalize_every=renormalize_every,
                        n_steps=n_steps,
                        steps_per_cell=steps_per_cell,
                        flrw_grid_size_mpc=flrw_grid_size_mpc,
                        seed=int(row.get("seed", 0)),

                    )
                else:
                    return Job(
                        jobid=int(row["jobid"]),
                        mass_msun=float(row["mass_msun"]),
                        radius_mpc=radius_mpc,
                        distance_mpc=float(row["distance_mpc"]),
                        flrw_grid_N=int(row["flrw_grid_N"]),
                        integrator=integrator,
                        sampling=row.get("sampling", "cone"),
                        cone_nphotons=int(row.get("cone_nphotons", 200)),
                        cone_angle_deg=float(row.get("cone_angle_deg", 10.0)),
                        bin_id=int(row.get("bin_id", 0)),
                        b_min_mpc=float(row.get("b_min_mpc", 0.0)),
                        b_max_mpc=float(row.get("b_max_mpc", 0.0)),
                        b_nbins=int(row.get("b_nbins", 0)),
                        b_nperbin=int(row.get("b_nperbin", 0)),
                        static=int(row.get("static", 1)),
                        local=int(row.get("local", 1)),
                        record_every=record_every,
                        progress_every=progress_every,
                        renormalize_every=renormalize_every,
                        n_steps=n_steps,
                        steps_per_cell=steps_per_cell,
                        flrw_grid_size_mpc=flrw_grid_size_mpc,
                        seed=int(row.get("seed", 0)),
                    )
    raise SystemExit(f"jobid={jobid} not found in {manifest}")


def run_capture(cmd, log_path: str) -> str:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    p = subprocess.run(cmd, text=True, capture_output=True)
    with open(log_path, "w") as f:
        f.write("CMD:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" + p.stdout + "\n\nSTDERR:\n" + p.stderr + "\n")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (see {log_path})")
    return p.stdout


def parse_outputs(stdout: str):
    s = re.search(r"Schwarzschild\s*->\s*(.+)", stdout)
    f = re.search(r"FLRW\s*->\s*(.+)", stdout)
    if not s or not f:
        raise RuntimeError("Could not parse output paths from runner stdout. Expected 'Schwarzschild ->' and 'FLRW ->'.")
    return s.group(1).strip(), f.group(1).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--jobid", type=int, required=True)
    p.add_argument("--runner", type=str, default="excalibur_run_compare_schwarzschild_vs_flrw.py")
    p.add_argument("--analyzer", type=str, default="analyze_compare_trajectories.py")
    p.add_argument("--outdir", type=str, default=os.path.join("_data", "output"))
    p.add_argument("--results", type=str, default="results")
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help=(
            "Python executable to use for running runner/analyzer. "
            "Defaults to the current interpreter (sys.executable)."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    job = read_job(args.manifest, args.jobid)

    per_run_csv = os.path.join(args.results, "per_run", f"job_{job.jobid:05d}.csv")
    log_runner = os.path.join(args.results, "logs", f"job_{job.jobid:05d}_runner.log")
    log_an = os.path.join(args.results, "logs", f"job_{job.jobid:05d}_analyze.log")

    os.makedirs(os.path.join(args.results, "per_run"), exist_ok=True)
    os.makedirs(os.path.join(args.results, "logs"), exist_ok=True)

    if (not args.overwrite) and os.path.exists(per_run_csv):
        print(f"[skip] {per_run_csv} exists")
        return

    cmd = [
        args.python,
        args.runner,
        "--mass-msun",
        str(job.mass_msun),
        "--radius-mpc",
        str(job.radius_mpc),
        "--distance-mpc",
        str(job.distance_mpc),
        "--integrator",
        str(job.integrator),
        "--flrw-grid-N",
        str(job.flrw_grid_N),
        "--output-dir",
        args.outdir,
        "--progress-every",
        str(job.progress_every),
        "--record-every",
        str(job.record_every),
        "--renormalize-every",
        str(job.renormalize_every),
        "--n-steps",
        str(job.n_steps),
        "--steps-per-cell",
        str(job.steps_per_cell),
        "--flrw-grid-size-mpc",
        str(job.flrw_grid_size_mpc),
    ]

    # Sampling-specific flags.
    if str(job.sampling).lower() == "impact":
        cmd += [
            "--sampling",
            "impact",
            "--b-min-mpc",
            str(job.b_min_mpc),
            "--b-max-mpc",
            str(job.b_max_mpc),
            "--b-nbins",
            str(job.b_nbins),
            "--b-nperbin",
            str(job.b_nperbin),
            # Keep runner metadata + filename generation consistent.
            "--n-photons",
            str(job.b_nperbin),
        ]
    else:
        # Default to cone sampling.
        cmd += [
            "--sampling",
            "cone",
            "--n-photons",
            str(job.cone_nphotons),
            "--cone-angle-deg",
            str(job.cone_angle_deg),
        ]

    if job.static:
        cmd.append("--static")
    if job.local:
        cmd.append("--flrw-local-coords")

    # If your runner supports a seed flag, pass it here (recommended).
    # cmd += ["--seed", str(job.seed)]

    stdout = run_capture(cmd, log_runner)
    sch_path, flrw_path = parse_outputs(stdout)

    cmd2 = [args.python, args.analyzer, "--schwarzschild", sch_path, "--flrw", flrw_path, "--csv", per_run_csv]
    run_capture(cmd2, log_an)

    print(f"[ok] job {job.jobid} -> {per_run_csv}")


if __name__ == "__main__":
    main()
