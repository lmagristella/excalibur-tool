#!/usr/bin/env python3
import csv, os, subprocess, re, itertools, math

RUNNER = "../_excalibur_runs/excalibur_run_compare_schwarzschild_vs_flrw.py"
ANALYZER = "../_postprocessing/analyze_compare_trajectories.py"

def run_cmd(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    p = subprocess.run(cmd, text=True, capture_output=True)
    with open(log_path, "w") as f:
        f.write("CMD:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" + p.stdout + "\n\nSTDERR:\n" + p.stderr + "\n")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)} (see {log_path})")
    return p.stdout

def parse_outputs(stdout):
    # matches lines printed by runner:
    # "Schwarzschild -> <path>"
    # "FLRW          -> <path>"
    s = re.search(r"Schwarzschild\s*->\s*(.+)", stdout)
    f = re.search(r"FLRW\s*->\s*(.+)", stdout)
    if not s or not f:
        raise RuntimeError("Could not parse output paths from runner stdout.")
    return s.group(1).strip(), f.group(1).strip()

def make_manifest(path):
    masses = [10**x for x in [14.0,14.2,14.4,14.6,14.8,15.0,15.2,15.4,15.6,15.8]]  # example
    distances = [100,150,200,300,400,500,600,700,800,900]  # Mpc
    Ns = [64,80,96,112,128,160,192,224,256,320]
    # bins in *min_r proxy* (meters) are done later; here we just label bin ids
    bin_ids = list(range(10))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["jobid","mass_msun","distance_mpc","flrw_grid_N","bin_id","seed"])
        jobid = 0
        for m,d,N,b in itertools.product(masses, distances, Ns, bin_ids):
            w.writerow([jobid, m, d, N, b, jobid])
            jobid += 1

def run_one(manifest_csv, jobid, base_out="_data/output", base_res="results"):
    rows = []
    with open(manifest_csv, newline="") as fp:
        r = csv.DictReader(fp)
        for row in r:
            if int(row["jobid"]) == int(jobid):
                rows = [row]
                break
    if not rows:
        raise SystemExit(f"jobid {jobid} not found")

    row = rows[0]
    outdir = os.path.join(base_out, f"M{row['mass_msun']}_D{row['distance_mpc']}_N{row['flrw_grid_N']}")
    log = os.path.join(base_res, "logs", f"job_{jobid:05d}.log")
    perrun_csv = os.path.join(base_res, "per_run", f"job_{jobid:05d}.csv")

    # Runner call
    cmd = [
        "python", RUNNER,
        "--mass-msun", row["mass_msun"],
        "--distance-mpc", row["distance_mpc"],
        "--flrw-grid-N", row["flrw_grid_N"],
        "--output-dir", outdir,
        "--n-photons", "500",
        "--cone-angle-deg", "10",
        "--static",
        "--flrw-local-coords",
        "--record-every", "50",
    ]
    stdout = run_cmd(cmd, log)
    s_path, f_path = parse_outputs(stdout)

    # Analyzer call
    cmd2 = ["python", ANALYZER, "--schwarzschild", s_path, "--flrw", f_path, "--csv", perrun_csv]
    run_cmd(cmd2, log_path=log.replace(".log", "_analyze.log"))

if __name__ == "__main__":
    # 1) make_manifest("results/manifest.csv")  # run once
    # 2) run_one("results/manifest.csv", jobid=os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    pass
