#!/bin/bash
#SBATCH -J excalibur_sweep
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 02:00:00
#SBATCH -o results/logs/%A_%a.out
#SBATCH -e results/logs/%A_%a.err

set -euo pipefail

MANIFEST="results/manifest.csv"

python run_one_job.py \
  --manifest "${MANIFEST}" \
  --jobid "${SLURM_ARRAY_TASK_ID}" \
  --runner excalibur_run_compare_schwarzschild_vs_flrw.py \
  --analyzer analyze_compare_trajectories.py \
  --outdir _data/output \
  --results results
