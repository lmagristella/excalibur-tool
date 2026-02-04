# Comparable runs: Schwarzschild vs FLRW

This mini-runner produces **two trajectory files** for the **same lens mass** and the **same observer–lens distance**:

- `schwarzschild` (exact, static, spherical coords)
- `perturbed_flrw` (Newtonian-gauge scalar perturbation from a `spherical_mass` potential on a finite box)

## Output naming / location

Outputs go to `_data/output/` by default and are named via `excalibur.io.filename_utils.generate_trajectory_filename`.

The filename includes:

- metric type (`schwarzschild`, `perturbed_flrw`, `perturbed_flrw_static`)
- mass `M...` (in $M_\odot$)
- radius `R...` (in Mpc)
- lens center `massX_Y_Z` (in Mpc; rounded)
- observer position `obsX_Y_Z` (in Mpc; rounded)
- photon count `N...`
- integrator (e.g. `rk4`)
- stop condition `S...` (steps)
- plus a couple of extra tags used by the compare runner:
  - `static0/1`
  - `D...` (observer–lens distance in Mpc)
  - `cone...deg`

## Run it

From the repo root:

```bash
./.venv/bin/python _excalibur_runs/excalibur_run_compare_schwarzschild_vs_flrw.py --static
```

A reasonably light but still “physics-like” run (tune as you want):

```bash
./.venv/bin/python _excalibur_runs/excalibur_run_compare_schwarzschild_vs_flrw.py \
  --mass-msun 1e15 \
  --radius-mpc 3 \
  --distance-mpc 500 \
  --n-photons 200 \
  --cone-angle-deg 10 \
  --n-steps 5000 \
  --integrator rk4 \
  --flrw-grid-N 256 \
  --flrw-grid-size-mpc 1200
```

Non-static FLRW (LCDM expansion):

```bash
./.venv/bin/python _excalibur_runs/excalibur_run_compare_schwarzschild_vs_flrw.py \
  --distance-mpc 500 \
  --n-photons 200 \
  --n-steps 5000
```

## Notes / assumptions

- The FLRW run uses a finite periodic grid. To avoid out-of-bounds during integration the script places the lens at the **box center** and the observer at a distance `D` along `-x`.
- The Schwarzschild run uses the exact same relative geometry (same `D`, same cone directions).
- The script prints null-condition relative error statistics before/after integration as a quick sanity check.
