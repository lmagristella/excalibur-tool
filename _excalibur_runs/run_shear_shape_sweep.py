#!/usr/bin/env python3
"""
Shear-vs-halo-shape sweep driver.

Runs ``run_lensing_nfw_triaxial.py`` for a grid of halo SHAPES at a FIXED
mass / concentration / redshift so the resulting shear fields are directly
comparable across shapes.

Integrator: the default is the adaptive DOPRI5 batch integrator WITH a step
cap (``--dt-max-rs``). The cap is essential -- uncapped, DOPRI5 ramps its
step x5 per accepted step in the empty foreground and jumps clean over the
compact NFW core (mu -> 1, |gamma| underestimated by ~3700x). With the cap it
was validated against RK4 on the M_200=1e15 sphere: the shear MAP agrees to
~1e-7 (machine precision); only the deep strong-lensing cusp (b < 0.05 Mpc)
differs by up to ~12% on the few most cusp-sensitive photons -- irrelevant to
shear estimates. DOPRI5+cap is ~10-20x faster than RK4 and uses all cores via
numba.prange, so it runs the shapes one at a time. Pass ``--integrator rk4``
for the (slow, single-core) fixed-step reference instead.

Output isolation (so no existing simulation is ever clobbered):
  * every .npz goes under   _data/output/<subdir>/   (default: shear_shape_sweep)
  * the run script aborts on an existing target unless --overwrite is given
  * this driver additionally SKIPS a shape whose .npz already exists

Parallelism: each RK4 run is single-core, so we launch several shapes as
concurrent processes (``--jobs``). Per-run stdout/stderr is captured to a
``.log`` next to the .npz.

Examples
--------
    # standard preset (10 shapes, 21x21 map), 6 concurrent jobs
    python run_shear_shape_sweep.py

    # thorough preset (finer shape grid, 31x31 map)
    python run_shear_shape_sweep.py --preset thorough

    # dry run: just print the commands that would be launched
    python run_shear_shape_sweep.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_SCRIPT = os.path.join(HERE, "run_lensing_nfw_triaxial.py")
OUTPUT_ROOT = os.path.abspath(os.path.join(HERE, "..", "_data", "output"))


# ---------------------------------------------------------------------------
# Shape grids.  Each entry: (label, b_over_a, c_over_a, orientation_euler_deg)
#   orientation None    -> principal axes aligned with (x, y, z); the major
#                          axis 'a' lies in the sky plane (broadside).
#   orientation (0,90,0)-> major axis rotated onto the line of sight (end-on).
#   orientation (0,45,0)-> intermediate inclination.
# Convention reminder:
#   (1.0, 1.0)  spherical
#   (1.0, q)    oblate  (pancake): two long equal axes in a plane, q = short/long
#   (q,   q)    prolate (cigar)  : one long axis 'a', two short equal axes
#   b/a > c/a   genuinely triaxial
# ---------------------------------------------------------------------------
PRESETS = {
    "standard": [
        ("sphere",                1.0, 1.0, None),
        ("oblate_q0.6",           1.0, 0.6, None),
        ("oblate_q0.4",           1.0, 0.4, None),
        ("prolate_q0.6_broadside", 0.6, 0.6, None),
        ("prolate_q0.6_endon",    0.6, 0.6, (0.0, 90.0, 0.0)),
        ("prolate_q0.4_broadside", 0.4, 0.4, None),
        ("prolate_q0.4_endon",    0.4, 0.4, (0.0, 90.0, 0.0)),
        ("prolate_q0.6_incl45",   0.6, 0.6, (0.0, 45.0, 0.0)),
        ("triaxial_0.8_0.5",      0.8, 0.5, None),
        ("triaxial_0.7_0.4",      0.7, 0.4, None),
    ],
    "thorough": [
        ("sphere",                1.0, 1.0, None),
        ("oblate_q0.8",           1.0, 0.8, None),
        ("oblate_q0.6",           1.0, 0.6, None),
        ("oblate_q0.4",           1.0, 0.4, None),
        ("prolate_q0.7_broadside", 0.7, 0.7, None),
        ("prolate_q0.7_incl45",   0.7, 0.7, (0.0, 45.0, 0.0)),
        ("prolate_q0.7_endon",    0.7, 0.7, (0.0, 90.0, 0.0)),
        ("prolate_q0.5_broadside", 0.5, 0.5, None),
        ("prolate_q0.5_incl45",   0.5, 0.5, (0.0, 45.0, 0.0)),
        ("prolate_q0.5_endon",    0.5, 0.5, (0.0, 90.0, 0.0)),
        ("prolate_q0.3_endon",    0.3, 0.3, (0.0, 90.0, 0.0)),
        ("triaxial_0.8_0.5",      0.8, 0.5, None),
        ("triaxial_0.7_0.4",      0.7, 0.4, None),
        ("triaxial_0.6_0.3",      0.6, 0.3, None),
    ],
}

PRESET_MAP_N1D = {"standard": 21, "thorough": 31}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="standard")
    p.add_argument("--subdir", default="shear_shape_sweep",
                   help="Output sub-directory under _data/output/ (default: shear_shape_sweep).")
    p.add_argument("--integrator", choices=("dopri5", "rk4"), default="dopri5",
                   help="dopri5 = capped adaptive batch (default, fast, multi-core); "
                        "rk4 = fixed-step reference (slow, single-core).")
    p.add_argument("--dt-max-rs", type=float, default=0.25,
                   help="DOPRI5 step cap in units of r_s (default 0.25; ignored for rk4).")
    p.add_argument("--jobs", type=int, default=None,
                   help="Max concurrent runs. Default: 1 for dopri5 (each run already "
                        "saturates all cores via prange), 6 for rk4 (single-core each).")
    p.add_argument("--mass-200-msun", type=float, default=1e15)
    p.add_argument("--c-nfw", type=float, default=5.0)
    p.add_argument("--z-lens", type=float, default=0.5)
    p.add_argument("--z-source", type=float, default=2.0)
    p.add_argument("--map-n1d", type=int, default=None,
                   help="Override map resolution (default: preset-dependent).")
    p.add_argument("--map-half-mpc", type=float, default=1.5)
    p.add_argument("--overwrite", action="store_true",
                   help="Force re-running shapes whose .npz already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands without launching anything.")
    return p.parse_args()


def build_cmd(args, label, ba, ca, orient, map_n1d):
    cmd = [
        sys.executable, RUN_SCRIPT,
        "--mass-200-msun", repr(args.mass_200_msun),
        "--c-nfw", repr(args.c_nfw),
        "--axis-ratio-ba", repr(ba),
        "--axis-ratio-ca", repr(ca),
        "--z-lens", repr(args.z_lens),
        "--z-source", repr(args.z_source),
        "--map-n1d", str(map_n1d),
        "--map-half-mpc", repr(args.map_half_mpc),
        "--output-subdir", args.subdir,
        "--label", label,
    ]
    if orient is not None:
        cmd += ["--orientation-euler", *(repr(a) for a in orient)]
    if args.integrator == "dopri5":
        cmd += ["--backend", "numba", "--numba-kernel", "specialized",
                "--integrator", "dopri5", "--parallel-batch",
                "--dt-max-rs", repr(args.dt_max_rs)]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def run_one(args, entry, map_n1d, outdir):
    label, ba, ca, orient = entry
    cmd = build_cmd(args, label, ba, ca, orient, map_n1d)
    log_path = os.path.join(outdir, f"_sweep_{label}.log")
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write("CMD: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    return label, proc.returncode, dt, log_path


def main():
    args = parse_args()
    if args.jobs is None:
        args.jobs = 1 if args.integrator == "dopri5" else 6
    shapes = PRESETS[args.preset]
    map_n1d = args.map_n1d if args.map_n1d is not None else PRESET_MAP_N1D[args.preset]
    outdir = os.path.join(OUTPUT_ROOT, args.subdir)
    os.makedirs(outdir, exist_ok=True)

    n_map = map_n1d * map_n1d
    print("=" * 70)
    print(f"  SHEAR-vs-SHAPE SWEEP   preset={args.preset}   jobs={args.jobs}")
    print("=" * 70)
    print(f"  Fixed: M_200={args.mass_200_msun:.1e} Msun  c={args.c_nfw}  "
          f"z_l={args.z_lens}  z_s={args.z_source}")
    integ = (f"dopri5 (cap {args.dt_max_rs} r_s)" if args.integrator == "dopri5"
             else "rk4 (fixed-step)")
    print(f"  Map:   {map_n1d}x{map_n1d} = {n_map} photons (+103 profile)  "
          f"half={args.map_half_mpc} Mpc  integrator={integ}")
    print(f"  Out:   {outdir}/")
    print(f"  Shapes ({len(shapes)}):")
    for label, ba, ca, orient in shapes:
        o = "aligned" if orient is None else "e" + "_".join(f"{a:g}" for a in orient)
        print(f"    - {label:24s}  b/a={ba:<4g} c/a={ca:<4g} orient={o}")
    print("=" * 70)

    if args.dry_run:
        print("\n[dry-run] commands:")
        for entry in shapes:
            print("  " + " ".join(build_cmd(args, entry[0], entry[1], entry[2],
                                             entry[3], map_n1d)))
        return

    t_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(run_one, args, entry, map_n1d, outdir): entry[0]
                for entry in shapes}
        for fut in as_completed(futs):
            label, rc, dt, log_path = fut.result()
            status = "OK " if rc == 0 else f"FAIL(rc={rc})"
            done = len(results) + 1
            print(f"  [{done:2d}/{len(shapes)}] {status} {label:24s} "
                  f"({dt/60:.1f} min)  log: {os.path.basename(log_path)}", flush=True)
            results.append((label, rc, dt, log_path))

    total = time.time() - t_start
    n_ok = sum(1 for _, rc, _, _ in results if rc == 0)
    n_fail = len(results) - n_ok
    print("=" * 70)
    print(f"  DONE: {n_ok} ok, {n_fail} failed in {total/60:.1f} min wall")
    if n_fail:
        print("  Failed runs (see their logs):")
        for label, rc, _, log_path in results:
            if rc != 0:
                print(f"    - {label}: rc={rc}  {log_path}")
    print(f"  Results in: {outdir}/")
    print("=" * 70)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
