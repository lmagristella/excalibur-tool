#!/usr/bin/env python3
r"""
Direction-dependent lensing through a quasi-spherical Szekeres structure.

Shoots a fan of null rays from a fixed observer in a range of sky directions
(varying the transverse momentum ``k^p`` while keeping the ray mostly radial),
integrates each with the Sachs/Jacobi machinery, and records, at a fixed target
redshift, the area distance ``D_A(theta)`` and the shear ``|gamma|(theta)``.

The hallmark of the (non-LTB) Szekeres geometry is that these are **anisotropic**
-- a dipole in the distance-redshift relation across the sky -- whereas an LTB or
FLRW background would give a flat response.  This is the observable the weak-lensing
program of Celerier (2024) ultimately targets.

Run in cosmo units:

    EXCALIBUR_UNITS=cosmo python _excalibur_runs/run_szekeres_lensing_map.py

Outputs ``szekeres_lensing_map.npz`` for ``_postprocessing/plot_szekeres_lensing_map.py``.
"""
import os
import sys

os.environ.setdefault("EXCALIBUR_UNITS", "cosmo")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import G, c
from excalibur.core.szekeres_model import SzekeresModel
from excalibur.metrics.szekeres_metric import SzekeresMetric
from excalibur.observables.szekeres_distances import integrate_area_distance

if c > 1e3:
    raise SystemExit("Run with EXCALIBUR_UNITS=cosmo.")

M0 = 2.0 / (9.0 * G)
T_OBS, R_OBS = 18.0, 3.8
Z_TARGET = 1.0
STOP = lambda s: s[0] < 1.3 or s[1] < 0.25 or s[1] > 4.4


def dipole_model():
    return SzekeresModel(
        M=lambda r: M0 * r ** 3 * (1.0 + 0.30 * np.exp(-((r - 2.0) / 0.7) ** 2)),
        k=lambda r: 0.05 * r ** 2 * np.exp(-((r - 2.0) / 0.7) ** 2),
        S=lambda r: 0.9 + 0.1 * r,
        P=lambda r: 0.25 * np.sin(0.7 * r),
        Q=lambda r: -0.12 * r,
        t_B=lambda r: 0.0, Lambda=0.0, epsilon=1,
        r_min=0.2, r_max=4.5, n_r=220, n_t=440, t_min=1.0, t_max=22.0)


def main(n_dir=25, kp_max=0.08):
    model = dipole_model()
    met = SzekeresMetric(model)

    kp_grid = np.linspace(-kp_max, kp_max, n_dir)
    D_A = np.full(n_dir, np.nan)
    gamma = np.full(n_dir, np.nan)
    z_reached = np.zeros(n_dir)

    for i, kp in enumerate(kp_grid):
        res = integrate_area_distance(met, model, [T_OBS, R_OBS, 0.0, 0.0],
                                      [-1.0, kp, 0.0], n_steps=6000,
                                      span_t=11.0, stop=STOP)
        z = res["z"]
        z_reached[i] = z[-1]
        if z[-1] >= Z_TARGET:
            D_A[i] = np.interp(Z_TARGET, z, res["D_A"])
            gamma[i] = np.interp(Z_TARGET, z, res["gamma"])
        print(f"  dir {i+1}/{n_dir}  k^p={kp:+.3f}  z_max={z[-1]:.2f}  "
              f"D_A(z={Z_TARGET})={D_A[i]:.4f}  |gamma|={gamma[i]:.2e}")

    np.savez("szekeres_lensing_map.npz", kp=kp_grid, D_A=D_A, gamma=gamma,
             z_target=Z_TARGET, z_reached=z_reached)
    good = np.isfinite(D_A)
    if good.sum() > 2:
        rel_spread = np.ptp(D_A[good]) / np.nanmean(D_A[good])
        print(f"\nD_A anisotropy across sky at z={Z_TARGET}: "
              f"{rel_spread*100:.2f}% peak-to-peak  (Szekeres dipole signature)")
    print("Saved szekeres_lensing_map.npz")


if __name__ == "__main__":
    main()
