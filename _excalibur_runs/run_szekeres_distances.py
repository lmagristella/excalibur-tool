#!/usr/bin/env python3
r"""
Redshift -> distance mapping through a quasi-spherical Szekeres model.

Backward-integrates a null ray from an observer and records ``z``, the area
distance ``D_A`` and the luminosity distance ``D_L`` (Sachs/Jacobi route, with
the analytic Szekeres tidal tensor).  Compares an inhomogeneous (dipole) model
to its Einstein--de Sitter background to expose the inhomogeneity imprint on the
distance-redshift relation.

Run in cosmo units (the Szekeres curvature is ill-conditioned in SI):

    EXCALIBUR_UNITS=cosmo python _excalibur_runs/run_szekeres_distances.py

Outputs ``szekeres_distances.npz`` for the post-processing script
``_postprocessing/plot_szekeres_dz.py``.
"""
import os
import sys

os.environ.setdefault("EXCALIBUR_UNITS", "cosmo")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import G, c, unit_system
from excalibur.core.szekeres_model import SzekeresModel
from excalibur.metrics.szekeres_metric import SzekeresMetric
from excalibur.observables.szekeres_distances import integrate_area_distance

if c > 1e3:
    raise SystemExit("Run with EXCALIBUR_UNITS=cosmo (Szekeres curvature is "
                     f"ill-conditioned in SI; got c={c:.3e}, units={unit_system!r}).")

M0 = 2.0 / (9.0 * G)
T_OBS, R_OBS = 18.0, 3.8
STOP = lambda s: s[0] < 1.3 or s[1] < 0.25 or s[1] > 4.4


def eds_model():
    fd = {"dM": lambda r: 3.0 * M0 * r * r, "dk": lambda r: 0.0,
          "dS": lambda r: 0.0, "dP": lambda r: 0.0, "dQ": lambda r: 0.0,
          "dt_B": lambda r: 0.0, "ddS": lambda r: 0.0,
          "ddP": lambda r: 0.0, "ddQ": lambda r: 0.0}
    return SzekeresModel(M=lambda r: M0 * r ** 3, k=lambda r: 0.0,
                         S=lambda r: 1.0, P=lambda r: 0.0, Q=lambda r: 0.0,
                         t_B=lambda r: 0.0, Lambda=0.0, epsilon=1,
                         r_min=0.2, r_max=4.5, n_r=220, n_t=440,
                         t_min=1.0, t_max=22.0, free_derivs=fd)


def dipole_model():
    """EdS background dressed with a smooth Szekeres dipole + curvature dip."""
    return SzekeresModel(
        M=lambda r: M0 * r ** 3 * (1.0 + 0.20 * np.exp(-((r - 2.0) / 0.8) ** 2)),
        k=lambda r: 0.04 * r ** 2 * np.exp(-((r - 2.0) / 0.8) ** 2),
        S=lambda r: 0.9 + 0.1 * r,
        P=lambda r: 0.15 * np.sin(0.6 * r),
        Q=lambda r: -0.1 * r,
        t_B=lambda r: 0.0, Lambda=0.0, epsilon=1,
        r_min=0.2, r_max=4.5, n_r=220, n_t=440, t_min=1.0, t_max=22.0)


def main():
    out = {}
    for label, model, k_spatial in [
        ("eds", eds_model(), [-1.0, 0.0, 0.0]),
        ("dipole", dipole_model(), [-1.0, 0.02, -0.015]),
    ]:
        met = SzekeresMetric(model)
        print(f"Integrating {label} ray ...")
        res = integrate_area_distance(met, model, [T_OBS, R_OBS, 0, 0], k_spatial,
                                      n_steps=8000, span_t=11.0, stop=STOP)
        out[f"{label}_z"] = res["z"]
        out[f"{label}_D_A"] = res["D_A"]
        out[f"{label}_D_L"] = res["D_L"]
        out[f"{label}_gamma"] = res["gamma"]
        print(f"  reached z={res['z'][-1]:.2f}, "
              f"D_A,max={res['D_A'].max():.4f} at z={res['z'][res['D_A'].argmax()]:.2f}, "
              f"max|gamma|={res['gamma'].max():.2e}")

    np.savez("szekeres_distances.npz", **out)
    print("Saved szekeres_distances.npz")


if __name__ == "__main__":
    main()
