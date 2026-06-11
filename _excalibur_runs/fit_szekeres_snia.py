#!/usr/bin/env python3
r"""
Fit a (flat-LambdaCDM-background) Szekeres model to the Pantheon+ SNIa data.

This is the first contact with **real data** (Celerier 2024's program): the
Szekeres free functions are constrained by observations.  We start with the
homogeneous (FLRW) limit -- ``M(r)=M0 r^3``, ``k=0``, ``Lambda=3 H0^2 (1-Om)/c^2``
-- whose only free parameters are ``(Omega_m, H0)``, fit to the Pantheon+
distance moduli via the *fast* Numba distance backend.  Recovering the standard
``Om ~ 0.3`` from 1580 supernovae validates the whole pipeline against the real
Hubble diagram; the inhomogeneous (void / dipole) extensions then plug into the
same machinery (``szekeres_void_fit`` TODO).

Data: ``_data/cosmo/pantheon_plus.npz`` (Pantheon+SH0ES Hubble-flow sample,
z>0.01, calibrators removed; MU_SH0ES distance moduli + diagonal errors).

    EXCALIBUR_UNITS=cosmo python _excalibur_runs/fit_szekeres_snia.py
"""
import os
import sys

os.environ.setdefault("EXCALIBUR_UNITS", "cosmo")

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import G, c
from excalibur.core.szekeres_model import SzekeresModel
from excalibur.metrics.szekeres_metric_fast import FastSzekeres, integrate_distance_fast

H0_CONV = 1.0227e-3        # (km/s/Mpc) -> (1/Gyr)
_FREE0 = {"dk": lambda r: 0.0, "dS": lambda r: 0.0, "dP": lambda r: 0.0,
          "dQ": lambda r: 0.0, "dt_B": lambda r: 0.0, "ddS": lambda r: 0.0,
          "ddP": lambda r: 0.0, "ddQ": lambda r: 0.0}


def lcdm_DL_Mpc(Om, H0_kmsMpc, z_query, n_r=200, n_t=300, n_steps=9000):
    r"""Luminosity distance (Mpc) of a flat-LambdaCDM Szekeres model at z_query."""
    H0 = H0_kmsMpc * H0_CONV
    OL = 1.0 - Om
    M0 = H0 ** 2 * Om / (2.0 * G)
    Lam = 3.0 * H0 ** 2 * OL / (c * c)
    t_o = (2.0 / (3.0 * H0 * np.sqrt(OL))) * np.arcsinh(np.sqrt(OL / Om))
    free = dict(_FREE0, dM=lambda r: 3.0 * M0 * r * r)
    model = SzekeresModel(
        M=lambda r: M0 * r ** 3, k=lambda r: 0.0, S=lambda r: 1.0,
        P=lambda r: 0.0, Q=lambda r: 0.0, t_B=lambda r: 0.0,
        Lambda=Lam, epsilon=1, r_min=0.05, r_max=9.0, n_r=n_r, n_t=n_t,
        t_min=0.3, t_max=t_o + 0.5, free_derivs=free)
    fast = FastSzekeres(model)
    res = integrate_distance_fast(fast, [t_o, 0.5, 0, 0], [1.0, 0, 0],
                                  n_steps=n_steps, span_t=t_o)
    if res["z"].max() < z_query.max():
        return None
    D_L_Gpc = np.interp(z_query, res["z"], res["D_L"])
    return D_L_Gpc * 1.0e3       # Gpc -> Mpc


def main():
    data = np.load(os.path.join(os.path.dirname(__file__), "..",
                                "_data", "cosmo", "pantheon_plus.npz"))
    z, mu, muerr = data["z"], data["mu"], data["muerr"]
    print(f"Pantheon+ Hubble-flow sample: {len(z)} SNe, z = {z.min():.3f}..{z.max():.3f}")
    inv_var = 1.0 / muerr ** 2

    def chi2(params):
        Om, H0 = params
        if not (0.05 < Om < 0.8 and 55.0 < H0 < 85.0):
            return 1e12
        D_L = lcdm_DL_Mpc(Om, H0, z)
        if D_L is None:
            return 1e12
        mu_model = 5.0 * np.log10(D_L) + 25.0
        return float(np.sum((mu_model - mu) ** 2 * inv_var))

    print("Fitting flat-LambdaCDM Szekeres (Omega_m, H0) to Pantheon+ ...")
    res = minimize(chi2, x0=[0.3, 72.0], method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-2, "maxiter": 200})
    Om, H0 = res.x
    dof = len(z) - 2
    print("\n=== best fit (flat-LambdaCDM Szekeres vs Pantheon+) ===")
    print(f"  Omega_m = {Om:.3f}")
    print(f"  H0      = {H0:.2f} km/s/Mpc")
    print(f"  chi^2   = {res.fun:.1f}   (dof = {dof},  chi^2/dof = {res.fun/dof:.3f})")
    print(f"  -> deceleration q0 = Om/2 - (1-Om) = {Om/2 - (1-Om):+.3f} "
          f"({'accelerating' if Om/2-(1-Om) < 0 else 'decelerating'})")

    np.savez("szekeres_snia_fit.npz", Om=Om, H0=H0, chi2=res.fun, dof=dof,
             z=z, mu=mu, muerr=muerr,
             mu_model=5.0 * np.log10(lcdm_DL_Mpc(Om, H0, z)) + 25.0)
    print("Saved szekeres_snia_fit.npz")


if __name__ == "__main__":
    main()
