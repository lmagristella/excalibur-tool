#!/usr/bin/env python3
r"""
Can a Lambda=0 *void* mimic dark energy?  -- LTB void fit to Pantheon+ SNIa.

The classic inhomogeneous-cosmology question (Celerier 2000; Garcia-Bellido &
Haugbolle 2008): if we sit near the centre of a large underdense region, the
enhanced *local* expansion rate can reproduce the SNIa Hubble diagram **without**
a cosmological constant.  We test it with an exact LTB model (the dipole-off
Szekeres sub-case), ``Lambda = 0``:

    M(r)  = M0 r^3                       (gauge; M0 = H0g^2/(2G), asymptotic EdS)
    k(r)  = -K r^2 exp(-(r/r_void)^2)    (open/underdense centre -> EdS outside)
    t_B   = 0,   S,P,Q = const           (dipole off)

Free parameters ``(H0g, K, r_void)`` are fit to Pantheon+ with the fast Numba
distance backend, observer near the void centre.  We compare the best-fit chi^2
to the flat-LambdaCDM fit (``fit_szekeres_snia.py``) and report the void depth /
size required -- the physical verdict on the void-vs-dark-energy degeneracy.

    EXCALIBUR_UNITS=cosmo python _excalibur_runs/fit_szekeres_void.py
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

H0_CONV = 1.0227e-3            # (km/s/Mpc) -> (1/Gyr)
R_OBS = 0.2                    # observer near the void centre (Gpc)


def _void_model(H0g_kmsMpc, K, r_void, n_r=210, n_t=300):
    H0g = H0g_kmsMpc * H0_CONV
    M0 = H0g ** 2 / (2.0 * G)          # asymptotic EdS (Omega_m -> 1)
    t_o = (2.0 / 3.0) / H0g            # EdS age scale
    rv2 = r_void * r_void

    def k(r):
        return -K * r * r * np.exp(-(r * r) / rv2)

    def dk(r):
        e = np.exp(-(r * r) / rv2)
        return -K * (2.0 * r * e + r * r * e * (-2.0 * r / rv2))

    free = {"dM": lambda r: 3.0 * M0 * r * r, "dk": dk,
            "dS": lambda r: 0.0, "dP": lambda r: 0.0, "dQ": lambda r: 0.0,
            "dt_B": lambda r: 0.0, "ddS": lambda r: 0.0,
            "ddP": lambda r: 0.0, "ddQ": lambda r: 0.0}
    model = SzekeresModel(
        M=lambda r: M0 * r ** 3, k=k, S=lambda r: 1.0, P=lambda r: 0.0,
        Q=lambda r: 0.0, t_B=lambda r: 0.0, Lambda=0.0, epsilon=1,
        r_min=0.02, r_max=14.0, n_r=n_r, n_t=n_t, t_min=0.3, t_max=t_o + 0.5,
        free_derivs=free)
    return model, t_o


def void_DL_Mpc(H0g, K, r_void, z_query, n_steps=11000):
    model, t_o = _void_model(H0g, K, r_void)
    fast = FastSzekeres(model)
    # Outward ray (r increases): disable the inward r-floor stop.
    res = integrate_distance_fast(fast, [t_o, R_OBS, 0, 0], [1.0, 0, 0],
                                  n_steps=n_steps, span_t=t_o, r_min_stop=0.03)
    if res["z"].size == 0 or res["z"].max() < z_query.max():
        return None
    return np.interp(z_query, res["z"], res["D_L"]) * 1.0e3, model, t_o


def central_density_contrast(model, t_o):
    """delta0 = rho(centre)/rho(outskirts) - 1 at the present time."""
    rho_in = model.rho(t_o, 0.1, 0.0, 0.0)
    rho_out = model.rho(t_o, 12.0, 0.0, 0.0)
    return rho_in / rho_out - 1.0


def main():
    data = np.load(os.path.join(os.path.dirname(__file__), "..",
                                "_data", "cosmo", "pantheon_plus.npz"))
    z, mu, muerr = data["z"], data["mu"], data["muerr"]
    inv_var = 1.0 / muerr ** 2
    print(f"Pantheon+ Hubble-flow sample: {len(z)} SNe, z = {z.min():.3f}..{z.max():.3f}")

    def chi2(params):
        H0g, K, r_void = params
        if not (45.0 < H0g < 78.0 and 0.0 <= K < 0.5 and 0.5 < r_void < 8.0):
            return 1e12
        out = void_DL_Mpc(H0g, K, r_void, z)
        if out is None:
            return 1e12
        D_L = out[0]
        mu_model = 5.0 * np.log10(D_L) + 25.0
        return float(np.sum((mu_model - mu) ** 2 * inv_var))

    print("Fitting Lambda=0 LTB void (H0g, K, r_void) to Pantheon+ ...")
    best = None
    for x0 in ([60.0, 0.05, 3.0], [55.0, 0.10, 4.0], [65.0, 0.02, 2.0]):
        r = minimize(chi2, x0=x0, method="Nelder-Mead",
                     options={"xatol": 1e-3, "fatol": 1e-2, "maxiter": 300})
        if best is None or r.fun < best.fun:
            best = r
    H0g, K, r_void = best.x

    D_L, model, t_o = void_DL_Mpc(H0g, K, r_void, z)
    mu_model = 5.0 * np.log10(D_L) + 25.0
    delta0 = central_density_contrast(model, t_o)
    # local Hubble at the observer (centre): H_loc = Phi_,t / Phi
    H_loc = model.Phi_t(t_o, R_OBS) / model.Phi(t_o, R_OBS) / H0_CONV   # km/s/Mpc
    dof = len(z) - 3

    print("\n=== best fit (Lambda=0 LTB void vs Pantheon+) ===")
    print(f"  H0_global (asymptotic) = {H0g:.2f} km/s/Mpc")
    print(f"  H0_local  (at centre)  = {H_loc:.2f} km/s/Mpc")
    print(f"  void depth   K         = {K:.4f}  -> central delta0 = {delta0:+.2f}")
    print(f"  void size    r_void    = {r_void:.2f} Gpc")
    print(f"  chi^2 = {best.fun:.1f}   (dof = {dof}, chi^2/dof = {best.fun/dof:.3f})")
    print(f"\n  Compare: flat-LambdaCDM fit chi^2 ~ 681 (Om=0.35, H0=73).")
    if best.fun < 681 * 1.05:
        print("  -> the void reproduces the SNIa Hubble diagram about as well as LambdaCDM")
        print(f"     BUT requires a ~{2*r_void:.0f} Gpc-scale void with delta0~{delta0:+.2f}")
        print("     (Gpc voids violate the Copernican principle & are excluded by CMB/kSZ).")
    else:
        print("  -> the void fits WORSE than LambdaCDM on SNIa alone.")

    np.savez("szekeres_void_fit.npz", H0g=H0g, H_loc=H_loc, K=K, r_void=r_void,
             delta0=delta0, chi2=best.fun, dof=dof, z=z, mu=mu, muerr=muerr,
             mu_model=mu_model)
    print("Saved szekeres_void_fit.npz")


if __name__ == "__main__":
    main()
