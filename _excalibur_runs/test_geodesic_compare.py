#!/usr/bin/env python3
"""
Compare compute_tensorial_acceleration vs compute_analytical_acceleration
for pure FLRW (Phi=0).  Should agree exactly.  If not -> tensorial is buggy.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from excalibur.metrics.perturbed_flrw_metric_fast import (
    compute_tensorial_acceleration,
    compute_analytical_acceleration,
)
from excalibur.core.constants import c

def main():
    print("Pure FLRW geodesic acceleration:  tensorial vs analytical")
    print("="*72)
    for a in (1.0, 0.9, 0.8, 0.7, 0.5):
        H0 = 70e3 / 3.086e22
        Om, Ol = 0.3, 0.7
        H_phys = H0 * np.sqrt(Om/a**3 + Ol)
        adot = a * a * H_phys     # da/deta in conformal time

        # null photon: u^0 = -1, |u_spatial|^2 = c^2 (u^0)^2  -> u^x = c
        u0, u1, u2, u3 = -1.0, c, 0.0, 0.0

        # Pure FLRW: phi=0, all derivatives 0
        kw_zero = dict(phi=0.0, phi_dot=0.0)

        ten = compute_tensorial_acceleration(
            u0, u1, u2, u3, a, adot,
            phi=0.0, grad_phi_x=0.0, grad_phi_y=0.0, grad_phi_z=0.0,
            phi_dot=0.0, c_val=c,
        )
        ana = compute_analytical_acceleration(
            u0, u1, u2, u3, a, adot,
            phi=0.0, dphidx=0.0, dphidy=0.0, dphidz=0.0,
            dphideta=0.0, c_val=c,
        )

        # Analytical FLRW prediction: dk^eta/dlambda = -2 H_co × (k^eta)^2
        # where H_co = a'/a = adot/a
        H_co = adot / a
        du0_predicted = -2.0 * H_co * u0 * u0
        # Spatial: dk^x/dlambda = -2 H_co × k^x × k^eta (from Gamma^x_x0)
        du1_predicted = -2.0 * H_co * u1 * u0

        print(f"\n  a={a:.2f}  H_co={H_co:.4e} 1/s")
        print(f"    expected du0  = {du0_predicted:+.4e}")
        print(f"    tensorial du0 = {ten[0]:+.4e}   (diff/expected = {(ten[0]-du0_predicted)/abs(du0_predicted):+.4e})")
        print(f"    analytical du0= {ana[0]:+.4e}   (diff/expected = {(ana[0]-du0_predicted)/abs(du0_predicted):+.4e})")
        print(f"    expected du1  = {du1_predicted:+.4e}")
        print(f"    tensorial du1 = {ten[1]:+.4e}")
        print(f"    analytical du1= {ana[1]:+.4e}")


if __name__ == "__main__":
    main()
