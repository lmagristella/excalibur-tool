#!/usr/bin/env python3
"""Test analytique : la formule de Sigma_cr dans excalibur est-elle correcte ?

Vérification :
  - calcul direct via Bartelmann-Schneider en distances ANGULAIRES
  - comparaison avec sigma_cr_conventions() du code excalibur
  - vérification de la relation Sigma_cr_phys vs Sigma_cr_comoving

Référence : Bartelmann & Schneider 2001, "Weak Gravitational Lensing", PR 340, 291.
  eq (11): Sigma_cr = c^2 / (4 pi G) * D_s / (D_l * D_ls)  in PHYSICAL angular dist.

Conversion comoving <-> angular for flat FLRW:
  D_A(z) = D_C(z) / (1 + z)
  D_A(z_l -> z_s) = (chi_s - chi_l) / (1 + z_s)   (flat K=0)
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import G, c, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.observables.lensing_conventions import sigma_cr_conventions


def main():
    print("=" * 70)
    print("  AUDIT Sigma_cr : excalibur vs Bartelmann-Schneider 2001")
    print("=" * 70)

    cosmo = LCDM_Cosmology(70.0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)

    z_l = 0.3
    z_s = 1.0

    # Comoving distances (excalibur's native distances)
    D_C_l  = cosmo.comoving_distance(z_l)
    D_C_s  = cosmo.comoving_distance(z_s)
    D_C_ls = D_C_s - D_C_l   # flat K=0

    # Angular diameter distances (physical Bartelmann-Schneider convention)
    D_A_l  = D_C_l  / (1.0 + z_l)
    D_A_s  = D_C_s  / (1.0 + z_s)
    D_A_ls = D_C_ls / (1.0 + z_s)   # flat K=0

    print(f"\nGeometry: z_l = {z_l}, z_s = {z_s}")
    print(f"  D_C_l  = {D_C_l/one_Mpc:.2f} Mpc   D_A_l  = {D_A_l/one_Mpc:.2f} Mpc")
    print(f"  D_C_s  = {D_C_s/one_Mpc:.2f} Mpc   D_A_s  = {D_A_s/one_Mpc:.2f} Mpc")
    print(f"  D_C_ls = {D_C_ls/one_Mpc:.2f} Mpc  D_A_ls = {D_A_ls/one_Mpc:.2f} Mpc")

    # =================================================================
    # Reference: Bartelmann-Schneider 2001 eq 11 (PHYSICAL angular dist)
    # =================================================================
    Sigma_cr_BS = (c**2 / (4.0 * np.pi * G)) * D_A_s / (D_A_l * D_A_ls)
    print(f"\nReference (BS-2001, eq 11, physical angular distances):")
    print(f"  Sigma_cr_phys_BS = c^2/(4 pi G) * D_A_s / (D_A_l * D_A_ls)")
    print(f"                  = {Sigma_cr_BS * one_Mpc**2 / one_Msun:.4e} Msun/Mpc^2")

    # =================================================================
    # Excalibur convention
    # =================================================================
    Sigma_cr_comoving_code, Sigma_cr_physical_code = sigma_cr_conventions(
        D_C_l, D_C_s, D_C_ls, z_l
    )
    print(f"\nExcalibur sigma_cr_conventions():")
    print(f"  Sigma_cr_comoving = c^2/(4 pi G) * D_C_s / (D_C_l * D_C_ls)")
    print(f"                    = {Sigma_cr_comoving_code * one_Mpc**2 / one_Msun:.4e} Msun/Mpc^2")
    print(f"  Sigma_cr_physical = Sigma_cr_comoving / (1+z_l)")
    print(f"                    = {Sigma_cr_physical_code * one_Mpc**2 / one_Msun:.4e} Msun/Mpc^2")

    # =================================================================
    # The CORRECT physical = comoving x (1+z_l) factor
    # =================================================================
    Sigma_cr_physical_correct = Sigma_cr_comoving_code * (1.0 + z_l)
    print(f"\nWhat the physical SHOULD be (per BS-2001):")
    print(f"  Sigma_cr_physical = Sigma_cr_comoving * (1+z_l)")
    print(f"                    = {Sigma_cr_physical_correct * one_Mpc**2 / one_Msun:.4e} Msun/Mpc^2")

    # =================================================================
    # Ratios
    # =================================================================
    print(f"\nRatios to BS-2001 reference:")
    print(f"  Sigma_cr_BS                      = {Sigma_cr_BS * one_Mpc**2 / one_Msun:.4e}")
    print(f"  Sigma_cr_comoving / Sigma_cr_BS  = {Sigma_cr_comoving_code/Sigma_cr_BS:.4f}")
    print(f"  Sigma_cr_physical_code / Sigma_cr_BS = {Sigma_cr_physical_code/Sigma_cr_BS:.4f}")
    print(f"  Sigma_cr_physical_correct / Sigma_cr_BS = {Sigma_cr_physical_correct/Sigma_cr_BS:.4f}")

    print(f"\nKey numeric checks:")
    print(f"  (1+z_l)   = {1+z_l}")
    print(f"  (1+z_l)^2 = {(1+z_l)**2}")
    print(f"  (1+z_s)   = {1+z_s}")

    # =================================================================
    # Verdict
    # =================================================================
    print("\n" + "=" * 70)
    print("VERDICT:")
    print("=" * 70)
    rel_phys_code = abs(Sigma_cr_physical_code - Sigma_cr_BS) / Sigma_cr_BS
    rel_phys_correct = abs(Sigma_cr_physical_correct - Sigma_cr_BS) / Sigma_cr_BS
    rel_comoving = abs(Sigma_cr_comoving_code - Sigma_cr_BS) / Sigma_cr_BS

    if rel_phys_correct < 1e-6:
        print(f"  [OK]  Sigma_cr_comoving * (1+z_l) matches BS-2001 exactly.")
    else:
        print(f"  [WARN] Sigma_cr_comoving * (1+z_l) doesn't match BS-2001 ({rel_phys_correct:.2e})")

    if rel_phys_code < 1e-6:
        print(f"  [OK]  excalibur 'physical' matches BS-2001 -- formula is correct.")
    elif rel_phys_code > 0.1:
        print(f"  [BUG] excalibur 'physical' is OFF by factor {Sigma_cr_physical_code/Sigma_cr_BS:.4f}")
        print(f"        -> Sigma_cr_physical = Sigma_cr_comoving / (1+z_l) is INCORRECT.")
        print(f"        -> Should be Sigma_cr_comoving * (1+z_l).")

    if rel_comoving > 1e-6:
        ratio_comoving = Sigma_cr_comoving_code / Sigma_cr_BS
        # Should this comoving be exactly (1+z_l)^-1 of physical?
        expected_ratio = 1.0 / (1.0 + z_l)
        print(f"\n  Note: Sigma_cr_comoving / Sigma_cr_BS = {ratio_comoving:.4f}")
        print(f"        Expected if comoving == phys / (1+z_l): {expected_ratio:.4f}")
        if abs(ratio_comoving - expected_ratio) < 1e-6:
            print(f"        -> 'comoving' is defined as phys / (1+z_l) (a self-consistent")
            print(f"           but non-standard convention)")


if __name__ == "__main__":
    main()
