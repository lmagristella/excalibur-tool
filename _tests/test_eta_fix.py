#!/usr/bin/env python3
"""Test conformal time calculation after fix."""

from excalibur.core.cosmology import LCDM_Cosmology
import numpy as np

cosmo = LCDM_Cosmology(70, 0.3, 0, 0.7)

# Force recomputation by deleting cache
if hasattr(cosmo, '_eta_to_a_base_computed'):
    delattr(cosmo, '_eta_to_a_base_computed')

# Call to compute eta_at_a1
a_test = cosmo.a_of_eta(1e18)

years_to_sec = 365.25 * 24 * 3600

print(f"eta à a=1: {cosmo._eta_at_a1:.3e} s")
print(f"           = {cosmo._eta_at_a1/years_to_sec:.3e} années")
print(f"           = {cosmo._eta_at_a1/(years_to_sec*1e9):.1f} milliards d'années")
print()
print(f"a(1e18 s = {1e18/(years_to_sec*1e9):.1f} Gyr) = {cosmo.a_of_eta(1e18):.4f}")
print(f"a(1.45e18 s = 46 Gyr) = {cosmo.a_of_eta(1.45e18):.4f}")
print(f"a(4.4e17 s = 14 Gyr) = {cosmo.a_of_eta(4.4e17):.4f}")
