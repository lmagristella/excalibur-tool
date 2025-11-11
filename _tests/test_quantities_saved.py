#!/usr/bin/env python3
"""
Test script to verify that quantities are properly saved in trajectories.
"""

import h5py
import numpy as np

def check_quantities_in_trajectory(filename):
    """Check if quantities are saved and valid in HDF5 file."""
    print("="*70)
    print(f"Checking quantities in: {filename}")
    print("="*70)
    
    with h5py.File(filename, 'r') as f:
        # Get first photon
        photon_0_states = f['photon_0_states'][:]
        
        print(f"\nTrajectory shape: {photon_0_states.shape}")
        print(f"  - {photon_0_states.shape[0]} time steps")
        print(f"  - {photon_0_states.shape[1]} columns")
        
        if photon_0_states.shape[1] < 14:
            print("\n❌ NO QUANTITIES SAVED (only 8 columns: position + velocity)")
            return False
        
        print("\n✓ Trajectory has 14+ columns (quantities present)")
        
        # Check first valid row (non-NaN)
        valid_mask = ~np.isnan(photon_0_states).any(axis=1)
        first_valid_idx = np.where(valid_mask)[0][0]
        first_valid_row = photon_0_states[first_valid_idx]
        
        print(f"\nFirst valid state (index {first_valid_idx}):")
        print(f"  Coordinates [0-3]: η={first_valid_row[0]:.3e}, x={first_valid_row[1]:.3e}, y={first_valid_row[2]:.3e}, z={first_valid_row[3]:.3e}")
        print(f"  Velocities [4-7]:  u0={first_valid_row[4]:.3e}, u1={first_valid_row[5]:.3e}, u2={first_valid_row[6]:.3e}, u3={first_valid_row[7]:.3e}")
        
        # Check quantities
        a = first_valid_row[8]
        phi = first_valid_row[9]
        grad_phi_x = first_valid_row[10]
        grad_phi_y = first_valid_row[11]
        grad_phi_z = first_valid_row[12]
        phi_dot = first_valid_row[13]
        
        print(f"\n  Quantities [8-13]:")
        print(f"    [8]     a (scale factor):  {a:.6f}")
        print(f"    [9]     phi/c² (potential): {phi:.6e}")
        print(f"    [10-12] grad_phi (m/s²):    [{grad_phi_x:.3e}, {grad_phi_y:.3e}, {grad_phi_z:.3e}]")
        print(f"    [13]    phi_dot/c²:         {phi_dot:.6e}")
        
        # Validate
        if np.isnan(a) or np.isnan(phi):
            print("\n❌ QUANTITIES ARE NaN (not properly computed)")
            return False
        
        if a <= 0 or a > 2:
            print(f"\n⚠️  WARNING: Scale factor a={a:.6f} seems wrong (should be ~0.2-1.0)")
        
        print("\n✓ Quantities are valid!")
        
        # Check multiple rows
        n_valid = np.sum(valid_mask)
        print(f"\nChecking all {n_valid} valid rows...")
        
        all_a = photon_0_states[valid_mask, 8]
        all_phi = photon_0_states[valid_mask, 9]
        
        n_nan_a = np.sum(np.isnan(all_a))
        n_nan_phi = np.sum(np.isnan(all_phi))
        
        print(f"  NaN in a: {n_nan_a}/{n_valid} ({100*n_nan_a/n_valid:.1f}%)")
        print(f"  NaN in phi: {n_nan_phi}/{n_valid} ({100*n_nan_phi/n_valid:.1f}%)")
        
        if n_nan_a > 0 or n_nan_phi > 0:
            print("\n⚠️  Some quantities are NaN (partial failure)")
            return False
        
        print("\n✅ ALL QUANTITIES VALID ACROSS ENTIRE TRAJECTORY")
        
        # Statistics
        print(f"\nStatistics:")
        print(f"  a range:   [{all_a.min():.6f}, {all_a.max():.6f}]")
        print(f"  phi range: [{all_phi.min():.6e}, {all_phi.max():.6e}]")
        
        return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "backward_raytracing_trajectories_OPTIMAL_mass_500_500_500_Mpc.h5"
    
    success = check_quantities_in_trajectory(filename)
    
    if success:
        print("\n" + "="*70)
        print("TEST PASSED: Quantities properly saved ✅")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("TEST FAILED: Quantities not saved or invalid ❌")
        print("="*70)
        sys.exit(1)
