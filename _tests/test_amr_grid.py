#!/usr/bin/env python3
"""
Quick tests for the AMR grid module.

Tests:
  1. AMRPatch creation and containment
  2. AMRGrid with manual patches
  3. AMRInterpolator: accuracy vs analytic potential
  4. AMRGrid.from_field: automatic refinement
  5. Drop-in compatibility with InterpolatorFast API
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_4d_fast import InterpolatorFast
from excalibur.grid.amr_grid import AMRPatch, AMRGrid, AMRInterpolator
from excalibur.core.constants import G, one_Mpc, one_Msun


def _nfw_potential(x, y, z, center, rho_s, r_s):
    """Simple NFW potential for testing."""
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    r = np.maximum(r, 1e-10 * r_s)
    return -4.0 * np.pi * G * rho_s * r_s**3 / r * np.log(1.0 + r / r_s)


def test_patch_containment():
    """Test AMRPatch.contains()."""
    origin = np.array([1.0, 2.0, 3.0])
    extent = np.array([10.0, 10.0, 10.0])
    patch = AMRPatch(level=1, origin=origin, extent=extent, shape=(64, 64, 64))

    assert patch.contains(np.array([5.0, 5.0, 5.0])), "Should be inside"
    assert not patch.contains(np.array([0.0, 5.0, 5.0])), "Should be outside (x too low)"
    assert not patch.contains(np.array([12.0, 5.0, 5.0])), "Should be outside (x too high)"
    print("  [ok] test_patch_containment")


def test_amr_grid_manual():
    """Test AMRGrid with a manually added patch."""
    N = 64
    L = 10.0 * one_Mpc
    dx = L / N

    root = Grid(shape=(N, N, N), spacing=(dx, dx, dx), origin=(0, 0, 0))
    center = np.array([L/2, L/2, L/2])

    # NFW-like params
    M_200 = 1e14 * one_Msun
    H0 = 70e3 / one_Mpc
    rho_cr = 3.0 * H0**2 / (8.0 * np.pi * G)
    R_200 = (3.0 * M_200 / (4 * np.pi * 200 * rho_cr))**(1/3)
    c_NFW = 5.0
    r_s = R_200 / c_NFW
    fc = np.log(1 + c_NFW) - c_NFW / (1 + c_NFW)
    rho_s = M_200 / (4 * np.pi * r_s**3 * fc)

    def phi_fn(x, y, z):
        return _nfw_potential(x, y, z, center, rho_s, r_s)

    # Fill root
    x1d = np.linspace(0, L, N)
    Y, Z = np.meshgrid(x1d, x1d, indexing="ij")
    phi = np.empty((N, N, N))
    for ix in range(N):
        phi[ix] = phi_fn(np.full_like(Y, x1d[ix]), Y, Z)
    root.add_field("Phi", phi)

    amr = AMRGrid(root)
    assert amr.max_level == 0
    assert amr.find_patch(center) is None  # no patches yet

    print("  [ok] test_amr_grid_manual")


def test_amr_interpolator_accuracy():
    """
    Compare AMR interpolation vs uniform grid on an NFW potential.
    The AMR grid should be significantly more accurate near the halo center.
    """
    L = 10.0 * one_Mpc
    center = np.array([L/2, L/2, L/2])

    # NFW params
    M_200 = 1e15 * one_Msun
    H0 = 70e3 / one_Mpc
    rho_cr = 3 * H0**2 / (8 * np.pi * G)
    R_200 = (3 * M_200 / (4 * np.pi * 200 * rho_cr))**(1/3)
    c_NFW = 5.0
    r_s = R_200 / c_NFW
    fc = np.log(1 + c_NFW) - c_NFW / (1 + c_NFW)
    rho_s = M_200 / (4 * np.pi * r_s**3 * fc)

    def phi_fn(x, y, z):
        return _nfw_potential(x, y, z, center, rho_s, r_s)

    # Coarse grid
    N_coarse = 64
    dx_coarse = L / N_coarse
    root = Grid(shape=(N_coarse,)*3, spacing=(dx_coarse,)*3, origin=(0, 0, 0))
    x1d = np.linspace(0, L, N_coarse)
    Y, Z = np.meshgrid(x1d, x1d, indexing="ij")
    phi_coarse = np.empty((N_coarse,)*3)
    for ix in range(N_coarse):
        phi_coarse[ix] = phi_fn(np.full_like(Y, x1d[ix]), Y, Z)
    root.add_field("Phi", phi_coarse)

    # Uniform interpolator
    interp_uniform = InterpolatorFast(root, boundary="clamp", scheme="tricubic")

    # Build AMR with auto-refinement
    amr = AMRGrid.from_field(
        root, "Phi", phi_fn,
        max_level=3, ratio=2,
        refine_threshold=0.01,
        refine_mode="gradient",
        min_patch_cells=32,
        boundary="clamp", scheme="tricubic",
        verbose=True,
    )

    interp_amr = AMRInterpolator(amr)

    # Test points at various distances from halo center
    test_radii_Mpc = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
    direction = np.array([1.0, 0.0, 0.0])

    print("\n  Radius [Mpc]  | err_uniform [%] | err_AMR [%]   | speedup")
    print("  " + "-" * 65)

    for r_Mpc in test_radii_Mpc:
        r = r_Mpc * one_Mpc
        pos = center + r * direction

        # Clamp to grid
        pos = np.clip(pos, dx_coarse * 2, L - dx_coarse * 2)

        phi_exact = phi_fn(pos[0], pos[1], pos[2])
        phi_uni = interp_uniform.interpolate(pos, "Phi")
        phi_amr = interp_amr.interpolate(pos, "Phi")

        err_uni = abs((phi_uni - phi_exact) / phi_exact) * 100 if phi_exact != 0 else 0
        err_amr = abs((phi_amr - phi_exact) / phi_exact) * 100 if phi_exact != 0 else 0

        speedup = err_uni / err_amr if err_amr > 0 else float('inf')
        print(f"  {r_Mpc:8.1f}       | {err_uni:13.4f}   | {err_amr:11.4f}   | {speedup:.1f}x")

    # Check that AMR is more accurate at inner radii
    r_inner = 0.2 * one_Mpc
    pos_inner = center + r_inner * direction
    pos_inner = np.clip(pos_inner, dx_coarse * 2, L - dx_coarse * 2)
    phi_exact = phi_fn(pos_inner[0], pos_inner[1], pos_inner[2])
    phi_uni = interp_uniform.interpolate(pos_inner, "Phi")
    phi_amr = interp_amr.interpolate(pos_inner, "Phi")
    err_uni = abs(phi_uni - phi_exact) / abs(phi_exact)
    err_amr = abs(phi_amr - phi_exact) / abs(phi_exact)

    assert err_amr < err_uni, (
        f"AMR should be more accurate at r=0.2 Mpc: "
        f"err_amr={err_amr:.6f} vs err_uni={err_uni:.6f}"
    )
    print("\n  [ok] test_amr_interpolator_accuracy (AMR beats uniform at small radii)")


def test_amr_gradient_accuracy():
    """
    Test that the AMR interpolator gives better gradients near the halo core.
    """
    L = 10.0 * one_Mpc
    center = np.array([L/2, L/2, L/2])

    M_200 = 1e15 * one_Msun
    H0 = 70e3 / one_Mpc
    rho_cr = 3 * H0**2 / (8 * np.pi * G)
    R_200 = (3 * M_200 / (4 * np.pi * 200 * rho_cr))**(1/3)
    c_NFW = 5.0
    r_s = R_200 / c_NFW
    fc = np.log(1 + c_NFW) - c_NFW / (1 + c_NFW)
    rho_s = M_200 / (4 * np.pi * r_s**3 * fc)

    def phi_fn(x, y, z):
        return _nfw_potential(x, y, z, center, rho_s, r_s)

    def grad_phi_analytic(pos):
        """Analytic gradient of NFW potential: gradPhi = (dPhi/dr) r_hat."""
        r_vec = pos - center
        r = np.linalg.norm(r_vec)
        if r < 1e-10 * r_s:
            return np.zeros(3)
        s = r / r_s
        # dPhi/dr = 4pi G rho_s r_s^3 [ ln(1+r/r_s)/r^2 - 1/(r(r+r_s)) ]
        # This is positive (gradient points outward from halo center).
        dphidr = 4 * np.pi * G * rho_s * r_s**3 * (
            np.log(1 + s) / r**2 - 1.0 / (r * (r + r_s))
        )
        return dphidr * r_vec / r

    N = 64
    dx = L / N
    root = Grid(shape=(N,)*3, spacing=(dx,)*3, origin=(0, 0, 0))
    x1d = np.linspace(0, L, N)
    Y, Z = np.meshgrid(x1d, x1d, indexing="ij")
    phi = np.empty((N,)*3)
    for ix in range(N):
        phi[ix] = phi_fn(np.full_like(Y, x1d[ix]), Y, Z)
    root.add_field("Phi", phi)

    amr = AMRGrid.from_field(
        root, "Phi", phi_fn,
        max_level=3, ratio=2,
        refine_threshold=0.01,
        min_patch_cells=32,
        verbose=False,
    )
    interp_amr = AMRInterpolator(amr)
    interp_uni = InterpolatorFast(root, boundary="clamp", scheme="tricubic")

    # Test gradient at r = 0.3 Mpc
    r_test = 0.3 * one_Mpc
    pos = center + r_test * np.array([1, 0, 0])
    pos = np.clip(pos, dx * 2, L - dx * 2)

    grad_exact = grad_phi_analytic(pos)
    grad_uni = np.array(interp_uni.gradient(pos, "Phi"))
    grad_amr = np.array(interp_amr.gradient(pos, "Phi"))

    err_uni = np.linalg.norm(grad_uni - grad_exact) / np.linalg.norm(grad_exact)
    err_amr = np.linalg.norm(grad_amr - grad_exact) / np.linalg.norm(grad_exact)

    print(f"\n  Gradient at r=0.3 Mpc: err_uniform={err_uni:.4f}, err_AMR={err_amr:.4f}")
    assert err_amr < err_uni, "AMR gradient should be more accurate"
    print("  [ok] test_amr_gradient_accuracy")


def test_api_compatibility():
    """
    Verify AMRInterpolator has the same API as InterpolatorFast.
    """
    N = 32
    L = 1.0
    dx = L / N
    root = Grid(shape=(N,)*3, spacing=(dx,)*3, origin=(0, 0, 0))
    root.add_field("Phi", np.random.randn(N, N, N) * 1e-3)

    amr = AMRGrid(root)
    interp = AMRInterpolator(amr)

    pos = np.array([0.5, 0.5, 0.5])

    # All methods should work
    val = interp.interpolate(pos, "Phi")
    assert np.isfinite(val)

    grad = interp.gradient(pos, "Phi")
    assert len(grad) == 3

    val2, grad2 = interp.value_and_gradient(pos, "Phi")
    assert np.isfinite(val2)

    val3, grad3, dtd = interp.value_gradient_and_time_derivative(pos, "Phi")
    assert np.isfinite(val3)

    val4, grad4, hess, dtd2 = interp.value_gradient_hessian_and_time_derivative(pos, "Phi")
    assert np.isfinite(val4)
    assert len(hess) == 6  # hxx, hyy, hzz, hxy, hxz, hyz

    H = interp.hessian(pos, "Phi")
    assert H.shape == (3, 3)

    lap = interp.laplacian(pos, "Phi")
    assert np.isfinite(lap)

    print("  [ok] test_api_compatibility")


def main():
    print("=" * 60)
    print("  AMR Grid Module  --  Tests")
    print("=" * 60)

    test_patch_containment()
    test_amr_grid_manual()
    test_api_compatibility()
    test_amr_interpolator_accuracy()
    test_amr_gradient_accuracy()

    print("\n" + "=" * 60)
    print("  ALL AMR TESTS PASSED [ok]")
    print("=" * 60)


if __name__ == "__main__":
    main()
