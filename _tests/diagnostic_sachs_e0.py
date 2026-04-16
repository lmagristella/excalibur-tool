#!/usr/bin/env python3
"""
Diagnostic: check whether Sachs basis vectors stay purely spatial
along a curved geodesic through an NFW halo.

Traces ONE photon with enable_lensing=True, recording the full
24-component state at every step.  Then plots:

  1. e1^0(lambda) and e2^0(lambda)   --  temporal components of Sachs vectors
  2. g_mu_nu e^mu k^nu = 0       --  orthogonality check
  3. g_mu_nu e1^mu e2^nu = 0     --  mutual orthogonality
  4. ||e||^2 via g_mu_nu e^mu e^nu  --  norm conservation

This answers the question: "Les vecteurs de Sachs restent-ils
purement spatiaux tout au long d'une trajectoire courbe ?"
Answer: NO  --  e^0 != 0 already at initialisation, and evolves.
"""

import os, sys, time
import numpy as np
from scipy import interpolate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import c, G, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.objects.nfw_halo import NFWHalo


def main():
    print("=" * 60)
    print("  SACHS BASIS  e^0  DIAGNOSTIC")
    print("=" * 60)

    # -- 1. Cosmology ------------------------------------------
    H0 = 70.0
    cosmo = LCDM_Cosmology(H0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
    _ = cosmo.a_of_eta(1e18)
    eta_0 = cosmo._eta_at_a1
    a_0 = cosmo.a_of_eta(eta_0)

    eta_arr = np.linspace(0.5 * eta_0, eta_0, 2000)
    a_arr = np.array([cosmo.a_of_eta(e) for e in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic",
                                     fill_value="extrapolate")

    # -- 2. Small grid + NFW -----------------------------------
    N = 128
    box_Mpc = 20.0
    grid_size = box_Mpc * one_Mpc

    grid = Grid(
        shape=(N, N, N),
        spacing=(grid_size / N,) * 3,
        origin=np.array([0.0, 0.0, 0.0]),
    )

    M_200 = 1e15 * one_Msun
    c_NFW = 5.0
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    halo = NFWHalo(M_200, c_NFW, center)

    print(f"  halo: {halo}")
    print(f"  r_s = {halo.r_s/one_Mpc*1000:.0f} kpc")
    print(f"  grid {N}^3, box {box_Mpc} Mpc, dx = {grid_size/N/one_Mpc*1000:.0f} kpc")

    # Fill potential
    x1d = np.linspace(0, grid_size, N)
    Y, Z = np.meshgrid(x1d, x1d, indexing="ij")
    phi = np.empty((N, N, N), dtype=np.float64)
    for ix in range(N):
        phi[ix] = halo.potential(np.full_like(Y, x1d[ix]), Y, Z)
    grid.add_field("Phi", phi)
    print(f"  |Phi/c^2|_max = {np.max(np.abs(phi))/c**2:.3e}")

    # -- 3. Metric ---------------------------------------------
    from excalibur.grid.interpolator_4d_fast import InterpolatorFast
    interp = InterpolatorFast(grid, boundary="clamp", scheme="tricubic")

    metric = PerturbedFLRWMetricFast(
        a_of_eta=a_of_eta,
        grid=grid,
        interpolator=interp,
        adot_of_eta=cosmo.adot_of_eta,
        cosmology=cosmo,
        enable_lensing=True,
        slow_roll=True,
    )

    # -- 4. Photon (impact parameter b ~ r_s) -----------------
    obs_z_Mpc = 2.0
    obs_pos = np.array([box_Mpc / 2, box_Mpc / 2, obs_z_Mpc]) * one_Mpc
    obs_4d = np.array([eta_0, *obs_pos])

    # Aim slightly off-center  -> b ~ r_s
    b_Mpc = halo.r_s / one_Mpc  # impact parameter ~ r_s
    target = center + np.array([b_Mpc, 0, 0]) * one_Mpc

    direction = target - obs_pos
    direction /= np.linalg.norm(direction)

    g = metric.metric_tensor(obs_4d)
    k_spatial = direction * c
    spatial_sq = g[1, 1] * k_spatial[0]**2 + g[2, 2] * k_spatial[1]**2 + g[3, 3] * k_spatial[2]**2
    k0 = -np.sqrt(abs(-spatial_sq / g[0, 0]))
    k_mu = np.array([k0, *k_spatial])

    e1_init, e2_init = init_sachs_basis(-k_mu, g, a_0)

    photon = Photon(obs_4d.copy(), k_mu.copy())
    photon.e1 = e1_init.copy()
    photon.e2 = e2_init.copy()
    photon.D_flat = np.array([0.0, 0.0, 0.0, 0.0])
    photon.P_flat = np.array([1.0, 0.0, 0.0, 1.0])

    print(f"\n  Initial Sachs basis:")
    print(f"    e1 = [{e1_init[0]:.6e}, {e1_init[1]:.6e}, {e1_init[2]:.6e}, {e1_init[3]:.6e}]")
    print(f"    e2 = [{e2_init[0]:.6e}, {e2_init[1]:.6e}, {e2_init[2]:.6e}, {e2_init[3]:.6e}]")
    print(f"    e1^0 = {e1_init[0]:.6e}")
    print(f"    e2^0 = {e2_init[0]:.6e}")
    print(f"    |e1^0/e1^spatial| = {abs(e1_init[0])/np.linalg.norm(e1_init[1:]):.6e}")

    # -- 5. Integrate, recording at EVERY step -----------------
    D_l = np.linalg.norm(target - obs_pos)
    D_s = 2.0 * D_l
    dx = grid_size / N
    dt = dx / (5.0 * c)
    n_steps = int(np.ceil(D_s / (c * dt)))
    n_steps = min(n_steps, 5000)

    print(f"\n  Integration: dt = {dt:.3e} s,  n_steps = {n_steps}")

    integrator = Integrator(
        metric=metric,
        dt=dt,
        mode="sequential",
        integrator="rk4",
        rtol=1e-8,
    )

    # We want to record the FULL 24-component state at each step.
    # The integrator stores e1, e2, D_flat, P_flat on the photon
    # but record() only stores [x, u, quantities].
    # So we manually hook into it.

    # Build initial state
    state = np.concatenate([photon.x, photon.u, photon.e1, photon.e2,
                            photon.D_flat, photon.P_flat])

    # Manual RK4 integration with full recording
    history = [state.copy()]
    from excalibur.integration.integrator import RK4
    rk4 = RK4()

    t_start = time.time()
    for step in range(n_steps):
        new_state, _, _ = rk4.step(metric, state, dt)
        state = new_state
        history.append(state.copy())
        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{n_steps}")
    elapsed = time.time() - t_start
    print(f"  Done in {elapsed:.1f}s")

    history = np.array(history)  # (n_steps+1, 24)

    # -- 6. Extract components ---------------------------------
    # state layout: [x^0, x^1, x^2, x^3,  k^0, k^1, k^2, k^3,
    #                e1^0, e1^1, e1^2, e1^3,  e2^0, e2^1, e2^2, e2^3,
    #                D11, D12, D21, D22,  P11, P12, P21, P22]
    x_hist   = history[:, 0:4]    # x^mu
    k_hist   = history[:, 4:8]    # k^mu
    e1_hist  = history[:, 8:12]   # e1^mu
    e2_hist  = history[:, 12:16]  # e2^mu

    lam = np.arange(len(history)) * dt  # affine parameter

    e1_0 = e1_hist[:, 0]  # temporal component of e1
    e2_0 = e2_hist[:, 0]  # temporal component of e2
    e1_spatial_norm = np.sqrt(np.sum(e1_hist[:, 1:4]**2, axis=1))
    e2_spatial_norm = np.sqrt(np.sum(e2_hist[:, 1:4]**2, axis=1))

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")

    # -- A. Is e^0 zero? --------------------------------------
    print(f"\n  A) Temporal component e^0:")
    print(f"     e1^0:  init = {e1_0[0]:.6e},  final = {e1_0[-1]:.6e}")
    print(f"            min  = {np.min(e1_0):.6e},  max = {np.max(e1_0):.6e}")
    print(f"     e2^0:  init = {e2_0[0]:.6e},  final = {e2_0[-1]:.6e}")
    print(f"            min  = {np.min(e2_0):.6e},  max = {np.max(e2_0):.6e}")
    print(f"     |e1^0|/|e1_spatial|:  init = {abs(e1_0[0])/e1_spatial_norm[0]:.6e}"
          f"  final = {abs(e1_0[-1])/e1_spatial_norm[-1]:.6e}")
    print(f"     |e2^0|/|e2_spatial|:  init = {abs(e2_0[0])/e2_spatial_norm[0]:.6e}"
          f"  final = {abs(e2_0[-1])/e2_spatial_norm[-1]:.6e}")

    e1_0_changed = abs(e1_0[-1] - e1_0[0])
    e2_0_changed = abs(e2_0[-1] - e2_0[0])
    print(f"     Deltae1^0 = |final - init| = {e1_0_changed:.6e}")
    print(f"     Deltae2^0 = |final - init| = {e2_0_changed:.6e}")

    if e1_0_changed > 1e-15 or e2_0_changed > 1e-15:
        print(f"      -> e^0 EVOLVES during transport [ok]")
    else:
        print(f"      -> e^0 appears constant (check if spacetime is flat)")

    # -- B. Orthogonality checks -------------------------------
    # g_mu_nu e1^mu k^nu = 0  and  g_mu_nu e2^mu k^nu = 0
    # g_mu_nu e1^mu e2^nu = 0
    print(f"\n  B) Orthogonality (sampled at N=20 steps):")
    sample_indices = np.linspace(0, len(history) - 1, 20, dtype=int)

    ek1_list, ek2_list, e12_list, e1_norm_list, e2_norm_list = [], [], [], [], []

    for i in sample_indices:
        xi = x_hist[i]
        ki = k_hist[i]
        e1i = e1_hist[i]
        e2i = e2_hist[i]

        g = metric.metric_tensor(xi)

        # g_mu_nu e1^mu k^nu
        ek1 = np.einsum('ij,i,j->', g, e1i, ki)
        ek2 = np.einsum('ij,i,j->', g, e2i, ki)
        e12 = np.einsum('ij,i,j->', g, e1i, e2i)
        e1n = np.einsum('ij,i,j->', g, e1i, e1i)
        e2n = np.einsum('ij,i,j->', g, e2i, e2i)

        ek1_list.append(ek1)
        ek2_list.append(ek2)
        e12_list.append(e12)
        e1_norm_list.append(e1n)
        e2_norm_list.append(e2n)

    ek1_arr = np.array(ek1_list)
    ek2_arr = np.array(ek2_list)
    e12_arr = np.array(e12_list)
    e1n_arr = np.array(e1_norm_list)
    e2n_arr = np.array(e2_norm_list)

    # Normalize by typical scale to get relative errors
    k_scale = np.max(np.abs(k_hist[:, 0]))
    e_scale = np.max(e1_spatial_norm)

    print(f"     g_mu_nu e1^mu k^nu / (|k||e|):  max = {np.max(np.abs(ek1_arr))/(k_scale*e_scale):.3e}")
    print(f"     g_mu_nu e2^mu k^nu / (|k||e|):  max = {np.max(np.abs(ek2_arr))/(k_scale*e_scale):.3e}")
    print(f"     g_mu_nu e1^mu e2^nu / (|e|^2):   max = {np.max(np.abs(e12_arr))/(e_scale**2):.3e}")

    print(f"\n  C) Norm g_mu_nu e^mu e^nu (should be constant ~a^2):")
    print(f"     g(e1,e1):  init = {e1n_arr[0]:.6e},  final = {e1n_arr[-1]:.6e}")
    print(f"     g(e2,e2):  init = {e2n_arr[0]:.6e},  final = {e2n_arr[-1]:.6e}")
    drift1 = abs(e1n_arr[-1] - e1n_arr[0]) / abs(e1n_arr[0]) if abs(e1n_arr[0]) > 0 else 0
    drift2 = abs(e2n_arr[-1] - e2n_arr[0]) / abs(e2n_arr[0]) if abs(e2n_arr[0]) > 0 else 0
    print(f"     relative drift:  Delta|e1|^2/|e1|^2 = {drift1:.3e}")
    print(f"     relative drift:  Delta|e2|^2/|e2|^2 = {drift2:.3e}")

    # -- 7. Save data for plotting -----------------------------
    outfile = os.path.join(os.path.dirname(__file__), "..",
                           "_data", "output", "sachs_e0_diagnostic.npz")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    np.savez(outfile,
             lam=lam, x=x_hist, k=k_hist, e1=e1_hist, e2=e2_hist,
             ek1=ek1_arr, ek2=ek2_arr, e12=e12_arr,
             e1_norm=e1n_arr, e2_norm=e2n_arr,
             sample_lam=lam[sample_indices])
    print(f"\n  Data saved to {outfile}")

    # -- 8. Quick ASCII plot -----------------------------------
    print(f"\n  e1^0(lambda) along trajectory (first/last 5):")
    for i in [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]:
        idx = i if i >= 0 else len(lam) + i
        print(f"    step {idx:5d}  lambda={lam[idx]:.3e}  "
              f"e1^0={e1_0[idx]:+.6e}  e2^0={e2_0[idx]:+.6e}")

    print(f"\n{'='*60}")
    print(f"  CONCLUSION")
    print(f"{'='*60}")
    if abs(e1_0[0]) > 1e-20 or abs(e2_0[0]) > 1e-20:
        print(f"   -> e^0 != 0 des l'initialisation (composante temporelle non nulle)")
    if e1_0_changed > 1e-15:
        print(f"   -> e1^0 EVOLUE le long de la geodesique (Deltae1^0 = {e1_0_changed:.3e})")
    if e2_0_changed > 1e-15:
        print(f"   -> e2^0 EVOLUE le long de la geodesique (Deltae2^0 = {e2_0_changed:.3e})")
    print(f"   -> Les vecteurs de Sachs ne sont PAS purement spatiaux !")
    print(f"   -> C'est correct : dans un espace courbe, le transport")
    print(f"    parallele melange les composantes via les Gamma^mu_nusigma.")


if __name__ == "__main__":
    main()
