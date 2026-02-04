import h5py
import numpy as np

path = "_data/output/excalibur_run_perturbed_flrw_static_M1.0e15_R3.0_mass500_500_500_obs4_500_500_N40_rk4_S762_static1_D500_cone10deg_local1_Mpc.h5"

with h5py.File(path, "r") as f:
    traj = f["trajectories"][0]      # photon 0
    print("traj shape:", traj.shape)
    print("first row:", traj[0])
    print("second row:", traj[1])
    print("delta row:", traj[1] - traj[0])

    # Print likely blocks
    print("\ncols 0..7 first row:", traj[0, :8])
    print("cols 0..7 second row:", traj[1, :8])

    # candidate positions: 1:4
    dx = traj[1,1:4] - traj[0,1:4]
    print("\ndelta (1:4):", dx, "norm:", np.linalg.norm(dx))

    # candidate positions: 4:7
    dx2 = traj[1,4:7] - traj[0,4:7]
    print("delta (4:7):", dx2, "norm:", np.linalg.norm(dx2))
