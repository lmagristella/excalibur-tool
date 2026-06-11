#!/usr/bin/env python3
r"""Plot the Szekeres distance-redshift relations from ``run_szekeres_distances.py``.

Reads ``szekeres_distances.npz`` and draws ``D_A(z)`` and ``D_L(z)`` for the EdS
background and the inhomogeneous (dipole) model, plus the relative residual of
the dipole vs the EdS background -- the inhomogeneity imprint on the
distance-redshift relation.

    python _postprocessing/plot_szekeres_dz.py [szekeres_distances.npz]
"""
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(path="szekeres_distances.npz"):
    d = np.load(path)
    z_e, DA_e, DL_e = d["eds_z"], d["eds_D_A"], d["eds_D_L"]
    z_d, DA_d, DL_d = d["dipole_z"], d["dipole_D_A"], d["dipole_D_L"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    ax[0].plot(z_e, DA_e, label="EdS background")
    ax[0].plot(z_d, DA_d, "--", label="Szekeres dipole")
    ax[0].set_xlabel("redshift z"); ax[0].set_ylabel(r"$D_A$ [Gpc]")
    ax[0].set_title("Area distance"); ax[0].legend()

    ax[1].plot(z_e, DL_e, label="EdS background")
    ax[1].plot(z_d, DL_d, "--", label="Szekeres dipole")
    ax[1].set_xlabel("redshift z"); ax[1].set_ylabel(r"$D_L$ [Gpc]")
    ax[1].set_title("Luminosity distance"); ax[1].legend()

    # residual on a common z-grid
    zmax = min(z_e.max(), z_d.max())
    zg = np.linspace(0.05, zmax, 300)
    DAe = np.interp(zg, z_e, DA_e)
    DAd = np.interp(zg, z_d, DA_d)
    ax[2].plot(zg, (DAd - DAe) / DAe * 100.0)
    ax[2].axhline(0.0, color="k", lw=0.5)
    ax[2].set_xlabel("redshift z"); ax[2].set_ylabel(r"$\Delta D_A / D_A$ [%]")
    ax[2].set_title("Dipole imprint vs EdS")

    fig.tight_layout()
    out = "szekeres_distances.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "szekeres_distances.npz")
