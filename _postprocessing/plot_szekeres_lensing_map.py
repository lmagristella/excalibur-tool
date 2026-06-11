#!/usr/bin/env python3
r"""Plot the Szekeres direction-dependent lensing scan.

Reads ``szekeres_lensing_map.npz`` (from ``run_szekeres_lensing_map.py``) and
draws ``D_A`` and ``|gamma|`` versus sky direction at the target redshift.  The
asymmetry of ``D_A`` under ``k^p -> -k^p`` is the Szekeres dipole signature
(absent for an LTB/FLRW background).

    python _postprocessing/plot_szekeres_lensing_map.py [szekeres_lensing_map.npz]
"""
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(path="szekeres_lensing_map.npz"):
    d = np.load(path)
    kp, D_A, gamma = d["kp"], d["D_A"], d["gamma"]
    z_t = float(d["z_target"])
    good = np.isfinite(D_A)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    DA_mean = np.nanmean(D_A[good])
    ax[0].plot(kp[good], (D_A[good] / DA_mean - 1.0) * 100.0, "o-")
    ax[0].axvline(0.0, color="k", lw=0.5)
    ax[0].set_xlabel(r"transverse direction $k^p$")
    ax[0].set_ylabel(r"$D_A/\langle D_A\rangle - 1$ [%]")
    ax[0].set_title(f"Area-distance anisotropy at z={z_t}\n(asymmetry = Szekeres dipole)")

    ax[1].plot(kp[good], gamma[good], "o-")
    ax[1].set_xlabel(r"transverse direction $k^p$")
    ax[1].set_ylabel(r"$|\gamma|$")
    ax[1].set_title(f"Shear at z={z_t}")

    fig.tight_layout()
    out = "szekeres_lensing_map.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "szekeres_lensing_map.npz")
