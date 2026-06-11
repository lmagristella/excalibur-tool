#!/usr/bin/env python3
r"""Hubble diagram of the Szekeres fit to Pantheon+ (from ``fit_szekeres_snia.py``).

Top: distance modulus mu(z) -- data + best-fit Szekeres (flat-LambdaCDM limit).
Bottom: residuals mu_data - mu_model.

    python _postprocessing/plot_szekeres_snia_fit.py [szekeres_snia_fit.npz]
"""
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(path="szekeres_snia_fit.npz"):
    d = np.load(path)
    z, mu, muerr, mu_model = d["z"], d["mu"], d["muerr"], d["mu_model"]
    Om, H0, chi2, dof = float(d["Om"]), float(d["H0"]), float(d["chi2"]), int(d["dof"])
    order = np.argsort(z)

    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].errorbar(z, mu, yerr=muerr, fmt=".", ms=3, alpha=0.3, color="grey",
                   label="Pantheon+ (1580 SNe)")
    ax[0].plot(z[order], mu_model[order], "-", color="C3", lw=2,
               label=fr"Szekeres flat-$\Lambda$CDM: $\Omega_m$={Om:.3f}, $H_0$={H0:.1f}")
    ax[0].set_ylabel(r"$\mu = 5\log_{10}(D_L/10\,\mathrm{pc})$")
    ax[0].set_xscale("log")
    ax[0].legend(loc="lower right")
    ax[0].set_title(fr"Szekeres fit to Pantheon+ SNIa  "
                    fr"($\chi^2$/dof = {chi2/dof:.2f}, $q_0$={Om/2-(1-Om):+.2f})")

    ax[1].errorbar(z, mu - mu_model, yerr=muerr, fmt=".", ms=3, alpha=0.3, color="grey")
    ax[1].axhline(0.0, color="C3", lw=1.5)
    ax[1].set_xlabel("redshift z")
    ax[1].set_ylabel(r"$\Delta\mu$")
    ax[1].set_ylim(-1.0, 1.0)

    fig.tight_layout()
    out = "szekeres_snia_fit.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "szekeres_snia_fit.npz")
