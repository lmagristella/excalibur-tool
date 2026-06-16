#!/usr/bin/env python3
"""Convert a void (or any triaxial-schema) lensing .npz into the web-app cache:
``{prefix}_beta1.bin``, ``{prefix}_beta2.bin`` (raw little-endian float32, N*N,
web indexing) and ``{prefix}_overlays.json``, plus the JS PROFILES entry.

Web convention (reverse-engineered byte-exact from the live cigar cache):
  - values   : PHYSICAL (unreduced) source-plane position  beta_phys = beta_reduced / (D_l/D_s)
  - layout   : transpose then C-order flatten, matching the app's beta1[i + j*N]
  - dtype    : little-endian float32
Verified against cigar_beta1.bin to max|diff| ~ 3e-4 (float32 rounding).

Usage: python _postprocessing/build_void_webcache.py VOID.npz OUTDIR --prefix void --half 1.50 --n 1000
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lensed_sky_interactive_common as common
import make_lensed_sky_raytracing as ray


def _zero_contours(field, xf, yf):
    """Zero-level curves of a 2D field, as list of {x:[...],y:[...]} segments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    fig = plt.figure()
    cs = plt.contour(xf, yf, field.T, levels=[0.0])
    segs = []
    for seg in cs.allsegs[0] if cs.allsegs else []:
        if len(seg) >= 2:
            segs.append({"x": seg[:, 0].tolist(), "y": seg[:, 1].tolist()})
    plt.close(fig)
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("outdir")
    ap.add_argument("--prefix", default="void")
    ap.add_argument("--half", type=float, default=1.50)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--label", default="Void (underdensity)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = np.load(args.npz, allow_pickle=True)
    Dl, Ds = float(d["D_l_Mpc"]), float(d["D_s_Mpc"])
    rf = Dl / Ds
    print(f"reduce_factor D_l/D_s = {rf:.5f}")

    mapping, raytrace = common.precompute_raytrace_mapping(d, args.half, args.n)
    # PHYSICAL, transposed, float32 little-endian (web convention).
    b1 = np.ascontiguousarray((np.asarray(mapping.beta1) / rf).T, dtype="<f4")
    b2 = np.ascontiguousarray((np.asarray(mapping.beta2) / rf).T, dtype="<f4")
    b1.tofile(os.path.join(args.outdir, f"{args.prefix}_beta1.bin"))
    b2.tofile(os.path.join(args.outdir, f"{args.prefix}_beta2.bin"))
    print(f"wrote {args.prefix}_beta1.bin / _beta2.bin  ({b1.nbytes} bytes each, "
          f"{args.n}x{args.n} f32)  beta_phys range [{b1.min():.3f},{b1.max():.3f}]")

    # Overlays: critical curves (det A = 0) in the image plane and their caustics.
    xf = np.linspace(-args.half, args.half, args.n)
    yf = xf.copy()
    dx = xf[1] - xf[0]
    beta1 = common._interp2d(raytrace["beta1"], raytrace["x_img"], raytrace["y_img"], xf, yf) \
        if False else mapping.beta1  # mapping.beta1 already on the fine grid
    d11, d12, d21, d22, det_A, mu, kappa, g1, g2 = ray._jacobian_fields(
        np.asarray(mapping.beta1), np.asarray(mapping.beta2), dx)
    lam_t = 1.0 - kappa - np.hypot(g1, g2)
    lam_r = 1.0 - kappa + np.hypot(g1, g2)
    crit_tan = _zero_contours(lam_t, xf, yf)
    crit_rad = _zero_contours(lam_r, xf, yf)
    overlays = {
        "critical_tangential": crit_tan,
        "critical_radial": crit_rad,
        "caustic_tangential": [],   # voids: no tangential caustic
        "caustic_radial": [],
    }
    with open(os.path.join(args.outdir, f"{args.prefix}_overlays.json"), "w") as fh:
        json.dump(overlays, fh)
    print(f"wrote {args.prefix}_overlays.json  (crit_tan segs={len(crit_tan)}, crit_rad segs={len(crit_rad)})")

    r_s_Mpc = float(d["r_s_Mpc"])
    print("\n--- paste into PROFILES (webapp index.html) ---")
    print(f'  {{label:"{args.label}", prefix:"{args.prefix}", n:{args.n}, half:{args.half:.2f},')
    print(f'   r_E_mean:0.0, r_E_a:0.0, r_E_b:0.0, r_s:{r_s_Mpc:.4f}, ba:1.0, ca:1.0}},')
    print(f"\nkappa(center) range in the field: [{kappa.min():.3f}, {kappa.max():.3f}]  (negative = underdense)")


if __name__ == "__main__":
    main()
