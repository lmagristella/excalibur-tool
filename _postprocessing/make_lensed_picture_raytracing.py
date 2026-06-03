#!/usr/bin/env python3
"""
Render an arbitrary input image through the explicit EXCALIBUR ray-tracing map.

The source image is placed on the reduced source plane used by
make_lensed_sky_raytracing.py, then sampled at the ray-traced coordinates
beta(theta) obtained from explicit ray/source-plane intersections.

Usage:
    python make_lensed_picture_raytracing.py path/to/results.npz --image source.png
    python make_lensed_picture_raytracing.py --image source.jpg --width 0.08 --center-x 0.02
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
if os.environ.get("EXCALIBUR_MPL_BACKEND"):
    matplotlib.use(os.environ["EXCALIBUR_MPL_BACKEND"])
elif __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import make_lensed_sky_raytracing as ray


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Lens an arbitrary PNG/JPG image with the explicit source-plane "
            "ray-tracing map stored in an EXCALIBUR .npz output."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a trajectory-enabled .npz file. Defaults to the latest NFW run.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the source image to place on the reduced source plane.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output filename. Defaults to the run output directory.",
    )
    parser.add_argument("--n-fine", type=int, default=1024,
                        help="Fine rendering resolution per axis.")
    parser.add_argument("--fov", type=float, default=None,
                        help="Half field-of-view in Mpc for the rendered image plane.")
    parser.add_argument("--center-x", type=float, default=0.0,
                        help="Source image center x-position on the reduced source plane [Mpc].")
    parser.add_argument("--center-y", type=float, default=0.0,
                        help="Source image center y-position on the reduced source plane [Mpc].")
    parser.add_argument("--width", type=float, default=None,
                        help="Physical width of the source image on the reduced source plane [Mpc].")
    parser.add_argument("--height", type=float, default=None,
                        help="Physical height of the source image on the reduced source plane [Mpc].")
    parser.add_argument("--angle-deg", type=float, default=0.0,
                        help="Counter-clockwise source rotation angle on the source plane [deg].")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Output figure DPI.")
    return parser.parse_args()


def _load_dataset(path_arg):
    if path_arg:
        path = path_arg
    else:
        path = ray.latest_run("lensing_nfw_analytic") or ray.latest_run("lensing_nfw")

    if not path:
        sys.exit("No .npz result file found")
    if not os.path.isfile(path):
        sys.exit("File not found: %s" % path)

    return np.load(path, allow_pickle=True), path


def _normalize_image(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError("Expected a 2D grayscale or 3D RGB/RGBA image")

    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
    else:
        arr = arr.astype(np.float32)
        if np.nanmax(arr) > 1.0:
            scale = 255.0 if np.nanmax(arr) <= 255.0 else np.nanmax(arr)
            arr = arr / scale

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] not in (3, 4):
        raise ValueError(
            "Unsupported channel count %d; expected grayscale, RGB, or RGBA"
            % arr.shape[2]
        )

    return np.clip(arr, 0.0, 1.0)


def load_source_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("Image file not found: %s" % path)
    return _normalize_image(mpimg.imread(path))


def prepare_source_image(image):
    return np.flipud(np.asarray(image, dtype=np.float32))


def _default_picture_width(d):
    return 0.6 * float(d["r_s_Mpc"])


def _physical_picture_size(args, d, image):
    width = args.width if args.width is not None else _default_picture_width(d)
    if width <= 0.0:
        raise ValueError("--width must be strictly positive")

    aspect = float(image.shape[0]) / float(image.shape[1])
    height = args.height if args.height is not None else width * aspect
    if height <= 0.0:
        raise ValueError("--height must be strictly positive")

    return float(width), float(height)


def build_fine_mapping(raytrace, half_view, n_fine):
    if n_fine < 2:
        raise ValueError("--n-fine must be at least 2")

    xf = np.linspace(-half_view, half_view, n_fine, dtype=np.float32)
    yf = xf.copy()
    t1, t2 = np.meshgrid(xf, yf, indexing="ij")

    beta1 = ray._interp2d(raytrace["beta1"], raytrace["x_img"], raytrace["y_img"],
                          xf, yf).astype(np.float32, copy=False)
    beta2 = ray._interp2d(raytrace["beta2"], raytrace["x_img"], raytrace["y_img"],
                          xf, yf).astype(np.float32, copy=False)

    dx = float(xf[1] - xf[0])
    _, _, _, _, _, mu, kappa, gamma1, gamma2 = ray._jacobian_fields(beta1, beta2, dx)

    return dict(
        xf=xf,
        yf=yf,
        theta1=t1,
        theta2=t2,
        beta1=beta1,
        beta2=beta2,
        shift=np.sqrt((t1 - beta1) ** 2 + (t2 - beta2) ** 2).astype(np.float32, copy=False),
        mu=np.asarray(mu, dtype=np.float32),
        kappa=np.asarray(kappa, dtype=np.float32),
        gamma1=np.asarray(gamma1, dtype=np.float32),
        gamma2=np.asarray(gamma2, dtype=np.float32),
        extent=[float(xf[0]), float(xf[-1]), float(yf[0]), float(yf[-1])],
    )


def _rotate_into_picture_frame(x, y, center, angle_deg):
    dx = x - center[0]
    dy = y - center[1]
    ang = np.deg2rad(angle_deg)
    c = np.cos(ang)
    s = np.sin(ang)
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return u, v


def sample_prepared_source_image(image, x, y, center, width, height, angle_deg=0.0):
    ny, nx, nchan = image.shape
    u, v = _rotate_into_picture_frame(x, y, center, angle_deg)
    px = (u / width + 0.5) * (nx - 1)
    py = (v / height + 0.5) * (ny - 1)
    inside = (
        (px >= 0.0) & (px <= nx - 1) &
        (py >= 0.0) & (py <= ny - 1)
    )

    px = np.clip(px, 0.0, nx - 1)
    py = np.clip(py, 0.0, ny - 1)

    x0 = np.floor(px).astype(np.intp)
    y0 = np.floor(py).astype(np.intp)
    x1 = np.minimum(x0 + 1, nx - 1)
    y1 = np.minimum(y0 + 1, ny - 1)

    wx = (px - x0)[..., None].astype(np.float32)
    wy = (py - y0)[..., None].astype(np.float32)

    top = (1.0 - wx) * image[y0, x0] + wx * image[y0, x1]
    bottom = (1.0 - wx) * image[y1, x0] + wx * image[y1, x1]
    sampled = ((1.0 - wy) * top + wy * bottom).astype(np.float32, copy=False)
    sampled *= inside[..., None]

    return np.clip(sampled, 0.0, 1.0)


def sample_source_image(image, x, y, center, width, height, angle_deg=0.0):
    return sample_prepared_source_image(
        prepare_source_image(image),
        x,
        y,
        center=center,
        width=width,
        height=height,
        angle_deg=angle_deg,
    )


def render_lensed_picture_prepared(mapping, image, center, width, height, angle_deg):
    source = sample_prepared_source_image(
        image,
        mapping["theta1"],
        mapping["theta2"],
        center=center,
        width=width,
        height=height,
        angle_deg=angle_deg,
    )
    lensed = sample_prepared_source_image(
        image,
        mapping["beta1"],
        mapping["beta2"],
        center=center,
        width=width,
        height=height,
        angle_deg=angle_deg,
    )
    return dict(source=source, lensed=lensed)


def render_lensed_only_prepared(mapping, image, center, width, height, angle_deg):
    return sample_prepared_source_image(
        image,
        mapping["beta1"],
        mapping["beta2"],
        center=center,
        width=width,
        height=height,
        angle_deg=angle_deg,
    )


def render_lensed_picture(mapping, image, center, width, height, angle_deg):
    return render_lensed_picture_prepared(
        mapping,
        prepare_source_image(image),
        center=center,
        width=width,
        height=height,
        angle_deg=angle_deg,
    )


def _imshow_xy(ax, image_xyc, extent):
    ax.imshow(np.swapaxes(image_xyc, 0, 1), origin="lower", extent=extent,
              interpolation="bilinear", aspect="equal")


def _plot_reference_overlays(ax, d, center):
    rs = float(d["r_s_Mpc"])
    kappa_s = ray.compute_kappa_s(d)
    r_E_nfw = ray.einstein_radius_nfw_Mpc(rs, kappa_s)
    th = np.linspace(0.0, 2.0 * np.pi, 300)

    if r_E_nfw > 0.0:
        ax.plot(r_E_nfw * np.cos(th), r_E_nfw * np.sin(th), "lime",
                ls="-", lw=1.4, alpha=0.85,
                label=r"$r_E^{\rm NFW}$ = %.4f Mpc" % r_E_nfw)
    ax.plot(rs * np.cos(th), rs * np.sin(th), "cyan", ls=":",
            lw=0.8, alpha=0.55, label=r"$r_s$ = %.3f Mpc" % rs)
    ax.plot(center[0], center[1], "w+", ms=13, mew=2, alpha=0.95)


def plot_picture(d, mapping, rendered, center, width, height, angle_deg,
                 image_path, output_path, dpi):
    ext = mapping["extent"]

    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(16, 7.5))
    fig.set_facecolor("black")

    _imshow_xy(ax_s, rendered["source"], ext)
    _imshow_xy(ax_l, rendered["lensed"], ext)

    _plot_reference_overlays(ax_s, d, center)
    _plot_reference_overlays(ax_l, d, center)

    ax_s.set_title("Source image on reduced source plane", fontsize=12, color="white", pad=8)
    ax_l.set_title("Lensed image (ray/plane intersection)", fontsize=12, color="white", pad=8)
    ax_s.set_xlabel(r"$b_1$ [Mpc]", fontsize=11, color="white")
    ax_s.set_ylabel(r"$b_2$ [Mpc]", fontsize=11, color="white")
    ax_l.set_xlabel(r"$b_1$ [Mpc]", fontsize=11, color="white")

    for ax in (ax_s, ax_l):
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("gray")
        ax.legend(fontsize=9, loc="upper right",
                  facecolor="black", edgecolor="gray",
                  labelcolor="white", framealpha=0.7)

    img_name = Path(image_path).name
    fig.suptitle(
        r"Explicit ray tracing | image=%s | size=%.4f x %.4f Mpc | angle=%.1f$^\circ$"
        % (img_name, width, height, angle_deg),
        fontsize=12,
        color="white",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    print("  [ok] %s" % output_path)
    plt.close(fig)


def main():
    args = parse_args()
    d, npz_path = _load_dataset(args.path)
    image = load_source_image(args.image)

    try:
        raytrace = ray.build_raytrace_map(d)
    except KeyError as exc:
        sys.exit(str(exc))

    fov = ray._view_fov(d, args.fov)
    width, height = _physical_picture_size(args, d, image)
    center = (float(args.center_x), float(args.center_y))
    mapping = build_fine_mapping(raytrace, fov, args.n_fine)
    rendered = render_lensed_picture(
        mapping,
        image,
        center=center,
        width=width,
        height=height,
        angle_deg=float(args.angle_deg),
    )

    namer = ray.RunNamer.from_npz(npz_path)
    if args.output:
        output_path = args.output
    else:
        stem = Path(args.image).stem.replace(" ", "_")
        output_path = namer.plot("lensed_picture_raytracing_%s" % stem)

    zl = float(d["z_l"])
    zs = float(d["z_source"])
    max_resid_kpc = 1e3 * np.max(np.abs(raytrace["hit_residual_Mpc"]))
    rms_resid_kpc = 1e3 * np.sqrt(np.mean(raytrace["hit_residual_Mpc"] ** 2))

    print("Loaded: %s" % os.path.basename(npz_path))
    print("  image=%s" % args.image)
    print("  z_l=%.4f  z_s=%.4f" % (zl, zs))
    print("  FoV = +/- %.4f Mpc" % fov)
    print("  Source image size = %.4f x %.4f Mpc" % (width, height))
    print("  Source image center = (%+.4f, %+.4f) Mpc" % center)
    print("  Source rotation = %.2f deg" % float(args.angle_deg))
    print("  Plane-intersection residuals: rms=%.3e kpc  max=%.3e kpc"
          % (rms_resid_kpc, max_resid_kpc))

    plot_picture(
        d,
        mapping,
        rendered,
        center=center,
        width=width,
        height=height,
        angle_deg=float(args.angle_deg),
        image_path=args.image,
        output_path=output_path,
        dpi=args.dpi,
    )

    print("Done.")


if __name__ == "__main__":
    main()