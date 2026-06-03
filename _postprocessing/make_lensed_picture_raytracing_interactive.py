#!/usr/bin/env python3
"""Interactive viewer for explicit ray-traced lensing of an input image."""
from __future__ import annotations

import argparse
import os
import threading

import numpy as np

import matplotlib
if os.environ.get("EXCALIBUR_MPL_BACKEND"):
    matplotlib.use(os.environ["EXCALIBUR_MPL_BACKEND"])
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Button, Slider

import make_lensed_picture_raytracing as pic
import make_lensed_sky_raytracing as ray
from lensed_sky_interactive_common import default_half_view, load_dataset, summarize_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive explicit-raytracing viewer for an arbitrary source image."
    )
    parser.add_argument("path", nargs="?", help="Path to a trajectory-enabled .npz file")
    parser.add_argument("--image", required=True,
                        help="Path to the source image to place on the reduced source plane.")
    parser.add_argument("--n-fine", type=int, default=768,
                        help="Fine rendering resolution per axis.")
    parser.add_argument("--fov", type=float, default=None,
                        help="Half field-of-view in Mpc for the rendered image plane.")
    parser.add_argument("--center-x", type=float, default=0.0,
                        help="Initial source-image center x-position [Mpc].")
    parser.add_argument("--center-y", type=float, default=0.0,
                        help="Initial source-image center y-position [Mpc].")
    parser.add_argument("--width", type=float, default=None,
                        help="Initial source-image width on the reduced source plane [Mpc].")
    parser.add_argument("--angle-deg", type=float, default=0.0,
                        help="Initial source-image rotation angle [deg].")
    parser.add_argument("--snapshot-out", type=str, default=None,
                        help="Save one snapshot and exit instead of opening the interactive window.")
    return parser.parse_args()


class InteractivePictureLensingFigure:
    def __init__(self, d, mapping, image, *, image_path: str, center, width: float,
                 angle_deg: float = 0.0, title: str, note: str | None = None):
        self.d = d
        self.mapping = mapping
        self.image = image
        self.image_prepared = pic.prepare_source_image(image)
        self.image_path = image_path
        self.aspect = float(image.shape[0]) / float(image.shape[1])
        self.drag_active = False
        self._render_interval_ms = 20
        self._angle_step_deg = 0.2
        self._xy_step = float(abs(self.mapping["xf"][1] - self.mapping["xf"][0])) if len(self.mapping["xf"]) > 1 else 1e-3
        self._pending_state = None
        self._last_submitted_state = None
        self._last_render_state = None
        self._render_scheduled = False
        self._worker_request_id = 0
        self._worker_result = None
        self._worker_result_id = -1
        self._worker_stop = False
        self._worker_condition = threading.Condition()

        self.rs = float(d["r_s_Mpc"])
        self.kappa_s = ray.compute_kappa_s(d)
        self.r_E_nfw = ray.einstein_radius_nfw_Mpc(self.rs, self.kappa_s)
        self.theta = np.linspace(0.0, 2.0 * np.pi, 300)

        self.fig, (self.ax_source, self.ax_lensed) = plt.subplots(1, 2, figsize=(15.0, 7.8))
        self.fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.22, wspace=0.16)
        self.fig.suptitle(title, fontsize=13)
        self._render_timer = self.fig.canvas.new_timer(interval=self._render_interval_ms)
        if hasattr(self._render_timer, "single_shot"):
            self._render_timer.single_shot = True
        self._render_timer.add_callback(self._flush_pending_render)
        self._result_timer = self.fig.canvas.new_timer(interval=self._render_interval_ms)
        self._result_timer.add_callback(self._poll_worker_result)
        if note:
            self.fig.text(0.06, 0.02, note, fontsize=9)

        self._build_sliders(center, width, angle_deg)
        self._build_images()
        self._worker_thread = threading.Thread(
            target=self._lensed_worker_loop,
            name="InteractiveLensedWorker",
            daemon=True,
        )
        self._worker_thread.start()
        self._result_timer.start()
        self._connect_events()
        self._request_render(force=True)

    def _build_sliders(self, center, width, angle_deg):
        half_view = float(self.mapping["extent"][1])
        center_lim = 0.95 * half_view
        width0 = float(width)
        width_min = max(1e-3, 0.15 * width0)
        width_max = max(6.0 * width0, 1.6 * half_view)

        self._slider_axes = []
        self._sliders = []

        slider_specs = [
            ("x", "Image x [Mpc]", -center_lim, center_lim, float(center[0])),
            ("y", "Image y [Mpc]", -center_lim, center_lim, float(center[1])),
            ("width", "Width [Mpc]", width_min, width_max, width0),
            ("angle", "Angle [deg]", -180.0, 180.0, float(angle_deg)),
        ]

        y0 = 0.16
        dy = 0.035
        for idx, (key, label, vmin, vmax, vinit) in enumerate(slider_specs):
            ax = self.fig.add_axes([0.16, y0 - idx * dy, 0.66, 0.022])
            slider = Slider(ax, label, vmin, vmax, valinit=vinit)
            slider.on_changed(self._request_render)
            self._slider_axes.append(ax)
            self._sliders.append(slider)
            setattr(self, f"slider_{key}", slider)

        self.ax_reset = self.fig.add_axes([0.85, 0.03, 0.10, 0.05])
        self.button_reset = Button(self.ax_reset, "Reset")
        self.button_reset.on_clicked(self._reset)

    def _build_images(self):
        extent = self.mapping["extent"]
        initial = np.zeros(self.mapping["theta1"].shape + (self.image.shape[2],), dtype=np.float32)

        self.source_im = self.ax_source.imshow(
            self.image_prepared,
            origin="lower",
            extent=[-0.5, 0.5, -0.5, 0.5],
            interpolation="bilinear",
            aspect="equal",
        )
        self.lensed_im = self.ax_lensed.imshow(
            np.swapaxes(initial, 0, 1),
            origin="lower",
            extent=extent,
            interpolation="bilinear",
            aspect="equal",
        )

        self.source_marker, = self.ax_source.plot([0.0], [0.0], "w+", ms=14, mew=2)
        self.lensed_marker, = self.ax_lensed.plot([0.0], [0.0], "w+", ms=10, mew=1.8, alpha=0.8)

        self.ax_source.set_title("Source image on reduced source plane", fontsize=12)
        self.ax_lensed.set_title("Lensed image (ray/plane intersection)", fontsize=12)

        self.ax_source.set_xlabel(r"$\beta_1$ [Mpc]")
        self.ax_source.set_ylabel(r"$\beta_2$ [Mpc]")
        self.ax_lensed.set_xlabel(r"$b_1$ [Mpc]")
        self.ax_lensed.set_ylabel(r"$b_2$ [Mpc]")
        self.ax_source.set_xlim(extent[0], extent[1])
        self.ax_source.set_ylim(extent[2], extent[3])
        self.ax_lensed.set_xlim(extent[0], extent[1])
        self.ax_lensed.set_ylim(extent[2], extent[3])

        for ax in (self.ax_source, self.ax_lensed):
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("gray")
            self._add_reference_circles(ax)
            ax.legend(fontsize=9, loc="upper right",
                      facecolor="black", edgecolor="gray",
                      labelcolor="white", framealpha=0.7)

        self.fig.set_facecolor("black")

    def _add_reference_circles(self, ax):
        if self.r_E_nfw > 0.0:
            ax.plot(self.r_E_nfw * np.cos(self.theta), self.r_E_nfw * np.sin(self.theta),
                    "lime", ls="-", lw=1.2, alpha=0.85,
                    label=r"$r_E^{\rm NFW}$ = %.4f Mpc" % self.r_E_nfw)
        ax.plot(self.rs * np.cos(self.theta), self.rs * np.sin(self.theta),
                "cyan", ls=":", lw=0.8, alpha=0.6,
                label=r"$r_s$ = %.3f Mpc" % self.rs)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _quantize_value(self, value, step, vmin, vmax):
        if step <= 0.0:
            return float(np.clip(value, vmin, vmax))
        snapped = step * round(float(value) / step)
        return float(np.clip(snapped, vmin, vmax))

    def _current_state(self):
        x = self._quantize_value(
            self.slider_x.val, self._xy_step, self.slider_x.valmin, self.slider_x.valmax)
        y = self._quantize_value(
            self.slider_y.val, self._xy_step, self.slider_y.valmin, self.slider_y.valmax)
        width = self._quantize_value(
            self.slider_width.val, self._xy_step, self.slider_width.valmin, self.slider_width.valmax)
        angle_deg = self._quantize_value(
            self.slider_angle.val, self._angle_step_deg, self.slider_angle.valmin, self.slider_angle.valmax)
        height = width * self.aspect
        return (x, y, width, height, angle_deg)

    def _apply_source_transform(self, center, width, height, angle_deg):
        transform = (
            Affine2D()
            .scale(width, height)
            .rotate_deg(angle_deg)
            .translate(center[0], center[1])
            + self.ax_source.transData
        )
        self.source_im.set_transform(transform)

    def _set_image_center(self, x_val, y_val):
        x = float(np.clip(x_val, self.slider_x.valmin, self.slider_x.valmax))
        y = float(np.clip(y_val, self.slider_y.valmin, self.slider_y.valmax))
        self.slider_x.eventson = False
        self.slider_y.eventson = False
        self.slider_x.set_val(x)
        self.slider_y.set_val(y)
        self.slider_x.eventson = True
        self.slider_y.eventson = True
        self._request_render()

    def _request_render(self, _event=None, force=False):
        self._pending_state = self._current_state()
        if force:
            self._flush_pending_render(force_sync=True)
            return
        if self._pending_state == self._last_submitted_state:
            return
        if self._render_scheduled:
            return
        self._render_scheduled = True
        self._render_timer.start()

    def _format_source_title(self, center, width):
        return "Source plane  |  x=%.3f  y=%.3f  width=%.3f Mpc" % (center[0], center[1], width)

    def _format_lensed_title(self, angle_deg, height, pending):
        suffix = "  |  rendering..." if pending else ""
        return "Lensed image  |  angle=%.1f deg  height=%.3f Mpc%s" % (angle_deg, height, suffix)

    def _apply_source_state(self, state, pending):
        x, y, width, height, angle_deg = state
        center = (x, y)
        self._apply_source_transform(center, width, height, angle_deg)
        self.source_marker.set_data([center[0]], [center[1]])
        self.lensed_marker.set_data([center[0]], [center[1]])
        self.ax_source.set_title(
            self._format_source_title(center, width),
            fontsize=12,
            color="white",
        )
        self.ax_lensed.set_title(
            self._format_lensed_title(angle_deg, height, pending=pending),
            fontsize=12,
            color="white",
        )

    def _submit_worker_state(self, state):
        with self._worker_condition:
            self._worker_request_id += 1
            self._pending_worker_state = state
            self._pending_worker_request_id = self._worker_request_id
            self._worker_condition.notify_all()

    def _render_lensed_sync(self, state):
        x, y, width, height, angle_deg = state
        return pic.render_lensed_only_prepared(
            self.mapping,
            self.image_prepared,
            center=(x, y),
            width=width,
            height=height,
            angle_deg=angle_deg,
        )

    def _apply_lensed_result(self, state, lensed):
        x, y, _width, height, angle_deg = state
        center = (x, y)
        self.lensed_im.set_data(np.swapaxes(lensed, 0, 1))
        self.lensed_marker.set_data([center[0]], [center[1]])
        self.ax_lensed.set_title(
            self._format_lensed_title(angle_deg, height, pending=False),
            fontsize=12,
            color="white",
        )
        self._last_render_state = state

    def _flush_pending_render(self, force_sync=False):
        if self._render_scheduled:
            self._render_timer.stop()
            self._render_scheduled = False
        if self._pending_state is None:
            return
        state = self._pending_state
        if not force_sync and state == self._last_submitted_state:
            return

        self._apply_source_state(state, pending=not force_sync)
        self._last_submitted_state = state
        if force_sync:
            lensed = self._render_lensed_sync(state)
            self._apply_lensed_result(state, lensed)
        else:
            self._submit_worker_state(state)
        self.fig.canvas.draw_idle()

    def _lensed_worker_loop(self):
        last_seen_request_id = 0
        while True:
            with self._worker_condition:
                while (not self._worker_stop and
                       getattr(self, "_pending_worker_request_id", 0) == last_seen_request_id):
                    self._worker_condition.wait()
                if self._worker_stop:
                    return
                request_id = self._pending_worker_request_id
                state = self._pending_worker_state

            lensed = self._render_lensed_sync(state)

            with self._worker_condition:
                if self._worker_stop:
                    return
                last_seen_request_id = request_id
                if request_id != self._pending_worker_request_id:
                    continue
                self._worker_result = (request_id, state, lensed)
                self._worker_result_id = request_id

    def _poll_worker_result(self):
        result = None
        with self._worker_condition:
            if self._worker_result is not None:
                request_id, state, lensed = self._worker_result
                if request_id > getattr(self, "_last_applied_result_id", -1):
                    self._last_applied_result_id = request_id
                    result = (state, lensed)

        if result is None:
            return
        state, lensed = result
        self._apply_lensed_result(state, lensed)
        self.fig.canvas.draw_idle()

    def _reset(self, _event):
        for slider in self._sliders:
            slider.reset()
        self._request_render(force=True)

    def _on_press(self, event):
        if event.inaxes != self.ax_source or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.drag_active = True
        self._set_image_center(event.xdata, event.ydata)

    def _on_motion(self, event):
        if not self.drag_active or event.inaxes != self.ax_source:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._set_image_center(event.xdata, event.ydata)

    def _on_release(self, _event):
        self.drag_active = False
        self._request_render()

    def _on_close(self, _event):
        self._render_timer.stop()
        self._result_timer.stop()
        with self._worker_condition:
            self._worker_stop = True
            self._worker_condition.notify_all()

    def save_snapshot(self, path: str, dpi: int = 180):
        self._request_render(force=True)
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="black")

    def show(self):
        plt.show()


def main():
    args = parse_args()
    d, npz_path = load_dataset(args.path)
    image = pic.load_source_image(args.image)
    half_view = default_half_view(d, args.fov)
    summarize_dataset(d, half_view, prefix=f"Loaded {npz_path}")

    raytrace = ray.build_raytrace_map(d)
    max_resid_kpc = 1e3 * float(abs(raytrace["hit_residual_Mpc"]).max())
    print(f"  Raytrace intersection max residual = {max_resid_kpc:.3e} kpc")

    width = float(args.width) if args.width is not None else pic._default_picture_width(d)
    center = (float(args.center_x), float(args.center_y))
    mapping = pic.build_fine_mapping(raytrace, half_view, args.n_fine)

    title = (
        "Interactive lensing | explicit ray/source-plane mapping"
        f" | image={os.path.basename(args.image)}"
        f" | z_l={float(d['z_l']):.3f}, z_s={float(d['z_source']):.3f}"
    )
    note = "Drag the image in the left panel or use the sliders below. Height follows the input aspect ratio."
    app = InteractivePictureLensingFigure(
        d,
        mapping,
        image,
        image_path=args.image,
        center=center,
        width=width,
        angle_deg=float(args.angle_deg),
        title=title,
        note=note,
    )

    if args.snapshot_out:
        app.save_snapshot(args.snapshot_out)
        print(f"  [ok] Snapshot saved to {args.snapshot_out}")
        return

    app.show()


if __name__ == "__main__":
    main()