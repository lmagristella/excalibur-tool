#!/usr/bin/env python3
"""Common helpers for interactive lensing viewers."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

import matplotlib
if os.environ.get("EXCALIBUR_MPL_BACKEND"):
    matplotlib.use(os.environ["EXCALIBUR_MPL_BACKEND"])
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import make_lensed_sky as sachs
import make_lensed_sky_raytracing as ray


@dataclass
class PrecomputedMapping:
    label: str
    beta1: np.ndarray
    beta2: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    xf: np.ndarray
    yf: np.ndarray
    extent: list[float]
    half_view: float


def resolve_npz_path(path: str | None) -> str:
    if path:
        return path
    latest = ray.latest_run("lensing_nfw_analytic") or ray.latest_run("lensing_nfw")
    if latest is None:
        raise FileNotFoundError("No .npz result file found")
    return latest


def load_dataset(path: str | None):
    npz_path = resolve_npz_path(path)
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return data, npz_path


def default_half_view(d, requested: float | None = None) -> float:
    return ray._view_fov(d, requested)


def default_source_params(d, half_view: float):
    rs = float(d["r_s_Mpc"])
    return dict(
        center=(0.0, 0.0),
        R_e=max(0.1 * rs, 0.02 * half_view),
        n=1.0,
        ellip=0.3,
        pa_deg=30.0,
        I0=1.0,
    )


def summarize_dataset(d, half_view: float, *, prefix: str = "Loaded"):
    rs = float(d["r_s_Mpc"])
    print(f"{prefix}: z_l={float(d['z_l']):.4f}, z_s={float(d['z_source']):.4f}")
    print(
        "  M_200=%.1fe15 Msun  c=%.0f  rs=%.4f Mpc  map=%dx%d  half=%.3f Mpc"
        % (
            float(d["M_200_Msun"]) / 1e15,
            float(d["c_NFW"]),
            rs,
            int(d["n_map_1d"]),
            int(d["n_map_1d"]),
            float(d["map_half_Mpc"]),
        )
    )
    print(f"  Interactive FoV = +/- {half_view:.4f} Mpc")


def precompute_raytrace_mapping(d, half_view: float, n_fine: int):
    raytrace = ray.build_raytrace_map(d)
    dummy = default_source_params(d, half_view)
    out = ray.make_lensed_image(d, raytrace, half_view, dummy, Nfine=n_fine)
    t1, t2 = np.meshgrid(out["xf"], out["yf"], indexing="ij")
    mapping = PrecomputedMapping(
        label="Ray tracing",
        beta1=out["beta1"],
        beta2=out["beta2"],
        t1=t1,
        t2=t2,
        xf=out["xf"],
        yf=out["yf"],
        extent=out["extent"],
        half_view=half_view,
    )
    return mapping, raytrace


def precompute_sachs_mapping(d, half_view: float, n_fine: int):
    dummy = default_source_params(d, half_view)
    out = sachs.make_lensed_image(d, half_view, dummy, Nfine=n_fine, mode="simulation")
    xf = out["xf"]
    yf = xf.copy()
    t1, t2 = np.meshgrid(xf, yf, indexing="ij")
    mapping = PrecomputedMapping(
        label="Sachs / D",
        beta1=t1 - out["alpha1"],
        beta2=t2 - out["alpha2"],
        t1=t1,
        t2=t2,
        xf=xf,
        yf=yf,
        extent=out["extent"],
        half_view=half_view,
    )
    return mapping


def _stretch_image(image: np.ndarray, vmax: float):
    return np.sqrt(np.clip(image / max(vmax, 1e-12), 0.0, 1.0))


def _render_source(mapping: PrecomputedMapping, source_kw):
    return ray.sersic_source(mapping.t1, mapping.t2, **source_kw)


def _render_lensed(mapping: PrecomputedMapping, source_kw):
    return ray.sersic_source(mapping.beta1, mapping.beta2, **source_kw)


class InteractiveLensingFigure:
    def __init__(self, d, mappings: list[PrecomputedMapping], *, title: str,
                 source_kw: dict, note: str | None = None):
        if not mappings:
            raise ValueError("At least one mapping is required")

        self.d = d
        self.mappings = mappings
        self.rs = float(d["r_s_Mpc"])
        self.kappa_s = ray.compute_kappa_s(d)
        self.r_E_nfw = ray.einstein_radius_nfw_Mpc(self.rs, self.kappa_s)
        self.theta = np.linspace(0.0, 2.0 * np.pi, 300)
        self.drag_active = False

        self.fig, axes = plt.subplots(1, 1 + len(mappings), figsize=(6.2 * (1 + len(mappings)), 7.8))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        self.axes = axes
        self.ax_source = axes[0]
        self.ax_lensed = list(axes[1:])
        self.fig.subplots_adjust(left=0.055, right=0.985, top=0.9, bottom=0.30, wspace=0.16)
        self.fig.suptitle(title, fontsize=13)
        if note:
            self.fig.text(0.055, 0.02, note, fontsize=9)

        self._build_sliders(source_kw)
        self._build_images()
        self._connect_events()
        self._update_images()

    def _build_sliders(self, source_kw):
        half_view = self.mappings[0].half_view
        center_lim = 0.95 * half_view
        re_min = max(1e-3, 0.01 * self.rs)
        re_max = max(0.30 * half_view, 0.8 * self.rs)

        self._slider_axes = []
        self._sliders = []

        slider_specs = [
            ("x", "Source x [Mpc]", -center_lim, center_lim, float(source_kw["center"][0])),
            ("y", "Source y [Mpc]", -center_lim, center_lim, float(source_kw["center"][1])),
            ("R_e", "R_e [Mpc]", re_min, re_max, float(source_kw["R_e"])),
            ("n", "Sersic n", 0.5, 6.0, float(source_kw["n"])),
            ("ellip", "Ellipticity", 0.0, 0.85, float(source_kw["ellip"])),
            ("pa", "PA [deg]", 0.0, 180.0, float(source_kw["pa_deg"])),
            ("I0", "I0", 0.1, 2.0, float(source_kw["I0"])),
        ]

        y0 = 0.235
        dy = 0.028
        for idx, (key, label, vmin, vmax, vinit) in enumerate(slider_specs):
            ax = self.fig.add_axes([0.16, y0 - idx * dy, 0.66, 0.018])
            slider = Slider(ax, label, vmin, vmax, valinit=vinit)
            slider.on_changed(self._update_images)
            self._slider_axes.append(ax)
            self._sliders.append(slider)
            setattr(self, f"slider_{key}", slider)

        self.ax_reset = self.fig.add_axes([0.85, 0.05, 0.10, 0.045])
        self.button_reset = Button(self.ax_reset, "Reset")
        self.button_reset.on_clicked(self._reset)

    def _build_images(self):
        extent = self.mappings[0].extent
        initial = np.zeros_like(self.mappings[0].t1)

        self.source_im = self.ax_source.imshow(
            initial.T, origin="lower", extent=extent, cmap="magma",
            vmin=0.0, vmax=1.0, aspect="equal"
        )
        self.source_marker, = self.ax_source.plot([0.0], [0.0], "w+", ms=14, mew=2)
        self.ax_source.set_title("Source plane", fontsize=12)
        self.ax_source.set_xlabel(r"$\beta_1$ [Mpc]")
        self.ax_source.set_ylabel(r"$\beta_2$ [Mpc]")

        self.lensed_ims = []
        self.lensed_markers = []
        for ax, mapping in zip(self.ax_lensed, self.mappings):
            im = ax.imshow(
                initial.T, origin="lower", extent=extent, cmap="magma",
                vmin=0.0, vmax=1.0, aspect="equal"
            )
            self.lensed_ims.append(im)
            marker, = ax.plot([0.0], [0.0], "w+", ms=8, mew=1.6, alpha=0.7)
            self.lensed_markers.append(marker)
            ax.set_title(mapping.label, fontsize=12)
            ax.set_xlabel(r"$b_1$ [Mpc]")
            ax.set_ylabel(r"$b_2$ [Mpc]")
            self._add_reference_circles(ax)

    def _add_reference_circles(self, ax):
        if self.r_E_nfw > 0.0:
            ax.plot(self.r_E_nfw * np.cos(self.theta), self.r_E_nfw * np.sin(self.theta),
                    "lime", ls="-", lw=1.1, alpha=0.85)
        ax.plot(self.rs * np.cos(self.theta), self.rs * np.sin(self.theta),
                "cyan", ls=":", lw=0.8, alpha=0.7)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    def _current_source_kw(self):
        return dict(
            center=(self.slider_x.val, self.slider_y.val),
            R_e=self.slider_R_e.val,
            n=self.slider_n.val,
            ellip=self.slider_ellip.val,
            pa_deg=self.slider_pa.val,
            I0=self.slider_I0.val,
        )

    def _update_images(self, _event=None):
        source_kw = self._current_source_kw()
        source = _render_source(self.mappings[0], source_kw)
        lensed_images = [_render_lensed(mapping, source_kw) for mapping in self.mappings]

        vmax = max(source.max(), *(img.max() for img in lensed_images), 1e-10)
        self.source_im.set_data(_stretch_image(source, vmax).T)
        self.source_marker.set_data([source_kw["center"][0]], [source_kw["center"][1]])
        self.ax_source.set_title(
            "Source plane  |  x=%.3f, y=%.3f" % source_kw["center"],
            fontsize=12,
        )

        for im, marker, ax, mapping, image in zip(
            self.lensed_ims, self.lensed_markers, self.ax_lensed, self.mappings, lensed_images
        ):
            im.set_data(_stretch_image(image, vmax).T)
            marker.set_data([source_kw["center"][0]], [source_kw["center"][1]])
            ax.set_title(mapping.label, fontsize=12)

        self.fig.canvas.draw_idle()

    def _reset(self, _event):
        for slider in self._sliders:
            slider.reset()

    def _set_source_center(self, x_val, y_val):
        x = float(np.clip(x_val, self.slider_x.valmin, self.slider_x.valmax))
        y = float(np.clip(y_val, self.slider_y.valmin, self.slider_y.valmax))
        self.slider_x.set_val(x)
        self.slider_y.set_val(y)

    def _on_press(self, event):
        if event.inaxes != self.ax_source or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.drag_active = True
        self._set_source_center(event.xdata, event.ydata)

    def _on_motion(self, event):
        if not self.drag_active or event.inaxes != self.ax_source:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._set_source_center(event.xdata, event.ydata)

    def _on_release(self, _event):
        self.drag_active = False

    def save_snapshot(self, path: str, dpi: int = 180):
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")

    def show(self):
        plt.show()
