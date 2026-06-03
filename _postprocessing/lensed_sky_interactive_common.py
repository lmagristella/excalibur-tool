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
from matplotlib import animation
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from scipy.interpolate import RectBivariateSpline


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


@dataclass
class LensProfile:
    """One selectable lens model: a label, its dataset, and its raytrace mapping.

    All profiles shown together must be precomputed on the *same* half_view and
    n_fine so they share the image/source-plane grid (the viewer just swaps the
    beta field when you switch profile)."""
    label: str
    d: object
    mapping: PrecomputedMapping


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
    def __init__(self, profiles, *, title: str | None,
                 source_kw: dict, note: str | None = None, show_controls: bool = True,
                 show_source_marker: bool = True):
        # Backward compatible: allow (d, [mapping]) by wrapping into one profile.
        if not isinstance(profiles, (list, tuple)) or (
            profiles and not isinstance(profiles[0], LensProfile)
        ):
            raise TypeError(
                "InteractiveLensingFigure expects a list of LensProfile; "
                "use make_profile(d, mapping, label) to build them."
            )
        if not profiles:
            raise ValueError("At least one lens profile is required")

        self.profiles = list(profiles)
        self.active_idx = 0
        self.d = self.profiles[0].d
        self.mappings = [self.profiles[0].mapping]
        self._load_active_params()
        self.theta = np.linspace(0.0, 2.0 * np.pi, 300)
        self.drag_active = False
        self._animation = None
        self.show_controls = show_controls
        self.show_source_marker = show_source_marker

        self.fig, axes = plt.subplots(1, 2, figsize=(12.4, 7.8))
        self.axes = axes
        self.ax_source = axes[0]
        self.ax_lensed = [axes[1]]
        bottom = 0.30 if show_controls else 0.08
        top = 0.9 if title else 0.97
        self.fig.subplots_adjust(left=0.055, right=0.985, top=top, bottom=bottom, wspace=0.16)
        if title:
            self.fig.suptitle(title, fontsize=13)
        if note:
            note_y = 0.02 if show_controls else 0.012
            self.fig.text(0.055, note_y, note, fontsize=9)

        # Attribution centered along the bottom of the window.
        self.fig.text(
            0.5, 0.006, "EXCALIBUR  ·  Laurent Magri-Stella",
            fontsize=11, color="0.30", alpha=0.92, ha="center", va="bottom",
            style="italic", weight="semibold",
        )

        self._build_sliders(source_kw)
        self._build_images()
        self._build_overlays()
        self._build_profile_selector()
        if not self.show_controls:
            self._set_controls_visible(False)
        self._connect_events()
        self._update_images()

    def _load_active_params(self):
        """(Re)load halo-derived quantities for the active profile's dataset."""
        d = self.d
        self.rs = float(d["r_s_Mpc"])
        self.kappa_s = ray.compute_kappa_s(d)
        self.r_E_nfw = ray.einstein_radius_nfw_Mpc(self.rs, self.kappa_s)

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
            ax = self.fig.add_axes([0.16, y0 - idx * dy, 0.58, 0.018])
            slider = Slider(ax, label, vmin, vmax, valinit=vinit)
            slider.on_changed(self._update_images)
            self._slider_axes.append(ax)
            self._sliders.append(slider)
            setattr(self, f"slider_{key}", slider)

        self.ax_reset = self.fig.add_axes([0.86, 0.018, 0.085, 0.036])
        self.button_reset = Button(self.ax_reset, "Reset")
        self.button_reset.on_clicked(self._reset)

    def _set_controls_visible(self, visible: bool):
        for ax in self._slider_axes:
            ax.set_visible(visible)
        self.ax_reset.set_visible(visible)
        self.ax_checks.set_visible(visible)
        if getattr(self, "ax_profile", None) is not None:
            self.ax_profile.set_visible(visible)

    def _build_images(self):
        extent = self.mappings[0].extent
        initial = np.zeros_like(self.mappings[0].t1)

        self.source_im = self.ax_source.imshow(
            initial.T, origin="lower", extent=extent, cmap="magma",
            vmin=0.0, vmax=1.0, aspect="equal"
        )
        self.source_marker, = self.ax_source.plot([0.0], [0.0], "w+", ms=14, mew=2)
        self.source_marker.set_visible(self.show_source_marker)
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
            ax.set_title(self.profiles[self.active_idx].label, fontsize=12)
            ax.set_xlabel(r"$b_1$ [Mpc]")
            ax.set_ylabel(r"$b_2$ [Mpc]")

    # ----- physical-quantity overlays + toggle checkboxes -----------------

    def _compute_critical_caustic(self):
        """Critical curve (det J = 0, image plane) and its caustic (source plane)."""
        mp = self.mappings[0]
        xf, yf = np.asarray(mp.xf), np.asarray(mp.yf)
        if xf.size < 3 or yf.size < 3:
            return [], []
        dx = float(xf[1] - xf[0])
        dy = float(yf[1] - yf[0])
        b1, b2 = mp.beta1, mp.beta2
        db1_dt1 = np.gradient(b1, dx, axis=0)
        db1_dt2 = np.gradient(b1, dy, axis=1)
        db2_dt1 = np.gradient(b2, dx, axis=0)
        db2_dt2 = np.gradient(b2, dy, axis=1)
        detJ = db1_dt1 * db2_dt2 - db1_dt2 * db2_dt1

        # Trim a small border: np.gradient uses one-sided differences at the
        # edges and the raytrace map flattens there, producing spurious det J
        # sign flips (stray caustic spikes at the FoV boundary). Drop them.
        m = max(2, xf.size // 100)
        if xf.size > 2 * m + 2 and yf.size > 2 * m + 2:
            detJ = detJ[m:-m, m:-m]
            xf_c, yf_c = xf[m:-m], yf[m:-m]
        else:
            xf_c, yf_c = xf, yf
        if not (detJ.min() < 0.0 < detJ.max()):
            return [], []  # no sign change -> no critical curve inside the FoV

        t1, t2 = np.meshgrid(xf_c, yf_c, indexing="ij")
        tmp = Figure()
        cs = tmp.subplots().contour(t1, t2, detJ, levels=[0.0])
        crit = []
        for path in cs.get_paths():
            crit.extend(s for s in path.to_polygons(closed_only=False) if len(s) >= 2)

        # Physical critical curves sit near the Einstein radius, never out at the
        # edge of the field. Spline overshoot in the high-magnification core can
        # spawn spurious zero-det segments at large image radius; drop those so
        # only the real (tangential + radial) critical curves remain.
        r_keep = 0.6 * float(self.mappings[0].half_view)
        crit = [s for s in crit if np.hypot(s[:, 0], s[:, 1]).max() <= r_keep]

        sp1 = RectBivariateSpline(xf, yf, b1)
        sp2 = RectBivariateSpline(xf, yf, b2)
        caustic = []
        for seg in crit:
            bx = sp1.ev(seg[:, 0], seg[:, 1])
            by = sp2.ev(seg[:, 0], seg[:, 1])
            caustic.append(np.column_stack([bx, by]))
        return crit, caustic

    _EINSTEIN_LABEL = r"$\theta_E$ (Einstein)"

    def _build_overlays(self):
        # Create persistent artists once (radii / segments filled per profile by
        # _rebuild_shape_overlays). Circles are reused across profile switches;
        # critical/caustic curves are rebuilt (their segment count changes).
        self._overlays = {}
        self._overlays["r_s (source)"] = self.ax_source.plot(
            [], [], color="cyan", ls=":", lw=0.9, alpha=0.75)
        rs_lens, einstein = [], []
        for ax in self.ax_lensed:
            rs_lens += ax.plot([], [], color="cyan", ls=":", lw=0.9, alpha=0.75)
            einstein += ax.plot([], [], color="lime", ls="-", lw=1.1, alpha=0.85)
        self._overlays["r_s (lentille)"] = rs_lens
        self._overlays[self._EINSTEIN_LABEL] = einstein
        self._overlays["courbe critique"] = []
        self._overlays["caustique"] = []

        self._overlay_defaults = {
            "r_s (source)": False,
            "r_s (lentille)": True,
            self._EINSTEIN_LABEL: True,
            "courbe critique": False,
            "caustique": False,
        }
        self._check_labels = list(self._overlay_defaults.keys())

        self.ax_checks = self.fig.add_axes([0.80, 0.065, 0.185, 0.090])
        self.ax_checks.set_facecolor("none")
        self.check = CheckButtons(
            self.ax_checks, self._check_labels,
            [self._overlay_defaults[l] for l in self._check_labels],
        )
        for txt in self.check.labels:
            txt.set_fontsize(7)
        self.check.on_clicked(self._toggle_overlay)

        self._rebuild_shape_overlays()

    def _set_circle(self, artists, radius):
        cos, sin = np.cos(self.theta), np.sin(self.theta)
        for art in artists:
            if radius and radius > 0.0:
                art.set_data(radius * cos, radius * sin)
            else:
                art.set_data([], [])

    def _rebuild_shape_overlays(self):
        """Refresh radius-dependent circles and recompute critical/caustic for
        the active profile, then re-apply the checkbox visibility."""
        self._set_circle(self._overlays["r_s (source)"], self.rs)
        self._set_circle(self._overlays["r_s (lentille)"], self.rs)
        self._set_circle(self._overlays[self._EINSTEIN_LABEL], self.r_E_nfw)

        for key in ("courbe critique", "caustique"):
            for art in self._overlays.get(key, []):
                art.remove()

        crit_segs, caustic_segs = self._compute_critical_caustic()
        crit = []
        for ax in self.ax_lensed:
            for seg in crit_segs:
                crit += ax.plot(seg[:, 0], seg[:, 1], color="yellow", lw=1.3, alpha=0.9)
        caustic = []
        for seg in caustic_segs:
            caustic += self.ax_source.plot(seg[:, 0], seg[:, 1],
                                           color="orange", lw=1.3, alpha=0.9)
        self._overlays["courbe critique"] = crit
        self._overlays["caustique"] = caustic
        self._apply_overlay_visibility()

    def _apply_overlay_visibility(self):
        status = dict(zip(self._check_labels, self.check.get_status()))
        for label, artists in self._overlays.items():
            vis = status.get(label, self._overlay_defaults.get(label, True))
            for art in artists:
                art.set_visible(vis)

    def _toggle_overlay(self, label):
        status = dict(zip(self._check_labels, self.check.get_status()))
        for art in self._overlays.get(label, []):
            art.set_visible(status[label])
        self.fig.canvas.draw_idle()

    # ----- lens-profile selector (spherical / elliptical / ...) -----------

    def _build_profile_selector(self):
        self.ax_profile = None
        if len(self.profiles) < 2:
            return
        self.ax_profile = self.fig.add_axes([0.815, 0.165, 0.175, 0.115])
        self.ax_profile.set_facecolor("none")
        self.ax_profile.set_title("Lens profile", fontsize=8, loc="left", pad=2)
        labels = [p.label for p in self.profiles]
        self.radio_profile = RadioButtons(self.ax_profile, labels, active=self.active_idx)
        for txt in self.radio_profile.labels:
            txt.set_fontsize(7)
        self.radio_profile.on_clicked(self._activate_profile)

    def _activate_profile(self, label):
        idx = next((i for i, p in enumerate(self.profiles) if p.label == label), None)
        if idx is None or idx == self.active_idx:
            return
        self.active_idx = idx
        self.d = self.profiles[idx].d
        self.mappings = [self.profiles[idx].mapping]
        self._load_active_params()
        self.ax_lensed[0].set_title(label, fontsize=12)
        self._rebuild_shape_overlays()
        self._update_images()
        self.fig.canvas.draw_idle()

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

        # Normalise against the I0=1 brightness so the I0 slider behaves as a
        # real exposure control. Computing vmax from the I0-scaled data instead
        # would cancel the factor out exactly (source and images scale alike),
        # which is why I0 used to have no visible effect.
        i0 = max(source_kw["I0"], 1e-12)
        vmax = max(source.max() / i0, *(img.max() / i0 for img in lensed_images), 1e-10)
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
            ax.set_title(self.profiles[self.active_idx].label, fontsize=12)

        self.fig.canvas.draw_idle()

    def _reset(self, _event):
        for slider in self._sliders:
            slider.reset()

    def set_source_center(self, x_val, y_val):
        x = float(np.clip(x_val, self.slider_x.valmin, self.slider_x.valmax))
        y = float(np.clip(y_val, self.slider_y.valmin, self.slider_y.valmax))
        sliders = (self.slider_x, self.slider_y)
        event_states = [slider.eventson for slider in sliders]
        try:
            for slider in sliders:
                slider.eventson = False
            self.slider_x.set_val(x)
            self.slider_y.set_val(y)
        finally:
            for slider, state in zip(sliders, event_states):
                slider.eventson = state
        self._update_images()

    def _set_source_center(self, x_val, y_val):
        self.set_source_center(x_val, y_val)

    def _build_strong_to_weak_track(self, frames: int, orbit_scale: float):
        if frames < 2:
            raise ValueError("Animation requires at least 2 frames")

        half_view = float(self.mappings[0].half_view)
        center_lim = 0.92 * min(
            abs(self.slider_x.valmin),
            abs(self.slider_x.valmax),
            abs(self.slider_y.valmin),
            abs(self.slider_y.valmax),
        )
        progress = np.linspace(0.0, 1.0, frames)
        eased = progress * progress * (3.0 - 2.0 * progress)
        radial_dir = np.array([np.cos(np.deg2rad(28.0)), np.sin(np.deg2rad(28.0))])
        tangent_dir = np.array([-radial_dir[1], radial_dir[0]])

        inner_radius = max(0.05 * max(self.r_E_nfw, self.slider_R_e.val), 0.012 * half_view)
        outer_radius = min(
            center_lim,
            max(orbit_scale * 2.9 * max(self.r_E_nfw, self.slider_R_e.val), orbit_scale * 0.58 * half_view),
        )
        radius = inner_radius + (outer_radius - inner_radius) * eased
        sway_amp = min(0.12 * max(self.r_E_nfw, self.slider_R_e.val), 0.03 * half_view)
        sway = sway_amp * (1.0 - eased) * np.sin(np.pi * eased)
        return radius[:, None] * radial_dir[None, :] + sway[:, None] * tangent_dir[None, :]

    def _build_animation_track(self, frames: int, orbit_scale: float, track_mode: str):
        if track_mode == "loop":
            return self._build_demo_track(frames, orbit_scale)
        if track_mode == "strong-to-weak":
            return self._build_strong_to_weak_track(frames, orbit_scale)
        raise ValueError(f"Unknown animation track mode: {track_mode}")

    def _build_demo_track(self, frames: int, orbit_scale: float):
        if frames < 2:
            raise ValueError("Animation requires at least 2 frames")

        phase = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
        raw_x = np.sin(phase) + 0.22 * np.sin(3.0 * phase + 0.45)
        raw_y = 0.82 * np.cos(2.0 * phase - 0.35) - 0.18 * np.sin(5.0 * phase + 0.15)
        swell = 0.18 + 0.64 * (0.5 + 0.5 * np.sin(phase - 0.8))
        coords = np.column_stack([raw_x * swell, raw_y * swell])
        coords /= max(float(np.linalg.norm(coords, axis=1).max()), 1e-12)

        half_view = float(self.mappings[0].half_view)
        center_lim = 0.92 * min(
            abs(self.slider_x.valmin),
            abs(self.slider_x.valmax),
            abs(self.slider_y.valmin),
            abs(self.slider_y.valmax),
        )
        lens_scale = max(1.0 * self.r_E_nfw, 3.5 * self.slider_R_e.val, 0.10 * half_view)
        radius = min(center_lim, max(orbit_scale * lens_scale, 0.05 * half_view))
        return radius * coords

    def stop_demo_animation(self):
        if self._animation is not None and self._animation.event_source is not None:
            self._animation.event_source.stop()
        self._animation = None

    def start_demo_animation(
        self,
        *,
        frames: int = 180,
        fps: int = 24,
        orbit_scale: float = 1.0,
        track_mode: str = "loop",
        repeat: bool = True,
    ):
        self.stop_demo_animation()
        track = self._build_animation_track(frames, orbit_scale, track_mode)
        interval_ms = max(1, int(round(1000.0 / max(fps, 1))))

        def _update(frame_idx):
            x_val, y_val = track[frame_idx % len(track)]
            self.set_source_center(x_val, y_val)
            return [self.source_im, self.source_marker, *self.lensed_ims, *self.lensed_markers]

        self._animation = animation.FuncAnimation(
            self.fig,
            _update,
            frames=len(track),
            interval=interval_ms,
            blit=False,
            repeat=repeat,
        )
        return self._animation

    def save_demo_animation(
        self,
        path: str,
        *,
        frames: int = 180,
        fps: int = 24,
        orbit_scale: float = 1.0,
        track_mode: str = "loop",
        dpi: int = 140,
    ):
        ext = os.path.splitext(path)[1].lower()
        restore_ffmpeg_path = None
        if ext in {".gif", ".webp"}:
            writer = animation.PillowWriter(fps=fps)
        else:
            if not animation.writers.is_available("ffmpeg"):
                try:
                    import imageio_ffmpeg
                except ImportError as exc:
                    raise RuntimeError(
                        "FFmpeg is required to save video formats; install ffmpeg or `pip install imageio-ffmpeg`."
                    ) from exc
                restore_ffmpeg_path = matplotlib.rcParams["animation.ffmpeg_path"]
                matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            writer = animation.FFMpegWriter(fps=fps)

        anim = self.start_demo_animation(
            frames=frames,
            fps=fps,
            orbit_scale=orbit_scale,
            track_mode=track_mode,
            repeat=False,
        )
        try:
            anim.save(path, writer=writer, dpi=dpi)
        finally:
            self.stop_demo_animation()
            if restore_ffmpeg_path is not None:
                matplotlib.rcParams["animation.ffmpeg_path"] = restore_ffmpeg_path

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
