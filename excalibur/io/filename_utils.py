"""
Centralised file-naming, directory management and parameter encoding
for EXCALIBUR simulation outputs.

Two naming systems coexist:

1. **RunNamer** (new)  --  for lensing / Sachs-equation simulations.
   Encodes physical parameters directly into filenames so that every
   ``.npz`` / ``.png`` carries a self-describing human-readable tag.
   Parameters can be **parsed back** from an existing filename.

2. **generate_trajectory_filename / parse_trajectory_filename** (legacy)  -- 
   for old-style geodesic trajectory files (``excalibur_run_*_Mpc.h5``).
   Kept for backward compatibility.

Naming convention
-----------------
::

    {run_type}_{integrator}[_dt{val}]_{metric}_{profile}[_M{val}][_c{val}]
    [_Rvir{val}][_rs{val}][_zl{val}][_zs{val}][_Dl{val}][_Ds{val}]
    [_obs{x}_{y}_{z}][_box{val}Mpc][_N{val}][_AMR{val}][_Nph{val}]
    _{suffix}.{ext}

Quick-start
-----------
::

    from excalibur.io import RunNamer

    # ---- In a simulation script ----
    namer = RunNamer("lensing_nfw_amr",
                     integrator="rk4", metric="FLRWP1", profile="nfw",
                     M=1e15, c_NFW=5, zl=0.022, zs=0.044,
                     Dl=95.0, Ds=190.0, obs=(100, 100, 5),
                     box_Mpc=200, N=256, AMR=5, Nph=1027)
    np.savez(namer.npz(), ...)

    # ---- In a postprocessing script ----
    namer = RunNamer.from_npz("_data/output/lensing_nfw_amr_rk4_..._results.npz")
    namer.params["M"]       # --> 1e+15
    namer.params["c_NFW"]   # --> 5.0
    namer.params["zl"]      # --> 0.022
    namer.integrator        # --> "rk4"
    namer.metric            # --> "FLRWP1"

    fig.savefig(namer.plot("profiles"))
    fig.savefig(namer.plot("maps"))

    # ---- Discovery ----
    from excalibur.io import latest_run
    latest_run("lensing_nfw_amr")        # most recent .npz
"""

from __future__ import annotations

import os
import re
import glob
from datetime import datetime
from collections import OrderedDict
from typing import Any

import numpy as np


# ------------------------------------------------------------------
#  Project root & default output directory
# ------------------------------------------------------------------
def _project_root():
    """Walk up from this file to find the directory containing ``_data/``."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "_data")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_ROOT = _project_root()
_OUTPUT_DIR = os.path.join(_ROOT, "_data", "output")


# ===================================================================
#  Parameter formatting / parsing helpers
# ===================================================================
# Ordered list of *known* parameter keys and their filename prefix.
# Order matters: it defines the canonical token order in the stem.
# Each entry:  (param_key,  filename_prefix,  human_label, format_fn)
_PARAM_SPEC: list[tuple[str, str, str]] = [
    # -- Mass / profile ---
    ("M",        "M",      "Mass [Msun]"),
    ("c_NFW",    "c",      "NFW concentration"),
    ("Rvir",     "Rvir",   "Virial radius [Mpc]"),
    ("rs",       "rs",     "Scale radius [Mpc]"),
    # -- Distances / redshifts ---
    ("zl",       "zl",     "Lens redshift"),
    ("zs",       "zs",     "Source redshift"),
    ("Dl",       "Dl",     "D_l [Mpc]"),
    ("Ds",       "Ds",     "D_s [Mpc]"),
    # -- Observer position ---
    ("obs",      "obs",    "Observer [Mpc]"),      # tuple (x, y, z)
    # -- Grid ---
    ("box_Mpc",  "box",    "Box size [Mpc]"),
    ("N",        "N",      "Grid cells per side"),
    ("AMR",      "AMR",    "AMR levels"),           # 0 or absent --> regular
    # -- Photons ---
    ("Nph",      "Nph",    "Number of photons"),
    ("Nshape",   "Nshape", "Number of compared shapes"),
    ("ref",      "ref",    "Reference profile"),
    # -- Time step (only for non-adaptive) ---
    ("dt",       "dt",     "Time step [s]"),
]

# Fast lookup: filename_prefix --> param_key
_PREFIX_TO_KEY = {prefix: key for key, prefix, _ in _PARAM_SPEC}
_KEY_TO_PREFIX = {key: prefix for key, prefix, _ in _PARAM_SPEC}
_KEY_TO_LABEL  = {key: label for key, _, label in _PARAM_SPEC}

# Known integrators and metrics (for parsing)
_KNOWN_INTEGRATORS = frozenset({
    "rk4", "rk45", "rkf45", "dopri5", "dopri54", "dop853", "euler", "leapfrog4", "dopri", "rkck",
})
_KNOWN_METRICS = frozenset({
    "FLRWP1", "schwarzschild", "kerr", "minkowski",
    "flrwp1",  # allow lowercase too
})
_KNOWN_PROFILES = frozenset({
    "nfw", "sis", "sphere", "point", "hernquist", "einasto",
    "simulation",  # for future N-body based runs
})
_KNOWN_SUFFIXES = frozenset({
    "results", "profiles", "ratio", "maps", "distance",
    "map", "shear", "convergence", "magnification", "log",
})

# Known run-type prefixes (longest first for greedy matching)
_KNOWN_TYPES = sorted([
    "lensing_mass_shape",
    "lensing_nfw_amr",
    "lensing_nfw_cosmo",
    "lensing_nfw",
    "lensing_cone",
    "lensing_sim",
    "geodesic",
    "flrw_test",
], key=len, reverse=True)


def _fmt_val(v: Any) -> str:
    """Format a single value for embedding in a filename token."""
    if isinstance(v, (tuple, list, np.ndarray)):
        # Observer position: (100, 100, 5) --> "100_100_5"
        return "_".join(_fmt_val(x) for x in v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if f == 0.0:
            return "0"
        # Use scientific for large / tiny, compact otherwise
        af = abs(f)
        if af >= 1e6 or af < 0.001:
            return f"{f:.1e}".replace("+", "")
        # Strip trailing zeros
        s = f"{f:.6g}"
        return s
    return str(v)


def _parse_val(s: str) -> int | float | str:
    """Parse a string value back to int or float."""
    try:
        f = float(s)
        if "." not in s and "e" not in s.lower():
            return int(s)
        return f
    except ValueError:
        return s


def _parse_obs_token(s: str) -> tuple[float, ...]:
    """Parse 'obs100_100_5' --> (100.0, 100.0, 5.0)."""
    parts = s.split("_")
    return tuple(_parse_val(p) for p in parts)


# ===================================================================
#  RunNamer
# ===================================================================
class RunNamer:
    """
    Generate and parse self-describing filenames for EXCALIBUR outputs.

    Every physical parameter supplied becomes part of the filename stem
    and can be recovered by :meth:`from_npz` / :meth:`from_path`.

    Parameters
    ----------
    run_type : str
        Simulation family: ``"lensing_nfw"``, ``"lensing_nfw_amr"``, etc.
    integrator : str
        ``"rk4"``, ``"rk45"``, ``"rkf45"``, ``"leapfrog4"``, ...
    metric : str or None
        ``"FLRWP1"`` (perturbed FLRW Psi=Phi), ``"schwarzschild"``, ...
    profile : str or None
        Mass profile: ``"nfw"``, ``"sis"``, ``"sphere"``, ``"simulation"``.
    subdir : str or None
        Sub-directory under ``_data/output/`` for parameter sweeps.
    output_dir : str or None
        Override the entire output directory.
    timestamp : bool
        Append ``_YYYYMMDD_HHMMSS`` to avoid overwriting.  Default False.
    **params
        Physical parameters  --  any of:
        ``M``, ``c_NFW``, ``Rvir``, ``rs``, ``zl``, ``zs``,
        ``Dl``, ``Ds``, ``obs``, ``box_Mpc``, ``N``, ``AMR``,
        ``Nph``, ``dt``.
        Unknown keys are accepted too (appended at the end of the stem).
    """

    def __init__(
        self,
        run_type: str,
        integrator: str = "rk4",
        metric: str | None = None,
        profile: str | None = None,
        subdir: str | None = None,
        output_dir: str | None = None,
        timestamp: bool = False,
        **params,
    ):
        self.run_type = run_type
        self.integrator = integrator
        self.metric = metric
        self.profile = profile
        self._timestamp = timestamp
        # Store params in canonical order (known keys first, extras last)
        self.params: OrderedDict[str, Any] = OrderedDict()
        known_order = [key for key, _, _ in _PARAM_SPEC]
        for key in known_order:
            if key in params:
                self.params[key] = params.pop(key)
        # Remaining (unknown) keys  --  alphabetical
        for key in sorted(params):
            self.params[key] = params[key]

        if output_dir is not None:
            self._outdir = output_dir
        elif subdir is not None:
            self._outdir = os.path.join(_OUTPUT_DIR, subdir)
        else:
            self._outdir = _OUTPUT_DIR

    # ------------------------------------------------------------------
    #  Stem
    # ------------------------------------------------------------------
    @property
    def stem(self) -> str:
        """
        Build the canonical filename stem.

        Structure::

            {run_type}_{integrator}[_dt{val}]_{metric}_{profile}_{param tokens}
        """
        parts: list[str] = [self.run_type, self.integrator]

        if self.metric:
            parts.append(self.metric)
        if self.profile:
            parts.append(self.profile)

        # Parameter tokens
        for key, val in self.params.items():
            prefix = _KEY_TO_PREFIX.get(key, key)
            parts.append(f"{prefix}{_fmt_val(val)}")

        if self._timestamp:
            parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))

        return "_".join(parts)

    @property
    def outdir(self) -> str:
        """Output directory (created lazily on first access)."""
        os.makedirs(self._outdir, exist_ok=True)
        return self._outdir

    # ------------------------------------------------------------------
    #  Path generators
    # ------------------------------------------------------------------
    def npz(self, suffix: str = "results") -> str:
        """``_data/output/{stem}_{suffix}.npz``"""
        return os.path.join(self.outdir, f"{self.stem}_{suffix}.npz")

    def plot(self, name: str, ext: str = ".png") -> str:
        """``_data/output/{stem}_{name}.png``"""
        return os.path.join(self.outdir, f"{self.stem}_{name}{ext}")

    def log(self) -> str:
        """``_data/output/{stem}.log``"""
        return os.path.join(self.outdir, f"{self.stem}.log")

    def path(self, filename: str) -> str:
        """Arbitrary file inside the output directory."""
        return os.path.join(self.outdir, filename)

    # ------------------------------------------------------------------
    #  Convenient param accessors (with defaults)
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Get a parameter value, with optional default."""
        return self.params.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.params[key]

    def __contains__(self, key: str) -> bool:
        return key in self.params

    # ------------------------------------------------------------------
    #  Parsing: reconstruct from an existing filename
    # ------------------------------------------------------------------
    @classmethod
    def from_path(cls, path: str) -> "RunNamer":
        """
        Parse an existing output filename and reconstruct a RunNamer
        with all encoded parameters.

        >>> n = RunNamer.from_path(".../lensing_nfw_amr_rk4_FLRWP1_nfw_M1.0e15_c5_zl0.022_results.npz")
        >>> n.integrator
        'rk4'
        >>> n.metric
        'FLRWP1'
        >>> n.params['M']
        1000000000000000.0
        >>> n.params['c_NFW']
        5.0
        """
        path = os.path.abspath(path)
        outdir = os.path.dirname(path)
        basename = os.path.basename(path)

        # Strip final suffix + extension
        stem = basename
        for sfx in ("_results.npz", ".npz", ".png", ".log"):
            if stem.endswith(sfx):
                stem = stem[: -len(sfx)]
                break

        # Remove trailing known suffix (e.g. "_profiles", "_maps")
        last_under = stem.rfind("_")
        if last_under > 0:
            candidate = stem[last_under + 1:]
            if candidate.lower() in _KNOWN_SUFFIXES:
                stem = stem[:last_under]

        tokens = stem.split("_")

        # 1. Extract run_type (longest greedy match)
        run_type = None
        remainder_tokens = tokens
        for rt in _KNOWN_TYPES:
            rt_tokens = rt.split("_")
            n = len(rt_tokens)
            if tokens[:n] == rt_tokens:
                run_type = rt
                remainder_tokens = tokens[n:]
                break
        if run_type is None:
            # Fallback: first token is run_type
            run_type = tokens[0]
            remainder_tokens = tokens[1:]

        # 2. Extract integrator (next token if it matches)
        integrator = "rk4"
        if remainder_tokens and remainder_tokens[0].lower() in _KNOWN_INTEGRATORS:
            integrator = remainder_tokens[0].lower()
            remainder_tokens = remainder_tokens[1:]

        # 3. Extract metric (next token if it matches)
        metric = None
        if remainder_tokens and remainder_tokens[0].lower() in {m.lower() for m in _KNOWN_METRICS}:
            metric = remainder_tokens[0]
            remainder_tokens = remainder_tokens[1:]

        # 4. Extract profile (next token if it matches)
        profile = None
        if remainder_tokens and remainder_tokens[0].lower() in _KNOWN_PROFILES:
            profile = remainder_tokens[0].lower()
            remainder_tokens = remainder_tokens[1:]

        # 5. Parse parameter tokens
        # Each token is {prefix}{value} or for obs: obs{x}_{y}_{z}
        params: dict[str, Any] = {}
        i = 0
        while i < len(remainder_tokens):
            tok = remainder_tokens[i]

            # Try to match against known prefixes (longest prefix first)
            matched = False
            for key, prefix, _ in sorted(_PARAM_SPEC, key=lambda x: len(x[1]), reverse=True):
                if tok.startswith(prefix) and len(tok) > len(prefix):
                    val_str = tok[len(prefix):]

                    if key == "obs":
                        # obs token: obs{x}_{y}_{z}  --  consumes next 2 tokens
                        obs_parts = [val_str]
                        # Look ahead for the next 2 numeric tokens
                        for j in range(1, 3):
                            if i + j < len(remainder_tokens):
                                next_tok = remainder_tokens[i + j]
                                # Check if it's a plain number
                                try:
                                    float(next_tok)
                                    obs_parts.append(next_tok)
                                except ValueError:
                                    break
                        params[key] = tuple(_parse_val(p) for p in obs_parts)
                        i += len(obs_parts)
                        matched = True
                        break
                    else:
                        params[key] = _parse_val(val_str)
                        i += 1
                        matched = True
                        break

            if not matched:
                # Unknown token  --  try generic key{value} pattern
                m = re.match(r'^([A-Za-z_]+)(.+)$', tok)
                if m and m.group(2):
                    try:
                        params[m.group(1)] = _parse_val(m.group(2))
                    except ValueError:
                        pass  # skip unparseable
                i += 1

        return cls(
            run_type=run_type,
            integrator=integrator,
            metric=metric,
            profile=profile,
            output_dir=outdir,
            **params,
        )

    @classmethod
    def from_npz(cls, npz_path: str) -> "RunNamer":
        """Alias for :meth:`from_path`  --  most common entry point."""
        return cls.from_path(npz_path)

    # ------------------------------------------------------------------
    #  Rich display
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [
            f"RunNamer: {self.stem}",
            f"  run_type   : {self.run_type}",
            f"  integrator : {self.integrator}",
        ]
        if self.metric:
            lines.append(f"  metric     : {self.metric}")
        if self.profile:
            lines.append(f"  profile    : {self.profile}")
        lines.append(f"  outdir     : {self._outdir}")
        if self.params:
            lines.append("  parameters :")
            for key, val in self.params.items():
                label = _KEY_TO_LABEL.get(key, key)
                lines.append(f"    {label:.<28s} {val}")
        lines.append(f"  npz        : {self.npz()}")
        lines.append(f"  plot(*)    : {self.plot('*')}")
        return "\n".join(lines)

    def title_line(self) -> str:
        """
        One-line summary suitable for plot suptitles.

        Example::

            'NFW  M=1.0e+15 Msun  c=5  zl=0.022  zs=0.044  200 Mpc  256^3+AMR5  rk4  FLRWP1'
        """
        parts = []
        if self.profile:
            parts.append(self.profile.upper())
        if "M" in self.params:
            parts.append(rf"$M = {_fmt_val(self.params['M'])}$ M$_\odot$")
        if "c_NFW" in self.params:
            parts.append(rf"$c = {self.params['c_NFW']:.0f}$")
        if "Rvir" in self.params:
            parts.append(rf"$R_{{vir}} = {self.params['Rvir']:.1f}$ Mpc")
        if "zl" in self.params:
            parts.append(rf"$z_l = {self.params['zl']:.4f}$")
        if "zs" in self.params:
            parts.append(rf"$z_s = {self.params['zs']:.4f}$")
        if "box_Mpc" in self.params:
            parts.append(f"{self.params['box_Mpc']:.0f} Mpc")
        grid_str = ""
        if "N" in self.params:
            grid_str = f"${int(self.params['N'])}^3$"
        if "AMR" in self.params and self.params["AMR"]:
            grid_str += f"+AMR{int(self.params['AMR'])}"
        elif grid_str:
            grid_str += " regular"
        if grid_str:
            parts.append(grid_str)
        parts.append(self.integrator)
        if self.metric:
            parts.append(self.metric)
        return "   --   ".join(parts)

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return (
            f"RunNamer({self.run_type!r}, integrator={self.integrator!r}, "
            f"metric={self.metric!r}, profile={self.profile!r}"
            f"{', ' + p if p else ''})"
        )


# ===================================================================
#  Directory-level discovery helpers
# ===================================================================
def list_runs(run_type: str | None = None, output_dir: str | None = None) -> list[str]:
    """
    List ``.npz`` result files, optionally filtered by run type prefix.

    Returns sorted list of absolute paths.
    """
    d = output_dir or _OUTPUT_DIR
    pattern = f"{run_type}*_results.npz" if run_type else "*_results.npz"
    return sorted(glob.glob(os.path.join(d, pattern)))


def latest_run(run_type: str | None = None, output_dir: str | None = None) -> str | None:
    """Most recently modified ``.npz`` result file (or ``None``)."""
    files = list_runs(run_type, output_dir)
    return max(files, key=os.path.getmtime) if files else None


def latest_namer(run_type: str | None = None, output_dir: str | None = None) -> RunNamer | None:
    """``RunNamer`` for the most recent result file."""
    path = latest_run(run_type, output_dir)
    return RunNamer.from_npz(path) if path else None


# ===================================================================
#  Legacy trajectory filename helpers  (generate / parse / format)
# ===================================================================
from excalibur.core.constants import one_Mpc, one_Msun  # noqa: E402


def generate_trajectory_filename(
    mass_kg,
    radius_m,
    mass_position_m,
    observer_position_m,
    metric_type="perturbed_flrw",
    n_photons=None,
    integrator=None,
    version=None,
    stop_mode=None,
    stop_value=None,
    output_dir="_data",
    extra_tags=None,
):
    r"""
    Generate a standardised filename for trajectory data.

    Parameters
    ----------
    mass_kg : float
        Mass in kilograms.
    radius_m : float
        Mass radius in metres.
    mass_position_m : array-like
        Mass centre position ``[x, y, z]`` in metres.
    observer_position_m : array-like
        Observer position ``[x, y, z]`` in metres.
    metric_type : str, optional
        ``"perturbed_flrw"``, ``"schwarzschild"``, etc.
    n_photons : int, optional
    integrator : str, optional
    version : str, optional
        Backward-compatible alias for *integrator*.
    stop_mode : str, optional
        ``"steps"``, ``"redshift"``, ``"a"``, ``"chi"``.
    stop_value : float | int, optional
    output_dir : str, optional
    extra_tags : dict, optional

    Returns
    -------
    str
        Full path to output file.
    """
    mass_msun = mass_kg / one_Msun
    radius_mpc = radius_m / one_Mpc
    mass_pos_mpc = np.array(mass_position_m) / one_Mpc
    obs_pos_mpc = np.array(observer_position_m) / one_Mpc

    mass_str = f"M{mass_msun:.1e}".replace("+", "")
    radius_str = f"R{radius_mpc:.1f}"
    mass_pos_str = (
        f"mass{int(round(mass_pos_mpc[0]))}_{int(round(mass_pos_mpc[1]))}"
        f"_{int(round(mass_pos_mpc[2]))}"
    )
    obs_pos_str = (
        f"obs{int(round(obs_pos_mpc[0]))}_{int(round(obs_pos_mpc[1]))}"
        f"_{int(round(obs_pos_mpc[2]))}"
    )

    if integrator is None and version is not None:
        integrator = version

    photon_str = f"N{n_photons}" if n_photons is not None else ""
    integrator_str = integrator or ""

    stop_str = ""
    if stop_mode is not None and stop_value is not None:
        _MAP = {"steps": "S", "redshift": "z", "a": "a", "chi": "chi"}
        prefix = _MAP.get(stop_mode, stop_mode)
        if stop_mode == "steps":
            stop_str = f"{prefix}{int(stop_value)}"
        else:
            stop_str = f"{prefix}{stop_value}"

    parts = ["excalibur_run", metric_type, mass_str, radius_str,
             mass_pos_str, obs_pos_str]
    for p in (photon_str, integrator_str, stop_str):
        if p:
            parts.append(p)

    if extra_tags:
        for k, v in extra_tags.items():
            if v is None:
                continue
            key = str(k)
            if isinstance(v, bool):
                val = "1" if v else "0"
            elif isinstance(v, (int, np.integer)):
                val = str(int(v))
            elif isinstance(v, (float, np.floating)):
                val = f"{float(v):.6g}".replace("+", "")
            else:
                val = str(v)
            parts.append(f"{key}{val}")

    filename = "_".join(parts) + "_Mpc.h5"
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def parse_trajectory_filename(filename):
    r"""Parse a standardised trajectory filename to extract parameters.

    Returns a dict or ``None`` if the filename doesn't match.
    """
    basename = os.path.basename(filename)

    pattern = (
        r"excalibur_run_"
        r"(?P<metric_type>\w+)_"
        r"M(?P<mass>[0-9.e+-]+)_"
        r"R(?P<radius>[0-9.]+)_"
        r"mass(?P<mass_x>-?[0-9]+)_(?P<mass_y>-?[0-9]+)_(?P<mass_z>-?[0-9]+)_"
        r"obs(?P<obs_x>-?[0-9]+)_(?P<obs_y>-?[0-9]+)_(?P<obs_z>-?[0-9]+)"
        r"(?:_N(?P<n_photons>[0-9]+))?"
        r"(?:_(?P<integrator>sequential|parallel|optimal|persistent"
        r"|rk45|rk4|leapfrog4|OPTIMAL|OPTIMIZED|standard))?"
        r"(?:_(?P<stop_condition>S[0-9]+|z[0-9.]+|a[0-9.]+|chi[0-9.]+))?"
        r"(?:_[A-Za-z0-9.]+)*"
        r"_Mpc\.h5"
    )
    match = re.match(pattern, basename)

    if not match:
        legacy = re.match(
            r"(backward_raytracing|excalibur_run).*?"
            r"mass_?(?P<mass_x>[0-9]+)_(?P<mass_y>[0-9]+)_(?P<mass_z>[0-9]+)_Mpc\.h5",
            basename,
        )
        if legacy:
            mass_pos = np.array([float(legacy.group("mass_x")),
                                 float(legacy.group("mass_y")),
                                 float(legacy.group("mass_z"))])
            return {
                "metric_type": "unknown",
                "mass_msun": None, "radius_mpc": None,
                "mass_position_mpc": mass_pos,
                "observer_position_mpc": None,
                "n_photons": None, "integrator": None, "version": None,
                "stop_mode": None, "stop_value": None,
                "mass_kg": None, "radius_m": None,
                "mass_position_m": mass_pos * one_Mpc,
                "observer_position_m": None, "distance_mpc": None,
            }
        return None

    metric_type = match.group("metric_type")
    mass_msun = float(match.group("mass"))
    radius_mpc = float(match.group("radius"))
    mass_position_mpc = np.array([float(match.group("mass_x")),
                                  float(match.group("mass_y")),
                                  float(match.group("mass_z"))])
    observer_position_mpc = np.array([float(match.group("obs_x")),
                                      float(match.group("obs_y")),
                                      float(match.group("obs_z"))])
    n_photons = int(match.group("n_photons")) if match.group("n_photons") else None
    integrator = match.group("integrator") or None

    stop_mode = stop_value = None
    sc = match.group("stop_condition")
    if sc:
        if sc.startswith("S"):
            stop_mode, stop_value = "steps", int(sc[1:])
        elif sc.startswith("z"):
            stop_mode, stop_value = "redshift", float(sc[1:])
        elif sc.startswith("a"):
            stop_mode, stop_value = "a", float(sc[1:])
        elif sc.startswith("chi"):
            stop_mode, stop_value = "chi", float(sc[3:])

    return {
        "metric_type": metric_type,
        "mass_msun": mass_msun,
        "radius_mpc": radius_mpc,
        "mass_position_mpc": mass_position_mpc,
        "observer_position_mpc": observer_position_mpc,
        "n_photons": n_photons,
        "integrator": integrator,
        "version": integrator,
        "stop_mode": stop_mode,
        "stop_value": stop_value,
        "mass_kg": mass_msun * one_Msun,
        "radius_m": radius_mpc * one_Mpc,
        "mass_position_m": mass_position_mpc * one_Mpc,
        "observer_position_m": observer_position_mpc * one_Mpc,
        "distance_mpc": float(np.linalg.norm(
            mass_position_mpc - observer_position_mpc)),
    }


def format_simulation_info(filename):
    """Human-readable summary of parameters encoded in a trajectory filename."""
    info = parse_trajectory_filename(filename)
    if info is None:
        return f"Could not parse filename: {filename}"

    lines = ["Simulation Parameters:"]
    lines.append(f"  Metric: {info['metric_type']}")
    if info["mass_msun"] is not None:
        lines.append(f"  Mass: {info['mass_msun']:.2e} M_sun")
    if info["radius_mpc"] is not None:
        lines.append(f"  Radius: {info['radius_mpc']:.1f} Mpc")
    if info["mass_position_mpc"] is not None:
        mp = info["mass_position_mpc"]
        lines.append(f"  Mass position: [{mp[0]:.0f}, {mp[1]:.0f}, {mp[2]:.0f}] Mpc")
    if info["observer_position_mpc"] is not None:
        op = info["observer_position_mpc"]
        lines.append(f"  Observer position: [{op[0]:.0f}, {op[1]:.0f}, {op[2]:.0f}] Mpc")
    if info["n_photons"] is not None:
        lines.append(f"  Number of photons: {info['n_photons']}")
    if info["integrator"] is not None:
        lines.append(f"  Integrator: {info['integrator']}")
    if info["stop_mode"] is not None:
        sm, sv = info["stop_mode"], info["stop_value"]
        _FMT = {"steps": f"{int(sv)} steps", "redshift": f"z = {sv}",
                "a": f"a = {sv}", "chi": f"chi = {sv}"}
        lines.append(f"  Stop condition: {_FMT.get(sm, f'{sm} = {sv}')}")
    if info["distance_mpc"] is not None:
        lines.append(f"  Observer-Mass distance: {info['distance_mpc']:.2f} Mpc")
    return "\n".join(lines)
