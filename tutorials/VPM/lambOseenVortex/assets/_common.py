"""Shared utilities for lambOseenVortex plot scripts.

Each plot script lives in assets/ and imports from here via::

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from _common import load_theme, build_style_map, ...
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# ── Directory layout ─────────────────────────────────────────────────────────
ASSETS_DIR = Path(__file__).resolve().parent  # …/assets/
SCRIPT_DIR = ASSETS_DIR.parent  # …/lambOseenVortex/
FIGURES_DIR = SCRIPT_DIR / "figures"
SOLUTION_DIR = SCRIPT_DIR / "solution"
REF_DIR = ASSETS_DIR / "references"
THEME_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"
FONT_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "DejaVuSerif.ttf"

SCHEMES = ("cs", "rwm", "dvh", "gbd")


def load_theme() -> tuple[dict[str, str], object | None]:
    """Load the OpenONDA matplotlib theme and return (COLORS dict, theme module)."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    theme = None
    if THEME_PATH.exists():
        spec = importlib.util.spec_from_file_location("mpl_setup", THEME_PATH)
        theme = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(theme)
        try:
            theme.set_style()
        except Exception:
            pass

    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = "DejaVu Serif"

    if theme is not None and hasattr(theme, "COLORS"):
        return dict(theme.COLORS), theme
    return {}, theme


def build_style_map(colors: dict[str, str]) -> dict[str, dict]:
    """Map scheme names to plot style dicts (color, marker, label)."""
    return {
        "cs": {"label": "CS", "color": colors.get("FVMorange", "#772953"), "marker": "o"},
        "rwm": {"label": "RWM", "color": colors.get("RefGray", "#6E8898"), "marker": "^"},
        "dvh": {"label": "DVH", "color": colors.get("TUDcyan", "#0E8A85"), "marker": "v"},
        "gbd": {"label": "GBD", "color": colors.get("VPMpurple", "#5C3D9B"), "marker": "D"},
    }


def build_arg_parser(description: str):
    """Base argument parser shared by all plot scripts."""
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--solution-dir", default=str(SOLUTION_DIR), help="Root solution directory.")
    p.add_argument(
        "--figures-dir", default=str(FIGURES_DIR), help="Output directory for PNG figures."
    )
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI (PNG only).")
    p.add_argument(
        "--format",
        choices=["png", "svg"],
        default="png",
        help="Output figure format (default: png).",
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        help="Specific time-step to plot (default: last available).",
    )
    return p


def add_physics_args(parser) -> None:
    """Add physical-parameter arguments with defaults matching vortex_setup.py."""
    _RE = 530.0  # Re_Γ = Γ/nu — matches allrun.sh and the C&W 2003 reference
    _NU = 1.0 / _RE
    _RC = 0.125
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=_NU)
    parser.add_argument("--t0", type=float, default=_RC**2 / (4.0 * _NU))
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--re", type=float, default=_RE)
    parser.add_argument("--circulation", type=float, default=1.0)
    parser.add_argument("--b0", type=float, default=1.0)
    parser.add_argument("--a0-over-b0", type=float, default=0.125)
    parser.add_argument("--total-time", type=float, default=40.0)


def read_flow_time(csv_path: Path) -> float | None:
    """Extract ``flow_time`` from the first comment line of a sampled CSV."""
    with open(csv_path) as handle:
        first_line = handle.readline().strip()
    if first_line.startswith("# flow_time="):
        return float(first_line.split("=", 1)[1])
    return None


def infer_run_nu(
    solution_dir: Path,
    gamma: float,
    fallback_nu: float,
    rc0: float,
) -> float:
    """Infer the run viscosity from the sampled single-vortex profile data.

    This keeps the plot scripts consistent with whatever data is already in
    ``solution/``, even if it was generated before the current defaults changed.
    """
    nu_candidates: list[float] = []
    for scheme in ("cs", "dvh", "gbd", "rwm"):
        samples_dir = solution_dir / f"vortex_{scheme}" / "samples"
        csv_files = sorted(samples_dir.glob(f"vortex_{scheme}_x_*.csv"))
        if not csv_files:
            continue
        flow_time = read_flow_time(csv_files[-1])
        if flow_time is None or flow_time <= 0.0:
            continue
        df = pd.read_csv(csv_files[-1], comment="#")
        if df.empty or "omega_z" not in df.columns:
            continue
        omega_peak = float(df["omega_z"].max())
        if omega_peak <= 0.0:
            continue
        rc_sq = gamma / (np.pi * omega_peak)
        nu = (rc_sq - rc0**2) / (4.0 * flow_time)
        if nu > 0.0:
            nu_candidates.append(nu)
    if not nu_candidates:
        return fallback_nu
    return float(np.median(nu_candidates))


def resolve_runtime_physics(
    solution_dir: Path,
    gamma: float,
    fallback_nu: float,
    b0: float,
    a0_over_b0: float,
) -> dict[str, float]:
    """Return the physical constants that match the data already on disk."""
    rc0 = a0_over_b0 * b0
    nu = infer_run_nu(solution_dir, gamma, fallback_nu, rc0)
    return {"nu": nu, "t0": rc0**2 / (4.0 * nu), "rc0": rc0}


def pvd_time_map(solution_dir: Path, prefix: str, scheme: str) -> dict[int, float]:
    """Read the surface-sample PVD to get a step → physical-time mapping.

    Useful as a fallback when HDF5 backup files are not present.
    """
    import re as _re
    import xml.etree.ElementTree as ET

    pvd = solution_dir / f"{prefix}_{scheme}" / "samples" / f"{prefix}_{scheme}_z0.pvd"
    if not pvd.exists():
        return {}
    tree = ET.parse(pvd)  # nosec B314
    result: dict[int, float] = {}
    for ds in tree.getroot().iter("DataSet"):
        fname = ds.attrib.get("file", "")
        m = _re.search(r"_(\d+)\.vts$", fname)
        if m:
            result[int(m.group(1))] = float(ds.attrib.get("timestep", 0.0))
    return result


# ── Gaussian core-radius utilities ────────────────────────────────────────────


def gaussian_model(r, omega0, a):
    """Lamb-Oseen vorticity using C&W convention: ω = ω0 · exp(-r² / (2a²))."""
    return omega0 * np.exp(-(r**2) / (2.0 * a**2))


def azimuthal_profile(xy, values, center, n_bins=50, r_max=None):
    """Azimuthally averaged |values| profile around *center*."""
    dxy = xy - center
    r = np.linalg.norm(dxy, axis=1)
    if r_max is None:
        r_max = float(r.max())
    mask = r < r_max
    r_m, v_m = r[mask], np.abs(values[mask])
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        in_bin = (r_m >= r_edges[i]) & (r_m < r_edges[i + 1])
        if in_bin.any():
            profile[i] = v_m[in_bin].mean()
    return r_centers, profile


def fit_gaussian_core(r_profile, omega_profile):
    """Return a² (sigma²) from a Gaussian fit (C&W convention)."""
    from scipy.optimize import curve_fit

    mask = omega_profile > 0.05 * omega_profile.max()
    if mask.sum() < 3:
        return np.nan
    r_fit, w_fit = r_profile[mask], omega_profile[mask]
    omega0_g = omega_profile.max()
    above = omega_profile > omega0_g / 2.0
    sigma_g = r_profile[above][-1] / np.sqrt(2.0 * np.log(2.0)) if above.any() else 0.1
    try:
        popt, _ = curve_fit(
            gaussian_model,
            r_fit,
            w_fit,
            p0=[omega0_g, max(sigma_g, 0.01)],
            bounds=([0, 0.001], [np.inf, 2.0]),
            maxfev=5000,
        )
        return popt[1] ** 2
    except (RuntimeError, ValueError):
        return np.nan


def core_radius_sigma(xy, values, center, n_bins=50, r_max=None):
    """Core radius sigma from Gaussian fit on a scalar field."""
    r_prof, w_prof = azimuthal_profile(xy, values, center, n_bins, r_max)
    if w_prof.max() < 1e-10:
        return np.nan
    a2 = fit_gaussian_core(r_prof, w_prof)
    return np.sqrt(a2) if not np.isnan(a2) else np.nan


def centroid(xy, values):
    """Scalar-weighted centroid of a 2-D point cloud."""
    w = np.abs(values)
    wt = w.sum()
    if wt < 1e-30:
        return np.full(2, np.nan)
    return np.array([np.dot(w, xy[:, 0]) / wt, np.dot(w, xy[:, 1]) / wt])
