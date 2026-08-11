"""Shared utilities for the Lamb--Oseen plot scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# -- Directory layout --------------------------------------------------------
ASSETS_DIR = Path(__file__).resolve().parent  # …/assets/
SCRIPT_DIR = ASSETS_DIR.parent  # …/lambOseenVortex/
FIGURES_DIR = SCRIPT_DIR / "figures"
SAMPLES_DIR = SCRIPT_DIR / "samples"
REF_DIR = ASSETS_DIR / "references"
THEME_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"

SCHEMES = ("cs", "rwm", "dvh", "gbd")

BETA_RMAX = 1.12
GAMMA = 1.0
REYNOLDS_NUMBER = 530.0
CORE_RADIUS = 0.125
SEPARATION = 1.0
PUBLICATION_WIDTH_CM = 12.5

_THEME_MODULE = None


def _theme():
    global _THEME_MODULE
    if _THEME_MODULE is None:
        if not THEME_PATH.exists():
            raise FileNotFoundError(f"OpenONDA matplotlib theme not found: {THEME_PATH}")
        spec = importlib.util.spec_from_file_location("openonda_matplotlib_setup", THEME_PATH)
        theme = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(theme)
        _THEME_MODULE = theme
    return _THEME_MODULE


def load_theme() -> tuple[dict[str, str], object | None]:
    """Load the OpenONDA matplotlib theme and return (COLORS dict, theme module)."""
    theme = _theme()
    theme.set_style()
    return dict(theme.COLORS), theme


def build_style_map(colors: dict[str, str]) -> dict[str, dict]:
    """Map scheme names to plot style dicts (color, marker, label)."""
    return {name: dict(style) for name, style in _theme().LAMB_OSEEN_SCHEME_STYLE.items()}


def publication_size(height_cm: float) -> tuple[float, float]:
    """Return a 12.5 cm-wide publication canvas in inches."""

    return PUBLICATION_WIDTH_CM / 2.54, height_cm / 2.54


def save_publication_figure(fig, path: Path, dpi: int) -> None:
    """Save without tight bounding-box cropping so physical size stays exact."""

    save_kwargs = {"dpi": dpi} if path.suffix == ".png" else {}
    fig.savefig(path, **save_kwargs)
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"  Saved: {path}")


def build_arg_parser(description: str):
    """Base argument parser shared by all plot scripts."""
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI (PNG only).")
    p.add_argument(
        "--format",
        choices=_theme().EXPORT_FORMATS,
        default="png",
        help="Output figure format (default: png).",
    )
    viscosity = GAMMA / REYNOLDS_NUMBER
    p.set_defaults(
        samples_dir=SAMPLES_DIR,
        figures_dir=FIGURES_DIR,
        gamma=GAMMA,
        nu=viscosity,
        b0=SEPARATION,
        a0_over_b0=CORE_RADIUS / SEPARATION,
    )
    return p


def read_flow_time(csv_path: Path) -> float | None:
    """Extract ``flow_time`` from the first comment line of a sampled CSV."""
    with open(csv_path) as handle:
        first_line = handle.readline().strip()
    if first_line.startswith("# flow_time="):
        return float(first_line.split("=", 1)[1])
    return None


def read_run_metadata(samples_dir: Path, prefix: str = "vortex") -> dict:
    """Load the physical constants stored with the sampled results.

    Plotting must not depend on a dense particle backup: those files are sparse
    restart checkpoints, while ``samples/<case>/run_metadata.json`` is written
    alongside the data used by each figure.
    """
    import json

    for scheme in SCHEMES:
        path = samples_dir / f"{prefix}_{scheme}" / "run_metadata.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return {}


def read_column_half_length(samples_dir: Path, prefix: str = "vortex") -> float | None:
    """Read the finite column half-length from sampled-result metadata."""

    value = read_run_metadata(samples_dir, prefix).get("column_half_length")
    return float(value) if value is not None else None


def resolve_runtime_physics(
    samples_dir: Path,
    gamma: float,
    fallback_nu: float,
    b0: float,
    a0_over_b0: float,
) -> dict[str, float]:
    """Return the physical constants for the analytic reference.

    ``nu`` and the core radius come from ``samples/<case>/run_metadata.json``.
    If a run has no metadata, the tutorial constants provide the reference.
    The analytic reference is never inferred from the schemes' own output.
    """
    metadata = read_run_metadata(samples_dir)
    ac0 = float(metadata.get("core_radius", a0_over_b0 * b0))
    nu = float(metadata.get("viscosity", fallback_nu))
    if nu <= 0.0:
        nu = fallback_nu
    sigma0 = ac0 / BETA_RMAX
    return {"nu": nu, "t0": sigma0**2 / (4.0 * nu), "ac0": ac0}


def pvd_time_map(samples_dir: Path, prefix: str, scheme: str) -> dict[int, float]:
    """Read the surface-sample PVD to get a step → physical-time mapping.

    This keeps sampled fields tied to physical time without reading checkpoints.
    """
    import re as _re
    import xml.etree.ElementTree as ET

    pvd = samples_dir / f"{prefix}_{scheme}" / f"{prefix}_{scheme}_z0.pvd"
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


# -- Gaussian core-radius utilities (C&W a convention) ------------------------


def gaussian_model(r, omega0, a):
    """Gaussian vorticity fit ω = ω0·exp(-r²/(2a²)).

    Here *a* is the statistical standard deviation, which equals sigma/√2 for the
    solver's omega ~ exp(-r²/sigma²) form.  It is NOT the C&W peak-velocity core
    radius (= BETA_RMAX·sigma); convert if you need to compare against C&W.
    """
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
    """Return a² from a Gaussian fit, where a = statistical std = sigma/√2."""
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
    """Core radius a from Gaussian fit on a scalar field."""
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
