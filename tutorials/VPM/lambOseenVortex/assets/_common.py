"""Shared utilities for the Lamb--Oseen plot scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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


def figure_size(name: str = "single") -> tuple[float, float]:
    """Return a named figure size in inches from the shared theme."""
    return _theme().figure_size(name)


def save_fig(fig, path: Path, dpi: int) -> None:
    """Save without tight layout or cropping; manual subplots_adjust() takes precedence."""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"  Saved: {out}")


def build_arg_parser(description: str):
    """Base argument parser shared by all plot scripts."""
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dpi", type=int, default=_theme().DEFAULT_DPI, help="Figure DPI (PNG only).")
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
