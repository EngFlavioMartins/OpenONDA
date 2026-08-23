"""Shared utilities for rotor_flow plot scripts."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

# -- Directory layout --------------------------------------------------------
ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
FIGURES_DIR = CASE_DIR / "figures"
SOLUTION_DIR = CASE_DIR / "solution"
SAMPLES_DIR = CASE_DIR / "samples" / "rotor"
THEME_PATH = CASE_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"

# -- Case definition ---------------------------------------------------------
# Single source of truth for the post-processing side.  These must track the
# physical parameters in ``rotor_setup.py``; keeping them here stops the three
# plot scripts from drifting apart from each other and from the case.
ROTOR_RADIUS = 6.0  # [m]
HUB_RADIUS = 1.0  # [m]
FREESTREAM_SPEED = 7.0  # [m/s]
TIP_SPEED_RATIO = 7.0
DENSITY = 1.225  # [kg/m³]
N_BLADES = 3
ANGULAR_VELOCITY = TIP_SPEED_RATIO * FREESTREAM_SPEED / ROTOR_RADIUS  # [rad/s]
ROTATION_PERIOD = 2.0 * math.pi / ANGULAR_VELOCITY  # [s]

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
    """Load the OpenONDA matplotlib theme and return (COLORS, theme module)."""
    theme = _theme()
    theme.set_style()
    return dict(theme.COLORS), theme


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Base argument parser shared by rotor_flow plot scripts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--format", choices=_theme().EXPORT_FORMATS, default="png")
    p.add_argument("--dpi", type=int, default=_theme().DEFAULT_DPI, help="Figure DPI (PNG only).")
    return p


def build_rotor_style_map(colors: dict[str, str]) -> dict[str, dict[str, object]]:
    """Map rotor-data sources to plot style dicts (color, marker, linestyle, label)."""
    return {name: dict(style) for name, style in _theme().ROTOR_STYLE.items()}


# ==============================================================================
# Sampled run data
# ==============================================================================


def read_time_step(samples_dir: Path | str) -> float | None:
    """Read the time step represented by a sampled force history."""
    csv_path = Path(samples_dir) / "vlm_forces.csv"
    if csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path)
        if not df.empty and "time" in df and "step" in df:
            step = float(df["step"].iloc[-1])
            if step > 0:
                return float(df["time"].iloc[-1]) / step

    return None


def read_operating_point(
    samples_dir: Path | str,
    *,
    density: float = DENSITY,
    freestream_speed: float = FREESTREAM_SPEED,
    rotor_radius: float = ROTOR_RADIUS,
    tip_speed_ratio: float = TIP_SPEED_RATIO,
    tail_fraction: float = 0.2,
) -> tuple[float, float] | None:
    """Return the run's own tail-mean (thrust_coefficient, power_coefficient), or None if the CSV is missing.

    The wake references must be drawn at the operating point the simulation
    actually reached, not at the Betz design point it was aiming for.
    """
    csv_path = Path(samples_dir) / "vlm_forces.csv"
    if not csv_path.exists():
        return None

    import numpy as np
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    angular_velocity = tip_speed_ratio * freestream_speed / rotor_radius
    dynamic_pressure_area = 0.5 * density * freestream_speed**2 * math.pi * rotor_radius**2
    thrust_coefficient = df["force_x"].to_numpy() / dynamic_pressure_area
    power_coefficient = (-df["moment_x"].to_numpy() * angular_velocity) / (
        dynamic_pressure_area * freestream_speed
    )

    tail = slice(max(0, int((1.0 - tail_fraction) * len(df))), None)
    return float(np.mean(thrust_coefficient[tail])), float(np.mean(power_coefficient[tail]))
