"""Shared utilities for rotorFlow plot scripts."""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
from pathlib import Path

# -- Directory layout --------------------------------------------------------
ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
FIGURES_DIR = CASE_DIR / "figures"
SOLUTION_DIR = CASE_DIR / "solution" / "rotor"
THEME_PATH = CASE_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"

# -- Case definition ---------------------------------------------------------
# Single source of truth for the post-processing side.  These must track the
# physical parameters in ``rotor_setup.py``; keeping them here stops the three
# plot scripts from drifting apart from each other and from the case.
ROTOR_RADIUS = 6.0  # [m]
HUB_RADIUS = 1.0  # [m]
FREESTREAM_VELOCITY = 7.0  # [m/s]
TIP_SPEED_RATIO = 7.0
DENSITY = 1.225  # [kg/m³]
NUM_BLADES = 3
ANGULAR_VELOCITY = TIP_SPEED_RATIO * FREESTREAM_VELOCITY / ROTOR_RADIUS  # [rad/s]
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
    """Base argument parser shared by rotorFlow plot scripts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--solution-dir", default=str(SOLUTION_DIR), help="Rotor solution directory.")
    p.add_argument("--figures-dir", default=str(FIGURES_DIR), help="Output figure directory.")
    p.add_argument("--format", choices=_theme().EXPORT_FORMATS, default="png")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI (PNG only).")
    return p


def add_case_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the rotor geometry/operating-point overrides shared by the plot scripts."""
    p.add_argument("--rotor-radius", type=float, default=ROTOR_RADIUS, help="Tip radius [m].")
    p.add_argument("--hub-radius", type=float, default=HUB_RADIUS, help="Hub radius [m].")
    p.add_argument("--u-inf", type=float, default=FREESTREAM_VELOCITY, help="Freestream [m/s].")
    p.add_argument("--tip-speed-ratio", type=float, default=TIP_SPEED_RATIO, help="TSR = wR/U.")
    p.add_argument("--rho", type=float, default=DENSITY, help="Fluid density [kg/m³].")
    return p


def build_rotor_style_map(colors: dict[str, str]) -> dict[str, dict[str, object]]:
    """Map rotor-data sources to plot style dicts (color, marker, linestyle, label)."""
    return {name: dict(style) for name, style in _theme().ROTOR_STYLE.items()}


# ==============================================================================
# Run introspection — read what the solver actually did, never assume
# ==============================================================================


def read_time_step(solution_dir: Path | str) -> float | None:
    """Return the time-step size the run actually used, or None if undeterminable.

    Hardcoding a default here is how the wake-plane averaging window silently
    ended up 17 % out of step with the case, so both sources are derived from
    the run itself: the force CSV first (exact, ``time / step``), then the
    solver log banner as a fallback.
    """
    solution_dir = Path(solution_dir)

    csv_path = solution_dir / "samples" / "vlm_forces.csv"
    if csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path)
        if not df.empty and "time" in df and "step" in df:
            step = float(df["step"].iloc[-1])
            if step > 0:
                return float(df["time"].iloc[-1]) / step

    log_path = solution_dir / "rotor.log"
    if log_path.exists():
        match = re.search(
            r"Time Step Size\s*:\s*([0-9.eE+-]+)",
            log_path.read_text(errors="replace"),
        )
        if match:
            return float(match.group(1))

    return None


def read_operating_point(
    solution_dir: Path | str,
    *,
    rho: float = DENSITY,
    u_inf: float = FREESTREAM_VELOCITY,
    rotor_radius: float = ROTOR_RADIUS,
    tip_speed_ratio: float = TIP_SPEED_RATIO,
    tail_fraction: float = 0.2,
) -> tuple[float, float] | None:
    """Return the run's own tail-mean (Ct, Cp), or None if the CSV is missing.

    The wake references must be drawn at the operating point the simulation
    actually reached, not at the Betz design point it was aiming for.
    """
    csv_path = Path(solution_dir) / "samples" / "vlm_forces.csv"
    if not csv_path.exists():
        return None

    import numpy as np
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    omega = tip_speed_ratio * u_inf / rotor_radius
    q_a = 0.5 * rho * u_inf**2 * math.pi * rotor_radius**2
    ct = df["Fx"].to_numpy() / q_a
    cp = (-df["Mx"].to_numpy() * omega) / (q_a * u_inf)

    tail = slice(max(0, int((1.0 - tail_fraction) * len(df))), None)
    return float(np.mean(ct[tail])), float(np.mean(cp[tail]))
