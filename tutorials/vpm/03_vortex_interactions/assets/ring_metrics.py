"""Shared data loading and plot styles for the two-ring cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
SAMPLES_DIR = CASE_DIR / "samples"
SOLUTION_DIR = CASE_DIR / "solution"
FIGURES_DIR = CASE_DIR / "figures"

RING_RADIUS = 1.0
RING_CIRCULATION = np.pi
REFERENCE_TIME = RING_RADIUS**2 / RING_CIRCULATION
FAMILIES = ("leapfrog",)
FAMILY_LABELS = {"leapfrog": "Leapfrogging"}
FAMILY_FILE_STEMS = {"leapfrog": "leapfrogging"}
CASES = (
    "baseline",
    "stretching_viscosity",
    "pedrizzetti",
    "splitting",
    "divergence_relaxation",
    "remeshing",
)
INTENDED_CASE_ORDER = {name: order for order, name in enumerate(CASES)}

_THEME_MODULE = None


def _theme():
    """Return the shared OpenONDA Matplotlib theme."""
    global _THEME_MODULE
    if _THEME_MODULE is None:
        from openonda import plotting as theme

        _THEME_MODULE = theme
    return _THEME_MODULE


def load_theme() -> tuple[dict[str, str], object | None]:
    """Apply the shared OpenONDA Matplotlib style."""
    theme = _theme()
    theme.set_thesis_style()
    return dict(theme.COLORS), theme


def figure_size(name: str = "single") -> tuple[float, float]:
    """Return a named figure size from the shared plot theme."""
    return _theme().figure_size(name)


def reference_style() -> dict:
    """Return the shared reference-line style."""
    return dict(_theme().REFERENCE_STYLE)


def reference_fill_style(kind: str = "normal") -> dict:
    """Return the shared reference-band style."""
    if kind == "strong":
        return dict(_theme().REFERENCE_STRONG_FILL_STYLE)
    return dict(_theme().REFERENCE_FILL_STYLE)


def legend_handle_style(style: dict) -> dict:
    """Return the shared legend-handle style."""
    return _theme().legend_handle_style(style)


def case_legend_handles(cases: list[str]) -> list:
    """Build one legend entry per plotted method."""
    from matplotlib.lines import Line2D

    handles = []
    labels = set()
    for name in cases:
        style = case_style(name)
        if style["label"] in labels:
            continue
        labels.add(style["label"])
        handles.append(Line2D([0], [0], **legend_handle_style(style)))
    return handles


def mark_every(name: str = "default") -> int:
    """Return the shared marker cadence for a plot kind."""
    return _theme().MARK_EVERY[name]


def secondary_line_style() -> dict:
    """Return the shared style for a secondary line."""
    theme = _theme()
    return {
        "linestyle": theme.SECONDARY_LINESTYLE,
        "linewidth": theme.SECONDARY_LINE_WIDTH,
    }


def build_arg_parser(description: str):
    """Build the minimal argument parser shared by the figures."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--format", choices=_theme().EXPORT_FORMATS, default="png")
    parser.set_defaults(solution_dir=str(SAMPLES_DIR), dpi=_theme().DEFAULT_DPI)
    return parser


def _case_parts(name: str) -> tuple[str, str]:
    """Return the common flow family and the varied stabilization method."""
    return "leapfrog", name


def case_style(name: str) -> dict:
    """Return a consistent style dictionary from the shared plot theme."""
    theme = _theme()
    _, variant = _case_parts(name)
    styles = {
        "baseline": {"label": "Baseline", "color": theme.PALETTE["dark"], "marker": "o"},
        "stretching_viscosity": {
            "label": "Stretching viscosity",
            "color": theme.PALETTE["teal"],
            "marker": "s",
        },
        "pedrizzetti": {
            "label": "Pedrizzetti relaxation",
            "color": theme.PALETTE["purple"],
            "marker": "D",
        },
        "splitting": {
            "label": "Filament refinement",
            "color": theme.PALETTE["orange"],
            "marker": "^",
        },
        "divergence_relaxation": {
            "label": "Divergence relaxation",
            "color": theme.PALETTE["green"],
            "marker": "v",
        },
        "remeshing": {
            "label": "Conservative regularization",
            "color": theme.PALETTE["red"],
            "marker": "P",
        },
    }
    style = styles[variant]
    return {
        "color": style["color"],
        "linestyle": "-",
        "linewidth": theme.LINE_WIDTH,
        "marker": style["marker"],
        "markersize": theme.MARKER_SIZE,
        "markeredgewidth": theme.MARKER_EDGE_WIDTH,
        "label": style["label"],
    }


def discover_cases(solution_dir=SAMPLES_DIR, family: str | None = None) -> list[Path]:
    """Return available cases in plotting order."""
    samples = Path(solution_dir)
    if not samples.is_dir():
        return []
    cases = []
    intended = INTENDED_CASE_ORDER
    for case_dir in samples.iterdir():
        if not case_dir.is_dir() or case_dir.name not in intended:
            continue
        case_family, _ = _case_parts(case_dir.name)
        if family and case_family != family:
            continue
        if (SOLUTION_DIR / case_dir.name / "vpm_metadata.json").exists() or (
            case_dir / "flow_integrals.csv"
        ).exists():
            cases.append(case_dir)
    return sorted(cases, key=lambda path: intended[path.name])


def _samples_dir(case_dir: str | Path) -> Path:
    """Return the sample directory for one case."""
    case = Path(case_dir)
    if case.parent.name == "samples":
        return case
    return CASE_DIR / "samples" / case.name


def load_metadata(case_dir: str | Path) -> dict:
    """Load the universal VPM metadata from the case's backup directory."""
    import json

    case = Path(case_dir)
    path = SOLUTION_DIR / case.name / "vpm_metadata.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_study_metadata(directory: str | Path) -> dict:
    """Load solver-owned metadata for one advanced study directory."""
    import json

    path = Path(directory) / "solution" / "vpm_metadata.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def metadata_settings(metadata: dict) -> dict:
    """Extract plotting controls from the universal VPM metadata schema."""
    configuration = metadata.get("configuration", {})
    numerics = configuration.get("numerics", {})
    initial_conditions = configuration.get("initial_conditions", [])
    first = initial_conditions[0] if initial_conditions else {}
    second = initial_conditions[1] if len(initial_conditions) > 1 else {}
    distribution = first.get("distribution", {})
    disturbance = first.get("disturbance", {}) or {}
    distribution_disturbance = distribution.get("disturbance")
    viscous = numerics.get("viscous", {})
    stabilization = numerics.get("stabilization", {})
    case_name = str(metadata.get("case_name") or "")
    known_methods = (
        "p_split_remesh",
        "solenoidal_remeshing",
        "divergence_relaxation",
        "stretching_viscosity",
        "p_relaxation",
        "p_moments",
        "p_remesh",
        "splitting",
        "remeshing",
        "pedrizzetti",
        "baseline",
    )
    method = next(
        (name for name in known_methods if name in case_name),
        "baseline",
    )
    if "halfdt" in case_name:
        method = "baseline"
    first_circulation = float(first.get("circulation", 0.0))
    second_circulation = float(second.get("circulation", first_circulation))
    molecular_viscosity = float(first.get("kinematic_viscosity", np.nan))
    time_step_size = float(numerics.get("time_step_size", np.nan))
    relaxation = float(stabilization.get("pedrizzetti_relaxation_factor", 0.0))
    return {
        "scenario": "collision" if first_circulation * second_circulation < 0.0 else "leapfrog",
        "method": method,
        "diffusion": viscous.get("scheme", "CS"),
        "gbd_remeshing": viscous.get("gbd_remeshing_kernel", "M4_PRIME"),
        "integrator": numerics.get("integrator", {}).get("name", "SSPRK3"),
        "spacing": float(distribution.get("spacing", viscous.get("particle_spacing", np.nan))),
        "core_ratio": float(
            distribution.get("core_radius_ratio", viscous.get("core_radius_ratio", np.nan))
        ),
        "support": "disturbed" if distribution_disturbance else "circular",
        "amplitude": float(disturbance.get("amplitude", 0.0)),
        "reynolds_number": (
            abs(first_circulation) / molecular_viscosity
            if molecular_viscosity != 0.0
            else float("inf")
        ),
        "dt": time_step_size,
        "smagorinsky": float(numerics.get("turbulence", {}).get("smagorinsky_coefficient", 0.0)),
        "frequency": relaxation / time_step_size if time_step_size > 0.0 else np.nan,
        "capacity": int(numerics.get("max_n_particles", 0)),
    }


def _trim_to_last_monotone_segment(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """Keep the latest monotone segment after a restart."""
    if time_column not in df.columns or len(df) <= 1:
        return df
    times = df[time_column].to_numpy(float)
    last_restart = 0
    for index in range(1, len(times)):
        if np.isfinite(times[index]) and times[index] < times[index - 1]:
            last_restart = index
    if last_restart:
        df = df.iloc[last_restart:].reset_index(drop=True)
    return df


def _merge_backup_restarts(df: pd.DataFrame, case_dir: str | Path) -> pd.DataFrame:
    """Merge identical backup continuations while keeping their latest samples."""
    if "step" not in df.columns:
        return df
    return (
        df.sort_values("step", kind="stable")
        .drop_duplicates("step", keep="last")
        .reset_index(drop=True)
    )


def read_integrals(case_dir) -> pd.DataFrame | None:
    """Return the VPM flow-integral history."""
    path = _samples_dir(case_dir) / "flow_integrals.csv"
    if not path.is_file():
        return None
    diagnostics = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    if "time" not in diagnostics or "total_kinetic_energy" not in diagnostics:
        return None
    diagnostics = diagnostics.dropna(subset=["time", "total_kinetic_energy"])
    if diagnostics.empty:
        return None
    diagnostics = diagnostics.reset_index(drop=True)
    diagnostics = _merge_backup_restarts(diagnostics, case_dir)
    return _trim_to_last_monotone_segment(diagnostics, "time")


def read_metric(case_dir, column: str):
    """Return nondimensional time and one global diagnostic."""
    diagnostics = read_integrals(case_dir)
    if diagnostics is None or column not in diagnostics:
        return np.array([]), np.array([])
    diagnostics = diagnostics.dropna(subset=["time", column])
    if diagnostics.empty:
        return np.array([]), np.array([])
    time = diagnostics["time"].to_numpy(float) / REFERENCE_TIME
    return time, diagnostics[column].to_numpy(float)


def read_ring_diagnostics(case_dir) -> pd.DataFrame | None:
    """Return the grouped vortex-ring history."""
    path = _samples_dir(case_dir) / "ring_diagnostics.csv"
    if not path.is_file():
        return None
    diagnostics = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    required = ["time", "step", "group_id", "vortex_centroid_x", "major_radius"]
    if diagnostics.empty or not set(required).issubset(diagnostics.columns):
        return None
    diagnostics = diagnostics.dropna(subset=required)
    diagnostics = _merge_backup_restarts(diagnostics, case_dir)
    diagnostics = _trim_to_last_monotone_segment(diagnostics, "time")
    return (
        diagnostics.sort_values(["step", "group_id"], kind="stable")
        .drop_duplicates(["step", "group_id"], keep="last")
        .reset_index(drop=True)
    )


def save_fig(
    fig,
    path,
    dpi: int | None = None,
    figure_format: str = "png",
    tight_rect: tuple[float, float, float, float] | None = None,
) -> None:
    """Save a figure through the shared theme."""
    _theme().save_fig(
        fig,
        path,
        figure_format=figure_format,
        dpi=dpi,
        tight_rect=tight_rect,
        bbox_inches=None,
    )
