"""Shared case styles and recorded settings for interaction figures."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from ..setup import CASES

CASE_DIR = Path(__file__).resolve().parents[1]


def _theme():
    from openonda import plotting

    return plotting


def case_style(name):
    label, palette, marker = {
        "baseline": ("Baseline", "TUDdark", "o"),
        "selective_eddy_viscosity": ("Selective eddy viscosity", "VPMpurple", "s"),
        "pedrizzetti_relaxation": ("Pedrizzetti relaxation", "TUDcyan", "D"),
        "particle_splitting": ("Particle splitting", "AccentGreen", "^"),
    }[name]
    return {"label": label, "color": _theme().COLORS[palette], "marker": marker}


def figure_size(height_cm):
    """Keep thesis-width exports compact without scaling their text."""
    theme = _theme()
    return theme.MAX_FIGURE_WIDTH_CM * theme.CM, height_cm * theme.CM


def plot_style_metadata():
    theme = _theme()
    return {
        "palette_source": "Thesis/thesis.tex",
        "width_cm": theme.MAX_FIGURE_WIDTH_CM,
        "font_size_pt": theme.THESIS_FONT_SIZE_PT,
        "outer_y_text_padding_pt": theme.MIN_TEXT_CANVAS_PADDING_PT + 0.5,
        "plotting_sha256": hashlib.sha256(Path(theme.__file__).read_bytes()).hexdigest(),
        "postprocess_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": {name: case_style(name) for name in CASES},
    }


def comparison_legend(fig, handles, labels=None):
    """Two compact rows accommodate all four methods at thesis text size."""
    return fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
        borderaxespad=0,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.25,
    )


def save_figure(fig, stem, axes, formats=("png",), *, fit_margins=True):
    """Export at the fixed thesis size after checking its text and margins."""
    theme = _theme()
    axes = (axes,) if hasattr(axes, "get_position") else tuple(axes)
    if fit_margins:
        theme.fit_thesis_y_label_margins(fig, axes)
    theme.validate_thesis_figure(fig, axes)
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.parent / f"{stem.name}.{fmt}", dpi=theme.DEFAULT_DPI, bbox_inches=None)


def load_metadata(name):
    path = CASE_DIR / "solution" / name / "vpm_metadata.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def metadata_settings(metadata):
    configuration = metadata.get("configuration", {})
    numerics = configuration.get("numerics", {})
    rings = configuration.get("initial_conditions", [])
    first = rings[0] if rings else {}
    second = rings[1] if len(rings) > 1 else first
    distribution = first.get("distribution") or {}
    disturbance = first.get("disturbance") or {}
    viscous = numerics.get("viscous", {})
    turbulence = numerics.get("turbulence", {})
    stabilization = numerics.get("stabilization", {})
    circulation = float(first.get("circulation", np.nan))
    viscosity = float(first.get("kinematic_viscosity", np.nan))
    dt = float(numerics.get("time_step_size", np.nan))
    relaxation = float(stabilization.get("pedrizzetti_relaxation_factor", 0.0))
    return {
        "scenario": "collision"
        if circulation * float(second.get("circulation", circulation)) < 0
        else "leapfrog",
        "method": metadata.get("case_name"),
        "reynolds_number": abs(circulation) / viscosity if viscosity > 0 else np.nan,
        "amplitude": float(disturbance.get("amplitude", 0.0)),
        "spacing": float(distribution.get("spacing", np.nan)),
        "core_ratio": float(distribution.get("core_radius_ratio", np.nan)),
        "dt": dt,
        "integrator": numerics.get("integrator", {}).get("name"),
        "diffusion": viscous.get("scheme"),
        "flow_model": turbulence.get("model"),
        "smagorinsky": (
            float(turbulence.get("smagorinsky_coefficient", 0.0))
            if turbulence.get("model") == "LES_SMAGORINSKY"
            else 0.0
        ),
        "frequency": relaxation / dt if dt > 0 else np.nan,
        "capacity": int(numerics.get("max_n_particles", 0)),
        "initial_conditions": rings,
        "induction": numerics.get("induction", {}),
        "stabilization": stabilization,
    }
