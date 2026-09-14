"""Shared case styles and recorded settings for interaction figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
CASES = ("baseline", "stretching_viscosity", "p_moments")


def _theme():
    from openonda import plotting

    return plotting


def case_style(name):
    label, palette, marker = {
        "baseline": ("Baseline", "black", "o"),
        "stretching_viscosity": ("Stretching viscosity", "purple", "s"),
        "p_moments": ("Moment-preserving relaxation", "yellow", "D"),
    }[name]
    colors = {"black": "#000000", "purple": "#5C3D9B", "yellow": "#B08A00"}
    return {"label": label, "color": colors[palette], "marker": marker}


def save_figure(fig, stem, axes, formats=("pdf", "png")):
    """Export at the fixed thesis size after checking its text and margins."""
    theme = _theme()
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
    }
