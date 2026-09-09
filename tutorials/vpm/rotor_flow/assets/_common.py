"""Plotting inputs from native solver metadata, geometry and sampled rotor motion."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from openonda import plotting as theme

ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
FIGURES_DIR = CASE_DIR / "figures"
SOLUTION_DIR = CASE_DIR / "solution"


def blade_geometry(vlm):
    """Read the actual first blade's radial chord/pitch schedule, without regenerating it."""
    geometry = theme.read_vlm_surface(vlm["surfaces"][0], ASSETS_DIR)
    rows = []
    # The tutorial's first blade is unrotated; its rotor axis is global x.
    for wing in geometry["wings"]:
        for segment in wing["segments"]:
            a, b, c, d = (np.array(segment["vertex_position"][key]) for key in "abcd")
            q_root, q_tip = 0.75 * a + 0.25 * d, 0.75 * b + 0.25 * c
            r_root, r_tip = np.linalg.norm(q_root[1:]), np.linalg.norm(q_tip[1:])
            point = 0.5 * (q_root + q_tip)
            chord_vector = 0.5 * (c + d - a - b)
            tangent = np.cross([1.0, 0.0, 0.0], point)
            tangent /= np.linalg.norm(tangent)
            pitch = np.arctan2(chord_vector[0], chord_vector @ tangent)
            rows.append(
                dict(
                    radial_position=0.5 * (r_root + r_tip),
                    width=r_tip - r_root,
                    chord=np.linalg.norm(chord_vector),
                    pitch=pitch,
                    inner_radius=r_root,
                    outer_radius=r_tip,
                )
            )
    return pd.DataFrame(rows).sort_values("radial_position")


@lru_cache(maxsize=1)
def rotor_inputs():
    """Load recorded inputs only when a plot runs; --help needs no saved results."""
    metadata = json.loads((SOLUTION_DIR / "vpm_metadata.json").read_text())
    configuration = metadata["configuration"]
    vlm = configuration["numerics"]["vlm"]
    blade = blade_geometry(vlm)
    omega = abs(float(vlm["surfaces"][0]["kinematics"]["angular_speed"]))
    speed = float(np.linalg.norm(configuration["numerics"]["freestream_velocity"]))
    radius = float(blade.outer_radius.max())
    return SimpleNamespace(
        metadata=metadata,
        blade=blade,
        density=vlm["density"],
        n_blades=len(vlm["surfaces"]),
        samples_dir=CASE_DIR / "samples" / configuration["samplers"]["directory"],
        freestream_speed=speed,
        angular_velocity=omega,
        rotor_radius=radius,
        hub_radius=float(blade.inner_radius.min()),
        rotation_period=2 * np.pi / omega,
        tip_speed_ratio=omega * radius / speed,
        time_step_size=configuration["numerics"]["time_step_size"],
    )


def bem_reference():
    from openonda.rotor_theory import solve_blade_element_momentum

    p = rotor_inputs()
    return solve_blade_element_momentum(
        p.blade.radial_position,
        p.blade.chord,
        p.blade.pitch,
        p.n_blades,
        p.rotor_radius,
        p.freestream_speed,
        p.angular_velocity,
        hub_radius=p.hub_radius,
        radial_widths=p.blade.width,
        density=p.density,
    )


def performance():
    """Actual shaft power includes the prescribed ramp and all individual blades."""
    p = rotor_inputs()
    data = pd.read_csv(p.samples_dir / "vlm_forces.csv")
    reference = 0.5 * p.density * p.freestream_speed**2 * np.pi * p.rotor_radius**2
    data["CT"] = data.force_x / reference
    data["CP"] = data.rotational_power / (reference * p.freestream_speed)
    data["nominal_revolutions"] = data.time / p.rotation_period
    return data


def read_operating_point(*, revolutions=6):
    data = performance()
    tail = data[data.time > data.time.max() - revolutions * rotor_inputs().rotation_period]
    return float(tail.CT.mean()), float(tail.CP.mean())


def read_time_step():
    return rotor_inputs().time_step_size


def load_theme():
    theme.set_thesis_style()
    return dict(theme.COLORS), theme


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--format", choices=theme.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=theme.DEFAULT_DPI)
    return parser


def build_rotor_style_map(colors):
    return {name: dict(style) for name, style in theme.ROTOR_STYLE.items()}
