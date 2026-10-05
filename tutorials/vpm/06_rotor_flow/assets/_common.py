"""Plotting inputs from native solver metadata, geometry and sampled rotor motion."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from openonda import plotting as theme
from openonda.results import read_csv_table, read_json
from source.solvers.vpm.io.postprocess import accepted_history

from ..setup import ROTOR_RADIUS, WAKE_PLANE_DIAMETERS

ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
FIGURES_DIR = CASE_DIR / "figures"
SOLUTION_DIR = CASE_DIR / "solution"
OPERATING_WINDOW_REVOLUTIONS = 5
FIELD_STATIONARITY_COMPARISON_REVOLUTIONS = 5
REQUIRED_PLANE_NAMES = tuple((f"wake_{distance}D" for distance in WAKE_PLANE_DIAMETERS))


def blade_geometry(vlm):
    """Read the actual first blade's radial chord/pitch schedule, without regenerating it."""
    geometry = theme.read_vlm_surface(vlm["surfaces"][0])
    rows = []
    for wing in geometry["wings"]:
        for segment in wing["segments"]:
            a, b, c, d = (np.array(segment["vertex_position"][key]) for key in "abcd")
            q_root, q_tip = (0.75 * a + 0.25 * d, 0.75 * b + 0.25 * c)
            r_root, r_tip = (np.linalg.norm(q_root[1:]), np.linalg.norm(q_tip[1:]))
            point = 0.5 * (q_root + q_tip)
            chord_vector = 0.5 * (c + d - a - b)
            tangent = np.cross([1.0, 0.0, 0.0], point)
            tangent /= np.linalg.norm(tangent)
            pitch = np.arctan2(chord_vector[0], chord_vector @ tangent)
            rows.append(
                {
                    "radial_position": 0.5 * (r_root + r_tip),
                    "width": r_tip - r_root,
                    "chord": np.linalg.norm(chord_vector),
                    "pitch": pitch,
                    "inner_radius": r_root,
                    "outer_radius": r_tip,
                }
            )
    return pd.DataFrame(rows).sort_values("radial_position")


def rotor_inputs():
    """Load recorded inputs only when a plot runs; --help needs no saved results."""
    metadata = read_json(SOLUTION_DIR / "vpm_metadata.json")
    configuration = metadata["configuration"]
    vlm = configuration["numerics"]["vlm"]
    blade = blade_geometry(vlm)
    omega = abs(float(vlm["surfaces"][0]["kinematics"]["angular_speed"]))
    speed = float(np.linalg.norm(configuration["numerics"]["freestream_velocity"]))
    radius = float(blade.outer_radius.max())
    return SimpleNamespace(
        metadata=metadata,
        solution_dir=SOLUTION_DIR,
        blade=blade,
        density=vlm["density"],
        n_blades=len(vlm["surfaces"]),
        samples_dir=CASE_DIR / "samples" / configuration["samplers"]["directory"],
        freestream_speed=speed,
        angular_velocity=omega,
        rotor_radius=radius,
        station_radius=ROTOR_RADIUS,
        station_diameter=2.0 * ROTOR_RADIUS,
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
    data = accepted_history(
        pd.DataFrame(read_csv_table(p.samples_dir / "vlm_forces.csv")), p.metadata
    )
    reference = 0.5 * p.density * p.freestream_speed**2 * np.pi * p.rotor_radius**2
    data["CT"] = data.force_x / reference
    data["CP"] = data.rotational_power / (reference * p.freestream_speed)
    data["nominal_revolutions"] = data.time / p.rotation_period
    return data


def read_operating_point(
    *, revolutions=OPERATING_WINDOW_REVOLUTIONS, window_start=None, window_end=None
):
    data = performance()
    if window_start is not None or window_end is not None:
        from openonda.validation import time_mean

        return tuple(time_mean(data.time, data[["CT", "CP"]], window_start, window_end))
    end = float(data.time.max())
    return read_operating_point(
        window_start=end - revolutions * rotor_inputs().rotation_period, window_end=end
    )


def read_time_step():
    return rotor_inputs().time_step_size


def load_theme():
    theme.set_thesis_style()
    return (dict(theme.COLORS), theme)


def rotor_subplots(nrows, *, height_cm, sharex=False):
    """Compact rotor panels using the authored OpenONDA Matplotlib style."""
    import matplotlib.pyplot as plt

    theme.set_thesis_style()
    fig, axes = plt.subplots(
        nrows,
        1,
        squeeze=False,
        sharex=sharex,
        figsize=(theme.MAX_FIGURE_WIDTH_CM * theme.CM, height_cm * theme.CM),
    )
    theme.centered_subplots_adjust(
        fig,
        outer=0.16,
        bottom=1.3 / height_cm,
        top=1 - 0.25 / height_cm,
        hspace=0.1 if sharex else 0.38,
    )
    return (fig, axes[:, 0])


def save_rotor_figure(fig, path, *, figure_format="both", dpi=None):
    outer = {
        "rotor_performance": 0.1,
        "rotor_loading_validation": 0.129,
        "rotor_wake_planes": 0.113,
        "rotor_streamwise": 0.139,
    }[Path(path).stem]
    top_padding_cm = 0.17 if Path(path).stem == "rotor_loading_validation" else 0.11
    theme.centered_subplots_adjust(
        fig, outer=outer, top=1 - top_padding_cm / (fig.get_figheight() / theme.CM)
    )
    return theme.export_figure(fig, path, figure_format=figure_format, dpi=dpi)


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--format", choices=theme.FORMAT_CHOICES, default="both")
    parser.add_argument("--dpi", type=int, default=theme.DEFAULT_DPI)
    return parser


def build_rotor_style_map(colors):
    return {name: dict(style) for name, style in theme.ROTOR_STYLE.items()}
