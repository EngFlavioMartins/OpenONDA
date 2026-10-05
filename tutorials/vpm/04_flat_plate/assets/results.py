"""Read solver force samples and derive the flat-plate comparison coordinates."""

from pathlib import Path

import numpy as np
import pandas as pd

from openonda.plotting import read_vlm_surface
from openonda.results import read_csv_table, read_json

from .kinematics import distance_travelled


def parameters(case_dir: Path, name: str | None = None) -> dict:
    """Read the recorded VLM motion and its actual input surface geometry."""
    record = (
        case_dir / "solution" / name / "vpm_metadata.json"
        if name is not None
        else next(iter(sorted((case_dir / "solution").glob("*/vpm_metadata.json"))))
    )
    metadata = read_json(record)
    vlm = metadata["configuration"]["numerics"]["vlm"]
    surface = vlm["surfaces"][0]
    refs = read_vlm_surface(surface)["refs"]
    motion = surface["kinematics"]
    return {
        "status": metadata["lifecycle"]["status"],
        "chord": float(refs["chord"]),
        "span": float(refs["span"]),
        "density": float(vlm["density"]),
        "speed": float(np.linalg.norm(vlm["freestream_velocity"])),
        "reference_velocity": np.asarray(vlm["freestream_velocity"], dtype=float),
        "kinematics": "ramp" if motion["type"] == "SmoothRampVLM" else "static",
        "ramp_time": float(motion["acceleration_time"])
        if motion["type"] == "SmoothRampVLM"
        else 0.0,
        "start_time": float(motion["start_time"]) if motion["type"] == "SmoothRampVLM" else 0.0,
    }


def load_forces(case_dir: Path, name: str) -> pd.DataFrame | None:
    """Derive travel from recorded motion without rewriting the solver's CSV."""
    path = case_dir / "samples" / name / "vlm_forces.csv"
    data = pd.DataFrame(read_csv_table(path))
    physics = parameters(case_dir, name)
    times = np.maximum(data["time"].to_numpy() - physics["start_time"], 0.0)
    data["nondimensional_distance_travelled"] = distance_travelled(
        times, physics["kinematics"], physics["ramp_time"], physics["speed"], physics["chord"]
    )
    return data


def settled_coefficients(case_dir, name, chord_lengths=5):
    """Mean final cruise loads with exact, fully bracketed travel boundaries."""
    from openonda.validation import time_mean

    data = load_forces(case_dir, name)
    distance = data.nondimensional_distance_travelled.to_numpy()
    end = float(distance[-1])
    columns = ["lift_coefficient", "drag_coefficient", "pitching_moment_coefficient_quarter_chord"]
    return time_mean(distance, data[columns].to_numpy(), end - chord_lengths, end)
