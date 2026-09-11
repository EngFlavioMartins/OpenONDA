"""Read solver force samples and derive the flat-plate comparison coordinates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from openonda.plotting import read_vlm_surface

from .kinematics import distance_travelled


def parameters(case_dir: Path, name: str | None = None) -> dict:
    """Read the recorded VLM motion and its actual input surface geometry."""
    record = (
        case_dir / "solution" / name / "vpm_metadata.json"
        if name is not None
        else next(iter(sorted((case_dir / "solution").glob("*/vpm_metadata.json"))))
    )
    metadata = json.loads(record.read_text())
    vlm = metadata["configuration"]["numerics"]["vlm"]
    surface = vlm["surfaces"][0]
    refs = read_vlm_surface(surface, case_dir / "assets" / "surfaces")["refs"]
    motion = surface["kinematics"]
    return {
        "status": metadata["lifecycle"]["status"],
        "chord": float(refs["chord"]),
        "span": float(refs["span"]),
        "density": float(vlm.get("density", 1.225)),
        "speed": float(np.linalg.norm(vlm["freestream_velocity"])),
        "reference_velocity": np.asarray(vlm["freestream_velocity"], dtype=float),
        "kinematics": "ramp" if motion["type"] == "SmoothRampVLM" else "static",
        "ramp_time": float(motion["acceleration_time"])
        if motion["type"] == "SmoothRampVLM"
        else 0.0,
        "start_time": float(motion.get("start_time", 0.0)),
    }


def load_forces(case_dir: Path, name: str) -> pd.DataFrame | None:
    """Derive travel from recorded motion without rewriting the solver's CSV."""
    path = case_dir / "samples" / name / "vlm_forces.csv"
    if not path.is_file():
        print(f"  [MISSING] {path}")
        return None
    data = pd.read_csv(path)
    physics = parameters(case_dir, name)
    if physics["status"] != "completed":
        print(f"  [INCOMPLETE] {name}: {physics['status']}")
        return None
    times = np.maximum(data["time"].to_numpy() - physics["start_time"], 0.0)
    data["nondimensional_distance_travelled"] = distance_travelled(
        times, physics["kinematics"], physics["ramp_time"], physics["speed"], physics["chord"]
    )
    return data
