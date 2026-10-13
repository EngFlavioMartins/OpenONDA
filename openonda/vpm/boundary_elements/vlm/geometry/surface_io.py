"""
Read/write VLM surface geometry to and from disk.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from .aircraft import Aircraft, Wing, WingSegment


def surface_to_dict(aircraft: Aircraft) -> dict[str, Any]:
    """Return the input geometry and reference values as a portable JSON mapping."""
    wings: list[dict[str, Any]] = []
    for wing in aircraft.wings.values():
        segments: list[dict[str, Any]] = []
        for segment in wing.segments.values():
            segments.append(
                {
                    "uid": segment.uid,
                    "vertex_position": {k: v.tolist() for k, v in segment.vertex_position.items()},
                    "n_chordwise_panels": segment.n_chordwise_panels,
                    "n_spanwise_panels": segment.n_spanwise_panels,
                    "airfoils": segment.airfoils,
                }
            )
        wings.append({"uid": wing.uid, "symmetry": wing.symmetry, "segments": segments})

    # Add reference values
    refs = aircraft.refs
    return {
        "uid": aircraft.uid,
        "wings": wings,
        "refs": {
            "area": refs.get("area", 1.0),
            "chord": refs.get("chord", 1.0),
            "span": refs.get("span", 1.0),
            "geometry_centre": np.asarray(refs.get("geometry_centre", np.zeros(3))).tolist(),
            "reference_point": np.asarray(refs.get("reference_point", np.zeros(3))).tolist(),
        },
    }


def save_surface(aircraft: Aircraft, filepath: str) -> str:
    """
    Save aircraft surface geometry to JSON file.

    Args:
        aircraft: Aircraft object to save
        filepath: Output filepath (without extension, .json will be added)

    Returns:
        str: Path to saved file
    """
    data = surface_to_dict(aircraft)

    output_path = filepath if filepath.endswith(".json") else f"{filepath}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def load_surface(filepath: str) -> Aircraft:
    """Load an aircraft surface geometry from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the surface JSON file (``.json`` extension is appended
        automatically if not present).

    Returns
    -------
    Aircraft
        Loaded aircraft object with wings, segments, and reference values.

    Examples
    --------
    >>> aircraft = load_surface("aircraft.json")
    >>> aircraft.uid
    'my_aircraft'
    """
    input_path = filepath if filepath.endswith(".json") else f"{filepath}.json"

    with open(input_path) as f:
        data = json.load(f)

    aircraft = Aircraft(uid=data["uid"])

    for wing_data in data["wings"]:
        wing = Wing(uid=wing_data["uid"], symmetry=wing_data.get("symmetry", 1))

        for seg_data in wing_data["segments"]:
            vertex_position = {k: np.array(v) for k, v in seg_data["vertex_position"].items()}
            segment = WingSegment(
                uid=seg_data["uid"],
                vertex_position=vertex_position,
                n_chordwise_panels=seg_data["n_chordwise_panels"],
                n_spanwise_panels=seg_data["n_spanwise_panels"],
                airfoils=seg_data.get("airfoils", {"inner": "flat", "outer": "flat"}),
            )
            wing.add_segment(segment)

        aircraft.add_wing(wing)

    # Restore reference values if present
    if "refs" in data:
        aircraft.refs = {
            "area": data["refs"].get("area", 1.0),
            "chord": data["refs"].get("chord", 1.0),
            "span": data["refs"].get("span", 1.0),
            "geometry_centre": np.array(data["refs"].get("geometry_centre", [0.0, 0.0, 0.0])),
            "reference_point": np.array(data["refs"].get("reference_point", [0.0, 0.0, 0.0])),
        }
    else:
        aircraft.compute_default_refs()

    return aircraft
