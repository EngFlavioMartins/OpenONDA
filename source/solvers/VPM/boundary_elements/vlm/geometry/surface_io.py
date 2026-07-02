"""
Read/write VLM surface geometry to and from disk.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import json

import numpy as np

from .aircraft import Aircraft, Wing, WingSegment

def save_surface(aircraft: "Aircraft", filepath: str) -> str:
    """
    Save aircraft surface geometry to JSON file.

    Args:
        aircraft: Aircraft object to save
        filepath: Output filepath (without extension, .json will be added)

    Returns:
        str: Path to saved file
    """
    data = {"uid": aircraft.uid, "wings": []}

    for wing in aircraft.wings.values():
        wing_data = {"uid": wing.uid, "symmetry": wing.symmetry, "segments": []}

        for segment in wing.segments.values():
            seg_data = {
                "uid": segment.uid,
                "vertices": {k: v.tolist() for k, v in segment.vertices.items()},
                "panels_chord": segment.panels_chord,
                "panels_span": segment.panels_span,
                "airfoils": segment.airfoils,
            }
            wing_data["segments"].append(seg_data)

        data["wings"].append(wing_data)

    # Add reference values
    refs = aircraft.refs
    data["refs"] = {
        "area": refs.get("area", 1.0),
        "chord": refs.get("chord", 1.0),
        "span": refs.get("span", 1.0),
        "gcenter": refs.get("gcenter", np.zeros(3)).tolist()
        if hasattr(refs.get("gcenter", np.zeros(3)), "tolist")
        else list(refs.get("gcenter", [0, 0, 0])),
        "rcenter": refs.get("rcenter", np.zeros(3)).tolist()
        if hasattr(refs.get("rcenter", np.zeros(3)), "tolist")
        else list(refs.get("rcenter", [0, 0, 0])),
    }

    output_path = filepath if filepath.endswith(".json") else f"{filepath}.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path

def load_surface(filepath: str) -> "Aircraft":
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
            vertices = {k: np.array(v) for k, v in seg_data["vertices"].items()}
            segment = WingSegment(
                uid=seg_data["uid"],
                vertices=vertices,
                panels_chord=seg_data["panels_chord"],
                panels_span=seg_data["panels_span"],
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
            "gcenter": np.array(data["refs"].get("gcenter", [0.0, 0.0, 0.0])),
            "rcenter": np.array(data["refs"].get("rcenter", [0.0, 0.0, 0.0])),
        }
    else:
        aircraft.compute_default_refs()

    return aircraft
