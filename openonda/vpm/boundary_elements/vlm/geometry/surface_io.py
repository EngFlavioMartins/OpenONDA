"""Read and write VLM surface geometry as JSON."""

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
                }
            )
        wings.append({"uid": wing.uid, "symmetry": wing.symmetry, "segments": segments})

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


def save_surface(aircraft: Aircraft, filepath: str) -> None:
    """Write the surface geometry and reference values of ``aircraft`` as JSON.

    Parameters
    ----------
    aircraft : Aircraft
        Surface to save.
    filepath : str
        Output file; missing parent directories are created.
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(surface_to_dict(aircraft), f, indent=2)


def load_surface(filepath: str) -> Aircraft:
    """Load a surface geometry written by :func:`save_surface`.

    Parameters
    ----------
    filepath : str
        Path to the surface JSON file.

    Returns
    -------
    Aircraft
        Aircraft with its wings, segments and reference values.

    Raises
    ------
    KeyError
        If the file lacks a key that :func:`save_surface` writes.
    """
    with open(filepath) as f:
        data = json.load(f)

    aircraft = Aircraft(uid=data["uid"])
    for wing_data in data["wings"]:
        wing = Wing(uid=wing_data["uid"], symmetry=wing_data["symmetry"])
        for seg_data in wing_data["segments"]:
            wing.add_segment(
                WingSegment(
                    uid=seg_data["uid"],
                    vertex_position={
                        k: np.array(v) for k, v in seg_data["vertex_position"].items()
                    },
                    n_chordwise_panels=seg_data["n_chordwise_panels"],
                    n_spanwise_panels=seg_data["n_spanwise_panels"],
                )
            )
        aircraft.add_wing(wing)

    refs = data["refs"]
    aircraft.refs = {
        "area": refs["area"],
        "chord": refs["chord"],
        "span": refs["span"],
        "geometry_centre": np.array(refs["geometry_centre"]),
        "reference_point": np.array(refs["reference_point"]),
    }
    return aircraft
