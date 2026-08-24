from __future__ import annotations

import json
from pathlib import Path
import re

import h5py
import numpy as np

from _reference_util import sample_vtu
from _frames_util import vpm_velocity
from _plotutil import load_vpm_particles

CASE_DIR = Path(__file__).resolve().parents[1]
TIME = 3.0
STATIONS = (0.75, 1.4, 1.6, 2.5)
EPS = 0.026


def _frame(pvd: Path, time: float) -> tuple[float, Path]:
    matches = re.finditer(r'timestep="([^"]+)"\s+[^>]*file="([^"]+)"', pvd.read_text())
    frames = [(float(match.group(1)), pvd.parent / match.group(2)) for match in matches]
    return min(frames, key=lambda item: abs(item[0] - time))


def _particles(solution: Path, time: float) -> dict:
    snapshots = []
    for path in solution.glob("vpm_*.h5"):
        with h5py.File(path, "r") as handle:
            attrs = handle["solver"].attrs
            snapshots.append((float(attrs["time"]), path))
    path = min(snapshots, key=lambda item: abs(item[0] - time))[1]
    return load_vpm_particles(path)


def _hybrid_velocity(vtu: Path, particles: dict, points: np.ndarray, box: dict):
    lower = np.array([box["xmin"], box["ymin"], box["zmin"]])
    upper = np.array([box["xmax"], box["ymax"], box["zmax"]])
    inside = np.all((points >= lower) & (points <= upper), axis=1)
    velocity = np.empty((len(points), 3))
    velocity[inside] = sample_vtu(vtu, points[inside])["velocity"]
    velocity[~inside] = vpm_velocity(particles, points[~inside])
    return velocity, inside


def extract_hybrid(solution: Path, output: Path, time: float = TIME) -> None:
    actual_time, vtu = _frame(solution / "coupled_replacement_flow.pvd", time)
    particles = _particles(solution, actual_time)
    metadata = json.loads((solution / "run_metadata.json").read_text())
    box = metadata["fvm_solver"]["fvm_domain"]

    x = np.linspace(-3.0, 10.0, 261)
    y = np.linspace(-2.0, 2.0, 161)
    stream_points = np.vstack(
        (
            np.column_stack((x, np.full_like(x, EPS), np.full_like(x, EPS))),
            np.column_stack((x, np.full_like(x, 0.75), np.full_like(x, EPS))),
        )
    )
    cross_points = np.vstack(
        [
            np.column_stack((np.full_like(y, station), y, np.full_like(y, EPS)))
            for station in STATIONS
        ]
    )
    points = np.vstack((stream_points, cross_points))
    velocity, inside = _hybrid_velocity(vtu, particles, points, box)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        time=actual_time,
        x=x,
        y=y,
        stations=np.asarray(STATIONS),
        velocity=velocity,
        inside_fvm=inside,
    )


def extract_reference(solution: Path, output: Path, time: float = TIME) -> None:
    actual_time, vtu = _frame(solution / "reference_flow.pvd", time)
    x = np.linspace(-3.0, 10.0, 261)
    y = np.linspace(-2.0, 2.0, 161)
    points = np.vstack(
        (
            np.column_stack((x, np.full_like(x, EPS), np.full_like(x, EPS))),
            np.column_stack((x, np.full_like(x, 0.75), np.full_like(x, EPS))),
            *[
                np.column_stack((np.full_like(y, station), y, np.full_like(y, EPS)))
                for station in STATIONS
            ],
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        time=actual_time,
        x=x,
        y=y,
        stations=np.asarray(STATIONS),
        velocity=sample_vtu(vtu, points)["velocity"],
    )


if __name__ == "__main__":
    extract_hybrid(CASE_DIR / "solution", CASE_DIR / "postprocessed" / "current_profiles.npz")
