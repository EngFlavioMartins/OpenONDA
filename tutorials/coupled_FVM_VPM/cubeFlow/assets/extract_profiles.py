from __future__ import annotations

import json
from pathlib import Path
import re

import h5py
import numpy as np

from _reference_util import sample_vtu
from source.coupler.core.helpers.interior_bs import gaussian_bs_velocity_treecode

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
            snapshots.append((float(handle["solver"].attrs["flow_time"]), path))
    path = min(snapshots, key=lambda item: abs(item[0] - time))[1]
    with h5py.File(path, "r") as handle:
        count = int(handle["solver"].attrs["number_of_particles"])
        return {
            name: np.asarray(handle[f"particles/{name}"][:count])
            for name in ("position", "circulation", "radius")
        }


def _hybrid_velocity(vtu: Path, particles: dict, points: np.ndarray, box: dict):
    lower = np.array([box["xmin"], box["ymin"], box["zmin"]])
    upper = np.array([box["xmax"], box["ymax"], box["zmax"]])
    inside = np.all((points >= lower) & (points <= upper), axis=1)
    velocity = np.empty((len(points), 3))
    velocity[inside] = sample_vtu(vtu, points[inside])["U"]
    velocity[~inside] = gaussian_bs_velocity_treecode(
        points[~inside],
        particles["position"],
        particles["circulation"],
        particles["radius"],
        theta=0.3,
    ) + np.array([1.0, 0.0, 0.0])
    return velocity, inside


def extract_hybrid(solution: Path, output: Path, time: float = TIME) -> None:
    actual_time, vtu = _frame(solution / "coupled_hybridFlow.pvd", time)
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
    actual_time, vtu = _frame(solution / "referenceFlow.pvd", time)
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
        velocity=sample_vtu(vtu, points)["U"],
    )


if __name__ == "__main__":
    extract_hybrid(CASE_DIR / "solution", CASE_DIR / "postprocessed" / "current_profiles.npz")
