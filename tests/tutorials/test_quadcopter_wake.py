"""Native plane collections must demonstrate field convergence, not just exist."""

from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pytest
import pyvista as pv

from tests._tutorial_helpers import load_tutorial_module

_validate_results = load_tutorial_module("vpm/quadcopter", "assets.validate_results")
validate_impulse = _validate_results.validate_impulse
validate_wake = _validate_results.validate_wake


def _published_planes(directory, defect=None):
    times = np.arange(1, 49) / 8
    if defect == "stale":
        times = times[:-3]
    elif defect == "gap":
        times = np.delete(times, 20)
    elif defect == "duplicate":
        times[20] = times[19]
    points = np.array([[-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0]], dtype=float)
    samplers = []
    for index, height in enumerate((-0.35, -0.7)):
        name = "sampled_zplane" + ("_deep" if index else "")
        collection = ElementTree.Element("VTKFile", type="Collection")
        frames = ElementTree.SubElement(collection, "Collection")
        for step, time in enumerate(times, start=1):
            grid = pv.PolyData(points + [0, 0, height])
            velocity = np.tile([0.0, 0.0, -101.0 + 0.1 * np.sin(2 * np.pi * time)], (4, 1))
            if defect == "cancelling_drift" and time > 3:
                # Spatial means still agree; a large background flow must not hide this change.
                velocity[:, 0] = [0.2, -0.2, 0.2, -0.2]
            if defect == "nonfinite" and step == len(times):
                velocity[0, 0] = np.nan
            grid["velocity"] = velocity
            filename = f"{name}_{step:06d}.vtp"
            grid.save(directory / filename)
            ElementTree.SubElement(frames, "DataSet", timestep=str(time), file=filename)
        ElementTree.ElementTree(collection).write(directory / f"{name}.pvd")
        samplers.append({"type": "SurfaceSampler", "file_name": name, "schedule": {"interval": 1}})
    return SimpleNamespace(
        period=1.0,
        samples_dir=directory,
        metadata={
            "configuration": {
                "numerics": {"time_step_size": 0.125, "freestream_velocity": [0.0, 0.0, -100.0]},
                "samplers": {"items": samplers},
            }
        },
    )


def test_complete_periodic_native_velocity_window_passes(tmp_path):
    parameters = _published_planes(tmp_path)
    assert validate_wake(parameters, end=6.0) == []


@pytest.mark.parametrize(
    "defect, message",
    [
        ("stale", "incomplete"),
        ("gap", "incomplete"),
        ("duplicate", "unordered"),
        ("nonfinite", "non-finite"),
        ("cancelling_drift", "drift exceeds"),
    ],
)
def test_plane_files_alone_cannot_qualify_a_wake(tmp_path, defect, message):
    parameters = _published_planes(tmp_path, defect)
    assert any(message in failure for failure in validate_wake(parameters, end=6.0))


@pytest.mark.parametrize("ratio", [1.0, 2.0])
def test_impulse_check_uses_native_coupled_moment_and_dense_force_integral(tmp_path, ratio):
    times = np.linspace(0, 6, 49)
    density = 2.0
    # An oscillating force tests integration over complete periods. Wake-only
    # impulse would predict 50% too much thrust; the bound term closes the budget.
    thrust = 4 + np.sin(2 * np.pi * times)
    pd.DataFrame(
        {
            "time": times,
            "linear_impulse_z": -3 * times,
            "coupled_linear_impulse_z": -ratio
            * (4 * times + (1 - np.cos(2 * np.pi * times)) / (2 * np.pi))
            / density,
        }
    ).to_csv(tmp_path / "flow_integrals.csv", index=False)
    p = SimpleNamespace(samples_dir=tmp_path, density=density, period=1.0)
    forces = pd.DataFrame({"time": times, "thrust": thrust})
    failures = validate_impulse(p, 6.0, forces)
    assert bool(failures) == (ratio != 1.0)
