"""Raw force histories stay immutable while plots use recorded motion."""

import json

import numpy as np
import pytest

import openonda.vpm as vpm
from source.solvers.vpm.io.manifest import _manifest_value
from tests._tutorial_helpers import load_tutorial_module

load_forces = load_tutorial_module("vpm/flat_plate", "assets.results").load_forces


@pytest.mark.parametrize("moving", [False, True])
def test_travel_uses_recorded_motion_and_geometry_without_rewriting_samples(tmp_path, moving):
    name = "edited_experiment"
    samples = tmp_path / "samples" / name
    solution = tmp_path / "solution" / name
    geometry = tmp_path / "assets/surfaces"
    for directory in (samples, solution, geometry):
        directory.mkdir(parents=True)
    raw = samples / "vlm_forces.csv"
    contents = "time,lift_coefficient,drag_coefficient\n0,0,0\n0.5,1,0.1\n1,1,0.1\n"
    raw.write_text(contents)
    (tmp_path / "setup.py").write_text("raise RuntimeError('Do not import changed inputs')\n")
    (geometry / f"{name}.json").write_text(json.dumps({"refs": {"chord": 2, "span": 8}}))
    motion = (
        vpm.SmoothRampVLM(final_velocity=[-4, 0, 0], acceleration_time=0.5)
        if moving
        else vpm.StaticVLM()
    )
    metadata = {
        "lifecycle": {"status": "completed"},
        "configuration": {
            "numerics": {
                "vlm": {
                    "freestream_velocity": [4, 0, 0],
                    "surfaces": [
                        {
                            "surface": f"/old/moved/case/assets/surfaces/{name}.json",
                            "kinematics": _manifest_value(motion),
                        }
                    ],
                }
            }
        },
    }
    (solution / "vpm_metadata.json").write_text(json.dumps(metadata))

    data = load_forces(tmp_path, name)

    np.testing.assert_allclose(
        data["nondimensional_distance_travelled"], [0, 0.5, 1.5] if moving else [0, 1, 2]
    )
    assert raw.read_text() == contents
    assert list(samples.iterdir()) == [raw]
