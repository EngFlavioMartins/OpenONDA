"""The merger scene must use the original cloud after solver resumes."""

import json

import h5py
import numpy as np
import pytest

from tests._tutorial_helpers import load_tutorial_module


@pytest.fixture
def scene():
    return load_tutorial_module("vpm/lamb_oseen_vortex", "assets.plot_merging_snapshots")


def write_initial(root, *, step=0, time=0.0, count=2):
    path = root / "solution/merging_gbd/vpm/vpm_000000.h5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as state:
        solver = state.create_group("solver")
        solver.attrs.update(step=step, time=time, n_particles_total=count)
        particles = state.create_group("particles")
        particles["position"] = np.arange(6, dtype=np.float32).reshape(2, 3)
        particles["vortex_strength"] = np.array([[0, 0, 1], [0, 0, 2]], dtype=np.float32)
        particles["core_radius"] = np.array([0.1, 0.2], dtype=np.float32)
    return path


def test_completed_resume_keeps_original_particle_cloud(scene, tmp_path):
    initial = write_initial(tmp_path)
    metadata = initial.parents[1] / "vpm_metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "lifecycle": {"status": "completed"},
                "state": {"initial_step": 927, "initial_n_particles_total": 0},
            }
        )
    )
    position, strength, radius = scene._initial_particle_state(tmp_path)
    with h5py.File(initial) as state:
        np.testing.assert_array_equal(position, state["particles/position"][:])
        np.testing.assert_array_equal(strength, state["particles/vortex_strength"][:])
        np.testing.assert_array_equal(radius, state["particles/core_radius"][:])


def test_missing_initial_backup_rejected(scene, tmp_path):
    with pytest.raises(FileNotFoundError, match="vpm_000000"):
        scene._initial_particle_state(tmp_path)
