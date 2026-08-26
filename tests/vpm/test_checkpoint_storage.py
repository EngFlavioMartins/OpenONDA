"""VPM checkpoints stay compact, restartable, and readable by ParaView."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import h5py
import numpy as np
import pytest

from source.solvers.vpm import ViscousConfig, VPMSetup, VPMSolver


def _load_ring_metrics():
    path = Path(__file__).parents[2] / "tutorials/vpm/vortex_ring/assets/ring_metrics.py"
    spec = importlib.util.spec_from_file_location("vortex_ring_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup(*, store_velocity_gradient: bool) -> VPMSetup:
    return VPMSetup(
        time_step_size=0.01,
        compute_device="CPU",
        max_n_particles=64,
        domain_bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        write_precision="f16",
        checkpoint_store_velocity_gradient=store_velocity_gradient,
        verbose=False,
        viscous=ViscousConfig.cs(kinematic_viscosity=0.01, particle_spacing=0.2),
    )


def _solver(case_dir, *, store_velocity_gradient: bool) -> VPMSolver:
    with contextlib.redirect_stdout(io.StringIO()):
        return VPMSolver(
            _setup(store_velocity_gradient=store_velocity_gradient),
            case_dir=case_dir,
        )


@pytest.mark.parametrize("store_velocity_gradient", [True, False])
def test_vpm_checkpoint_precision_and_gradient_policy(tmp_path, store_velocity_gradient):
    solver = _solver(tmp_path / "writer", store_velocity_gradient=store_velocity_gradient)
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0], [0.11, 0.0, 0.0]], dtype=np.float32),
        velocity=np.zeros((2, 3), dtype=np.float32),
        vortex_strength=np.array([[0.0, 0.0, 0.01], [0.0, 0.02, 0.0]], dtype=np.float32),
        core_radius=np.array([0.05, 0.05], dtype=np.float32),
        particle_volume=np.array([0.008, 0.008], dtype=np.float32),
        kinematic_viscosity=np.array([0.01, 0.01], dtype=np.float32),
    )
    checkpoint = tmp_path / "checkpoint"
    with contextlib.redirect_stdout(io.StringIO()):
        solver.save_state(str(checkpoint))

    with h5py.File(f"{checkpoint}.h5", "r") as archive:
        particles = archive["particles"]
        assert particles["position"].dtype == np.float32
        assert particles["position"].compression == "gzip"
        assert particles["position"].shuffle
        assert ("velocity_gradient" in particles) is store_velocity_gradient
        assert "strain_rate" not in particles
        assert (
            bool(archive["solver"].attrs["checkpoint_store_velocity_gradient"])
            is store_velocity_gradient
        )

    xdmf = (tmp_path / "checkpoint.xdmf").read_text(encoding="utf-8")
    assert ('Name="velocity_gradient"' in xdmf) is store_velocity_gradient

    pv = pytest.importorskip("pyvista")
    visual = pv.read(tmp_path / "checkpoint.xdmf")
    assert "velocity" in visual.point_data
    assert ("velocity_gradient" in visual.point_data) is store_velocity_gradient

    restored = _solver(tmp_path / "reader", store_velocity_gradient=store_velocity_gradient)
    with contextlib.redirect_stdout(io.StringIO()):
        restored.load_numerical_state(str(checkpoint))

    np.testing.assert_allclose(
        restored.particle_position,
        solver.particle_position,
        rtol=0.0,
        atol=4.0e-5,
    )
    assert np.isfinite(restored.particles.velocity_gradient_cpu()).all()

    if not store_velocity_gradient:
        ring_data = _load_ring_metrics().load_ring_data([f"{checkpoint}.h5"])
        assert len(ring_data[0]) == 1


def test_vpm_checkpoint_storage_policy_survives_setup_serialization():
    setup = _setup(store_velocity_gradient=False)
    restored = VPMSetup.from_dict(setup.to_dict())

    assert restored.write_precision == "f16"
    assert not restored.checkpoint_store_velocity_gradient


def test_empty_vpm_checkpoint_is_still_paraview_readable(tmp_path):
    solver = _solver(tmp_path / "writer", store_velocity_gradient=False)
    checkpoint = tmp_path / "empty"
    with contextlib.redirect_stdout(io.StringIO()):
        solver.save_state(str(checkpoint))

    pv = pytest.importorskip("pyvista")
    visual = pv.read(tmp_path / "empty.xdmf")

    assert visual.n_points == 0
    assert {"velocity", "vortex_strength", "vorticity"} <= set(visual.point_data)
