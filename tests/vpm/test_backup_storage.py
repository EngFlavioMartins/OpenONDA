"""VPM backups stay compact, restartable, and readable by ParaView."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import h5py
import numpy as np
import pytest

from source.solvers.vpm import Backup, ViscousConfig, VPMSetup, VPMSolver


def _load_ring_metrics():
    path = Path(__file__).parents[2] / "tutorials/vpm/vortex_ring/assets/ring_metrics.py"
    spec = importlib.util.spec_from_file_location("vortex_ring_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup(output_directory: Path) -> VPMSetup:
    return VPMSetup(
        time_step_size=0.01,
        compute_device="CPU",
        max_n_particles=64,
        domain_bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        write_precision="f16",
        backup=Backup(
            interval_steps=0,
            directory=str(output_directory),
            log_directory=str(output_directory),
        ),
        verbose=False,
        viscous=ViscousConfig.cs(kinematic_viscosity=0.01, particle_spacing=0.2),
    )


def _solver(case_dir) -> VPMSolver:
    with contextlib.redirect_stdout(io.StringIO()):
        return VPMSolver(
            _setup(case_dir / "solution"),
            case_dir=case_dir,
        )


def test_vpm_backup_has_one_fixed_restart_schema(tmp_path):
    solver = _solver(tmp_path / "writer")
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0], [0.11, 0.0, 0.0]], dtype=np.float32),
        velocity=np.zeros((2, 3), dtype=np.float32),
        vortex_strength=np.array([[0.0, 0.0, 0.01], [0.0, 0.02, 0.0]], dtype=np.float32),
        core_radius=np.array([0.05, 0.05], dtype=np.float32),
        particle_volume=np.array([0.008, 0.008], dtype=np.float32),
        kinematic_viscosity=np.array([0.01, 0.01], dtype=np.float32),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver.save_backup()
    backup = tmp_path / "writer" / "solution" / "vpm_000000"

    with h5py.File(f"{backup}.h5", "r") as archive:
        particles = archive["particles"]
        assert particles["position"].dtype == np.float32
        assert particles["position"].compression == "gzip"
        assert particles["position"].shuffle
        assert "velocity_gradient" not in particles
        assert "strain_rate" not in particles
        assert "backup_store_velocity_gradient" not in archive["solver"].attrs
        assert archive["solver"].attrs["backup_format_version"] == "9.0"

    xdmf = Path(f"{backup}.xdmf").read_text(encoding="utf-8")
    assert 'Name="velocity_gradient"' not in xdmf

    pv = pytest.importorskip("pyvista")
    visual = pv.read(f"{backup}.xdmf")
    assert "velocity" in visual.point_data
    assert "velocity_gradient" not in visual.point_data

    restored = _solver(tmp_path / "reader")
    with contextlib.redirect_stdout(io.StringIO()):
        restored.load_backup(str(backup))

    np.testing.assert_allclose(
        restored.particle_position,
        solver.particle_position,
        rtol=0.0,
        atol=4.0e-5,
    )
    assert np.isfinite(restored.particles.velocity_gradient_cpu()).all()

    ring_data = _load_ring_metrics().load_ring_data([f"{backup}.h5"])
    assert len(ring_data[0]) == 1


def test_vpm_backup_configuration_survives_setup_serialization(tmp_path):
    setup = _setup(tmp_path / "custom-output")
    restored = VPMSetup.from_dict(setup.to_dict())

    assert restored.write_precision == "f16"
    assert restored.backup == setup.backup
    assert setup.to_dict()["backup"] == {
        "interval_steps": 0,
        "directory": str(tmp_path / "custom-output"),
        "log_directory": str(tmp_path / "custom-output"),
    }


def test_backup_refresh_does_not_compute_velocity_gradients(tmp_path, monkeypatch):
    solver = _solver(tmp_path / "refresh")
    calls = []
    monkeypatch.setattr(
        solver.stepper,
        "_update_velocity_gradients",
        lambda: calls.append("gradient"),
    )

    solver._refresh_backup_particle_fields()

    assert calls == []


def test_empty_vpm_backup_is_still_paraview_readable(tmp_path):
    solver = _solver(tmp_path / "writer")
    with contextlib.redirect_stdout(io.StringIO()):
        solver.save_backup()
    backup = tmp_path / "writer" / "solution" / "vpm_000000"

    pv = pytest.importorskip("pyvista")
    visual = pv.read(f"{backup}.xdmf")

    assert visual.n_points == 0
