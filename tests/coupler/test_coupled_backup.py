"""Focused tests for atomic coupled backups."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from source.coupler import boundary as boundary_module
from source.coupler.backup import (
    _backup_config,
    checkpoint_path_hash,
    config_difference_paths,
    load_coupled_backup,
    publish_vpm_snapshot,
    save_coupled_backup,
)


def test_backup_config_records_stable_wall_geometry_revision():
    setup = _MappingSetup({"coupler": {}})
    vpm = SimpleNamespace(setup=_MappingSetup({"viscous": {"scheme": "GBD"}}))
    transfer = SimpleNamespace(_solid_bodies=(SimpleNamespace(revision="wall-a"),))
    coupler = SimpleNamespace(setup=setup, vpm_solver=vpm, vorticity_transfer=transfer)
    first = _backup_config(coupler)
    transfer._solid_bodies = (SimpleNamespace(revision="wall-b"),)
    second = _backup_config(coupler)
    assert config_difference_paths(first, second) == {"solid_geometry.wall_revisions"}


def test_backup_config_records_transfer_lattice_phase():
    setup = _MappingSetup({"coupler": {}})
    vpm = SimpleNamespace(setup=_MappingSetup({"viscous": {"scheme": "GBD"}}))
    transfer = SimpleNamespace(_lattice_anchor=np.array([0.1, 0.2, 0.3]))
    coupler = SimpleNamespace(setup=setup, vpm_solver=vpm, vorticity_transfer=transfer)
    first = _backup_config(coupler)
    transfer._lattice_anchor[0] += 0.01
    second = _backup_config(coupler)
    assert config_difference_paths(first, second) == {"transfer_lattice.anchor"}


@pytest.fixture(autouse=True)
def _serialize_the_minimal_fake_vpm_setup(monkeypatch):
    """Explicit fake schema/inspection; real native validation has separate tests."""
    monkeypatch.setattr(
        "source.coupler.backup._vpm_numerical_config",
        lambda setup: {
            key: value
            for key, value in setup.to_dict().items()
            if key not in {"backup", "output", "step", "time"}
        },
    )
    monkeypatch.setattr("source.coupler.backup._inspect_coupled_vpm_checkpoint", lambda *args: None)


class _MappingSetup:
    def __init__(self, mapping: dict):
        self.mapping = deepcopy(mapping)

    def to_dict(self) -> dict:
        return deepcopy(self.mapping)


class _CouplerSetup(_MappingSetup):
    def __init__(self, *, interface_normal_tolerance=1e-6):
        super().__init__({"coupler": {"interface_normal_tolerance": interface_normal_tolerance}})
        self.coupling_patch = "numericalBoundary"
        self.freestream_velocity = [1.0, 0.0, 0.0]


class _FVM:
    def __init__(self):
        self.parallel = SimpleNamespace(is_partitioned=False)
        self.step = 2
        self.time = 0.1

    def save_state(self, path: Path) -> None:
        with path.open("wb") as stream:
            np.savez(stream, step=np.asarray(self.step), time=np.asarray(self.time))

    def load_state(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as state:
            self.step = int(state["step"])
            self.time = float(state["time"])


class _VPM:
    def __init__(
        self,
        *,
        gbd_threshold: float,
        backup_directory: str,
        velocity: np.ndarray,
    ):
        self.setup = _MappingSetup(
            {
                "time_step_size": 0.1,
                "time": 0.0,
                "step": 0,
                "viscous": {
                    "scheme": "GBD",
                    "gbd_threshold": gbd_threshold,
                },
                "precision": "f32",
                "backup": {
                    "interval_steps": 0,
                    "directory": backup_directory,
                    "log_directory": backup_directory,
                },
            }
        )
        self.particles = SimpleNamespace(n_particles_total=0)
        self.step = 1
        self.time = 0.1
        self.current_velocity = np.asarray(velocity, dtype=np.float64).copy()

    def _save_backup_to(self, filename: str) -> None:
        import h5py

        with h5py.File(f"{filename}.h5", "w") as archive:
            solver = archive.create_group("solver")
            solver.attrs["step"] = int(self.step)
            solver.attrs["time"] = float(self.time)

        from vtk import vtkPoints, vtkUnstructuredGrid, vtkXMLUnstructuredGridWriter
        from vtk.util.numpy_support import numpy_to_vtk

        points = vtkPoints()
        points.SetData(numpy_to_vtk(np.zeros((0, 3), dtype=np.float32), deep=True))
        grid = vtkUnstructuredGrid()
        grid.SetPoints(points)
        for name in ("time", "TimeValue"):
            array = numpy_to_vtk(np.array([self.time], dtype=np.float64), deep=True)
            array.SetName(name)
            grid.GetFieldData().AddArray(array)
        writer = vtkXMLUnstructuredGridWriter()
        writer.SetFileName(f"{filename}.vtu")
        writer.SetInputData(grid)
        writer.SetDataModeToAppended()
        writer.EncodeAppendedDataOff()
        writer.Write()

    def _load_backup_from(
        self, filename: str, *, allowed_config_differences=(), expected_config_differences=None
    ) -> None:
        assert Path(filename).is_file()

    def compute_velocity_at_points(self, points: np.ndarray, **_kwargs) -> np.ndarray:
        assert len(points) == len(self.current_velocity)
        return self.current_velocity.copy()


def _make_coupler(
    *,
    gbd_threshold: float = 1.0e-5,
    interface_normal_tolerance: float = 1e-6,
    backup_directory: str = "original-output",
):
    previous_velocity = np.array([[1.0, 0.1, 0.0], [1.0, 0.1, 0.0]])
    vpm = _VPM(
        gbd_threshold=gbd_threshold,
        backup_directory=backup_directory,
        velocity=previous_velocity,
    )
    coupler = SimpleNamespace(
        setup=_CouplerSetup(interface_normal_tolerance=interface_normal_tolerance),
        fvm_solver=_FVM(),
        vpm_solver=vpm,
        vorticity_transfer=SimpleNamespace(step=0),
        n_fvm_substeps=2,
        _is_master=True,
        _velocity_boundary_condition_old=previous_velocity.copy(),
        _normal_velocity_boundary_condition_old=None,
        _normal_velocity_boundary_condition=np.ones(2),
        _tangential_gradient_boundary_condition_old=None,
        _tangential_gradient_boundary_condition=np.ones((2, 3)),
        density=1.0,
        kinematic_viscosity=0.01,
        fvm_time_step_size=0.05,
        vpm_time_step_size=0.1,
        vpm_particle_spacing=0.1,
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        fvm_box=np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
        _last_vpm_boundary_condition_flux_diagnostics={},
    )
    return coupler


def _make_mixed_coupler():
    coupler = _make_coupler()
    coupler._normal_velocity_boundary_condition_old = np.array([-1.0, 1.0])
    coupler._tangential_gradient_boundary_condition_old = np.tile([0.0, 0.2, 0.0], (2, 1))

    def mixed_trace(points, normals, *, particle_spacing):
        del points, normals, particle_spacing
        velocity = coupler.vpm_solver.current_velocity.copy()
        tangent = np.zeros_like(velocity)
        tangent[:, 1] = 2 * velocity[:, 1]
        return velocity, tangent

    coupler.vpm_solver.compute_velocity_and_tangential_normal_gradient_at_points = mixed_trace
    return coupler


def test_mixed_history_survives_restart_subcycling_and_replacement(tmp_path, monkeypatch):
    uninterrupted = _make_mixed_coupler()
    backup = tmp_path / "combined"
    save_coupled_backup(uninterrupted, backup, coupling_step=1)
    restarted = _make_mixed_coupler()
    for field in (
        "_normal_velocity_boundary_condition_old",
        "_tangential_gradient_boundary_condition_old",
    ):
        setattr(restarted, field, None)
    load_coupled_backup(restarted, backup)
    for field in (
        "_normal_velocity_boundary_condition_old",
        "_tangential_gradient_boundary_condition_old",
    ):
        np.testing.assert_array_equal(getattr(restarted, field), getattr(uninterrupted, field))

    points = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    normals = points.copy()
    area = np.ones(2)
    captured = []

    def capture(_coupler, _patch, velocity, *, normal_velocity, tangential_gradient):
        captured.append(
            tuple(
                np.array(value, copy=True)
                for value in (velocity, normal_velocity, tangential_gradient)
            )
        )

    monkeypatch.setattr(boundary_module, "apply_fvm_boundary", capture)
    for coupler in (uninterrupted, restarted):
        coupler.vpm_solver.current_velocity = np.tile([1.2, 0.15, 0], (2, 1))
        previous, current, _ = boundary_module.evaluate_vpm_boundary(coupler, points, normals, area)
        boundary_module.advance_fvm_substeps(
            coupler,
            "numericalBoundary",
            points,
            normals,
            area,
            previous,
            current,
            coupler._normal_velocity_boundary_condition_old,
            coupler._normal_velocity_boundary_condition,
            coupler._tangential_gradient_boundary_condition_old,
            coupler._tangential_gradient_boundary_condition,
        )
    assert len(captured) == 4
    for left, right in zip(captured[:2], captured[2:], strict=True):
        for a, b in zip(left, right, strict=True):
            np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(captured[0][1], [-1.1, 1.1])
    np.testing.assert_allclose(captured[0][2], np.tile([0, 0.25, 0], (2, 1)))
    np.testing.assert_allclose(captured[1][1], [-1.2, 1.2])
    np.testing.assert_allclose(captured[1][2], np.tile([0, 0.3, 0], (2, 1)))

    restarted.vpm_solver.current_velocity[:, 1] = 0.25
    boundary_module.update_boundary_history_after_replacement(restarted, points, normals, area)
    np.testing.assert_allclose(
        restarted._tangential_gradient_boundary_condition_old, np.tile([0, 0.5, 0], (2, 1))
    )


def test_mixed_boundary_initializes_both_histories_on_worker():
    coupler = _make_mixed_coupler()
    coupler._is_master = False
    coupler.vpm_solver = None
    for field in (
        "_velocity_boundary_condition_old",
        "_normal_velocity_boundary_condition_old",
        "_tangential_gradient_boundary_condition_old",
    ):
        setattr(coupler, field, None)
    boundary_module.evaluate_vpm_boundary(coupler, np.empty((0, 3)), np.empty((0, 3)), np.empty(0))
    assert coupler._normal_velocity_boundary_condition_old.shape == (0,)
    assert coupler._tangential_gradient_boundary_condition_old.shape == (0, 3)


def test_mixed_boundary_rejects_missing_trace_before_changing_fvm_state():
    coupler = _make_mixed_coupler()
    # This minimal FVM has no setters: entering one before rejecting an
    # incomplete pressure trace would fail with AttributeError instead.
    with pytest.raises(RuntimeError, match="require normal velocity and tangential gradient"):
        boundary_module.apply_fvm_boundary(
            coupler,
            "numericalBoundary",
            np.ones((2, 3)),
            normal_velocity=np.ones(2),
        )


def test_post_renewal_particle_history_is_published_outside_the_rolling_backup(tmp_path):
    backup = tmp_path / "backup"
    output = tmp_path / "solution"
    coupler = _make_coupler()
    for step in (1, 2):
        coupler.fvm_solver.step = 2 * step
        coupler.fvm_solver.time = 0.1 * step
        coupler.vpm_solver.step = step
        coupler.vpm_solver.time = 0.1 * step
        save_coupled_backup(coupler, backup, coupling_step=step)
        publish_vpm_snapshot(backup, output)

    vpm_frames = output / "vpm"
    assert sorted(path.name for path in vpm_frames.glob("vpm_*.h5")) == [
        "vpm_000001.h5",
        "vpm_000002.h5",
    ]
    assert sorted(path.name for path in vpm_frames.glob("vpm_*.vtu")) == [
        "vpm_000001.vtu",
        "vpm_000002.vtu",
    ]
    entries = ET.parse(output / "vpm.pvd").findall(".//DataSet")
    assert [(float(entry.attrib["timestep"]), entry.attrib["file"]) for entry in entries] == [
        (0.1, "vpm/vpm_000001.vtu"),
        (0.2, "vpm/vpm_000002.vtu"),
    ]
    generations = list(backup.glob("checkpoint-*"))
    assert len(generations) == 1
    assert sorted(path.name for path in generations[0].glob("vpm_*.h5")) == ["vpm_000002.h5"]
    assert sorted(path.name for path in generations[0].glob("fvm_*")) == ["fvm_000002.npz"]
    assert sorted(path.name for path in generations[0].glob("vpm_boundary_condition_*")) == [
        "vpm_boundary_condition_000002.npz"
    ]


@pytest.mark.parametrize("defect", ["fvm_time", "fvm_step", "vpm_time", "vpm_step", "nan_time"])
def test_backup_rejects_unsynchronized_state_before_writing(tmp_path, defect):
    coupler = _make_coupler()
    save_coupled_backup(coupler, tmp_path, coupling_step=1)
    before = checkpoint_path_hash(tmp_path)
    if defect == "fvm_time":
        coupler.fvm_solver.time += 0.05
    elif defect == "fvm_step":
        coupler.fvm_solver.step += 1
    elif defect == "vpm_time":
        coupler.vpm_solver.time += 0.1
    elif defect == "vpm_step":
        coupler.vpm_solver.step += 1
    else:
        coupler.fvm_solver.time = np.nan
    with pytest.raises(ValueError, match="backup"):
        save_coupled_backup(coupler, tmp_path, coupling_step=1)
    assert checkpoint_path_hash(tmp_path) == before


@pytest.mark.parametrize("step", [-1, 0, 1.5, True])
def test_backup_rejects_wrong_exchange_label(tmp_path, step):
    destination = tmp_path / "backup"
    with pytest.raises(ValueError):
        save_coupled_backup(_make_coupler(), destination, coupling_step=step)
    assert not destination.exists()


def test_long_run_clock_roundoff_survives_save_and_restart(tmp_path):
    coupler = _make_coupler()
    coupler.n_fvm_substeps = 5
    coupler.fvm_solver.step, coupler.vpm_solver.step = 12500, 2500
    coupler.fvm_solver.time = 0.0
    for _ in range(coupler.fvm_solver.step):
        coupler.fvm_solver.time += 0.008
    coupler.vpm_solver.time = 100.0
    assert abs(coupler.fvm_solver.time - coupler.vpm_solver.time) > 1e-12
    save_coupled_backup(coupler, tmp_path, coupling_step=2500)
    assert load_coupled_backup(coupler, tmp_path) == 2500
    before = checkpoint_path_hash(tmp_path)
    coupler.fvm_solver.time += 1e-8
    with pytest.raises(ValueError, match="not synchronized"):
        save_coupled_backup(coupler, tmp_path, coupling_step=2500)
    assert checkpoint_path_hash(tmp_path) == before


@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("failure", ["vpm", "checkpoint_info"])
def test_failed_save_preserves_committed_pair_even_at_same_step(
    tmp_path, monkeypatch, step, failure
):
    import source.coupler.backup as backup_module

    coupler = _make_coupler()
    save_coupled_backup(coupler, tmp_path, coupling_step=1)
    original_checkpoint_info = (tmp_path / "checkpoint_info.json").read_bytes()
    checkpoint_info = json.loads(original_checkpoint_info)
    coupler.fvm_solver.step = 2 * step
    coupler.vpm_solver.step = step
    coupler.fvm_solver.time = coupler.vpm_solver.time = 0.1 * step
    with monkeypatch.context() as patch:
        if failure == "vpm":

            def fail_vpm(filename):
                Path(f"{filename}.h5").write_bytes(b"incomplete")
                raise OSError("injected VPM write failure")

            patch.setattr(coupler.vpm_solver, "_save_backup_to", fail_vpm)
        else:
            replace = backup_module.os.replace

            def fail_checkpoint_info(source, destination):
                if Path(destination) == tmp_path / "checkpoint_info.json":
                    raise OSError("injected checkpoint_info commit failure")
                return replace(source, destination)

            patch.setattr(backup_module.os, "replace", fail_checkpoint_info)
        with pytest.raises(OSError, match="injected"):
            save_coupled_backup(coupler, tmp_path, coupling_step=step)
    assert (tmp_path / "checkpoint_info.json").read_bytes() == original_checkpoint_info
    for name, relative in checkpoint_info["checkpoint_files"].items():
        assert checkpoint_path_hash(tmp_path / relative) == checkpoint_info["file_sha256"][name]
    assert load_coupled_backup(_make_coupler(), tmp_path) == 1
    save_coupled_backup(coupler, tmp_path, coupling_step=step)
    assert len(list(tmp_path.glob("checkpoint-*"))) == 1


def test_config_difference_paths_are_recursive_and_distinguish_missing_from_none():
    stored = {
        "vpm": {
            "viscous": {"scheme": "GBD", "gbd_threshold": 1.0e-5},
            "optional": None,
        }
    }
    current = {"vpm": {"viscous": {"scheme": "GBD", "gbd_threshold": 2.0e-5}}}

    assert config_difference_paths(stored, current) == {
        "vpm.viscous.gbd_threshold",
        "vpm.optional",
    }


def test_hash_verified_coupled_checkpoint_info_requires_matching_stabilization(
    tmp_path, monkeypatch
):
    import json

    from source.coupler.backup import config_mapping_digest

    writer = _make_coupler()
    settings = {
        "selective_eddy_viscosity_coefficient": 0.5,
        "regularization_preserve_groups": False,
    }
    writer.vpm_solver.setup.mapping["stabilization"] = dict(settings)
    save_coupled_backup(writer, tmp_path, coupling_step=1)
    reader = _make_coupler()
    reader.vpm_solver.setup.mapping["stabilization"] = settings
    assert load_coupled_backup(reader, tmp_path) == 1
    settings["regularization_preserve_groups"] = True
    with pytest.raises(ValueError, match="regularization_preserve_groups"):
        load_coupled_backup(reader, tmp_path)
    settings["regularization_preserve_groups"] = False
    settings["selective_eddy_viscosity_coefficient"] = 0.75
    with pytest.raises(ValueError, match="vpm.stabilization.selective_eddy_viscosity_coefficient"):
        load_coupled_backup(reader, tmp_path)

    metadata_path = tmp_path / "checkpoint_info.json"
    checkpoint_info = json.loads(metadata_path.read_text())
    settings["selective_eddy_viscosity_coefficient"] = 0.5
    checkpoint_info["config"]["vpm"]["stabilization"]["selective_eddy_viscosity_coefficient"] = 0.75
    # A valid checksum does not authorize a changed numerical configuration.
    checkpoint_info["config_sha256"] = config_mapping_digest(checkpoint_info["config"])
    metadata_path.write_text(json.dumps(checkpoint_info))
    monkeypatch.setattr(
        reader.fvm_solver, "load_state", lambda *args: pytest.fail("premature state load")
    )
    monkeypatch.setattr(
        reader.vpm_solver, "_load_backup_from", lambda *args: pytest.fail("premature state load")
    )
    with pytest.raises(ValueError, match="vpm.stabilization.selective_eddy_viscosity_coefficient"):
        load_coupled_backup(reader, tmp_path)


@pytest.mark.parametrize(
    ("path", "changed"),
    [
        ("vpm.viscous.gbd_threshold", {"gbd_threshold": 2.0e-5}),
        ("coupler.interface_normal_tolerance", {"interface_normal_tolerance": 2e-6}),
    ],
)
def test_restart_config_changes_require_the_exact_allowed_setting_path(
    tmp_path, path, changed, caplog
):
    backup = tmp_path / "backup"
    save_coupled_backup(_make_coupler(), backup, coupling_step=1)

    strict = _make_coupler(**changed)
    with pytest.raises(ValueError, match=re.escape(path)):
        load_coupled_backup(strict, backup)

    allowed = _make_coupler(**changed)
    with caplog.at_level("WARNING", logger="coupler"):
        restored_step = load_coupled_backup(allowed, backup, allowed_config_differences={path})
    assert path in caplog.text
    assert restored_step == 1
    assert allowed.vorticity_transfer.step == 2
