"""Portable fixtures for the frozen Delta checkpoint's native/VLM prefix."""

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from source.solvers.vpm.io.checkpoint_audit import _frame_time, audit_checkpoint
from tests._tutorial_helpers import write_vtu_time_frame

CLOCKS = [(400, 1.0), (800, 2.0), (1200, 3.0), (1600, 4.0)]


def test_native_binary_vtp_clock_is_readable(tmp_path):
    import pyvista as pv

    from source.vtk_output import write_vtk_dataset

    mesh = pv.PolyData(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), faces=[3, 0, 1, 2]
    )
    mesh.field_data["TimeValue"] = np.array([0.123456789012], dtype=np.float64)
    path = tmp_path / "native.vtp"
    write_vtk_dataset(mesh, path)
    assert _frame_time(path) == 0.123456789012


def _write_vtu(directory: Path, step: int, time: float) -> None:
    write_vtu_time_frame(directory / f"vpm_{step:06d}.vtu", time)


def _write_native(directory: Path, step: int, time: float) -> Path:
    frame_directory = directory / "vpm"
    frame_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = frame_directory / f"vpm_{step:06d}.h5"
    with h5py.File(checkpoint, "w") as archive:
        solver = archive.create_group("solver")
        solver.attrs.update(
            step=step,
            time=time,
            time_step_size=0.0025,
            n_particles_total=1,
            backup_format_version="10.0",
            numerical_configuration="{}",
            numerical_configuration_sha256=hashlib.sha256(b"{}").hexdigest(),
        )
        particles = archive.create_group("particles")
        for name in (
            "position",
            "velocity",
            "vortex_strength",
            "core_radius",
            "particle_volume",
            "kinematic_viscosity",
            "eddy_viscosity",
            "effective_viscosity",
            "group_id",
            "vorticity",
            "zone_id",
        ):
            particles.create_dataset(name, data=np.zeros((1, 3)))
        vlm = solver.create_group("vlm")
        vlm.attrs.update(version=7, time=time, identity="fixture")
        vlm.create_group("motion")
        for name in (
            "panel_corner_position",
            "vortex_point_position",
            "collocation_point",
            "bound_vortex_midpoint",
            "normal",
            "circulation",
            "circulation_old",
            "cumulative_circulation",
            "cumulative_circulation_old",
            "velocity",
            "bound_vortex_velocity",
            "kinematic_velocity",
            "external_velocity",
            "bound_kinematic_velocity",
            "bound_external_velocity",
            "panel_force",
            "unsteady_panel_force",
            "unsteady_pressure_jump_coefficient",
            "panel_moment_correction",
            "pressure_coefficient",
            "leading_edge_suction_parameter",
            "area",
            "relative_velocity",
            "bound_relative_velocity",
        ):
            vlm.create_dataset(name, data=np.zeros(1))
    _write_vtu(frame_directory, step, time)
    return checkpoint


def _write_index(directory: Path, clocks: list[tuple[int, float]]) -> None:
    entries = "".join(
        f'<DataSet timestep="{time:.17g}" file="vlm/vlm_{step:06d}.vtp"/>' for step, time in clocks
    )
    (directory / "vlm.pvd").write_text(
        f'<VTKFile type="Collection"><Collection>{entries}</Collection></VTKFile>',
        encoding="utf-8",
    )


def _write_surfaces(directory: Path, clocks: list[tuple[int, float]]) -> None:
    frame_directory = directory / "vlm"
    frame_directory.mkdir(parents=True, exist_ok=True)
    for step, time in clocks:
        (frame_directory / f"vlm_{step:06d}.vtp").write_text(
            '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">'
            '<PolyData><FieldData><DataArray type="Float64" Name="TimeValue" '
            f'NumberOfTuples="1" format="ascii">{time:.17g}</DataArray></FieldData>'
            '<Piece NumberOfPoints="0" NumberOfPolys="0"/></PolyData></VTKFile>',
            encoding="utf-8",
        )
    _write_index(directory, clocks)


@pytest.fixture
def published_prefix(tmp_path):
    for step, time in CLOCKS:
        _write_native(tmp_path, step, time)
    _write_surfaces(tmp_path, CLOCKS)
    return tmp_path


def _checkpoint(directory: Path, step: int = 1600) -> Path:
    return directory / "vpm" / f"vpm_{step:06d}.h5"


def test_rejects_legacy_160_surface_frames_on_four_native_backups(tmp_path):
    for step, time in CLOCKS:
        _write_native(tmp_path, step, time)
    _write_surfaces(tmp_path, [(step, step * 0.0025) for step in range(10, 1601, 10)])

    with pytest.raises(ValueError, match=r"native VPM backup steps \(extra=\[10,"):
        audit_checkpoint(_checkpoint(tmp_path))


def test_matching_native_prefix_is_accepted(published_prefix):
    evidence = audit_checkpoint(_checkpoint(published_prefix))

    assert evidence["step"] == 1600
    assert evidence["time"] == 4.0
    assert evidence["vlm_identity"] == "fixture"
    assert evidence["vtp_series"]["frame_count"] == 4
    assert evidence["vtp_series"]["pvd_entries"] == 4
    assert evidence["vtp_series"]["native_checkpoint_count"] == 4
    assert evidence["vtp_series"]["last_frame"] == "vlm_001600.vtp"
    assert evidence["vtp_series"]["through_step"] == 1600


def test_unrelated_vpm_resources_in_prefix_are_ignored(published_prefix):
    (published_prefix / "vpm/vpm_notes.txt").write_text("not a checkpoint frame")

    evidence = audit_checkpoint(_checkpoint(published_prefix))

    assert evidence["vtp_series"]["native_checkpoint_count"] == 4
    assert evidence["vtp_series"]["frame_count"] == 4


def test_rejects_malformed_numeric_native_filename(published_prefix):
    (published_prefix / "vpm/vpm_000400_extra.vtu").write_bytes(b"invalid frame filename")

    with pytest.raises(ValueError, match="invalid vpm series filename"):
        audit_checkpoint(_checkpoint(published_prefix))


@pytest.mark.parametrize("source", ["h5", "vtu", "pvd"])
@pytest.mark.parametrize("bad_time", [1.0 + 1.0e-12, float("nan")])
def test_rejects_wrong_native_or_index_clock(published_prefix, source, bad_time):
    if source == "h5":
        with h5py.File(published_prefix / "vpm/vpm_000400.h5", "r+") as archive:
            archive["solver"].attrs["time"] = bad_time
    elif source == "vtu":
        _write_vtu(published_prefix / "vpm", 400, bad_time)
    else:
        _write_index(published_prefix, [(400, bad_time), *CLOCKS[1:]])

    with pytest.raises(ValueError, match=r"(time.*native VPM clock|native VPM step/time)"):
        audit_checkpoint(_checkpoint(published_prefix))


def test_rejects_native_step_that_disagrees_with_filename(published_prefix):
    with h5py.File(published_prefix / "vpm/vpm_000400.h5", "r+") as archive:
        archive["solver"].attrs["step"] = 401

    with pytest.raises(ValueError, match="native VPM step/time conflicts with filename"):
        audit_checkpoint(_checkpoint(published_prefix))


@pytest.mark.parametrize("missing", ["h5", "vtu", "vtp", "pvd_entry", "pvd"])
def test_rejects_missing_prefix_companions(published_prefix, missing):
    if missing == "pvd_entry":
        _write_index(published_prefix, CLOCKS[1:])
    elif missing == "pvd":
        (published_prefix / "vlm.pvd").unlink()
    else:
        prefix = "vlm" if missing == "vtp" else "vpm"
        (published_prefix / prefix / f"{prefix}_000400.{missing}").unlink()

    with pytest.raises(ValueError):
        audit_checkpoint(_checkpoint(published_prefix))


def test_rejects_native_backup_omitted_from_surface_series(published_prefix):
    (published_prefix / "vlm/vlm_000400.vtp").unlink()
    _write_index(published_prefix, CLOCKS[1:])

    with pytest.raises(ValueError, match=r"extra=\[\], missing=\[400\]"):
        audit_checkpoint(_checkpoint(published_prefix))


@pytest.mark.parametrize("suffix", ["h5.tmp", "vtu.tmp"])
def test_rejects_unfinished_publication_inside_prefix(published_prefix, suffix):
    (published_prefix / "vpm" / f"vpm_000400.{suffix}").write_bytes(b"unfinished")

    with pytest.raises(ValueError, match="native HDF5/VTU prefix is not fully published"):
        audit_checkpoint(_checkpoint(published_prefix))


def test_ignores_later_publications_before_reading_their_content(published_prefix):
    # A later native backup, an unindexed surface, and a not-yet-written indexed
    # surface can all coexist while the solver is publishing its next outputs.
    (published_prefix / "vpm/vpm_002000.h5").write_bytes(b"unfinished HDF5")
    (published_prefix / "vpm/vpm_002000.vtu.tmp").write_bytes(b"unfinished VTU")
    (published_prefix / "vpm/vpm_002400.vtu").write_bytes(b"unfinished VTU")
    (published_prefix / "vlm/vlm_002000.vtp").write_bytes(b"unfinished VTP")
    _write_index(published_prefix, [*CLOCKS, (2400, float("nan"))])

    evidence = audit_checkpoint(_checkpoint(published_prefix))

    assert evidence["vtp_series"]["frame_count"] == 4
    assert evidence["vtp_series"]["pvd_entries"] == 4
    assert evidence["vtp_series"]["native_checkpoint_count"] == 4
    assert evidence["vtp_series"]["last_frame"] == "vlm_001600.vtp"


def test_auditing_earlier_checkpoint_reports_only_its_prefix(published_prefix):
    evidence = audit_checkpoint(_checkpoint(published_prefix, 800))

    assert evidence["vtp_series"]["frame_count"] == 2
    assert evidence["vtp_series"]["pvd_entries"] == 2
    assert evidence["vtp_series"]["native_checkpoint_count"] == 2
    assert evidence["vtp_series"]["last_frame"] == "vlm_000800.vtp"
    assert evidence["vtp_series"]["through_step"] == 800
