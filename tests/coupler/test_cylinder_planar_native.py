"""Bounded native tests of the unit-span, one-layer periodic cylinder case.

These temporary coarse cases exercise the authored physics and native restart.
Two exchanges establish state/clock consistency, not developed force accuracy.
"""

from __future__ import annotations

import csv
from functools import partial
from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from openonda import coupler
from openonda.tutorial_runner import load_case_module
from source.coupler.backup import artifact_digest, config_mapping_digest

CASE = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def _physical_case(*, cores=1, h=0.1, end_time=0.08):
    module = load_case_module(CASE)
    flow, particles, exchange, mesh = module.build_case(
        end_time=end_time,
        overrides={
            "hxy": h,
            "cores": cores,
            "compute_device": "CPU",
            "particle_limit": 20000,
        },
    )
    assert pytest.approx(1.0) == module.FVM_RESOLVED_SPAN
    assert mesh.levels == pytest.approx((-0.5, 0.5))
    assert particles.numerics.induction.planar_span == pytest.approx(1.0)
    assert particles.numerics.induction.plane_z == 0.0
    assert particles.numerics.viscous.particle_spacing == pytest.approx(h)
    assert particles.numerics.viscous.gbd_threshold == pytest.approx(0.01 * h**2)
    patches = {boundary.name: boundary for boundary in flow.boundaries}
    for name, neighbour in (("zmin", "zmax"), ("zmax", "zmin")):
        assert patches[name].velocity_type == patches[name].pressure_type == "cyclic"
        assert patches[name].neighbour_patch == neighbour
    force = next(sample for sample in flow.samplers if sample.name == "forces_history")
    assert force.reference_area == force.reference_velocity == force.reference_length == 1.0
    seed = partial(
        module.cylinder_initial_velocity,
        freestream_velocity=module.STARTUP_FREESTREAM_VELOCITY,
        **module.INITIAL_PERTURBATION,
    )
    return flow, particles, exchange, mesh, seed


def test_both_default_cases_share_unit_span_and_planar_initial_field():
    flow, particles, _exchange, mesh, seed = _physical_case(h=0.04)
    reference = load_case_module(CASE / "reference_flow")
    reference_flow, reference_mesh = reference.build_case(
        reference.DEFAULT_NAME, reference.DEFAULT_H
    )
    assert pytest.approx(1.0) == reference.SPAN
    assert reference_mesh.levels == pytest.approx(mesh.levels)
    assert reference_flow.samplers[0].reference_area == flow.samplers[0].reference_area == 1.0
    patches = {boundary.name: boundary for boundary in reference_flow.boundaries}
    assert patches["zmin"].neighbour_patch == "zmax"
    assert patches["zmax"].neighbour_patch == "zmin"
    assert patches["zmin"].velocity_type == patches["zmax"].velocity_type == "cyclic"
    points = np.array([[0.8, 0.12, -0.3], [0.8, 0.12, 0.0], [0.8, 0.12, 0.3]])
    velocity = seed(points)
    reference_velocity = reference.cylinder_initial_velocity(
        points,
        freestream_velocity=reference.STARTUP_FREESTREAM_VELOCITY,
        **reference.INITIAL_PERTURBATION,
    )
    np.testing.assert_array_equal(velocity[:, 2], 0.0)
    np.testing.assert_array_equal(velocity, np.tile(velocity[0], (3, 1)))
    np.testing.assert_array_equal(reference_velocity, velocity)
    assert particles.numerics.induction.planar_span == 1.0


def _particle_snapshot(vpm, volume=0.1**2):
    positions = vpm.particle_position.copy()
    strengths = vpm.particle_vortex_strength.copy()
    assert len(positions) > 0
    assert np.isfinite(positions).all() and np.isfinite(strengths).all()
    np.testing.assert_array_equal(positions[:, 2], 0.0)
    np.testing.assert_array_equal(strengths[:, :2], 0.0)
    np.testing.assert_allclose(vpm.particle_volume, volume, rtol=2e-7, atol=0)
    return {"positions": positions, "strengths": strengths, "vpm_step": vpm.step}


def _native_restart_state(driver):
    """Primary and temporal state, before restart-derived boundary refresh."""
    native = driver.fvm_solver
    result = {
        name: np.asarray(getattr(native, name)).copy()
        for name in (
            "velocity",
            "kinematic_pressure",
            "volumetric_face_flux",
            "velocity_old",
            "velocity_older",
            "volumetric_face_flux_old",
            "volumetric_face_flux_older",
        )
    }
    for name in (
        "time",
        "step",
        "_n_committed_time_steps",
        "time_step_size",
        "_accepted_time_step_size",
        "_previous_time_step_size",
        "_kinematic_viscosity",
        "max_courant_number",
    ):
        result[name] = np.asarray(getattr(native, name)).copy()
    result["eddy_viscosity"] = (
        np.asarray([]) if native.eddy_viscosity is None else native.eddy_viscosity.copy()
    )
    result["accepted_counters"] = np.asarray(
        [
            native._n_consecutive_accepted_steps[name]
            for name in sorted(native._n_consecutive_accepted_steps)
        ]
    )
    for name in (
        "_velocity_boundary_condition_old",
        "_normal_velocity_boundary_condition_old",
        "_tangential_gradient_boundary_condition_old",
    ):
        value = getattr(driver, name) if driver._is_master else None
        if driver._comm is not None:
            value = driver._comm.bcast(value, root=0)
        result[name] = np.asarray(value).copy()
    particles = driver.apply_vpm(
        lambda vpm: {
            name: np.asarray(getattr(vpm, name)).copy()
            for name in (
                "particle_position",
                "particle_vortex_strength",
                "particle_core_radius",
                "particle_volume",
                "particle_kinematic_viscosity",
                "particle_group_id",
                "particle_zone_id",
                "particle_eddy_viscosity",
                "particle_effective_viscosity",
                "freestream_velocity",
                "time",
                "step",
            )
        }
    )
    result.update({f"vpm_{name}": value for name, value in particles.items()})
    return result


def _advance(
    directory,
    *,
    limit,
    start_from,
    cores=1,
    mesh_path=None,
    expected_restart=None,
    h=0.1,
    end_time=0.08,
):
    flow, particles, exchange, mesh, seed = _physical_case(cores=cores, h=h, end_time=end_time)
    if start_from == "latest" and mesh_path is None:
        mesh_path = directory / "solution/fvm/mesh.npz"
    if mesh_path is not None:
        mesh = mesh_path
    with coupler.create_coupler(flow, particles, exchange, mesh=mesh, case_dir=directory) as driver:
        start_arguments = {"start_from": start_from}
        if expected_restart is not None:
            driver.initialize()
            restored_step = driver.load_backup(directory / "solution/backups")
            restored = _native_restart_state(driver)
            assert restored.keys() == expected_restart.keys()
            for name in restored:
                np.testing.assert_array_equal(restored[name], expected_restart[name], err_msg=name)
            assert driver.vorticity_transfer.step == restored_step + 1
            start_arguments = {"start_step": restored_step}
        accepted = driver.run(
            **start_arguments,
            max_coupling_steps=limit,
            backup_at_stop=True,
            initial_velocity=seed,
        )
        native = driver.fvm_solver
        count = native.mesh_data["n_cells"]
        centres = native.geo_data["cell_centre"][:count]
        _xy, column_count = np.unique(np.round(centres[:, :2], 12), axis=0, return_counts=True)
        np.testing.assert_array_equal(column_count, 1)
        np.testing.assert_allclose(centres[:, 2], 0.0, atol=2e-14, rtol=0)
        patches = {patch["name"]: patch for patch in native.mesh_data["boundary"]}
        for name in ("zmin", "zmax"):
            patch = patches[name]
            faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
            np.testing.assert_array_equal(
                native.mesh_data["boundary_neighbour_cell"][faces],
                native.mesh_data["owners"][faces],
            )
        velocity = np.asarray(native.velocity)[:count].copy()
        pressure = np.asarray(native.kinematic_pressure)[:count].copy()
        assert np.isfinite(velocity).all() and np.isfinite(pressure).all()
        np.testing.assert_allclose(velocity[:, 2], 0, atol=1e-12, rtol=0)
        snapshot = driver.apply_vpm(_particle_snapshot, h**2)
        assert native.step == accepted * 5
        assert native.time == pytest.approx(accepted * 0.04, abs=1e-12)
        assert snapshot["vpm_step"] == accepted
        assert native.parallel.size == cores
        if cores > 1:
            assert native.parallel.mode == "petsc_replicated"
        result = {
            "velocity": velocity,
            "pressure": pressure,
            **snapshot,
            "accepted": accepted,
            "cells": count,
            "rank": native.parallel.rank,
            "parallel_mode": native.parallel.mode,
            "restart_state": _native_restart_state(driver),
        }
    records = [
        json.loads(line)
        for line in (directory / "solution/diagnostics.jsonl").read_text().splitlines()
    ]
    assert records[-1]["n_nonfinite_values"] == 0
    assert records[-1]["max_continuity_error"] < 1e-6
    assert abs(records[-1]["net_boundary_volumetric_flux"]) < 1e-10
    exchanges = [
        json.loads(line)
        for line in (directory / "solution/coupler_diagnostics.jsonl").read_text().splitlines()
    ]
    for record in exchanges:
        iteration = record["interface_iteration"]
        assert iteration["converged"], iteration
        last = next(
            residual for residual in reversed(iteration["residuals"]) if residual["accepted"]
        )
        assert last["normal_residual_rms"] <= exchange.interface_normal_tolerance
        assert last["gradient_residual_rms"] <= exchange.interface_gradient_tolerance
        recovery = record["gbd_moment_recovery"]
        assert recovery["correction_fraction"] < 0.08
        for name in (
            "normalized_vortex_strength_residual",
            "normalized_linear_impulse_residual",
            "normalized_angular_impulse_residual",
        ):
            assert recovery[name] < 1e-10
    result["exchanges"] = exchanges
    manifest = json.loads((directory / "solution/backups/manifest.json").read_text())
    assert manifest["coupling_step"] == accepted
    assert manifest["time"] == pytest.approx(accepted * 0.04)
    assert manifest["config_sha256"] == config_mapping_digest(manifest["config"])
    for name, relative in manifest["artifacts"].items():
        artifact = directory / "solution/backups" / relative
        assert artifact_digest(artifact) == manifest["artifact_sha256"][name]
    if result["rank"] == 0:
        shutil.copytree(
            directory / "solution/backups", directory / f"checkpoint_after_{accepted:06d}"
        )
        np.savez(
            directory / f"accepted_state_{accepted:06d}.npz",
            **{name: result[name] for name in ("velocity", "pressure", "positions", "strengths")},
        )
    result["manifest"] = manifest
    histories = list(directory.glob("samples/**/forces_history.csv"))
    assert len(histories) == 1
    with histories[0].open(newline="") as stream:
        result["forces"] = list(csv.DictReader(stream))
    return result


@pytest.mark.integration
def test_actual_periodic_planar_cylinder_lossless_native_resume(tmp_path):
    continuous = _advance(tmp_path / "continuous", limit=2, start_from="initial")
    saved_mesh = tmp_path / "continuous/solution/fvm/mesh.npz"
    first = _advance(tmp_path / "split", limit=1, start_from="initial", mesh_path=saved_mesh)
    assert first["accepted"] == 1
    resumed = _advance(
        tmp_path / "split",
        limit=1,
        start_from="latest",
        expected_restart=first["restart_state"],
    )
    assert resumed["accepted"] == continuous["accepted"] == 2
    assert resumed["cells"] == continuous["cells"]
    # The lossless native roundtrip is asserted before advancing above.
    # Optimization histories and linear preconditioners intentionally start
    # cold, so residual-converged trajectories need not be bitwise equal.
    differences = {
        field: float(np.max(np.abs(resumed[field] - continuous[field])))
        for field in ("velocity", "pressure", "positions", "strengths")
    }
    for result in (continuous, resumed):
        times = np.array([float(row["time"]) for row in result["forces"]])
        np.testing.assert_allclose(times, [0.04, 0.08], atol=1e-12, rtol=0)
        for row in result["forces"]:
            assert all(np.isfinite(float(value)) for key, value in row.items() if key != "patch")
    (tmp_path / "continuation_differences.json").write_text(
        json.dumps(differences, indent=2) + "\n"
    )


@pytest.mark.integration
def test_four_rank_actual_cyclic_cylinder_backup_and_resume(tmp_path):
    if find_spec("mpi4py") is None or find_spec("petsc4py") is None:
        pytest.skip("MPI and PETSc are required")
    if not (Path(sys.executable).with_name("mpiexec").is_file() or shutil.which("mpiexec")):
        pytest.skip("mpiexec is required")
    script = Path(__file__).with_name("_cylinder_planar_mpi_smoke.py")
    environment = os.environ.copy()
    environment.pop("_OPENONDA_MPI_CHILD", None)
    environment["PYTHONPATH"] = str(CASE.parents[2])
    environment["TI_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path / "taichi-cache")
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "four-rank")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-10000:] + result.stderr[-10000:]
    root_fields = np.load(tmp_path / "four-rank/rank-0-state.npz")
    for rank in range(4):
        report = json.loads((tmp_path / "four-rank" / f"rank-{rank}.json").read_text())
        assert report["accepted"] == 2
        assert report["parallel_mode"] == "petsc_replicated"
        assert report["force_rows"] == 2
        assert report["cells"] > 0
        with np.load(tmp_path / "four-rank" / f"rank-{rank}-state.npz") as fields:
            for name in root_fields.files:
                np.testing.assert_array_equal(fields[name], root_fields[name])
    root_fields.close()
