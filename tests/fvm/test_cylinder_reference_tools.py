"""Regression tests for the cylinder reference restart and analysis tools."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.fvm.fields.spanwise_projection import SpanwiseInvariantProjector
from source.solvers.fvm.sampling.base import sampler_from_dict, sampler_to_dict
from source.solvers.fvm.sampling.fields import LineSampler, SurfaceSampler
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.assets.analyse_grid_study import (
    build_report,
)
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.assets.analyse_reference import (
    _extrema,
    analyse,
)
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.assets.audit_reference_samples import (
    _expected_spanwise_centres,
    _spanwise_field_metrics,
)
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.assets.check_grid_independence import (
    _spatial_convergence,
)
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.assets.prune_restart_tail import (
    checkpoint_position,
    prune_csv,
    prune_json_lines,
    reconcile_surface_indices,
)
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.assets.save_verification_case import (
    preserve,
)
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.benchmark_config import (
    GRID_SPECS,
    grid_study_spec,
)
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.seed_perturbation import (
    _affine_interpolate_2d,
)


def test_reference_analysis_requires_stable_complete_cycles(tmp_path):
    samples = tmp_path / "samples"
    solution = tmp_path / "solution"
    samples.mkdir()
    solution.mkdir()

    time = np.arange(0.0, 40.0 + 0.05, 0.1)
    omega = 2.0 * np.pi * 0.2
    drag = 1.25 + 0.02 * np.cos(2.0 * omega * time)
    lift = 0.5 * np.sin(omega * time)
    with (samples / "forces_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "time",
                "step",
                "patch",
                "pressure_force_x",
                "pressure_force_y",
                "pressure_force_z",
                "viscous_force_x",
                "viscous_force_y",
                "viscous_force_z",
                "total_force_x",
                "total_force_y",
                "total_force_z",
                "drag_coefficient",
                "lift_coefficient",
                "side_force_coefficient",
            )
        )
        writer.writerows(
            (
                sample_time,
                step,
                "cylinder",
                1.6 * cd,
                1.8 * cl,
                0.0,
                0.4 * cd,
                0.2 * cl,
                0.0,
                2.0 * cd,
                2.0 * cl,
                0.0,
                cd,
                cl,
                0.0,
            )
            for step, (sample_time, cd, cl) in enumerate(
                zip(time, drag, lift, strict=True), start=1
            )
        )

    with (solution / "diagnostics.jsonl").open("w", encoding="utf-8") as stream:
        for step, sample_time in enumerate(time, start=1):
            stream.write(
                json.dumps(
                    {
                        "step": step,
                        "time": sample_time,
                        "max_courant_number": 0.2,
                        "max_continuity_error": 1.0e-10,
                        "n_nonfinite_values": 0,
                        "residuals": {
                            "velocity": 1.0e-8,
                            "kinematic_pressure": 1.0e-8,
                        },
                    }
                )
                + "\n"
            )

    report = analyse(tmp_path)

    assert report["status"] == "statistically_ready"
    assert report["saturation"]["passed"] is True
    assert report["history_alignment"]["passed"] is True
    np.testing.assert_allclose(report["latest_strouhal_number"], 0.2, atol=2.0e-3)
    assert len(report["complete_cycles"]) >= 10
    assert "NaN" not in (solution / "reference_diagnostics.json").read_text()


def test_grid_study_force_report_uses_one_common_statistics_window(tmp_path):
    reference = tmp_path / "reference_flow"
    time = np.arange(0.0, 40.0 + 1.0e-12, 0.1)
    omega = 2.0 * np.pi * 0.2
    for number, (case, dx) in enumerate(
        (("very_coarse", 1 / 12), ("coarse", 1 / 24), ("medium", 1 / 36), ("fine", 1 / 48)),
        start=1,
    ):
        samples = reference / "samples" / case
        solution = reference / "solution" / case
        samples.mkdir(parents=True)
        solution.mkdir(parents=True)
        drag = 1.3 + (0.03 / number) * np.cos(2.0 * omega * time)
        lift = (0.4 / number) * np.sin(omega * time)
        with (samples / "forces_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("time", "patch", "drag_coefficient", "lift_coefficient")
            )
            writer.writeheader()
            writer.writerows(
                {
                    "time": sample_time,
                    "patch": "cylinder",
                    "drag_coefficient": cd,
                    "lift_coefficient": cl,
                }
                for sample_time, cd, cl in zip(time, drag, lift, strict=True)
            )
        (solution / "benchmark_metadata.json").write_text(
            json.dumps(
                {
                    "mesh": {
                        "cell_count": 1_000 * number,
                        "grid_study": {
                            "wall_dx": dx,
                            "near_body_dx": 2 * dx,
                            "wake_dx": 4 * dx,
                            "far_field_dx": 12 * dx,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

    report = build_report(reference, ("very_coarse", "coarse", "medium", "fine"), 30.0)

    assert report["status"] == "evidence_ready"
    assert report["common_statistics_window"] == {"start": 10.0, "end": 40.0, "duration": 30.0}
    np.testing.assert_allclose(report["cases"]["fine"]["force_statistics"]["mean_cd"], 1.3)
    np.testing.assert_allclose(report["cases"]["medium"]["force_statistics"]["strouhal"], 0.2)
    assert len(report["sequential_relative_changes"]) == 3


def test_grid_study_spec_enforces_the_exact_1_2_4_12_contract():
    grid = grid_study_spec(1 / 24, "coarse")
    assert (grid.surface, grid.shear_layer, grid.near_wake, grid.background) == (
        1 / 24,
        2 / 24,
        4 / 24,
        12 / 24,
    )
    with pytest.raises(ValueError, match="D/\\(12 n\\)"):
        grid_study_spec(1 / 16, "invalid")


def test_reference_extrema_accepts_single_startup_sample():
    assert _extrema(np.asarray([0.1]), np.asarray([0.0])) == ([], [])


def test_reference_audit_uses_configured_spanwise_slab_centres(tmp_path):
    solution = tmp_path / "solution"
    solution.mkdir()
    (solution / "benchmark_metadata.json").write_text(
        json.dumps(
            {
                "mesh": {
                    "effective_domain": [-8.0, 20.0, -8.0, 8.0, -2.0, 2.0],
                    "spanwise_cell_size": 0.25,
                    "spanwise_cells": 16,
                }
            }
        ),
        encoding="utf-8",
    )

    expected = np.linspace(-1.875, 1.875, 16)
    np.testing.assert_allclose(_expected_spanwise_centres(tmp_path), expected)


def test_reference_field_transfer_is_affine_exact_and_spanwise_invariant():
    source_xy = np.asarray(
        [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]]
    )
    source = np.column_stack(
        (
            1.0 + 0.5 * source_xy[:, 0] - 0.2 * source_xy[:, 1],
            -0.3 + 0.1 * source_xy[:, 0] + 0.4 * source_xy[:, 1],
        )
    )
    target_xy = np.repeat(np.asarray([[-0.5, 0.25], [0.5, -0.25]]), 8, axis=0)
    transferred = _affine_interpolate_2d(source_xy, source, target_xy, k=6)
    expected = np.column_stack(
        (
            1.0 + 0.5 * target_xy[:, 0] - 0.2 * target_xy[:, 1],
            -0.3 + 0.1 * target_xy[:, 0] + 0.4 * target_xy[:, 1],
        )
    )
    np.testing.assert_allclose(transferred, expected, atol=2.0e-12)
    np.testing.assert_allclose(transferred[:8], np.tile(transferred[0], (8, 1)))
    np.testing.assert_allclose(transferred[8:], np.tile(transferred[8], (8, 1)))


def test_full_field_coherence_metrics_detect_off_probe_spanwise_mode():
    xy = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    centres = np.asarray([[*point, z] for point in xy for z in (-0.5, 0.5)])
    velocity = np.zeros((4, 3))
    pressure = np.zeros(4)
    velocity[2:, 2] = (-2.0e-3, 2.0e-3)
    pressure[2:] = (-0.1, 0.1)
    metrics = _spanwise_field_metrics(centres, velocity, pressure)
    assert metrics["maximum_absolute_velocity_z"] == 2.0e-3
    assert metrics["velocity_deviation_rms"]["velocity_z"] > 1.0e-3
    assert metrics["pressure_deviation_rms"] > 0.05


def test_spanwise_projector_averages_cells_and_conservative_fluxes():
    class SerialParallel:
        is_partitioned = False
        n_owned = None
        rank = 0

        @staticmethod
        def bcast(value, root=0):
            del root
            return value

        @staticmethod
        def global_sum(value):
            return value

        @staticmethod
        def global_max(value):
            return value

    solver = SimpleNamespace(
        parallel=SerialParallel(),
        mesh_data={"n_cells": 4, "n_faces": 4},
        velocity=np.asarray([[1.0, 2.0, 0.1], [3.0, 4.0, -0.1], [5.0, 6.0, 0.2], [7.0, 8.0, -0.2]]),
        kinematic_pressure=np.asarray([1.0, 3.0, 5.0, 7.0]),
        volumetric_face_flux=np.asarray([2.0, 4.0, 0.3, -0.4]),
    )
    layout = {
        "cell_groups": np.asarray([0, 0, 1, 1]),
        "face_groups": np.asarray([0, 0, -1, -1]),
        "horizontal_faces": np.asarray([False, False, True, True]),
        "face_authority": np.zeros(4, dtype=np.int32),
        "n_cell_groups": 2,
        "n_face_groups": 1,
    }
    projector = SpanwiseInvariantProjector(solver, layout)
    projector(solver)

    np.testing.assert_allclose(
        solver.velocity,
        [[2.0, 3.0, 0.0], [2.0, 3.0, 0.0], [6.0, 7.0, 0.0], [6.0, 7.0, 0.0]],
    )
    np.testing.assert_allclose(solver.kinematic_pressure, [2.0, 2.0, 6.0, 6.0])
    np.testing.assert_allclose(solver.volumetric_face_flux, [3.0, 3.0, 0.0, 0.0])


def test_restart_tail_pruning_stops_at_committed_step(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"
    np.savez(checkpoint, step=np.asarray(3), time=np.asarray(0.3))
    assert checkpoint_position(checkpoint) == (3, 0.3)

    diagnostics = tmp_path / "diagnostics.jsonl"
    diagnostics.write_text(
        "".join(json.dumps({"step": step}) + "\n" for step in range(1, 6)),
        encoding="utf-8",
    )
    forces = tmp_path / "forces_history.csv"
    forces.write_text(
        "time,step,value\n" + "".join(f"{0.1 * step},{step},{step}\n" for step in range(1, 6)),
        encoding="utf-8",
    )

    assert prune_json_lines(diagnostics, 3) == 2
    assert prune_csv(forces, 3) == 2
    assert [json.loads(line)["step"] for line in diagnostics.read_text().splitlines()] == [
        1,
        2,
        3,
    ]
    assert forces.read_text().splitlines()[-1].split(",")[1] == "3"


def test_restart_tail_rebuilds_complete_surface_index(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    for step in (50, 100, 150, 200):
        (samples / f"slice_z0_{step:06d}.vts").write_bytes(b"vtk")
    (samples / "slice_z0.pvd").write_text("stale", encoding="utf-8")

    removed, entries = reconcile_surface_indices(samples, 150, 0.01)

    assert removed == 1
    assert entries == 3
    index = (samples / "slice_z0.pvd").read_text(encoding="utf-8")
    assert "slice_z0_000050.vts" in index
    assert "slice_z0_000150.vts" in index
    assert "slice_z0_000200.vts" not in index


def test_surface_sampler_body_mask_uses_vtk_point_order(tmp_path):
    import pyvista as pv

    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 3), np.linspace(-1.0, 1.0, 3))
    centres = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))

    class Context:
        parallel = SimpleNamespace(is_partitioned=False)
        mesh_data = {"n_cells": centres.shape[0]}
        geo_data = {"cell_centre": centres}
        velocity = np.column_stack((1.0 + centres[:, 0], centres[:, 1], centres[:, 2]))
        kinematic_pressure = np.zeros(centres.shape[0])

        @staticmethod
        def _vorticity_field():
            return np.column_stack((centres[:, 0], centres[:, 1], np.ones(centres.shape[0])))

    sampler = SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[-1.0, 1.0, -1.0, 1.0],
        spacing=0.5,
        body_bounds=[-0.6, 0.6, -0.6, 0.6, -1.0, 1.0],
        body_geometry="cylinder_z",
    )
    destination = tmp_path / "slice.vts"
    sampler.save_vts(Context(), destination)
    grid = pv.read(destination)

    invalid = np.asarray(grid.point_data["vtkValidPointMask"]) == 0
    assert np.count_nonzero(invalid) == 5
    assert np.all(np.linalg.norm(grid.points[invalid, :2], axis=1) < 0.6)
    corner = np.all(np.isclose(grid.points[:, :2], [0.5, 0.5]), axis=1)
    assert np.count_nonzero(corner) == 1
    assert not invalid[corner][0]
    velocity = np.asarray(grid.point_data["velocity"])
    assert np.all(np.isfinite(velocity[~invalid]))
    assert np.all(np.isnan(velocity[invalid]))


def test_point_sampler_reproduces_affine_fields_exactly():
    rng = np.random.default_rng(7)
    centres = rng.uniform(-1.0, 1.0, size=(200, 3))
    sampler = LineSampler(
        start=[-0.4, 0.2, -0.3],
        end=[0.5, -0.1, 0.4],
        n_points=9,
        k=16,
        reconstruction="affine",
    )
    field = 1.7 + 0.8 * centres[:, 0] - 1.2 * centres[:, 1] + 0.4 * centres[:, 2]
    expected = (
        1.7 + 0.8 * sampler.points[:, 0] - 1.2 * sampler.points[:, 1] + 0.4 * sampler.points[:, 2]
    )
    np.testing.assert_allclose(sampler._interpolate(field, centres), expected, atol=2.0e-12)


def test_grid_convergence_reports_observed_order_and_gci():
    g0 = {
        "mean_cd": 1.00,
        "cd_peak_to_peak": 0.080,
        "cl_first_harmonic": 0.50,
        "strouhal": 0.1800,
    }
    g1 = {
        "mean_cd": 0.99,
        "cd_peak_to_peak": 0.0784,
        "cl_first_harmonic": 0.495,
        "strouhal": 0.1791,
    }
    g2 = {
        "mean_cd": 0.9875,
        "cd_peak_to_peak": 0.0780,
        "cl_first_harmonic": 0.49375,
        "strouhal": 0.178875,
    }

    report = _spatial_convergence(g0, g1, g2)

    for metric in report.values():
        assert metric["passed"] is True
        assert metric["monotone"] is True
        np.testing.assert_allclose(metric["observed_order"], 2.0)
        assert metric["fine_grid_gci"] > 0.0


def test_reference_grid_hierarchy_concentrates_resolution():
    for name in ("g0", "g1", "g2"):
        grid = GRID_SPECS[name]
        np.testing.assert_allclose(grid.shear_layer, 2.0 * grid.surface)
        np.testing.assert_allclose(grid.near_wake, 2.0 * grid.surface)
        np.testing.assert_allclose(grid.downstream_wake, 4.0 * grid.surface)
        assert grid.background >= grid.downstream_wake
    for name in ("g0", "g1", "g2"):
        grid = GRID_SPECS[name]
        np.testing.assert_allclose(grid.first_cell_height, grid.surface / 8.0)


def test_preserve_verification_case_copies_validated_authority(tmp_path):
    case = tmp_path / "case"
    destination = tmp_path / "verification"
    (case / "samples").mkdir(parents=True)
    (case / "solution").mkdir()
    (case / "samples" / "forces_history.csv").write_text("time,step\n60,30000\n")
    (case / "solution" / "performance.jsonl").write_text('{"step": 1}\n')
    (case / "solution" / "benchmark_metadata.json").write_text(
        json.dumps(
            {
                "mesh": {"grid": "g1", "domain": "baseline"},
                "time": {"dt_scale": 1.0},
            }
        )
    )
    (case / "solution" / "reference_diagnostics.json").write_text(
        json.dumps({"status": "statistically_ready"})
    )
    (case / "solution" / "sample_quality.json").write_text(json.dumps({"status": "passed"}))

    outputs = preserve(case, destination, "g1")

    assert {path.name for path in outputs} == {
        "g1_forces_history.csv",
        "g1_performance.jsonl",
        "g1_metadata.json",
        "g1_diagnostics.json",
        "g1_sample_quality.json",
    }
    assert all(path.is_file() for path in outputs)


def test_affine_sampler_configuration_round_trip():
    sampler = SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[-1.0, 1.0, -1.0, 1.0],
        spacing=0.25,
        k=12,
        inverse_distance_power=1.5,
        reconstruction="affine",
        body_bounds=[-0.5, 0.5, -0.5, 0.5, -2.0, 2.0],
        body_geometry="cylinder_z",
    )
    assert sampler_from_dict(sampler_to_dict(sampler)) == sampler
