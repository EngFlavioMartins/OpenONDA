"""Independent launcher and comparison contracts for the gated cylinder study."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "studies/panel_removal" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_planar_launcher_has_matching_span_and_strength_measure(tmp_path):
    module = load("run_cylinder")
    fv, vp, coupling, mesh, manifest = module.configuration(tmp_path)
    assert vp.numerics.panel_solver is None and not vp.numerics.bodies
    assert vp.numerics.induction.planar_span == 1.0
    assert vp.numerics.viscous.gbd_threshold == pytest.approx(0.01 * 0.05**2)
    assert mesh.source.effective_cell_size(0.04) == pytest.approx(0.03)
    assert len(mesh.levels) == 5
    assert tuple(mesh.levels)[::4] == (-0.5, 0.5)
    assert coupling.transfer_region_bounds[1] > 4.0
    assert fv.time.time_step_size == pytest.approx(0.004)
    assert manifest["span_projection"] is False
    assert coupling.transfer_vorticity_cutoff == manifest["transfer_vorticity_cutoff"] == 0.05
    assert manifest["gbd_vorticity_floor"] == 0.01
    for name in ("fvm_transverse_x1", "fvm_transverse_x2", "fvm_transverse_x4", "span_probe"):
        assert name in [s.file_name for s in fv.samplers]


def test_planar_replay_checks_actual_mesh_domain_spacing_and_span(tmp_path):
    module = load("run_cylinder")
    *_, manifest = module.configuration(tmp_path)
    path = tmp_path / "mesh.npz"
    bounds = manifest["fvm_box"]
    vertices = np.array([bounds[::2], bounds[1::2]])
    generation = {
        "resolved_surface_patch_sizes": {"cylinder": 0.03},
        "extrusion_levels": np.linspace(-0.5, 0.5, 5).tolist(),
    }

    def write():
        np.savez(
            path, metadata=json.dumps({"mesh_generation": generation}), vertex_position=vertices
        )

    write()
    module.validate_native_mesh(path, manifest)
    generation["resolved_surface_patch_sizes"]["cylinder"] = 0.04
    write()
    with pytest.raises(ValueError, match="wall spacing"):
        module.validate_native_mesh(path, manifest)
    generation["resolved_surface_patch_sizes"]["cylinder"] = 0.03
    generation["extrusion_levels"] = [-0.5, 0.0, 0.5]
    write()
    with pytest.raises(ValueError, match="span/layers"):
        module.validate_native_mesh(path, manifest)
    vertices[0, 0] -= 0.1
    write()
    with pytest.raises(ValueError, match="domain"):
        module.validate_native_mesh(path, manifest)


def test_planar_pruning_cutoffs_are_explicit_and_dimensionally_consistent(tmp_path):
    module = load("run_cylinder")
    _, vp, coupling, _, manifest = module.configuration(
        tmp_path, particle_spacing=0.1, transfer_cutoff=0.02, gbd_vorticity_floor=0.001
    )
    assert coupling.transfer_vorticity_cutoff == 0.02
    assert vp.numerics.viscous.gbd_threshold == pytest.approx(0.001 * 0.1**2)
    assert manifest["transfer_absolute_strength_cutoff"] == pytest.approx(0.02 * 0.1**2)
    for transfer, gbd in ((0.001, 0.01), (-1, 0), (np.nan, 0)):
        with pytest.raises(ValueError, match="vorticity floor"):
            module.configuration(tmp_path, transfer_cutoff=transfer, gbd_vorticity_floor=gbd)


def test_cylinder_gate_cannot_pass_before_mature_window(tmp_path):
    module = load("compare_cylinder_run")
    samples = tmp_path / "run/samples"
    reference = tmp_path / "reference"
    samples.mkdir(parents=True)
    reference.mkdir()
    history = pd.DataFrame(
        {"time": [0.0, 4.0], "drag_coefficient": [1.3, 1.3], "lift_coefficient": [0.0, 0.1]}
    )
    for directory in (samples, reference):
        history.to_csv(directory / "forces_history.csv", index=False)
    result = module.report(tmp_path / "run", reference)
    assert result["status"] == "before_mature_window"
    assert not result["mature_gate_passed"]


def write_complete_mature_records(tmp_path):
    run = tmp_path / "run"
    samples = run / "samples"
    reference = tmp_path / "reference"
    samples.mkdir(parents=True)
    reference.mkdir()
    time = np.arange(80.0, 160.0001, 0.04)
    history = pd.DataFrame(
        {
            "time": time,
            "drag_coefficient": 1.37 + 0.01 * np.cos(2 * np.pi * 0.4 * time),
            "lift_coefficient": 0.5 * np.sin(2 * np.pi * 0.2 * time),
            "side_force_coefficient": np.zeros_like(time),
        }
    )
    # Match the real ForceSampler CSV schema, including textual patch metadata.
    history["step"] = np.arange(len(time))
    history["accepted_time_step_size"] = 0.004
    history["patch"] = "cylinder"
    for column in (
        "pressure_force_x",
        "pressure_force_y",
        "pressure_force_z",
        "viscous_force_x",
        "viscous_force_y",
        "viscous_force_z",
        "total_force_x",
        "total_force_y",
        "total_force_z",
        "moment_x",
        "moment_y",
        "moment_z",
        "pitching_moment_coefficient",
    ):
        history[column] = 0.0
    history = history[
        [
            "time",
            "step",
            "accepted_time_step_size",
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
            "moment_x",
            "moment_y",
            "moment_z",
            "drag_coefficient",
            "lift_coefficient",
            "side_force_coefficient",
            "pitching_moment_coefficient",
        ]
    ]
    for directory in (samples, reference):
        history.to_csv(directory / "forces_history.csv", index=False)
    columns = [
        "time",
        "position_x",
        "position_y",
        "position_z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
    ]
    for name in ("centreline", "transverse_x1", "transverse_x2", "transverse_x4"):
        rows = []
        for t in np.round(np.arange(80.0, 160.0001, 0.2), 8):
            for q in np.linspace(-2, 6, 21) if name == "centreline" else np.linspace(-3, 3, 21):
                x, y = (q, 0.0) if name == "centreline" else (float(name[-1]), q)
                rows.append([t, x, y, 0.0, 0.8, 0.0, 0.0])
        frame = pd.DataFrame(rows, columns=columns)
        frame.to_csv(samples / f"fvm_{name}.csv", index=False)
        if name == "centreline":
            extra = pd.DataFrame(
                [
                    [t, x, 0.0, 0.0, 0.8, 0.0, 0.0]
                    for t in np.round(np.arange(80.0, 160.0001, 0.2), 8)
                    for x in (-0.56, 0.56, 12.0)
                ],
                columns=columns,
            )
            frame = pd.concat([frame, extra], ignore_index=True).sort_values(["time", "position_x"])
        frame.to_csv(reference / f"{name}.csv", index=False)
        # Use the exact public VPM sampler's float32 lattice and CSV precision.
        a, b = (
            ([0.6, 0.0, 0.0], [12.0, 0.0, 0.0])
            if name == "centreline"
            else ([float(name[-1]), -3.0, 0.0], [float(name[-1]), 3.0, 0.0])
        )
        a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        count = int(np.ceil(np.linalg.norm(b - a) / 0.08)) + 1
        points = (
            a[None, :] + np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None] * (b - a)[None, :]
        )
        times = np.round(np.arange(80.0, 160.0001, 0.2), 8)
        vpm_data = np.column_stack(
            (
                np.repeat(times, count),
                np.tile(points, (len(times), 1)),
                np.tile([0.8, 0.0, 0.0], (len(times) * count, 1)),
            )
        )
        pd.DataFrame(vpm_data, columns=columns).to_csv(samples / f"vpm_{name}.csv", index=False)
    pd.DataFrame(
        [
            [t, 1.5, 0.0, z, 0.8, 0.0, 0.0]
            for t in np.round(np.arange(80.0, 160.0001, 0.2), 8)
            for z in np.linspace(-0.45, 0.45, 9)
        ],
        columns=columns,
    ).to_csv(samples / "span_probe.csv", index=False)
    (run / "solution").mkdir()
    (run / "solution/coupler_diagnostics.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "time": round(float(t), 8),
                    "interface_iteration": {"converged": True},
                    "planar_spanwise_consistency": {
                        "span_velocity_max": 0.0002,
                        "span_variation_max": 0.0004,
                        "velocity_scale": 2.0,
                    },
                }
            )
            + "\n"
            for t in time
        )
    )
    (run / "experiment.json").write_text(
        json.dumps(
            {
                "panel": False,
                "geometry": "span-invariant cylinder Re150",
                "span": 1.0,
                "particle_strength_measure": "omega_z*h^2*span",
                "coupling_dt": 0.04,
                "transfer_box": [-2.0, 6.0, -3.0, 3.0, -0.375, 0.375],
            }
        )
    )
    return run, reference


def test_identical_mature_records_pass_without_phase_fit(tmp_path):
    module = load("compare_cylinder_run")
    run, reference = write_complete_mature_records(tmp_path)
    result = module.report(run, reference)
    assert result["mature_gate_passed"], result
    assert result["phase_adjustment"] is False
    assert result["shedding_cycles"] >= 8.0
    whole_field = result["full_donor_spanwise_consistency"]
    assert whole_field["complete"]
    assert whole_field["max_normalized_span_velocity"] == pytest.approx(0.0001)
    assert whole_field["max_normalized_span_variation"] == pytest.approx(0.0002)
    (run / "experiment.json").unlink()
    unverified = module.report(run, reference)
    assert not unverified["mature_gate_passed"]
    assert not unverified["criteria"]["verified_panel_free_planar_experiment"]


def test_physical_cylinder_execution_requires_verified_cube_gate(tmp_path):
    module = load("run_cylinder")
    gate = tmp_path / "cube_gate.json"
    gate.write_text(
        json.dumps(
            {
                "gate_passed": True,
                "latest_time": 4.0,
                "criteria": {"verified_panel_free_experiment": False, "four_seconds_reached": True},
            }
        )
    )
    with pytest.raises(RuntimeError, match="verified panel-free"):
        module.require_cube_gate(gate)
    gate.write_text(
        json.dumps(
            {
                "gate_passed": True,
                "latest_time": 4.0,
                "criteria": {"verified_panel_free_experiment": True, "four_seconds_reached": True},
            }
        )
    )
    assert module.require_cube_gate(gate)["gate_passed"]


def test_near_wall_profile_targets_never_extrapolate_reference_fluid_support(tmp_path):
    module = load("compare_cylinder_run")
    trial, reference = tmp_path / "trial", tmp_path / "reference"
    trial.mkdir()
    reference.mkdir()
    columns = [
        "time",
        "position_x",
        "position_y",
        "position_z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
    ]
    # The trial's near-wall probes have intentionally enormous values. They
    # cannot be compared: the corresponding reference fluid sides stop earlier.
    target_x = [-2.0, -1.0, -0.55, 0.0, 0.55, 1.0, 2.0]
    source_x = [-2.0, -1.0, -0.6, 0.6, 1.0, 2.0]
    left = [
        [t, x, 0.0, 0.0, 100.0 if abs(x) == 0.55 else x, 0.0, 0.0]
        for t in (0.0, 1.0)
        for x in target_x
    ]
    right = [[t, x, 0.0, 0.0, x, 0.0, 0.0] for t in (0.0, 1.0) for x in source_x]
    pd.DataFrame(left, columns=columns).to_csv(trial / "fvm_centreline.csv", index=False)
    pd.DataFrame(right, columns=columns).to_csv(reference / "centreline.csv", index=False)
    result = module.profile_comparison(trial, reference, "centreline", 0.0, 1.0)
    assert result["mean_velocity_rms_Uinf"] == pytest.approx(0.0)
    assert result["covered_length_D"] == pytest.approx(2.0)
    assert result["points_excluded_solid"] == 1
    assert result["points_excluded_outside_reference_support"] == 2
    assert result["points_excluded"] == 3
    assert [item["compared_extent_D"] for item in result["coverage"]] == [[-2.0, -1.0], [1.0, 2.0]]


def test_irregular_profile_times_use_physical_time_weighting(tmp_path):
    module = load("compare_cylinder_run")
    trial, reference = tmp_path / "trial", tmp_path / "reference"
    trial.mkdir()
    reference.mkdir()
    columns = [
        "time",
        "position_x",
        "position_y",
        "position_z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
    ]
    times = (0.0, 1.0, 10.0)
    mean = 45.5  # Integral of the piecewise-linear t² samples divided by10s.
    left = [[t, 1.0, y, 0.0, t * t, 0.0, 0.0] for t in times for y in (-1.0, 1.0)]
    right = [[t, 1.0, y, 0.0, mean, 0.0, 0.0] for t in times for y in (-1.0, 1.0)]
    pd.DataFrame(left, columns=columns).to_csv(trial / "fvm_transverse_x1.csv", index=False)
    pd.DataFrame(right, columns=columns).to_csv(reference / "transverse_x1.csv", index=False)
    result = module.profile_comparison(trial, reference, "transverse_x1", 0.0, 10.0)
    assert result["mean_velocity_rms_Uinf"] == pytest.approx(0.0)
    assert result["time_window"] == [0.0, 10.0]
    assert "trapezoidal" in result["time_average"]


@pytest.mark.parametrize(
    "field,value",
    [("panel", True), ("geometry", "3D cylinder"), ("particle_strength_measure", "omega*h^3")],
)
def test_nonplanar_or_panelled_identity_cannot_qualify(tmp_path, field, value):
    module = load("compare_cylinder_run")
    manifest = {
        "panel": False,
        "geometry": "span-invariant cylinder Re150",
        "span": 1.0,
        "particle_strength_measure": "omega_z*h^2*span",
    }
    manifest[field] = value
    (tmp_path / "experiment.json").write_text(json.dumps(manifest))
    assert not module.experiment_identity(tmp_path, tmp_path / "reference")["verified"]


@pytest.mark.parametrize("component", ["interfaces", "forces", "profiles"])
def test_missing_middle_sample_rejects_mature_gate(tmp_path, component):
    module = load("compare_cylinder_run")
    run, reference = write_complete_mature_records(tmp_path)
    if component == "interfaces":
        path = run / "solution/coupler_diagnostics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows if row["time"] != 100.0))
        criterion = "complete_interface_cadence"
    else:
        name = "forces_history" if component == "forces" else "fvm_transverse_x2"
        path = run / "samples" / (name + ".csv")
        data = pd.read_csv(path)
        data.loc[~np.isclose(data.time, 100.0, atol=1.0e-8, rtol=0.0)].to_csv(path, index=False)
        criterion = (
            "complete_force_cadence" if component == "forces" else "complete_profile_cadence"
        )
    result = module.report(run, reference)
    assert not result["mature_gate_passed"]
    assert not result["criteria"][criterion]


@pytest.mark.parametrize(
    "corruption", ["late_missing", "middle_missing", "partial_span", "nonfinite"]
)
def test_incomplete_span_evidence_rejects_mature_gate(tmp_path, corruption):
    module = load("compare_cylinder_run")
    run, reference = write_complete_mature_records(tmp_path)
    path = run / "samples/span_probe.csv"
    data = pd.read_csv(path)
    at100 = np.isclose(data.time, 100.0, atol=1.0e-8, rtol=0.0)
    if corruption == "late_missing":
        data = data[data.time < 140.0]
    elif corruption == "middle_missing":
        data = data[~at100]
    elif corruption == "partial_span":
        data = data[~(at100 & (data.position_z > 0.4))]
    else:
        data.loc[at100, "velocity_x"] = np.nan
    data.to_csv(path, index=False)
    result = module.report(run, reference)
    assert not result["mature_gate_passed"]
    assert not result["criteria"]["complete_span_probe_evidence"]


def test_missing_coupling_dt_prevents_mature_admission(tmp_path):
    module = load("compare_cylinder_run")
    run, reference = write_complete_mature_records(tmp_path)
    path = run / "experiment.json"
    manifest = json.loads(path.read_text())
    manifest.pop("coupling_dt")
    path.write_text(json.dumps(manifest))
    result = module.report(run, reference)
    assert not result["mature_gate_passed"]
    assert not result["criteria"]["verified_panel_free_planar_experiment"]
    assert not result["criteria"]["complete_interface_cadence"]


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_metric",
        "missing_field",
        "nonfinite",
        "zero_scale",
        "span_velocity_exceeded",
        "span_variation_exceeded",
    ],
)
def test_whole_donor_planarity_required_throughout_mature_window(tmp_path, corruption):
    module = load("compare_cylinder_run")
    run, reference = write_complete_mature_records(tmp_path)
    path = run / "solution/coupler_diagnostics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(row for row in rows if row["time"] == 100.0)
    values = row["planar_spanwise_consistency"]
    if corruption == "missing_metric":
        row.pop("planar_spanwise_consistency")
    elif corruption == "missing_field":
        values.pop("span_variation_max")
    elif corruption == "nonfinite":
        values["span_velocity_max"] = float("nan")
    elif corruption == "zero_scale":
        values["velocity_scale"] = 0.0
    else:
        field = (
            "span_velocity_max" if corruption == "span_velocity_exceeded" else "span_variation_max"
        )
        values[field] = 0.004  # scale2 -> normalized.002 exceeds the existing1e-3 limit.
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = module.report(run, reference)
    assert not result["mature_gate_passed"]
    if corruption.endswith("exceeded"):
        assert not result["criteria"]["full_donor_spanwise_invariance_below_1e_3_scaled"]
    else:
        assert not result["criteria"]["complete_full_donor_span_evidence"]
        assert result["full_donor_spanwise_consistency"]["first_invalid_times"] == [100.0]


@pytest.mark.parametrize(
    "column,value",
    [
        ("drag_coefficient", "not numeric"),
        ("lift_coefficient", float("inf")),
        ("side_force_coefficient", float("nan")),
    ],
)
def test_real_force_schema_validates_every_consumed_coefficient(tmp_path, column, value):
    module = load("compare_cylinder_run")
    run, reference = write_complete_mature_records(tmp_path)
    path = run / "samples/forces_history.csv"
    table = pd.read_csv(path)
    assert table["patch"].eq("cylinder").all()
    table[column] = table[column].astype(object)
    table.loc[np.isclose(table.time, 100.0, atol=1.0e-8, rtol=0.0), column] = value
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="force"):
        module.force_comparison(run / "samples", reference, 80.0, 160.0)


def test_off_bin_cycle_frequency_resolves_clean_shedding():
    module = load("compare_cylinder_run")
    time = np.arange(80.0, 160.00001, 0.04)
    frequency = 0.1931
    result = module.cycle_frequency(time, np.sin(2 * np.pi * frequency * time + 0.27))
    assert result["qualified"], result
    assert result["frequency"] == pytest.approx(frequency, rel=2.0e-5)
    assert result["fft_relative_bin_width"] > 0.03
    low, high = result["sampling_frequency_interval"]
    assert low <= frequency <= high
    assert result["relative_frequency_half_width"] < 0.002
    a0, b0 = result["endpoint_crossing_brackets"][0]
    aN, bN = result["endpoint_crossing_brackets"][1]
    assert low == pytest.approx(result["complete_periods"] / (bN - a0))
    assert high == pytest.approx(result["complete_periods"] / (aN - b0))


@pytest.mark.parametrize("kind", ["drifting", "noisy"])
def test_nonstationary_or_noisy_shedding_is_unqualified(kind):
    module = load("compare_cylinder_run")
    time = np.arange(0.0, 80.00001, 0.04)
    if kind == "drifting":
        phase = 2 * np.pi * (0.16 * time + 0.5 * (0.24 - 0.16) * time**2 / 80.0)
        lift = np.sin(phase)
    else:
        lift = np.sin(2 * np.pi * 0.2 * time) + 0.45 * np.sin(2 * np.pi * 2.93 * time + 0.2)
    result = module.cycle_frequency(time, lift)
    assert not result["qualified"], result
    assert result["reasons"]


def test_matching_fft_bins_do_not_establish_three_percent_frequency_agreement(tmp_path):
    module = load("compare_cylinder_run")
    trial, reference = tmp_path / "trial", tmp_path / "reference"
    trial.mkdir()
    reference.mkdir()
    time = np.arange(80.0, 160.00001, 0.04)
    for directory, frequency in ((trial, 0.202), (reference, 0.195)):
        pd.DataFrame(
            {
                "time": time,
                "patch": "cylinder",
                "drag_coefficient": 1.37,
                "lift_coefficient": np.sin(2 * np.pi * frequency * time),
            }
        ).to_csv(directory / "forces_history.csv", index=False)
    result = module.force_comparison(trial, reference, 80.0, 160.0)
    assert result["relative_changes"]["strouhal_lift_fft_bin"] == pytest.approx(0.0)
    assert result["frequency_comparison"]["qualified"]
    assert result["relative_changes"]["strouhal_lift"] > 0.03
    assert result["frequency_comparison"]["conservative_relative_difference_bound"] > 0.03


@pytest.mark.parametrize("name", ["centreline", "transverse_x1", "transverse_x2", "transverse_x4"])
def test_missing_vpm_profile_prevents_mature_gate(tmp_path, name):
    module = load("compare_cylinder_run")
    run, reference = write_complete_mature_records(tmp_path)
    (run / "samples" / f"vpm_{name}.csv").unlink()
    result = module.report(run, reference)
    assert not result["mature_gate_passed"]
    assert not result["criteria"]["complete_vpm_profile_evidence"]
    assert not result["vpm_profiles"][name]["available"]


def test_exterior_vpm_error_cannot_hide_in_good_fvm_or_whole_line_mean(tmp_path):
    module = load("compare_cylinder_run")
    run, reference = write_complete_mature_records(tmp_path)
    path = run / "samples/vpm_centreline.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.position_x > 6.0, "velocity_y"] = 0.04
    frame.to_csv(path, index=False)
    result = module.report(run, reference)
    assert result["criteria"]["mean_profiles_within_3pct_Uinf"]
    assert result["criteria"]["vpm_mean_profiles_within_3pct_Uinf"]
    assert result["criteria"]["complete_vpm_profile_evidence"]
    assert not result["criteria"]["vpm_exterior_centreline_within_3pct_Uinf"]
    assert not result["mature_gate_passed"]
    assert result["vpm_profiles"]["centreline_exterior"]["mean_velocity_rms_Uinf"] == pytest.approx(
        0.04
    )
    assert result["vpm_profiles"]["centreline_exterior"]["minimum_coordinate_exclusive"] == 6.0


@pytest.mark.parametrize("corruption", ["middle_time", "missing_endpoint", "reference_too_short"])
def test_vpm_requires_complete_time_and_spatial_reference_coverage(tmp_path, corruption):
    module = load("compare_cylinder_run")
    run, reference = write_complete_mature_records(tmp_path)
    path = (
        reference / "centreline.csv"
        if corruption == "reference_too_short"
        else run / "samples/vpm_centreline.csv"
    )
    frame = pd.read_csv(path)
    if corruption == "middle_time":
        frame = frame[~np.isclose(frame.time, 100.0, rtol=0.0, atol=1.0e-8)]
    else:
        frame = frame[frame.position_x < 12.0]
    frame.to_csv(path, index=False)
    result = module.report(run, reference)
    assert not result["mature_gate_passed"]
    assert not result["criteria"]["complete_vpm_profile_evidence"]


def test_serialized_endpoint_roundoff_is_not_a_missing_vpm_probe(tmp_path):
    module = load("compare_cylinder_run")
    trial, reference = tmp_path / "trial", tmp_path / "reference"
    trial.mkdir()
    reference.mkdir()
    columns = [
        "time",
        "position_x",
        "position_y",
        "position_z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
    ]
    for directory, name, positions in (
        (trial, "vpm_centreline", [0.6, 1.0, 2.0]),
        (reference, "centreline", [0.6000000000000001, 1.0, 2.0]),
    ):
        values = [[t, x, 0.0, 0.0, x, 0.0, 0.0] for t in (0.0, 1.0) for x in positions]
        pd.DataFrame(values, columns=columns).to_csv(directory / (name + ".csv"), index=False)
    result = module.profile_comparison(trial, reference, "centreline", 0.0, 1.0, prefix="vpm_")
    assert result["compared_points"] == 3
    assert result["points_excluded_outside_reference_support"] == 0
    assert result["roundoff_snapped_points"] == 1
    assert result["mean_velocity_rms_Uinf"] == pytest.approx(0.0, abs=1.0e-14)


def test_span_probe_uses_raw_cells_without_changing_velocity_profile_stencils(tmp_path):
    module = load("run_cylinder")
    fv, *_ = module.configuration(tmp_path)
    lines = {sampler.file_name: sampler for sampler in fv.samplers}
    assert lines["span_probe"].k == 1
    assert lines["span_probe"].reconstruction == "idw"
    from source.solvers.fvm.io.backup import _setup_dict

    numerical = _setup_dict(fv)
    lines["span_probe"].k = 12
    lines["span_probe"].reconstruction = "affine"
    assert _setup_dict(fv) == numerical
    for name in ("fvm_centreline", "fvm_transverse_x1", "fvm_transverse_x2", "fvm_transverse_x4"):
        assert lines[name].k == 12 and lines[name].reconstruction == "affine"


def test_span_diagnostic_does_not_create_variation_from_planar_nonlinear_field():
    from source.solvers.fvm.sampling.fields import _PointProbe

    centres = np.array(
        [
            [x, y, z]
            for z in (-0.375, -0.125, 0.125, 0.375)
            for x in (1.35, 1.43, 1.51, 1.59, 1.67)
            for y in (-0.13, -0.05, 0.03, 0.11)
        ]
    )
    values = (centres[:, 0] - 1.5) ** 2 + centres[:, 1] ** 2
    points = np.array([[1.5, 0, z] for z in np.linspace(-0.45, 0.45, 9)])
    affine = _PointProbe(points, k=12, reconstruction="affine")
    assert np.ptp(affine._interpolate(values, centres)) > 0.001
    raw = _PointProbe(points, k=1, reconstruction="idw")
    indices, _ = raw._interpolation_stencil(centres)
    assert len(np.unique(centres[indices[:, 0], :2], axis=0)) == 1
    assert len(np.unique(centres[indices[:, 0], 2])) == 4
    assert np.ptp(raw._interpolate(values, centres)) < 1e-14
    # A true z-dependent field is still detected; this is not a span projection.
    assert np.ptp(raw._interpolate(values + centres[:, 2] * 0.01, centres)) > 0.001
