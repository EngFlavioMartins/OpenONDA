"""Analytical checks for the cube comparison, independent of any simulation."""

import importlib
import json

import numpy as np
import pytest

PACKAGE = "tutorials.coupled_fvm_vpm.02_cube_flow.assets."


@pytest.fixture
def modules():
    return (
        importlib.import_module(PACKAGE + "plot_coupled_fvm_vpm_fields"),
        importlib.import_module(PACKAGE + "postprocess"),
    )


def test_vector_norm_includes_transverse_components_and_area_weights(modules):
    fields, _ = modules
    x, y = np.meshgrid([1.0, 2.0, 4.0], [1.0, 2.0])
    left = np.zeros((*x.shape, 3))
    right = np.broadcast_to([0.0, 0.03, 0.04], left.shape).copy()
    error, valid, stats = fields.differences(x, y, left, right)
    np.testing.assert_allclose(error, 5.0)
    assert stats["rms_percent"] == pytest.approx(5)
    assert stats["sampled_max_percent"] == pytest.approx(5)
    assert stats["covered_area_D2"] == pytest.approx(3)
    np.testing.assert_allclose(
        fields.area_weights(x, y, valid), [[0.25, 0.75, 0.5], [0.25, 0.75, 0.5]]
    )


def test_missing_support_is_not_zero_error_or_extrapolated(modules):
    fields, _ = modules
    x, y = np.meshgrid([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    left = np.zeros((*x.shape, 3))
    right = np.ones_like(left)
    left[1, 1] = np.nan
    error, valid, stats = fields.differences(x, y, left, right)
    assert np.isnan(error[1, 1])
    assert stats["covered_area_D2"] == pytest.approx(5)
    _, _, dense = fields.display_grid(x, y, right, valid)
    assert np.isnan(dense[4, 4]).all()
    assert stats["rms_percent"] == pytest.approx(100 * np.sqrt(3))


def test_cube_mask_has_no_artificial_near_wall_halo(modules):
    fields, prepare = modules
    x, y = np.meshgrid([-0.7, -0.52, -0.4, 0.4, 0.52, 0.7], [-0.7, 0.0, 0.7])
    velocity = np.zeros((*x.shape, 3))
    _, valid, _ = fields.differences(x, y, velocity, velocity)
    assert valid[1, 1] and valid[1, 4]
    assert not valid[1, 2] and not valid[1, 3]
    np.testing.assert_array_equal(
        prepare.fluid_points(np.array([[0.5, 0, 0], [0.52, 0, 0]])), [False, True]
    )


def test_display_interpolates_vectors_before_norm_and_does_not_change_statistics(modules):
    fields, _ = modules
    x, y = np.meshgrid([1.0, 2.0], [1.0, 2.0])
    left = np.zeros((*x.shape, 3))
    right = np.zeros_like(left)
    right[..., 2] = [[-1, 1], [-1, 1]]
    _, valid, stats = fields.differences(x, y, left, right)
    for subdivisions in (2, 4, 8):
        _, _, dense = fields.display_grid(x, y, right - left, valid, subdivisions)
        assert np.linalg.norm(dense[subdivisions // 2, subdivisions // 2]) == pytest.approx(0)
    assert stats["rms_percent"] == 100
    assert stats["sampled_max_percent"] == 100


def test_sampled_max_is_not_replaced_with_a_percentile(modules):
    fields, _ = modules
    x, y = np.meshgrid(np.arange(1.0, 12.0), np.arange(1.0, 12.0))
    left = np.zeros((*x.shape, 3))
    right = np.zeros_like(left)
    right[5, 5, 2] = 2.0
    error, _, stats = fields.differences(x, y, left, right)
    assert np.percentile(error, 95) == 0.0
    assert stats["sampled_max_percent"] == 200.0


def test_coordinate_check_accepts_roundoff_only_and_respects_validity(modules):
    fields, _ = modules
    x, y = np.meshgrid([1.0, 2.0], [1.0, 2.0])
    target = {"x": x, "y": y}
    source = {**target, "velocity": np.ones((2, 2, 3)), "valid": np.array([[1, 0], [1, 1]], bool)}
    source["x"] = x + 1e-7
    assert np.isnan(fields._on_grid(source, target)[0, 1]).all()
    source["x"] = x + 0.001
    with pytest.raises(ValueError, match="coordinates differ"):
        fields._on_grid(source, target)


def test_native_sampling_recovers_affine_3d_field_in_mpi_cell_order(modules, tmp_path):
    _, prepare = modules
    pv = pytest.importorskip("pyvista")
    grid = pv.ImageData(
        dimensions=(5, 5, 5), spacing=(0.25, 0.25, 0.25), origin=(2.0, 2.0, 2.0)
    ).cast_to_unstructured_grid()
    centres = grid.cell_centers().points.copy()
    ids = np.random.default_rng(42).permutation(grid.n_cells)
    # Native IDs need not follow VTK cell order, as with merged MPI pieces.
    native_centres = np.empty_like(centres)
    native_centres[ids] = centres
    grid.cell_data["global_cell_id"] = ids
    matrix = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0], [3.0, 1.0, -1.0]])
    grid.cell_data["velocity"] = centres @ matrix.T + [1.0, 2.0, 3.0]
    path = tmp_path / "frame.vtu"
    grid.save(path)
    sampler = prepare.NativeVelocity.__new__(prepare.NativeVelocity)
    sampler.centres, sampler.probes = native_centres, {}
    points = np.array([[2.36, 2.48, 2.55], [2.61, 2.72, 2.69], [3.5, 2.5, 2.5]])
    values = sampler.sample(path, {"query": points})["query"]
    np.testing.assert_allclose(values[:2], points[:2] @ matrix.T + [1.0, 2.0, 3.0], atol=1e-12)
    assert np.isnan(values[2]).all()
    grid.cell_data["global_cell_id"][1] = grid.cell_data["global_cell_id"][0]
    grid.save(path)
    with pytest.raises(ValueError, match="exactly once"):
        sampler.sample(path, {"query": points})


def test_raw_drag_spikes_are_preserved_and_duplicate_times_rejected(modules, tmp_path, monkeypatch):
    fields, _ = modules
    util = fields.util
    p = tmp_path / "forces_history.csv"
    original = "step,time,patch,drag_coefficient,accepted_time_step_size\n1,8.35,cube,1.1,.01\n2,8.4,cube,9.557578655,.0000129627\n3,8.45,cube,1.2,.01\n"
    p.write_text(original)
    monkeypatch.setitem(util.SOURCES, "reference", {"dir": tmp_path})
    profile = importlib.import_module(PACKAGE + "plot_velocity_profiles")
    times, cd = profile._force_series("reference", 8.4)
    np.testing.assert_allclose(times, [8.35, 8.4])
    np.testing.assert_allclose(cd, [1.1, 9.557578655])
    assert p.read_text() == original
    p.write_text(original.replace("1,8.35,cube,1.1,.01\n", "1,8.35,cube,1.1,.01\n" * 2))
    assert len(util.load_forces("reference")["time"]) == 3
    p.write_text(original + "4,8.45,cube,1.2,.01\n")
    with pytest.raises(ValueError, match="duplicate"):
        util.load_forces("reference")


def test_neighbouring_times_are_never_substituted(modules):
    fields, _ = modules
    np.testing.assert_allclose(
        fields.util.common_times(np.array([1.0, 2.0]), np.array([1.0 + 1e-12, 2.01])), [1.0]
    )


def test_pimple_comparison_uses_current_executable_controls(modules):
    fields, _ = modules
    util = fields.util
    current = {name: index for index, name in enumerate(util._PIMPLE_COMPARISON_FIELDS)}
    recorded_before_alias_removal = {**current, "n_orthogonal_correctors": 1}
    assert util._pimple_configurations_match(current, recorded_before_alias_removal)
    for name in util._PIMPLE_COMPARISON_FIELDS:
        changed = {**recorded_before_alias_removal, name: object()}
        assert not util._pimple_configurations_match(current, changed)
    missing = current.copy()
    missing.pop("n_nonorthogonal_correctors")
    assert not util._pimple_configurations_match(current, missing)


def test_fvm_artifacts_follow_the_saved_solution_layout(modules, tmp_path):
    _, prepare = modules
    current = tmp_path / "current"
    current.mkdir()
    (current / "fvm_metadata.json").write_text(json.dumps({"case_name": "current_case"}))
    (current / "fvm").mkdir()
    (current / "fvm.pvd").touch()
    (current / "fvm/mesh.npz").touch()
    assert prepare._fvm_artifacts(current) == (current / "fvm.pvd", current / "fvm/mesh.npz")

    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "fvm_metadata.json").write_text(json.dumps({"case_name": "fine"}))
    (reference / "fine.pvd").touch()
    (reference / "mesh.npz").touch()
    assert prepare._fvm_artifacts(reference) == (reference / "fine.pvd", reference / "mesh.npz")


def test_plotter_selects_finest_complete_reference_run(modules, tmp_path, monkeypatch):
    _, prepare = modules
    root = tmp_path / "reference_flow"
    monkeypatch.setattr(prepare, "CASE_DIR", tmp_path)

    def archived_run(name):
        solution = root / "solution" / name
        samples = root / "samples" / name
        (solution / "fvm").mkdir(parents=True)
        samples.mkdir(parents=True)
        (solution / "fvm_metadata.json").write_text(json.dumps({"case_name": name}))
        (solution / "fvm.pvd").touch()
        (solution / "fvm" / "mesh.npz").touch()
        for basename in ("forces_history", "centreline", "offaxis_y075"):
            (samples / f"{basename}.csv").touch()
        return solution, samples

    archived_run("grid_h010125")
    archived_run("grid_h0045")
    incomplete, _ = archived_run("grid_h003")
    (incomplete / "fvm" / "mesh.npz").unlink()

    selected = prepare.reference_run()
    assert selected.name == "grid_h0045"
    assert selected.target_spacing == pytest.approx(0.045)
    assert prepare._path("reference", "centreline", ".csv") == selected.samples / "centreline.csv"

    archived_run("fine")
    assert prepare.reference_run().name == "fine"


@pytest.mark.parametrize("state", ["missing_volume_index", "no_common_time"])
def test_preparation_waits_for_saved_common_states_without_traceback_or_stale_plots(
    modules,
    tmp_path,
    monkeypatch,
    capsys,
    state,
):
    _, prepare = modules
    util = prepare
    samples, solution = tmp_path / "samples", tmp_path / "solution"
    reference_samples = tmp_path / "reference_flow/samples/fine"
    reference_solution = tmp_path / "reference_flow/solution/fine"
    comparison = samples / "comparison"
    for path in (samples, solution, reference_samples, reference_solution, comparison):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(util, "CASE_DIR", tmp_path)
    monkeypatch.setattr(util, "SOLUTION", solution)
    monkeypatch.setattr(util, "SAMPLES", samples)
    monkeypatch.setattr(util, "REFERENCE_SAMPLES", reference_samples)
    monkeypatch.setattr(util, "COMPARISON", comparison)
    monkeypatch.setitem(util.SOURCES, "vpm", {"dir": samples, "prefix": "vpm_"})
    (reference_samples / "grid_run.json").write_text(
        json.dumps({"case": "fine", "cell_size": 0.06})
    )
    (solution / "fvm_metadata.json").write_text(json.dumps({"case_name": "coupled_cube_flow"}))
    previous = comparison / "manifest.json"
    previous.write_text('{"frames": [{"time": 9.0, "file": "previous.npz"}]}')
    original = previous.read_bytes()
    if state == "no_common_time":
        # Samples exist before the next full-volume output. Never substitute
        # the reference at t=1 for the coupled state at t=.5.
        for path, time in (
            (solution / "fvm.pvd", 0.5),
            (reference_solution / "fvm.pvd", 1.0),
            (samples / "fvm_slice_z0.pvd", 0.5),
            (samples / "vpm_slice_z0.pvd", 0.5),
        ):
            path.write_text(
                f'<VTKFile><Collection><DataSet timestep="{time}" file="state.vtu"/></Collection></VTKFile>'
            )
        for name in ("centreline", "offaxis_y075"):
            (samples / f"vpm_{name}.csv").write_text("time,step\n0.5,50\n")
    assert prepare.main() == 2
    output = capsys.readouterr()
    assert "Comparison plots are not ready yet" in output.err
    assert "Rerun ./allplot.sh" in output.err
    assert "Traceback" not in output.err
    assert previous.read_bytes() == original
    assert not list(comparison.glob("*.npz"))


def test_all_field_pairings_use_identical_sample_support(modules, monkeypatch):
    fields, _ = modules
    x, y = np.meshgrid(np.arange(1.0, 5.0), np.arange(1.0, 5.0))
    data = {}
    for source in ("fvm", "reference", "vpm"):
        data[source] = {
            "x": x,
            "y": y,
            "velocity": np.ones((*x.shape, 3)),
            "valid": np.ones(x.shape, bool),
        }
    data["fvm"]["valid"][1, 1] = False
    monkeypatch.setattr(fields.util, "load_slice", lambda source, time: data[source])
    areas = []

    def record(time, xx, yy, left, right, *args, **kwargs):
        _, _, stats = fields.differences(xx, yy, left, right)
        areas.append(stats["covered_area_D2"])
        return stats

    monkeypatch.setattr(fields, "_field_figure", record)
    for comparison in fields.COMPARISONS:
        fields.plot_frame(
            1,
            {"reference_length": 1, "freestream_speed": 1},
            comparison=comparison,
        )
    assert areas == pytest.approx([5, 5, 5])


def test_coupling_costs_are_normalized_to_one_fvm_step():
    diagnostics = importlib.import_module(PACKAGE + "plot_coupling_diagnostics")
    records = [
        {"n_fvm_substeps": 5, "timing_seconds": {"vpm": 10.0}},
        {"n_fvm_substeps": 4, "timing_seconds": {"vpm": 8.0}},
    ]
    np.testing.assert_allclose(diagnostics._timing_per_fvm_step(records, "vpm"), [2.0, 2.0])


def test_coupling_diagnostics_figure_keeps_only_requested_population_series(
    monkeypatch,
):
    diagnostics = importlib.import_module(PACKAGE + "plot_coupling_diagnostics")
    records = []
    for time, total, injected in ((0.05, 10, 4), (0.10, 20, 5)):
        records.append(
            {
                "time": time,
                "n_fvm_substeps": 5,
                "timing_seconds": {
                    "vpm": 10.0,
                    "fvm": 5.0,
                    "vpm_boundary_condition": 1.0,
                    "transfer": 2.0,
                },
                "transfer": {
                    "n_particles_after": total,
                    "n_particles_injected": injected,
                    "state_change_vortex_strength_net_x": 1.0e-5,
                    "state_change_vortex_strength_net_y": 0.0,
                    "state_change_vortex_strength_net_z": 0.0,
                },
            }
        )
    saved = {}
    monkeypatch.setattr(diagnostics, "_records", lambda: records)
    monkeypatch.setattr(
        diagnostics.util, "save", lambda figure, *args: saved.setdefault("figure", figure)
    )

    diagnostics.plot("png", dpi=72)

    figure = saved["figure"]
    assert len(figure.axes) == 3
    assert [line.get_label() for line in figure.axes[1].lines] == ["total", "injected"]
    assert figure.axes[0].get_ylabel() == "Cost [s]"
    assert figure.axes[2].get_title() == "(c) Net change in total vortex strength"


def test_field_comparison_places_panel_labels_at_top_and_keeps_x_axes_visible(
    modules,
    monkeypatch,
):
    fields, _ = modules
    coordinates = np.linspace(-1.0, 1.0, 9)
    x, y = np.meshgrid(coordinates, coordinates)
    left = np.zeros((*x.shape, 3))
    right = np.zeros_like(left)
    saved = {}
    monkeypatch.setattr(
        fields.util, "save", lambda figure, *args: saved.setdefault("figure", figure)
    )

    fields._field_figure(
        1.0,
        x,
        y,
        left,
        right,
        "(a) Left",
        "(b) Right",
        "test_fields",
        "png",
        72,
    )

    axes = saved["figure"].axes[:3]
    assert [axis.get_title() for axis in axes] == ["(a) Left", "(b) Right", "(c) Difference"]
    assert [axis.get_xlabel() for axis in axes] == ["", "", r"$x/D$"]
    assert all(label.get_visible() for axis in axes for label in axis.get_xticklabels())
