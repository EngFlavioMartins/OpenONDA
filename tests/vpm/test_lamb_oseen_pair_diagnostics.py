"""Tests for the unified field-based Lamb--Oseen vortex diagnostics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tutorials.VPM.lambOseenVortex.assets.plot_vortex_comparison import (
    lamb_oseen_profile,
    load_profile,
)
from tutorials.VPM.lambOseenVortex.assets.vortex_diagnostics import (
    BETA_RMAX,
    FIELD_CSV_COLUMNS,
    _core_radius_diagnostic,
    _core_radius_utheta,
    _diagnostics_row,
    average_field_diagnostics,
    resolve_runtime_physics,
    unwrap_pair_orientation,
)


def _superposed_lamb_oseen_field(vortices, extent=1.2, spacing=0.02) -> dict:
    """Synthetic z=const velocity/vorticity plane from superposed Lamb-Oseen
    vortices. ``vortices``: list of ``(cx, cy, gamma, core_radius)``."""
    xs = np.arange(-extent, extent + spacing / 2, spacing)
    ys = np.arange(-extent, extent + spacing / 2, spacing)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    ux = np.zeros_like(x)
    uy = np.zeros_like(x)
    wz = np.zeros_like(x)
    for cx, cy, gamma, core_radius in vortices:
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx**2 + dy**2)
        vel, oz, _ = lamb_oseen_profile(r, core_radius**2, gamma, 0.25)
        r_safe = np.where(r > 1e-12, r, 1.0)
        ux += vel * (-dy / r_safe)
        uy += vel * (dx / r_safe)
        wz += oz
    return {"x": x, "y": y, "Ux": ux, "Uy": uy, "omega_z": wz}


def _row_dict(field: dict, physics: str) -> dict:
    row = _diagnostics_row(field, physics)
    return dict(zip(FIELD_CSV_COLUMNS, row, strict=True))


def test_strength_cutoff_normalization_preserves_requested_circulation():
    from tutorials.VPM.lambOseenVortex.lambossen_setup import (
        normalize_retained_circulation,
    )

    particle_circulation = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
    normalized, raw_per_length, factor = normalize_retained_circulation(
        particle_circulation,
        np.array([True, False]),
        requested_circulation_per_length=1.0,
        column_length=4.0,
    )

    assert raw_per_length == pytest.approx(0.5)
    assert factor == pytest.approx(2.0)
    assert normalized[:, 2].sum() / 4.0 == pytest.approx(1.0)


def test_core_radius_utheta_recovers_known_lamb_oseen_core():
    field = _superposed_lamb_oseen_field([(0.0, 0.0, 1.0, 0.15)], extent=1.0, spacing=0.01)
    a_c = _core_radius_utheta(field, np.array([0.0, 0.0]), r_max=0.45)
    assert a_c == pytest.approx(BETA_RMAX * 0.15, rel=0.05)


def test_field_diagnostics_recovers_dipole_centers_and_core_radius():
    field = _superposed_lamb_oseen_field([(0.0, 0.5, 1.0, 0.10), (0.0, -0.5, -1.0, 0.10)])
    field["time"] = 0.25
    field["step"] = 1

    result = _row_dict(field, "dipole")

    assert result["center0_x"] == pytest.approx(0.0, abs=0.02)
    assert result["center0_y"] == pytest.approx(0.5, abs=0.02)
    assert result["center1_y"] == pytest.approx(-0.5, abs=0.02)
    assert result["separation"] == pytest.approx(1.0, abs=0.03)
    assert result["a_c0"] == pytest.approx(BETA_RMAX * 0.10, abs=0.02)
    assert result["a_c1"] == pytest.approx(BETA_RMAX * 0.10, abs=0.02)
    assert result["merged"] is False


def test_field_diagnostics_flags_merged_pair_when_cores_overlap():
    field = _superposed_lamb_oseen_field(
        [(0.0, 0.05, 1.0, 0.15), (0.0, -0.05, 1.0, 0.15)], extent=0.8
    )
    field["time"] = 1.0
    field["step"] = 3

    result = _row_dict(field, "merging")

    # Overlapping cores collapse to a single detected peak: no second center,
    # separation is NaN, and "merged" (defined as "no valid pair separation")
    # follows.
    assert np.isnan(result["center1_x"])
    assert np.isnan(result["separation"])
    assert result["merged"] is True


def test_merging_diagnostics_retains_two_peaks_after_high_vorticity_regions_connect():
    field = _superposed_lamb_oseen_field(
        [(0.0, 0.125, 1.0, 0.15), (0.0, -0.125, 1.0, 0.15)],
        extent=0.8,
        spacing=0.01,
    )
    field["time"] = 1.0
    field["step"] = 3

    result = _row_dict(field, "merging")

    # The high-vorticity regions are already connected, but two local maxima
    # still exist.  Peak tracking must retain the late two-core branch.
    assert np.isfinite(result["center0_x"])
    assert np.isfinite(result["center1_x"])
    assert result["separation"] > 0.1
    assert result["merged"] is False


def test_production_configuration_uses_converged_spacing_and_dvh_subcycling():
    from tutorials.VPM.lambOseenVortex.lambossen_setup import (
        CORE_RADIUS,
        DVH_MAX_NODES,
        DVH_RD_RATIO,
        FIELD_SPACING,
        GAUSSIAN_CORE_RADIUS,
        SPACING,
        TIME_STEP,
        scheme_time_control,
        viscous_config,
    )

    viscosity = 1.0 / 530.0
    viscous = viscous_config("dvh", viscosity, SPACING)
    dt, steps, final_time, diffusion_interval, substeps = scheme_time_control(
        "dvh", viscous, TIME_STEP, 30.0
    )

    assert pytest.approx(0.3375) == SPACING / CORE_RADIUS
    assert pytest.approx(0.15) == FIELD_SPACING / CORE_RADIUS
    assert np.isclose(BETA_RMAX * GAUSSIAN_CORE_RADIUS, CORE_RADIUS)
    assert viscous.dvh_rd_ratio == DVH_RD_RATIO == 4
    assert viscous.dvh_max_nodes == DVH_MAX_NODES == 300_000
    assert dt <= TIME_STEP
    assert diffusion_interval == pytest.approx(0.291)
    assert substeps == 30
    assert steps == 3090
    assert final_time == pytest.approx(29.973)


def test_field_diagnostics_handles_lone_vortex():
    field = _superposed_lamb_oseen_field([(0.0, 0.0, 1.0, 0.15)], extent=1.0, spacing=0.02)
    field["time"] = 5.0
    field["step"] = 10

    result = _row_dict(field, "vortex")

    assert result["center0_x"] == pytest.approx(0.0, abs=0.02)
    assert result["center0_y"] == pytest.approx(0.0, abs=0.02)
    assert np.isnan(result["center1_x"])
    assert result["a_c0"] == pytest.approx(BETA_RMAX * 0.15, rel=0.05)
    # Merger state is meaningful only for the co-rotating pair.
    assert np.isnan(result["separation"])
    assert result["merged"] is False


def test_pair_orientation_is_pi_periodic_across_center_label_swaps():
    physical = np.deg2rad([80.0, 95.0, 110.0, 125.0, 140.0])
    labeled = physical.copy()
    labeled[2:] -= np.pi

    unwrapped = unwrap_pair_orientation(labeled)

    assert np.degrees(unwrapped - unwrapped[0]) == pytest.approx([0, 15, 30, 45, 60])


def test_core_radius_marks_a_search_boundary_peak_invalid():
    field = _superposed_lamb_oseen_field([(0.0, 0.0, 1.0, 0.25)], extent=0.8, spacing=0.01)

    core, limited = _core_radius_diagnostic(field, np.array([0.0, 0.0]), r_max=0.16)

    assert np.isfinite(core)
    assert limited is True


def test_runtime_physics_distinguishes_legacy_and_defined_core_radius(tmp_path):
    legacy = tmp_path / "merging_cs"
    legacy.mkdir()
    base = {"viscosity": 0.01, "core_radius": 0.125, "column_half_length": 3.0}
    (legacy / "run_metadata.json").write_text(json.dumps(base), encoding="utf-8")

    runtime = resolve_runtime_physics(tmp_path, 1.0, 0.01, 1.0, 0.125, prefix="merging")

    assert runtime["ac0"] == pytest.approx(0.125 / BETA_RMAX)
    assert runtime["velocity_peak_radius0"] == pytest.approx(0.125)
    assert runtime["column_length"] == pytest.approx(6.0)

    base["core_radius_definition"] = "gaussian_1_over_e_vorticity_radius"
    (legacy / "run_metadata.json").write_text(json.dumps(base), encoding="utf-8")
    runtime = resolve_runtime_physics(tmp_path, 1.0, 0.01, 1.0, 0.125, prefix="merging")
    assert runtime["ac0"] == pytest.approx(0.125)
    assert runtime["velocity_peak_radius0"] == pytest.approx(BETA_RMAX * 0.125)


def test_runtime_physics_fallback_uses_paper_velocity_peak_radius(tmp_path):
    runtime = resolve_runtime_physics(tmp_path, 1.0, 0.01, 1.0, 0.125, prefix="merging")

    assert runtime["ac0"] == pytest.approx(0.125 / BETA_RMAX)
    assert runtime["velocity_peak_radius0"] == pytest.approx(0.125)


def test_surface_top_tile_puts_zero_boundary_at_first_row():
    from tutorials.VPM.lambOseenVortex.assets.plot_vortex_surface_fields import _tile

    field = np.arange(16).reshape(4, 4)
    bcol = np.array([10.0, 11.0, 12.0, 13.0])
    brow = np.array([20.0, 21.0, 22.0, 23.0])
    corner = 99.0

    top_left = _tile(field, "TL", 2, 2, bcol, brow, corner)
    top_right = _tile(field, "TR", 2, 2, bcol, brow, corner)

    assert top_left[0].tolist() == [20.0, 21.0, 99.0]
    assert top_right[0].tolist() == [99.0, 22.0, 23.0]


def test_energy_initial_point_uses_finite_column_length():
    from tutorials.VPM.lambOseenVortex.assets.plot_lamboseen_energy import prepend_initial_point

    data = {"t": np.array([1.0]), "dedt": np.array([-1.0]), "nu_omega": np.array([-1.0])}
    result = prepend_initial_point(data, gamma=2.0, t0=0.5, n_vortices=1, column_length=3.0)

    expected = -(2.0**2) * 3.0 / (8.0 * np.pi * 0.5)
    assert result["dedt"][0] == pytest.approx(expected)
    assert result["nu_omega"][0] == pytest.approx(expected)


def test_final_vortex_profile_slices_the_y_zero_row(tmp_path, monkeypatch):
    import tutorials.VPM.lambOseenVortex.assets.plot_vortex_comparison as plot_vortex_comparison

    field = _superposed_lamb_oseen_field([(0.0, 0.0, 1.0, 0.15)], extent=0.6, spacing=0.02)
    monkeypatch.setattr(plot_vortex_comparison, "pvd_time_map", lambda *a, **k: {50: 20.0})
    monkeypatch.setattr(
        plot_vortex_comparison,
        "read_surface_field",
        lambda path: field,
    )
    case_dir = tmp_path / "vortex_cs"
    case_dir.mkdir()
    (case_dir / "vortex_cs_zq_000050.vts").write_text("", encoding="utf-8")

    result = load_profile(tmp_path, "cs")

    assert result is not None
    x, uy, oz, time = result
    assert time == 20.0
    # The y row nearest 0 should be the y=0.0 row itself (grid includes it
    # exactly here since extent/spacing are symmetric around 0).
    j0 = int(np.argmin(np.abs(field["y"][0, :])))
    assert np.array_equal(x, field["x"][:, j0])
    assert np.array_equal(uy, field["Uy"][:, j0])
    assert np.array_equal(oz, field["omega_z"][:, j0])


def test_vortex_profile_returns_none_without_a_pvd_index(tmp_path, monkeypatch):
    import tutorials.VPM.lambOseenVortex.assets.plot_vortex_comparison as plot_vortex_comparison

    monkeypatch.setattr(plot_vortex_comparison, "pvd_time_map", lambda *a, **k: {})

    assert load_profile(tmp_path, "cs") is None


def test_live_partial_pvd_index_is_nonfatal(tmp_path):
    from tutorials.VPM.lambOseenVortex.assets.vortex_diagnostics import pvd_time_map

    case = tmp_path / "vortex_cs"
    case.mkdir()
    (case / "vortex_cs_zq.pvd").write_text("<VTKFile><Collection>", encoding="utf-8")

    assert pvd_time_map(tmp_path, "vortex", "cs") == {}


def test_rwm_field_ensemble_averages_independent_histories(tmp_path):
    member_dirs = []
    for member, offset in enumerate((-0.1, 0.1)):
        member_dir = tmp_path / f"member-{member}" / "samples" / "dipole_rwm"
        member_dir.mkdir(parents=True)
        field = pd.DataFrame(
            {
                "flow_time": [1.0, 2.0],
                "time_step": [5, 10],
                "center0_x": np.array([0.2, 0.4]) + offset,
                "center0_y": [0.5, 0.5],
                "center1_x": [np.nan, np.nan],
                "center1_y": [np.nan, np.nan],
                "separation": [np.nan, np.nan],
                "a_c0": [0.2, 0.3],
                "a_c1": [np.nan, np.nan],
                "a_c_mean": [0.2, 0.3],
                "angle_rad": [np.nan, np.nan],
                "merged": [False, False],
            }
        )
        field.to_csv(member_dir / "field_diagnostics.csv", index=False)
        pd.DataFrame(
            {
                "time": [1.0, 2.0],
                "step": [5, 10],
                "kinetic_energy": np.array([1.0, 0.8]) + offset,
            }
        ).to_csv(member_dir / "flow_integrals.csv", index=False)
        (member_dir / "run_metadata.json").write_text("{}", encoding="utf-8")
        member_dirs.append(member_dir)

    average_field_diagnostics(member_dirs[0], member_dirs, realizations=2)

    field = pd.read_csv(member_dirs[0] / "field_diagnostics.csv")
    integrals = pd.read_csv(member_dirs[0] / "flow_integrals.csv")
    metadata = json.loads((member_dirs[0] / "run_metadata.json").read_text(encoding="utf-8"))
    assert field["center0_x"].to_list() == pytest.approx([0.2, 0.4])
    assert field["separation"].isna().all()
    assert integrals["kinetic_energy"].to_list() == pytest.approx([1.0, 0.8])
    assert metadata["rwm_realizations"] == 2


def test_pair_plots_render_all_four_methods_at_publication_width(tmp_path):
    from tutorials.VPM.lambOseenVortex.assets.plot_dipole_comparison import plot_dipole_case
    from tutorials.VPM.lambOseenVortex.assets.plot_merging_comparison import plot_merging_case

    samples_dir = tmp_path / "samples"
    figures_dir = tmp_path / "figures"
    time = np.linspace(0.25, 20.0, 80)

    for method_index, scheme in enumerate(("cs", "rwm", "dvh", "gbd")):
        offset = 0.002 * method_index
        for case_name in ("dipole", "merging"):
            case_dir = samples_dir / f"{case_name}_{scheme}"
            case_dir.mkdir(parents=True)
            a_c0 = 0.11 * np.sqrt(1.0 + time / 20.0)
            pd.DataFrame(
                {
                    "flow_time": time,
                    "time_step": np.arange(1, len(time) + 1),
                    "center0_x": 0.02 * time + offset,
                    "center0_y": np.full_like(time, 0.5),
                    "center1_x": np.zeros_like(time),
                    "center1_y": np.full_like(time, -0.5),
                    "separation": 1.0 - 0.01 * time,
                    "a_c0": a_c0,
                    "a_c1": a_c0,
                    "a_c_mean": a_c0,
                    "angle_rad": 0.1 * time,
                    "merged": np.zeros_like(time, dtype=bool),
                }
            ).to_csv(case_dir / "field_diagnostics.csv", index=False)

    args = SimpleNamespace(
        samples_dir=samples_dir,
        figures_dir=figures_dir,
        format="png",
        dpi=100,
        gamma=1.0,
        nu=1.0 / 530.0,
        b0=1.0,
        a0_over_b0=0.125,
    )
    assert plot_dipole_case(args) == 0
    assert plot_merging_case(args) == 0

    from PIL import Image

    # Both figures render at the shared publication width (MAX_FIGURE_WIDTH_CM).
    from tutorials.VPM.lambOseenVortex.assets.plot_style import figure_size

    width_cm = figure_size("trajectory")[0] * 2.54
    for name in ("dipole_comparison.png", "merging_comparison.png"):
        with Image.open(figures_dir / name) as image:
            assert image.width == int(width_cm / 2.54 * args.dpi)


def test_merging_plot_keeps_partial_data_without_all_schemes(tmp_path):
    from tutorials.VPM.lambOseenVortex.assets.plot_merging_comparison import plot_merging_case

    samples_dir = tmp_path / "samples"
    figures_dir = tmp_path / "figures"
    case_dir = samples_dir / "merging_cs"
    case_dir.mkdir(parents=True)
    time = np.array([0.0, 0.5, 1.0])
    pd.DataFrame(
        {
            "flow_time": time,
            "time_step": [0, 1, 2],
            "center0_x": [0.0, 0.1, np.nan],
            "center0_y": [0.5, 0.49, np.nan],
            "center1_x": [0.0, -0.1, np.nan],
            "center1_y": [-0.5, -0.49, np.nan],
            "separation": [1.0, 1.0, np.nan],
            "a_c0": [0.125, 0.13, np.nan],
            "a_c1": [0.125, 0.13, np.nan],
            "a_c_mean": [0.125, 0.13, np.nan],
            "angle_rad": [np.pi / 2, np.pi / 2 + 0.2, np.nan],
            "merged": [False, False, True],
        }
    ).to_csv(case_dir / "field_diagnostics.csv", index=False)
    (case_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "viscosity": 1.0 / 530.0,
                "core_radius": 0.125,
                "core_radius_definition": "gaussian_1_over_e_vorticity_radius",
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        samples_dir=samples_dir,
        figures_dir=figures_dir,
        format="png",
        dpi=80,
        gamma=1.0,
        nu=1.0 / 530.0,
        b0=1.0,
        a0_over_b0=0.125,
    )

    assert plot_merging_case(args) == 0
    assert (figures_dir / "merging_comparison.png").is_file()


def test_cs_grid_convergence_metrics_use_exact_solution_errors():
    from tutorials.VPM.lambOseenVortex.assets.grid_independence_cs import (
        add_convergence_metrics,
        json_compatible,
    )

    rows = []
    for h in (0.6, 0.45, 0.3375):
        rows.append(
            {
                "spacing_ratio": h,
                "spacing": h,
                "velocity_l2": h**2,
                "vorticity_l2": 2.0 * h**2,
                "velocity_gradient_l2": 3.0 * h**2,
            }
        )

    orders = add_convergence_metrics(rows)

    assert all(order == pytest.approx(2.0) for order in orders.values())
    assert np.isnan(rows[-1]["velocity_l2_change_to_finer"])
    assert json_compatible({"order": np.nan}) == {"order": None}


def test_postprocessing_manifest_does_not_mark_failed_run_complete(tmp_path):
    from tutorials.VPM.lambOseenVortex.assets.postprocessing_manifest import build_manifest

    samples = tmp_path / "samples"
    figures = tmp_path / "figures"
    case = samples / "vortex_cs"
    case.mkdir(parents=True)
    figures.mkdir()
    (case / "run_metadata.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "completed": False,
                "wall_time_seconds": 1.0,
                "total_time": 30.0,
            }
        ),
        encoding="utf-8",
    )

    run = build_manifest(samples, figures)["runs"]["vortex_cs"]

    assert run["status"] == "failed"
    assert run["complete"] is False
