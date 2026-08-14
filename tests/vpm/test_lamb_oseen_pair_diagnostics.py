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
    FIELD_CSV_COLUMNS,
    _core_radius_utheta,
    _diagnostics_row,
    average_field_diagnostics,
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


def test_core_radius_utheta_recovers_known_lamb_oseen_core():
    field = _superposed_lamb_oseen_field([(0.0, 0.0, 1.0, 0.15)], extent=1.0, spacing=0.01)
    a_c = _core_radius_utheta(field, np.array([0.0, 0.0]), r_max=0.45)
    assert a_c == pytest.approx(0.15, rel=0.05)


def test_field_diagnostics_recovers_dipole_centers_and_core_radius():
    field = _superposed_lamb_oseen_field([(0.0, 0.5, 1.0, 0.10), (0.0, -0.5, -1.0, 0.10)])
    field["time"] = 0.25
    field["step"] = 1

    result = _row_dict(field, "dipole")

    assert result["center0_x"] == pytest.approx(0.0, abs=0.02)
    assert result["center0_y"] == pytest.approx(0.5, abs=0.02)
    assert result["center1_y"] == pytest.approx(-0.5, abs=0.02)
    assert result["separation"] == pytest.approx(1.0, abs=0.03)
    assert result["a_c0"] == pytest.approx(0.10, abs=0.02)
    assert result["a_c1"] == pytest.approx(0.10, abs=0.02)
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


def test_field_diagnostics_handles_lone_vortex():
    field = _superposed_lamb_oseen_field([(0.0, 0.0, 1.0, 0.15)], extent=1.0, spacing=0.02)
    field["time"] = 5.0
    field["step"] = 10

    result = _row_dict(field, "vortex")

    assert result["center0_x"] == pytest.approx(0.0, abs=0.02)
    assert result["center0_y"] == pytest.approx(0.0, abs=0.02)
    assert np.isnan(result["center1_x"])
    assert result["a_c0"] == pytest.approx(0.15, rel=0.05)
    # No second vortex to be separated from — separation/angle are NaN, and
    # "merged" (only meaningful for a pair) follows the NaN-separation rule.
    assert np.isnan(result["separation"])
    assert result["merged"] is True


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
