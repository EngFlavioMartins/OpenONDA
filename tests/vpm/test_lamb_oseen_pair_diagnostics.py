"""Tests for compact Lamb--Oseen pair diagnostics."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tutorials.VPM.lambOseenVortex.assets.pair_diagnostics import PairDiagnosticsSampler
from tutorials.VPM.lambOseenVortex.assets.rwm_ensemble import average_pair_diagnostics


class _ParticlePair:
    def __init__(self, signs: tuple[float, float]) -> None:
        offsets = np.array(((0.1, 0.0), (-0.1, 0.0), (0.0, 0.1), (0.0, -0.1)))
        xy = np.vstack((offsets + (0.0, 0.5), offsets + (0.0, -0.5)))
        self.particles_positions = np.column_stack((xy, np.zeros(len(xy))))
        circulation = np.repeat(signs, len(offsets))
        self.particles_strengths = np.column_stack(
            (np.zeros(len(xy)), np.zeros(len(xy)), circulation)
        )
        self.particles_radii = np.full(len(xy), 0.05)
        self.particles_group_ids = np.repeat((0, 1), len(offsets))


def test_dipole_sampler_writes_dense_compact_history_without_duplicate_steps(tmp_path):
    path = tmp_path / "pair_diagnostics.csv"
    sampler = PairDiagnosticsSampler("dipole", 1.0, 1.0)
    solver = _ParticlePair((1.0, -1.0))

    sampler.save_csv(solver, path, time=0.25, step=1)
    sampler.save_csv(solver, path, time=0.25, step=1)

    data = pd.read_csv(path)
    assert len(data) == 1
    assert data.loc[0, "x_core"] == pytest.approx(0.0)
    assert data.loc[0, "y_core"] == pytest.approx(0.5)
    assert data.loc[0, "core_radius"] == pytest.approx(np.sqrt(0.0125))


def test_merging_sampler_tracks_both_initial_cores(tmp_path):
    path = tmp_path / "pair_diagnostics.csv"
    sampler = PairDiagnosticsSampler("merging", 1.0, 1.0)

    sampler.save_csv(_ParticlePair((1.0, 1.0)), path, time=0.25, step=1)

    data = pd.read_csv(path)
    assert len(data) == 1
    assert bool(data.loc[0, "merged"]) is False
    assert data.loc[0, "separation"] == pytest.approx(1.0)
    assert data.loc[0, "core_area"] == pytest.approx(0.00625)


def test_pair_geometry_uses_midplane_while_circulation_uses_full_column(tmp_path):
    path = tmp_path / "pair_diagnostics.csv"
    solver = _ParticlePair((1.0, -1.0))
    solver.particles_positions[:4, 2] = 0.0
    solver.particles_positions[4:, 2] = 1.0
    solver.particles_positions[4:, 0] = 10.0
    sampler = PairDiagnosticsSampler("dipole", 1.0, 2.0, slab_half_width=0.1)

    sampler.save_csv(solver, path, time=0.25, step=1)

    data = pd.read_csv(path)
    assert data.loc[0, "x_core"] == pytest.approx(0.0)
    assert data.loc[0, "surface_circulation"] == pytest.approx(2.0)


def test_rwm_pair_ensemble_averages_independent_histories(tmp_path):
    member_dirs = []
    for member, offset in enumerate((-0.1, 0.1)):
        member_dir = tmp_path / f"member-{member}" / "samples" / "dipole_rwm"
        member_dir.mkdir(parents=True)
        pair = pd.DataFrame(
            {
                "flow_time": [1.0, 2.0],
                "time_step": [5, 10],
                "x_core": np.array([0.2, 0.4]) + offset,
                "y_core": [0.5, 0.5],
                "core_radius": [0.2, 0.3],
                "separation": [np.nan, np.nan],
                "core_area": [np.nan, np.nan],
                "angle_rad": [np.nan, np.nan],
                "surface_circulation": [0.99, 0.99],
                "merged": [False, False],
                "core_0_x": np.array([0.2, 0.4]) + offset,
                "core_0_y": [0.5, 0.5],
                "core_1_x": [np.nan, np.nan],
                "core_1_y": [np.nan, np.nan],
            }
        )
        pair.to_csv(member_dir / "pair_diagnostics.csv", index=False)
        pd.DataFrame(
            {
                "time": [1.0, 2.0],
                "step": [5, 10],
                "kinetic_energy": np.array([1.0, 0.8]) + offset,
            }
        ).to_csv(member_dir / "flow_integrals.csv", index=False)
        (member_dir / "run_metadata.json").write_text("{}", encoding="utf-8")
        member_dirs.append(member_dir)

    average_pair_diagnostics(member_dirs[0], member_dirs, realizations=2)

    pair = pd.read_csv(member_dirs[0] / "pair_diagnostics.csv")
    integrals = pd.read_csv(member_dirs[0] / "flow_integrals.csv")
    metadata = json.loads(
        (member_dirs[0] / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert pair["x_core"].to_list() == pytest.approx([0.2, 0.4])
    assert pair["separation"].isna().all()
    assert integrals["kinetic_energy"].to_list() == pytest.approx([1.0, 0.8])
    assert metadata["rwm_realizations"] == 2


def test_allrun_contains_the_complete_four_by_three_matrix():
    script = (Path(__file__).parents[2] / "tutorials/VPM/lambOseenVortex/allrun.sh").read_text(
        encoding="utf-8"
    )
    loop_values = {}
    for line in script.splitlines():
        words = line.strip().split()
        if len(words) >= 5 and words[:2] == ["for", "physics"]:
            loop_values["physics"] = tuple(word.rstrip(";") for word in words[3:-1])
        if len(words) >= 5 and words[:2] == ["for", "scheme"]:
            loop_values["scheme"] = tuple(word.rstrip(";") for word in words[3:-1])
    commands = set(itertools.product(loop_values["physics"], loop_values["scheme"]))
    expected = set(itertools.product(("vortex", "dipole", "merging"), ("cs", "rwm", "dvh", "gbd")))
    assert commands == expected


def test_final_vortex_profile_reads_comment_time_and_field_header(tmp_path):
    from tutorials.VPM.lambOseenVortex.assets.plot_vortex_comparison import load_profile

    case_dir = tmp_path / "vortex_cs"
    case_dir.mkdir()
    profile = case_dir / "vortex_cs_x.csv"
    profile.write_text(
        "# flow_time=20.0\nx,y,z,Ux,Uy,Uz,omega_z\n-0.1,0,0,0,-0.2,0,1.0\n0.1,0,0,0,0.2,0,1.0\n",
        encoding="utf-8",
    )

    data, time = load_profile(tmp_path, "cs")

    assert time == 20.0
    assert data.columns.tolist() == ["x", "y", "z", "Ux", "Uy", "Uz", "omega_z"]


def test_vortex_profile_rejects_noncurrent_csv_format(tmp_path):
    from tutorials.VPM.lambOseenVortex.assets.plot_vortex_comparison import load_profile

    case_dir = tmp_path / "vortex_cs"
    case_dir.mkdir()
    (case_dir / "vortex_cs_x.csv").write_text(
        "flow_time,time_step,x,Uy,omega_z\n20.0,400,0.0,0.2,1.0\n",
        encoding="utf-8",
    )

    assert load_profile(tmp_path, "cs") is None


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
            pd.DataFrame(
                {
                    "flow_time": time,
                    "time_step": np.arange(1, len(time) + 1),
                    "x_core": 0.02 * time + offset,
                    "y_core": np.full_like(time, 0.5),
                    "core_radius": 0.11 * np.sqrt(1.0 + time / 20.0),
                    "separation": 1.0 - 0.01 * time,
                    "core_area": 0.006 + 0.0005 * time,
                    "angle_rad": 0.1 * time,
                    "surface_circulation": np.full_like(time, 2.0),
                    "merged": np.zeros_like(time, dtype=bool),
                    "core_0_x": np.zeros_like(time),
                    "core_0_y": np.full_like(time, 0.5),
                    "core_1_x": np.zeros_like(time),
                    "core_1_y": np.full_like(time, -0.5),
                }
            ).to_csv(case_dir / "pair_diagnostics.csv", index=False)

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

    for name in ("dipole_comparison.png", "merging_comparison.png"):
        with Image.open(figures_dir / name) as image:
            assert image.width == round(12.5 / 2.54 * args.dpi)
