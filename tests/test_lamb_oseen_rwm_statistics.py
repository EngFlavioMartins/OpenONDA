from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tutorials" / "vpm" / "01_lamb_oseen_vortex" / "assets"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ASSETS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_column_projection_recovers_one_gaussian_blob_and_circulation(tmp_path):
    statistics = _load("postprocess")
    axis = np.linspace(-1.0, 1.0, 201)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    template = {"x": x, "y": y}
    sigma = 0.10
    column_length = 5.0
    backup = tmp_path / "vpm_vortex_rwm_000000.h5"
    with h5py.File(backup, "w") as handle:
        particles = handle.create_group("particles")
        particles.create_dataset("position", data=np.array([[0.0, 0.0, 0.0]]))
        particles.create_dataset("vortex_strength", data=np.array([[0.0, 0.0, column_length]]))
        particles.create_dataset("core_radius", data=np.array([sigma]))
        solver = handle.create_group("solver")
        solver.attrs["step"] = 0
        solver.attrs["time"] = 0.0

    field, quality = statistics.project_backup(backup, template, column_length)
    exact_vorticity = np.exp(-(x * x + y * y) / sigma**2) / (np.pi * sigma**2)
    relative_error = np.linalg.norm(field["vorticity_z"] - exact_vorticity) / np.linalg.norm(
        exact_vorticity
    )
    represented_circulation = np.trapezoid(
        np.trapezoid(field["vorticity_z"], axis, axis=1), axis, axis=0
    )

    assert relative_error < 0.02
    np.testing.assert_allclose(represented_circulation, 1.0, rtol=2.0e-3)
    assert quality["absolute_circulation_capture_fraction"] > 0.999
    positive_x = int(np.argmin(np.abs(axis - 0.2)))
    centre_y = int(np.argmin(np.abs(axis)))
    assert field["velocity_y"][positive_x, centre_y] > 0.0


def test_merging_pair_requires_peak_saddle_contrast_above_ensemble_noise():
    diagnostics = _load("postprocess")
    axis = np.linspace(-1.0, 1.0, 201)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    width = 0.12
    vorticity = np.exp(-(x * x + (y - 0.30) ** 2) / width**2) + np.exp(
        -(x * x + (y + 0.30) ** 2) / width**2
    )
    base = {
        "x": x,
        "y": y,
        "velocity_x": np.zeros_like(x),
        "velocity_y": np.zeros_like(x),
        "vorticity_z": vorticity,
        "confidence_multiplier": 2.365,
        "time": 0.0,
        "step": 0,
    }

    resolved = dict(base, vorticity_standard_error_z=np.full_like(x, 0.01))
    unresolved = dict(base, vorticity_standard_error_z=np.full_like(x, 0.60))
    resolved_row = diagnostics.diagnostics_row(resolved, "merging")
    unresolved_row = diagnostics.diagnostics_row(unresolved, "merging")

    assert resolved_row[11] is False
    assert np.isfinite(resolved_row[6])
    snr_index = diagnostics.FIELD_CSV_COLUMNS.index("peak_saddle_signal_to_noise")
    assert resolved_row[snr_index] > resolved["confidence_multiplier"]
    assert unresolved_row[11] is True
    assert np.isnan(unresolved_row[6])
    assert not bool(unresolved_row[diagnostics.FIELD_CSV_COLUMNS.index("is_peak_coalesced")])


def test_dynamic_fourier_energy_rate_is_not_reported_as_a_time_derivative(tmp_path):
    diagnostics = _load("postprocess")
    csv_path = tmp_path / "flow_integrals.csv"
    csv_path.write_text(
        "time,kinetic_energy_rate,kinetic_energy_rate_source,"
        "viscous_kinetic_energy_rate,n_particles_total\n"
        "0.5,-0.2,direct_energy_backward_difference,-0.19,49000\n"
        "1.0,0.8,undefined_dynamic_fourier_box,-0.17,51000\n",
        encoding="utf-8",
    )

    data = diagnostics.read_flow_integrals(csv_path)

    assert data is not None
    assert data["kinetic_energy_rate"][0] == -0.2
    assert np.isnan(data["kinetic_energy_rate"][1])
    assert data["viscous_kinetic_energy_rate"][1] == -0.17


def test_energy_audit_reports_an_unrun_ensemble_instead_of_crashing(tmp_path):
    diagnostics = _load("postprocess")

    audit = diagnostics.energy_balance_audit(tmp_path, schemes=("rwm",))

    assert set(audit["runs"]) == {
        "vortex_rwm",
        "dipole_rwm",
        "merging_rwm",
    }
    assert all(run["status"] == "missing" for run in audit["runs"].values())
    assert all(run["comparable_samples"] == 0 for run in audit["runs"].values())


def test_merging_separation_reference_uses_original_figure_four_samples():
    diagnostics = _load("postprocess")
    dimensional = np.loadtxt(
        diagnostics.SEPARATION_DIMENSIONAL_REFERENCE,
        delimiter=",",
    )

    reference = diagnostics.load_merging_references(0.125, 1.0)["separation"]

    assert len(reference) == len(dimensional)
    np.testing.assert_allclose(reference[:, 1], dimensional[:, 1])
    np.testing.assert_allclose(
        reference[:, 0],
        dimensional[:, 0] * diagnostics.REFERENCE_VISCOUS_TIME_PER_SECOND / 0.125**2,
    )
    np.testing.assert_allclose(reference[-1, 0], 0.04744 / 0.125**2)


@pytest.mark.parametrize("separator", ["  ", " | "])
def test_gbd_closure_reads_both_solver_log_layouts(tmp_path, monkeypatch, separator):
    diagnostics = _load("postprocess")
    monkeypatch.setattr(diagnostics, "SOLUTION_DIR", tmp_path)
    folder = tmp_path / "vortex_gbd"
    folder.mkdir()
    log = folder / "vpm.log"
    log.write_text(f"  net residual, after{separator}2.136376e-10\n")
    assert diagnostics._gbd_moment_recovery_failures(("vortex",)) == []
    for invalid in ("1e-2", "nan", "inf", "unreadable"):
        log.write_text(f"  net residual, after{separator}{invalid}\n")
        assert diagnostics._gbd_moment_recovery_failures(("vortex",))


def test_rwm_precision_plans_additional_independent_members_without_relaxing_gate():
    from tests._tutorial_helpers import load_tutorial_module

    required_ensemble_size = load_tutorial_module(
        "vpm/lamb_oseen_vortex", "assets.rwm_ensemble"
    ).required_ensemble_size

    assert required_ensemble_size(10, 0.093436, 0.075) == 18
    assert required_ensemble_size(18, 0.07, 0.075) == 18
    with pytest.raises(ValueError, match="finite"):
        required_ensemble_size(10, float("nan"), 0.075)


def test_rwm_convergence_extends_only_the_missing_members(tmp_path, monkeypatch):
    from tests._tutorial_helpers import load_tutorial_module

    postprocess = load_tutorial_module("vpm/lamb_oseen_vortex", "assets.postprocess")
    rwm_ensemble = load_tutorial_module("vpm/lamb_oseen_vortex", "assets.rwm_ensemble")

    monkeypatch.setattr(rwm_ensemble, "TUTORIAL_DIR", tmp_path)
    batches = []
    monkeypatch.setattr(
        rwm_ensemble,
        "run_ensemble",
        lambda case, count, seed, **kw: batches.append((count, kw["first_realization"])),
    )
    output = tmp_path / "samples/vortex_rwm"
    output.mkdir(parents=True)

    def aggregate(solution, samples, case, count):
        error = 0.093436 if count == 10 else 0.07
        (output / "rwm_convergence.csv").write_text(
            "relative_standard_error_l2_velocity,relative_standard_error_l2_vorticity\n"
            f"0.01,{error}\n"
        )

    monkeypatch.setattr(postprocess, "aggregate_case", aggregate)
    rwm_ensemble.run_converged_ensemble("vortex", 10, 42000, 80)
    assert batches == [(10, 0), (18, 10)]
    with pytest.raises(RuntimeError, match="increase --maximum-realizations"):
        rwm_ensemble.run_converged_ensemble("vortex", 10, 42000, 10)
