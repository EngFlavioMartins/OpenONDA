from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from studies.vpm.fmm_induction import setup as study_setup

VERIFY_PATH = Path(__file__).parents[2] / "studies/vpm/fmm_induction/assets/verify_results.py"
VERIFY_SPEC = importlib.util.spec_from_file_location("fmm_result_verifier", VERIFY_PATH)
assert VERIFY_SPEC is not None and VERIFY_SPEC.loader is not None
verifier = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(verifier)
VALID_SHA = "a" * 40


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True)


def _temporary_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test User")
    (repository / "source.py").write_text("value = 1\n", encoding="utf-8")
    _git(repository, "add", "source.py")
    _git(repository, "commit", "--quiet", "-m", "initial")
    return repository


def test_source_revision_reports_clean_temporary_repository(tmp_path):
    repository = _temporary_repository(tmp_path)
    assert study_setup.source_revision(repository) == (
        study_setup.source_revision(repository)[0],
        False,
        (),
    )


def test_clean_initialization_writes_canonical_manifest(tmp_path):
    repository = _temporary_repository(tmp_path)
    results = repository / "studies/vpm/fmm_induction/results"
    figures = repository / "studies/vpm/fmm_induction/figures"
    study_setup.initialize_results(
        repository_root=repository,
        results_dir=results,
        figures_dir=figures,
    )
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_commit"] == study_setup.source_revision(repository)[0]
    assert manifest["source_dirty"] is False
    assert "commit" not in manifest and "dirty" not in manifest


def test_source_revision_ignores_only_study_outputs(tmp_path):
    repository = _temporary_repository(tmp_path)
    for directory in ("results", "figures"):
        output = repository / "studies/vpm/fmm_induction" / directory
        output.mkdir(parents=True)
        (output / "record.txt").write_text("generated\n", encoding="utf-8")
        assert study_setup.source_revision(repository)[1:] == (False, ())
        (output / "record.txt").write_text("changed\n", encoding="utf-8")
        assert study_setup.source_revision(repository)[1:] == (False, ())


def test_source_revision_reports_tracked_and_untracked_source_changes(tmp_path):
    repository = _temporary_repository(tmp_path)
    source = repository / "source.py"
    source.write_text("value = 2\n", encoding="utf-8")
    dirty, changes = study_setup.source_revision(repository)[1:]
    assert dirty and any("source.py" in change for change in changes)

    source.write_text("value = 1\n", encoding="utf-8")
    (repository / "new_source.py").write_text("value = 3\n", encoding="utf-8")
    dirty, changes = study_setup.source_revision(repository)[1:]
    assert dirty and any("new_source.py" in change for change in changes)


def test_dirty_initialization_preserves_prior_results(tmp_path):
    repository = _temporary_repository(tmp_path)
    results = repository / "studies/vpm/fmm_induction/results"
    figures = repository / "studies/vpm/fmm_induction/figures"
    results.mkdir(parents=True)
    figures.mkdir(parents=True)
    prior = results / "prior.json"
    prior.write_text("prior evidence\n", encoding="utf-8")
    (repository / "source.py").write_text("value = 2\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="before deleting"):
        study_setup.initialize_results(
            repository_root=repository,
            results_dir=results,
            figures_dir=figures,
        )
    assert prior.read_text(encoding="utf-8") == "prior evidence\n"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _valid_results(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text(
        json.dumps(
            {
                "created_utc": "2026-09-03T00:00:00+00:00",
                "source_commit": VALID_SHA,
                "source_dirty": False,
                "python_version": "3.11.0",
                "numpy_version": "2.0.0",
                "taichi_version": "1.7.4",
                "platform": "test",
                "seed": 1,
                "kernels": list(verifier.REQUIRED_KERNELS),
                "distributions": list(verifier.REQUIRED_DISTRIBUTIONS),
                "accuracy_gates": verifier.ACCURACY_LIMITS,
            }
        ),
        encoding="utf-8",
    )
    fields = {
        "source_commit": VALID_SHA,
        "source_dirty": "False",
        "method": "FMM",
        "backend": "CPU",
        "kernel": "GAUSSIAN",
        "distribution": "uniform",
        "count": 14080,
        "velocity_relative_l2": 1.0e-4,
        "gradient_relative_l2": 2.0e-4,
        "rate_relative_l2": 3.0e-4,
        "rate_particle_p95": 4.0e-4,
        "raw_rate_defect": 5.0e-5,
        "host_particle_transfers": 0,
        "direct_strength_rate_fallbacks": 0,
    }
    accuracy_rows = []
    for kernel in verifier.REQUIRED_KERNELS:
        for backend in ("CPU", "VULKAN"):
            row = fields.copy()
            row.update(kernel=kernel, backend=backend)
            accuracy_rows.append(row)
    for distribution in verifier.REQUIRED_DISTRIBUTIONS:
        row = fields.copy()
        row.update(distribution=distribution)
        accuracy_rows.append(row)
    scaling_rows = []
    for count in verifier.REQUIRED_COUNTS:
        row = fields.copy()
        row.update(count=count)
        scaling_rows.append(row)
    _write_csv(results / "accuracy.csv", accuracy_rows)
    _write_csv(results / "scaling.csv", scaling_rows)
    common = {"source_commit": VALID_SHA, "source_dirty": False}
    (results / "direct_fmm_14080_10_leapfrog.json").write_text(
        json.dumps(
            {
                **common,
                "comparison_gate_passed": True,
                "fmm_host_particle_transfers": 0,
                "fmm_direct_strength_rate_fallbacks": 0,
            }
        ),
        encoding="utf-8",
    )
    (results / "coupled_vlm_comparison.json").write_text(
        json.dumps(
            {
                **common,
                "comparison_gate_passed": True,
                "fmm_zero_host_transfer_passed": True,
                "fmm_zero_fallback_passed": True,
                "fmm_scheduled_output_passed": True,
            }
        ),
        encoding="utf-8",
    )
    (results / "coupled_fvm_comparison.json").write_text(
        json.dumps(
            {
                **common,
                "comparison_gate_passed": True,
                "finite_fields": True,
                "fmm_host_particle_transfers": 0,
                "fmm_direct_strength_rate_fallbacks": 0,
            }
        ),
        encoding="utf-8",
    )
    return results


def test_verifier_accepts_minimal_complete_result_set(tmp_path):
    results = _valid_results(tmp_path)
    assert verifier.main(results) == 0


def test_verifier_rejects_dirty_or_mismatched_provenance(tmp_path):
    results = _valid_results(tmp_path)
    manifest_path = results / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_dirty"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verifier.verify_results(results)

    manifest["source_dirty"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rows = list(csv.DictReader((results / "accuracy.csv").open(encoding="utf-8")))
    rows[0]["source_commit"] = "b" * 40
    _write_csv(results / "accuracy.csv", rows)
    assert verifier.verify_results(results)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_kernel",
        "missing_backend",
        "missing_count",
        "missing_distribution",
        "failed_accuracy",
        "nonzero_transfer",
        "nonzero_fallback",
    ],
)
def test_verifier_rejects_missing_rows_and_failed_fmm_gates(tmp_path, mutation):
    results = _valid_results(tmp_path)
    path = results / ("scaling.csv" if mutation == "missing_count" else "accuracy.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if mutation == "missing_kernel":
        rows = [row for row in rows if row["kernel"] != "WINCKELMANS"]
    elif mutation == "missing_backend":
        rows = [row for row in rows if row["backend"] != "CPU"]
    elif mutation == "missing_count":
        rows = [row for row in rows if row["count"] != "70200"]
    elif mutation == "missing_distribution":
        rows = [row for row in rows if row["distribution"] != "rotor"]
    elif mutation == "failed_accuracy":
        rows[0]["velocity_relative_l2"] = "1.0"
    elif mutation == "nonzero_transfer":
        rows[0]["host_particle_transfers"] = "1"
    elif mutation == "nonzero_fallback":
        rows[0]["direct_strength_rate_fallbacks"] = "1"
    _write_csv(path, rows)
    assert verifier.verify_results(results)
