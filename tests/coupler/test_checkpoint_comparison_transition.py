"""Exact, read-only transition evidence; no solver or numerical runtime import."""

import copy
import hashlib
import json

import h5py
import numpy as np
import pytest

from tests.coupler.test_checkpoint_comparison_asset import _checkpoint, _csv, _encode, comparison

POLICY = {"backend": "cupy_cuda", "mesh": {"order": 10, "spacing_over_tau": 0.25},
          "tail_contract": "gaussian_interval_remainder_v1"}


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native(path, *, mesh=False, step=276):
    """Construct tiny independent native files, never production evidence."""
    fields, manifest = _checkpoint(path)
    config = manifest["config"]
    config["vpm"]["induction"] = {"method": "SLIP_SLAB", "tail_tolerance": 1e-5}
    if mesh:
        config["vpm"]["induction"]["gaussian_mesh_policy"] = copy.deepcopy(POLICY)
    manifest.update(backend="fvm", config_sha256=comparison.mapping_digest(config),
                    coupling_step=step, vpm_step=step, fvm_step=5*step, time=step*0.04)
    with h5py.File(path / "vpm.h5", "r+") as saved:
        saved["solver"].attrs.update(
            step=step, time=manifest["time"], numerical_configuration=json.dumps(config["vpm"]),
            numerical_configuration_sha256=comparison.mapping_digest(config["vpm"]))
    fields.update(step=np.asarray(5*step), n_committed_time_steps=np.asarray(5*step),
                  time=np.asarray(manifest["time"]))
    np.savez(path / "fvm/rank.npz", **_encode(fields))
    (path / "vpm.vtu").write_bytes(b"tiny complete immutable VTU fixture")
    manifest["artifacts"]["vpm_vtu"] = "vpm.vtu"
    manifest["artifact_sha256"] = {
        name: comparison.artifact_digest(path / relative)
        for name, relative in manifest["artifacts"].items()}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return comparison.load_checkpoint(path)


@pytest.fixture
def corpus(tmp_path):
    source = _native(tmp_path / "source", step=275)
    control = _native(tmp_path / "control")
    candidate = _native(tmp_path / "candidate", mesh=True)
    manifest = source["manifest"]
    evidence = {
        "manifest_path": str(source["directory"] / "manifest.json"),
        "manifest_sha256": source["manifest_sha256"],
        "expected_manifest_sha256": source["manifest_sha256"],
        "source_configuration_sha256": manifest["config_sha256"],
        "source_vpm_configuration_sha256": comparison.mapping_digest(manifest["config"]["vpm"]),
        "current_vpm_configuration_sha256": comparison.mapping_digest(candidate["manifest"]["config"]["vpm"]),
        "source_artifacts": {name: {"path": str(source["directory"] / relative),
                                    "sha256": manifest["artifact_sha256"][name]}
                             for name, relative in manifest["artifacts"].items()},
        "permissions": [{"path": comparison._MESH_POLICY_PATH, "stored": {"present": False},
                         "current": {"present": True, "value": copy.deepcopy(POLICY)}}],
    }
    report = {"status": "complete", "checkpoint": str(source["directory"]),
              "source_checkpoint_unchanged": True, "max_coupling_steps": 1, "final_step": 276,
              "exchanges": [{"step": 276, "time": 11.04}],
              "qualification_controls": {"gaussian_mesh_explicit_opt_in": True},
              "restart_admission": evidence, "loaded_source_files_changed_during_run": [],
              "source_hashes_before_initialization": {"/p/source/a.py": "a"*64},
              "source_inventory_at_completion": {"/p/source/a.py": "a"*64},
              "source_hashes_at_completion": {"/p/source/a.py": "a"*64}}
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(report))
    return control, candidate, source, report, path


def _admit(corpus):
    control, candidate, _, report, path = corpus
    path.write_text(json.dumps(report))
    return comparison.admit_comparison_identity(control, candidate, transition_report=path,
                                                expected_transition_report_sha256=_digest(path))


def test_default_is_strict_and_explicit_optin_is_required(corpus):
    control, candidate, _, _, path = corpus
    assert comparison.admit_comparison_identity(control, control) is None
    with pytest.raises(ValueError, match="Unmatched.*config"):
        comparison.admit_comparison_identity(control, candidate)
    with pytest.raises(ValueError, match="explicit report SHA"):
        comparison.admit_comparison_identity(control, candidate, transition_report=path)
    with pytest.raises(ValueError, match="requires an explicit report"):
        comparison.admit_comparison_identity(control, candidate, expected_transition_report_sha256="a"*64)
    with pytest.raises(ValueError, match="report SHA-256 mismatch"):
        comparison.admit_comparison_identity(control, candidate, transition_report=path,
                                            expected_transition_report_sha256="b"*64)


def test_valid_transition_displays_exact_delta_without_rewriting(corpus):
    control, candidate, source, _, path = corpus
    trees = (control["directory"], candidate["directory"], source["directory"])
    before = {file: file.read_bytes() for tree in trees for file in tree.rglob("*") if file.is_file()}
    loaded = copy.deepcopy((control["manifest"], candidate["manifest"], source["manifest"]))
    detail = _admit(corpus)
    assert detail["configuration_changes"] == corpus[3]["restart_admission"]["permissions"]
    assert detail["source_manifest_sha256"] == source["manifest_sha256"]
    assert detail["candidate_configuration_sha256"] == candidate["manifest"]["config_sha256"]
    assert (control["manifest"], candidate["manifest"], source["manifest"]) == loaded
    assert {file: file.read_bytes() for file in before} == before
    samples = path.parent / "samples"
    samples.mkdir()
    _csv(samples / "force.csv", [{"time": 11.04, "step": 1380, "drag": 1.0}])
    report = comparison.compare_checkpoints(
        control, candidate, samples, samples, path.parent / "absent-a", path.parent / "absent-b",
        transition_report=path, expected_transition_report_sha256=_digest(path))
    assert report["configuration_transition"] == detail
    assert report["status"] == "comparison_complete_no_accuracy_pass_inferred"
    assert report["control"]["config_sha256"] != report["candidate"]["config_sha256"]


@pytest.mark.parametrize("failure", ["failed", "mutable_source", "not_opted_in", "changed_code",
                                    "different_inventory", "missing_loaded_inventory", "extra_permission",
                                    "wrong_policy", "wrong_source_sha", "wrong_config_sha",
                                    "wrong_artifact_sha", "wrong_artifact_path", "wrong_final_clock",
                                    "wrong_final_step", "missing_exchange", "wrong_checkpoint_path"])
def test_authenticated_report_cannot_grant_unrecorded_or_incomplete_transition(corpus, failure):
    report, path = corpus[3:]
    evidence = report["restart_admission"]
    if failure == "failed":
        report["status"] = "failed"
    elif failure == "mutable_source":
        report["source_checkpoint_unchanged"] = False
    elif failure == "not_opted_in":
        report["qualification_controls"]["gaussian_mesh_explicit_opt_in"] = False
    elif failure == "changed_code":
        report["loaded_source_files_changed_during_run"] = ["a.py"]
    elif failure == "different_inventory":
        report["source_inventory_at_completion"]["/p/source/a.py"] = "b"*64
    elif failure == "missing_loaded_inventory":
        report["source_hashes_at_completion"] = {}
    elif failure == "extra_permission":
        evidence["permissions"].append({"path": "vpm.time_step_size"})
    elif failure == "wrong_policy":
        evidence["permissions"][0]["current"]["value"]["mesh"]["order"] = 12
    elif failure == "wrong_source_sha":
        evidence["expected_manifest_sha256"] = "b"*64
    elif failure == "wrong_config_sha":
        evidence["current_vpm_configuration_sha256"] = "b"*64
    elif failure == "wrong_artifact_sha":
        evidence["source_artifacts"]["vpm"]["sha256"] = "b"*64
    elif failure == "wrong_artifact_path":
        evidence["source_artifacts"]["vpm"]["path"] = str(path)
    elif failure == "wrong_final_clock":
        report["exchanges"][0]["time"] = 11.08
    elif failure == "wrong_final_step":
        report["final_step"] = 277
    elif failure == "missing_exchange":
        report["exchanges"] = []
    else:
        report["checkpoint"] = str(path.parent)
    with pytest.raises(ValueError):
        _admit(corpus)


@pytest.mark.parametrize("failure", ["other_config", "boolean_number", "already_present",
                                    "null_to_policy", "clock", "step", "direction"])
def test_optin_never_hides_other_config_or_clock_changes(corpus, failure):
    control, candidate = corpus[:2]
    left, right = control["manifest"], candidate["manifest"]
    if failure == "other_config":
        right["config"]["coupler"]["interface_normal_tolerance"] *= 2
    elif failure == "boolean_number":
        right["config"]["vpm"]["health_limits"]["lagrangian_cfl"]["maximum"] = True
    elif failure == "already_present":
        left["config"]["vpm"]["induction"]["gaussian_mesh_policy"] = copy.deepcopy(POLICY)
    elif failure == "null_to_policy":
        left["config"]["vpm"]["induction"]["gaussian_mesh_policy"] = None
    elif failure == "clock":
        right["time"] += .04
    elif failure == "step":
        right["coupling_step"] += 1
    else:
        corpus = (candidate, control, *corpus[2:])
    for manifest in (left, right):
        manifest["config_sha256"] = comparison.mapping_digest(manifest["config"])
    with pytest.raises(ValueError):
        _admit(corpus)


def test_source_artifact_corruption_is_rejected_even_with_pinned_report(corpus):
    source = corpus[2]
    (source["directory"] / "vpm.vtu").write_bytes(b"corrupted evidence")
    with pytest.raises(ValueError, match="artifact SHA256"):
        _admit(corpus)


def test_report_change_during_source_read_is_rejected(corpus, monkeypatch):
    original = comparison.load_checkpoint
    path = corpus[4]

    def changed(directory):
        result = original(directory)
        path.write_bytes(path.read_bytes()+b" ")
        return result

    monkeypatch.setattr(comparison, "load_checkpoint", changed)
    with pytest.raises(ValueError, match="changed while being read"):
        _admit(corpus)
