"""Qualification-driver controls only; no solver, real tutorial writes or GPU."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from source.coupler.backup import BACKUP_FORMAT_VERSION, artifact_digest, config_mapping_digest
from source.solvers.vpm.config import Numerics, ViscousConfig, VPMCase
from source.solvers.vpm.config.fingerprint import numerical_configuration
from source.solvers.vpm.config.restart_changes import MISSING_CONFIGURATION_VALUE
from source.solvers.vpm.physics.induction.fmm.device import FMMInduction
from source.solvers.vpm.physics.induction.gaussian_mesh.session import GaussianSlabPolicy
from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction

ASSET = (Path(__file__).resolve().parents[2] /
    "tests/support/cylinder/benchmark_coupled_checkpoint.py")
SPEC = importlib.util.spec_from_file_location("benchmark_mesh_transition_asset", ASSET)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def _case():
    return VPMCase(numerics=Numerics(time_step_size=.04, compute_device="CUDA",
        viscous=ViscousConfig.inviscid(), induction=SlipSlabInduction(FMMInduction(),
            z_min=-.48, z_max=.48, tail_tolerance=1e-4, max_shells=129)))


def _bundle(tmp_path, config):
    checkpoint = tmp_path / "bundle"
    checkpoint.mkdir()
    artifacts = {"fvm": "fvm", "vpm": "vpm.h5", "vpm_vtu": "particles.vtu",
                 "vpm_boundary_condition": "boundary.npz"}
    (checkpoint / "fvm").mkdir()
    (checkpoint / "fvm" / "state.bin").write_bytes(b"immutable fake FVM artifact")
    for filename in ("vpm.h5", "particles.vtu", "boundary.npz"):
        (checkpoint / filename).write_bytes(("immutable fixture "+filename).encode())
    configuration = {"vpm": deepcopy(config), "coupler": {"step_size": .04}}
    manifest = {"format_version": BACKUP_FORMAT_VERSION, "kind": "openonda.coupled_backup",
        "backend": "fvm", "config": configuration, "config_sha256": config_mapping_digest(configuration),
        "artifacts": artifacts,
        "artifact_sha256": {key: artifact_digest(checkpoint / value) for key, value in artifacts.items()}}
    path = checkpoint / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint, BENCH._digest(path)


def test_opt_in_replaces_frozen_values_preserving_every_other_numerical_control():
    original = _case()
    before = numerical_configuration(original.numerics)
    new = BENCH._with_gaussian_mesh(original)
    after = numerical_configuration(new.numerics)
    assert new is not original and new.numerics is not original.numerics
    assert new.numerics.induction.base is not original.numerics.induction.base
    assert original.numerics.induction.gaussian_mesh_policy is None
    assert numerical_configuration(original.numerics) == before
    policy = after["induction"].pop("gaussian_mesh_policy")
    assert policy["mesh"] == {"broadening_ratio": 3., "spacing_over_tau": .25,
                               "correction_radius_over_tau": 5., "order": 10}
    assert after == before
    assert new.numerics.induction.gaussian_mesh_policy == GaussianSlabPolicy()


def test_missing_to_mesh_requires_explicit_digest_and_one_exact_grant(tmp_path):
    original = _case()
    new = BENCH._with_gaussian_mesh(original)
    checkpoint, digest = _bundle(tmp_path, numerical_configuration(original.numerics))
    immutable = {p: p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="expected-manifest"):
        BENCH._mesh_restart_admission(checkpoint, new.numerics)
    with pytest.raises(ValueError, match="explicit expectation"):
        BENCH._mesh_restart_admission(checkpoint, new.numerics, "0"*64)
    kwargs, evidence = BENCH._mesh_restart_admission(checkpoint, new.numerics, digest)
    assert kwargs["restart_allowed_config_differences"] == ("vpm.induction.gaussian_mesh_policy",)
    expectation = kwargs["restart_expected_config_differences"]["vpm.induction.gaussian_mesh_policy"]
    assert expectation[0] is MISSING_CONFIGURATION_VALUE
    assert expectation[1] == numerical_configuration(new.numerics)["induction"]["gaussian_mesh_policy"]
    assert evidence["permissions"] == [{"path": "vpm.induction.gaussian_mesh_policy",
        "stored": {"present": False}, "current": {"present": True, "value": expectation[1]}}]
    assert evidence["manifest_sha256"] == digest and len(evidence["source_artifacts"]) == 4
    BENCH._assert_checkpoint_evidence(evidence)
    assert {p: p.read_bytes() for p in immutable} == immutable


def test_identical_mesh_restart_has_no_permissions(tmp_path):
    new = BENCH._with_gaussian_mesh(_case())
    checkpoint, digest = _bundle(tmp_path, numerical_configuration(new.numerics))
    for expected in (None, digest):
        kwargs, evidence = BENCH._mesh_restart_admission(checkpoint, new.numerics, expected)
        assert kwargs == {} and evidence["permissions"] == []


@pytest.mark.parametrize("defect", ["dt", "tail", "kernel", "existing_policy", "none_policy"])
def test_mesh_permission_cannot_hide_other_changes(tmp_path, defect):
    original = _case()
    config = numerical_configuration(original.numerics)
    new = BENCH._with_gaussian_mesh(original)
    if defect == "dt":
        config["time_step_size"] *= 2
    elif defect == "tail":
        config["induction"]["tail_tolerance"] *= 2
    elif defect == "kernel":
        config["particle_kernel"] = "WINCKELMANS"
    elif defect == "existing_policy":
        config["induction"]["gaussian_mesh_policy"] = {"backend": "other"}
    else:
        config["induction"]["gaussian_mesh_policy"] = None
    checkpoint, digest = _bundle(tmp_path, config)
    with pytest.raises(ValueError, match="configuration mismatch"):
        BENCH._mesh_restart_admission(checkpoint, new.numerics, digest)


@pytest.mark.parametrize("corruption", ["configuration", "artifact", "artifact_escape", "missing"])
def test_original_authentication_is_not_repaired_or_bypassed(tmp_path, corruption):
    original = _case()
    new = BENCH._with_gaussian_mesh(original)
    checkpoint, _ = _bundle(tmp_path, numerical_configuration(original.numerics))
    path = checkpoint / "manifest.json"
    manifest = json.loads(path.read_text())
    if corruption == "configuration":
        manifest["config_sha256"] = "0"*64
    elif corruption == "artifact_escape":
        manifest["artifacts"]["vpm"] = "../outside.h5"
    elif corruption == "missing":
        del manifest["artifacts"]["vpm"]
        del manifest["artifact_sha256"]["vpm"]
    else:
        (checkpoint / "vpm.h5").write_bytes(b"corruption remains untouched")
    path.write_text(json.dumps(manifest))
    before = {p: p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        BENCH._mesh_restart_admission(checkpoint, new.numerics, BENCH._digest(path))
    assert {p: p.read_bytes() for p in before} == before


def test_verified_inputs_are_rechecked_without_rewriting(tmp_path):
    original = _case()
    checkpoint, digest = _bundle(tmp_path, numerical_configuration(original.numerics))
    _, evidence = BENCH._mesh_restart_admission(checkpoint, BENCH._with_gaussian_mesh(original).numerics, digest)
    artifact = checkpoint / "vpm.h5"
    artifact.write_bytes(b"later change")
    with pytest.raises(RuntimeError, match="artifact changed"):
        BENCH._assert_checkpoint_evidence(evidence)
    assert artifact.read_bytes() == b"later change"


def test_authored_nondefault_mesh_policy_is_not_overwritten():
    original = _case()
    policy = GaussianSlabPolicy(max_query_points=200)
    induction = SlipSlabInduction(original.numerics.induction.base.build(), z_min=-.48,
                                  z_max=.48, gaussian_mesh_policy=policy)
    original = replace(original, numerics=replace(original.numerics, induction=induction))
    with pytest.raises(ValueError, match="authored nondefault"):
        BENCH._with_gaussian_mesh(original)


def test_hash_inventory_covers_unimported_lazy_sources_and_exact_external_exception(tmp_path, monkeypatch):
    root, case = tmp_path / "selected", tmp_path / "case"
    package = root / "source" / "solvers" / "vpm" / "physics" / "induction" / "gaussian_mesh"
    package.mkdir(parents=True)
    lazy = package / "fields.py"
    lazy.write_text("# lazy module, never imported")
    asset = tmp_path / "benchmark.py"
    asset.write_text("# explicit artifact")
    monkeypatch.setattr(BENCH, "_case_assets", lambda _: {asset})
    outside = tmp_path / "_fenv.so"
    outside.write_bytes(b"identity fixture only; never dynamically loaded")
    identity = {"module": BENCH._FENV_MODULE, "path": str(outside), "sha256": BENCH._digest(outside)}
    monkeypatch.setattr(BENCH, "sys", SimpleNamespace(modules={
        BENCH._FENV_MODULE: SimpleNamespace(__file__=str(outside))}))
    with pytest.raises(RuntimeError, match="outside"):
        BENCH._loaded_source_hashes(root, case)
    inventory = BENCH._source_hashes(root, case, identity)
    assert str(lazy) in inventory and inventory[str(outside)] == identity["sha256"]
    lazy.write_text("# changed before lazy import")
    assert BENCH._source_hashes(root, case, identity)[str(lazy)] != inventory[str(lazy)]
    BENCH.sys.modules["source.unrelated"] = SimpleNamespace(__file__=str(outside))
    with pytest.raises(RuntimeError, match="outside"):
        BENCH._loaded_source_hashes(root, case, identity)
    del BENCH.sys.modules["source.unrelated"]
    outside.write_bytes(b"mutated extension")
    with pytest.raises(RuntimeError, match="outside"):
        BENCH._loaded_source_hashes(root, case, identity)


def test_read_only_collective_action_has_one_root_result_and_shared_failure():
    calls, messages = [], []
    class Channel:
        def __init__(self, rank):
            self.rank = rank
        def Get_rank(self):
            return self.rank
        def Get_size(self):
            return 2
        def bcast(self, value, root):
            if self.rank == root:
                messages.append(value)
            return messages[-1]
    def action():
        calls.append("root")
        return {"read_only": True}
    assert BENCH._collective_read(Channel(0), action) == {"read_only": True}
    assert BENCH._collective_read(Channel(1), action) == {"read_only": True}
    assert calls == ["root"]
    def interrupted():
        raise KeyboardInterrupt("before any solver initialization")
    for rank in (0, 1):
        with pytest.raises(ValueError, match="KeyboardInterrupt"):
            BENCH._collective_read(Channel(rank), interrupted)
