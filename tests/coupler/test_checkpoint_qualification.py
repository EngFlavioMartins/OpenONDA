"""Strict checkpoint qualification controls, using temporary native-file fixtures."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from source.coupler.backup import BACKUP_FORMAT_VERSION, artifact_digest, config_mapping_digest
from source.solvers.vpm.config import Numerics, ViscousConfig, VPMCase
from source.solvers.vpm.config.fingerprint import numerical_configuration
from source.solvers.vpm.physics.induction.fmm.device import FMMInduction
from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction

ASSET = (
    Path(__file__).resolve().parents[2] / "tests/support/cylinder/benchmark_coupled_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_checkpoint_asset", ASSET)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def _case():
    return VPMCase(
        numerics=Numerics(
            time_step_size=0.04,
            compute_device="CUDA",
            viscous=ViscousConfig.inviscid(),
            induction=SlipSlabInduction(
                FMMInduction(), z_min=-0.48, z_max=0.48, tail_tolerance=1e-4, max_shells=129
            ),
        )
    )


def _bundle(tmp_path, config):
    checkpoint = tmp_path / "bundle"
    checkpoint.mkdir()
    artifacts = {
        "fvm": "fvm",
        "vpm": "vpm.h5",
        "vpm_vtu": "particles.vtu",
        "vpm_boundary_condition": "boundary.npz",
    }
    (checkpoint / "fvm").mkdir()
    (checkpoint / "fvm" / "state.bin").write_bytes(b"immutable fake FVM artifact")
    for filename in ("vpm.h5", "particles.vtu", "boundary.npz"):
        (checkpoint / filename).write_bytes(("immutable fixture " + filename).encode())
    configuration = {"vpm": deepcopy(config), "coupler": {"step_size": 0.04}}
    manifest = {
        "format_version": BACKUP_FORMAT_VERSION,
        "kind": "openonda.coupled_backup",
        "backend": "fvm",
        "config": configuration,
        "config_sha256": config_mapping_digest(configuration),
        "artifacts": artifacts,
        "artifact_sha256": {
            key: artifact_digest(checkpoint / value) for key, value in artifacts.items()
        },
    }
    path = checkpoint / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint, BENCH._digest(path)


def test_identical_configuration_is_authenticated_without_rewriting(tmp_path):
    case = _case()
    checkpoint, digest = _bundle(tmp_path, numerical_configuration(case.numerics))
    immutable = {p: p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()}
    evidence = BENCH._checkpoint_admission(checkpoint, case.numerics)
    assert evidence["manifest_sha256"] == digest
    assert len(evidence["source_artifacts"]) == 4
    BENCH._assert_checkpoint_evidence(evidence)
    assert {p: p.read_bytes() for p in immutable} == immutable


@pytest.mark.parametrize("defect", ["dt", "tail", "kernel", "missing_policy", "none_policy"])
def test_different_numerics_cannot_be_replayed(tmp_path, defect):
    case = _case()
    config = numerical_configuration(case.numerics)
    if defect == "dt":
        config["time_step_size"] *= 2
    elif defect == "tail":
        config["induction"]["tail_tolerance"] *= 2
    elif defect == "kernel":
        config["particle_kernel"] = "WINCKELMANS"
    elif defect == "missing_policy":
        del config["induction"]["gaussian_mesh_policy"]
    else:
        config["induction"]["gaussian_mesh_policy"] = None
    checkpoint, _ = _bundle(tmp_path, config)
    with pytest.raises(ValueError, match="configuration mismatch"):
        BENCH._checkpoint_admission(checkpoint, case.numerics)


@pytest.mark.parametrize("corruption", ["configuration", "artifact", "artifact_escape", "missing"])
def test_corrupt_checkpoint_is_rejected_without_repair(tmp_path, corruption):
    case = _case()
    checkpoint, _ = _bundle(tmp_path, numerical_configuration(case.numerics))
    path = checkpoint / "manifest.json"
    manifest = json.loads(path.read_text())
    if corruption == "configuration":
        manifest["config_sha256"] = "0" * 64
    elif corruption == "artifact_escape":
        manifest["artifacts"]["vpm"] = "../outside.h5"
    elif corruption == "missing":
        del manifest["artifacts"]["vpm"]
        del manifest["artifact_sha256"]["vpm"]
    else:
        (checkpoint / "vpm.h5").write_bytes(b"corrupt artifact")
    path.write_text(json.dumps(manifest))
    before = {p: p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        BENCH._checkpoint_admission(checkpoint, case.numerics)
    assert {p: p.read_bytes() for p in before} == before


def test_authenticated_artifacts_are_rechecked(tmp_path):
    case = _case()
    checkpoint, _ = _bundle(tmp_path, numerical_configuration(case.numerics))
    evidence = BENCH._checkpoint_admission(checkpoint, case.numerics)
    artifact = checkpoint / "vpm.h5"
    artifact.write_bytes(b"later change")
    with pytest.raises(RuntimeError, match="artifact changed"):
        BENCH._assert_checkpoint_evidence(evidence)
    assert artifact.read_bytes() == b"later change"


def test_hash_inventory_covers_unimported_lazy_sources_and_exact_external_exception(
    tmp_path, monkeypatch
):
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
    identity = {
        "module": BENCH._FENV_MODULE,
        "path": str(outside),
        "sha256": BENCH._digest(outside),
    }
    monkeypatch.setattr(
        BENCH,
        "sys",
        SimpleNamespace(modules={BENCH._FENV_MODULE: SimpleNamespace(__file__=str(outside))}),
    )
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
