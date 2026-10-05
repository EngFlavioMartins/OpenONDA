"""Strict checkpoint qualification controls, using temporary native-file fixtures."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from source.coupler.backup import BACKUP_FORMAT_VERSION, checkpoint_path_hash, config_mapping_digest
from source.solvers.vpm.config import Numerics, ViscousConfig, VPMCase
from source.solvers.vpm.config.configuration_values import numerical_configuration
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
    checkpoint_files = {
        "fvm": "fvm",
        "vpm": "vpm.h5",
        "vpm_vtu": "particles.vtu",
        "vpm_boundary_condition": "boundary.npz",
    }
    (checkpoint / "fvm").mkdir()
    (checkpoint / "fvm" / "state.bin").write_bytes(b"immutable fake FVM checkpoint_file")
    for filename in ("vpm.h5", "particles.vtu", "boundary.npz"):
        (checkpoint / filename).write_bytes(("immutable fixture " + filename).encode())
    configuration = {"vpm": deepcopy(config), "coupler": {"step_size": 0.04}}
    checkpoint_info = {
        "format_version": BACKUP_FORMAT_VERSION,
        "kind": "openonda.coupled_backup",
        "backend": "fvm",
        "config": configuration,
        "config_sha256": config_mapping_digest(configuration),
        "checkpoint_files": checkpoint_files,
        "file_sha256": {
            key: checkpoint_path_hash(checkpoint / value) for key, value in checkpoint_files.items()
        },
    }
    path = checkpoint / "checkpoint_info.json"
    path.write_text(json.dumps(checkpoint_info), encoding="utf-8")
    return checkpoint, BENCH._digest(path)


def test_identical_configuration_is_hash_verified_without_rewriting(tmp_path):
    case = _case()
    checkpoint, digest = _bundle(tmp_path, numerical_configuration(case.numerics))
    immutable = {p: p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()}
    evidence = BENCH._validate_checkpoint(checkpoint, case.numerics)
    assert evidence["checkpoint_info_sha256"] == digest
    assert len(evidence["source_checkpoint_files"]) == 4
    BENCH._assert_checkpoint_evidence(evidence)
    assert {p: p.read_bytes() for p in immutable} == immutable


@pytest.mark.parametrize("defect", ["dt", "tail", "kernel", "missing_settings", "none_settings"])
def test_different_numerics_cannot_be_replayed(tmp_path, defect):
    case = _case()
    config = numerical_configuration(case.numerics)
    if defect == "dt":
        config["time_step_size"] *= 2
    elif defect == "tail":
        config["induction"]["tail_tolerance"] *= 2
    elif defect == "kernel":
        config["particle_kernel"] = "WINCKELMANS"
    elif defect == "missing_settings":
        del config["induction"]["gaussian_mesh_settings"]
    else:
        config["induction"]["gaussian_mesh_settings"] = None
    checkpoint, _ = _bundle(tmp_path, config)
    with pytest.raises(ValueError, match="configuration mismatch"):
        BENCH._validate_checkpoint(checkpoint, case.numerics)


@pytest.mark.parametrize(
    "corruption", ["configuration", "checkpoint_file", "checkpoint_file_escape", "missing"]
)
def test_corrupt_checkpoint_is_rejected_without_repair(tmp_path, corruption):
    case = _case()
    checkpoint, _ = _bundle(tmp_path, numerical_configuration(case.numerics))
    path = checkpoint / "checkpoint_info.json"
    checkpoint_info = json.loads(path.read_text())
    if corruption == "configuration":
        checkpoint_info["config_sha256"] = "0" * 64
    elif corruption == "checkpoint_file_escape":
        checkpoint_info["checkpoint_files"]["vpm"] = "../outside.h5"
    elif corruption == "missing":
        del checkpoint_info["checkpoint_files"]["vpm"]
        del checkpoint_info["file_sha256"]["vpm"]
    else:
        (checkpoint / "vpm.h5").write_bytes(b"corrupt checkpoint_file")
    path.write_text(json.dumps(checkpoint_info))
    before = {p: p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        BENCH._validate_checkpoint(checkpoint, case.numerics)
    assert {p: p.read_bytes() for p in before} == before


def test_hash_verified_checkpoint_files_are_rechecked(tmp_path):
    case = _case()
    checkpoint, _ = _bundle(tmp_path, numerical_configuration(case.numerics))
    evidence = BENCH._validate_checkpoint(checkpoint, case.numerics)
    checkpoint_file = checkpoint / "vpm.h5"
    checkpoint_file.write_bytes(b"later change")
    with pytest.raises(RuntimeError, match="checkpoint file changed"):
        BENCH._assert_checkpoint_evidence(evidence)
    assert checkpoint_file.read_bytes() == b"later change"


def test_hash_inventory_covers_unimported_lazy_sources_and_exact_external_exception(
    tmp_path, monkeypatch
):
    root, case = tmp_path / "selected", tmp_path / "case"
    package = root / "source" / "solvers" / "vpm" / "physics" / "induction" / "gaussian_mesh"
    package.mkdir(parents=True)
    lazy = package / "fields.py"
    lazy.write_text("# lazy module, never imported")
    asset = tmp_path / "benchmark.py"
    asset.write_text("# explicit checkpoint_file")
    monkeypatch.setattr(BENCH, "_case_assets", lambda _: {asset})
    outside = tmp_path / "_fenv.so"
    outside.write_bytes(b"configuration fixture only; never dynamically loaded")
    configuration = {
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
    inventory = BENCH._source_hashes(root, case, configuration)
    assert str(lazy) in inventory and inventory[str(outside)] == configuration["sha256"]
    lazy.write_text("# changed before lazy import")
    assert BENCH._source_hashes(root, case, configuration)[str(lazy)] != inventory[str(lazy)]
    BENCH.sys.modules["source.unrelated"] = SimpleNamespace(__file__=str(outside))
    with pytest.raises(RuntimeError, match="outside"):
        BENCH._loaded_source_hashes(root, case, configuration)
    del BENCH.sys.modules["source.unrelated"]
    outside.write_bytes(b"mutated extension")
    with pytest.raises(RuntimeError, match="outside"):
        BENCH._loaded_source_hashes(root, case, configuration)


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
