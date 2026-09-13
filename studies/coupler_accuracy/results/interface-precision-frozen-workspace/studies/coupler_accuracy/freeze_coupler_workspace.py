#!/usr/bin/env python3
"""Copy code and explicit immutable cube inputs for isolated coupling experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    code = []
    excluded = {"__pycache__", ".pytest_cache", ".ruff_cache", "results", "samples", "solution", "reference_flow"}
    for directory in ("source", "openonda", "tutorials/coupled_fvm_vpm/02_cube_flow"):
        code += [path for path in (ROOT / directory).rglob("*") if path.is_file()
                 and not any(part in excluded for part in path.relative_to(ROOT / directory).parts)]
    code += list((ROOT / "studies/coupler_accuracy").glob("*.py"))
    for name in ("pyproject.toml", "setup.py", "setup.cfg"):
        if (ROOT / name).exists():
            code.append(ROOT / name)
    code = sorted(set(code))
    records = []
    for path in code:
        relative = path.relative_to(ROOT)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes()
        target.write_bytes(content)
        records.append({"path": str(relative), "sha256": hashlib.sha256(content).hexdigest(), "kind": "code"})
    # The snapshot is coherent over the copy window, not a mixture silently
    # accepted while source files were being edited.
    changed = [row["path"] for row in records if digest(ROOT / row["path"]) != row["sha256"]]
    if changed:
        (output / "source-copy-failure.json").write_text(json.dumps({"changed": changed}, indent=2)+"\n")
        raise RuntimeError("Source changed during snapshot: "+", ".join(changed))
    oracle = ROOT / "studies/coupler_accuracy/results/cube-3d-medium-laminar-oracle"
    inputs = [path for path in oracle.iterdir() if path.is_file() and path.suffix in (".npz", ".json")]
    qualification = ROOT / "studies/coupler_accuracy/results/cube-3d-panel-derivative-precision-coarse-qualified/panel-derivative-precision-3d.json"
    q = json.loads(qualification.read_text())
    assert q["status"] == "complete"
    inputs += [qualification, ROOT / q["fields"]["path"]]
    for row in q["sources"]:
        path = ROOT / row["path"]
        assert digest(path) == row["sha256"], row["path"]
        if path not in code:
            inputs.append(path)
    for path in sorted(set(inputs)):
        relative = path.relative_to(ROOT)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = digest(path)
        shutil.copyfile(path, target)
        assert digest(target) == expected
        records.append({"path": str(relative), "sha256": expected, "kind": "input"})
    result = {"schema": "openonda-frozen-coupler-workspace/1", "status": "complete", "source_root": str(ROOT),
              "snapshot_root": str(output), "created_at": datetime.now(timezone.utc).isoformat(),
              "code_files": len(code), "input_files": len(set(inputs)), "records": records,
              "limitations": ["Source stability is checked across the copy window. Subsequent shared-workspace edits do not change these copied bytes.",
                              "Installed Python and native dependencies still come from the OpenONDA environment. This is a source snapshot, not a fully isolated operating-system image.",
                              "New experiments must use this root as working directory and PYTHONPATH. Prior runs are not reclassified as having used this snapshot."]}
    (output / "frozen-workspace.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "snapshot_root", "code_files", "input_files")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())
