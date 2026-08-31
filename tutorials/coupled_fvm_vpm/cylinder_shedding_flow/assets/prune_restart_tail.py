#!/usr/bin/env python3
"""Trim incomplete reference histories back to the latest FVM checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import numpy as np


def checkpoint_position(checkpoint: Path) -> tuple[int, float]:
    """Read the committed step and time from a serial or MPI checkpoint."""
    if checkpoint.is_dir():
        manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
        state_file = checkpoint / manifest["files"][0]
    else:
        state_file = checkpoint
    with np.load(state_file, allow_pickle=False) as state:
        return int(state["step"].item()), float(state["time"].item())


def _atomic_replace(path: Path, retained: list[str]) -> int:
    original_count = sum(1 for _ in path.open(encoding="utf-8", errors="strict"))
    temporary = path.with_name(f".{path.name}.restart-tail.tmp")
    temporary.write_text("".join(retained), encoding="utf-8")
    os.replace(temporary, path)
    return original_count - len(retained)


def prune_json_lines(path: Path, checkpoint_step: int) -> int:
    """Remove JSON-line records whose solver step is not checkpointed."""
    if not path.is_file():
        return 0
    retained: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if int(payload.get("step", checkpoint_step)) <= checkpoint_step:
                retained.append(line)
    return _atomic_replace(path, retained)


def prune_csv(path: Path, checkpoint_step: int) -> int:
    """Remove sampled rows newer than the checkpoint, preserving text exactly."""
    retained: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            if line_number == 0:
                retained.append(line)
                continue
            fields = line.split(",", 2)
            if len(fields) < 2:
                raise ValueError(f"Malformed sample row in {path}: {line.rstrip()}")
            if int(fields[1]) <= checkpoint_step:
                retained.append(line)
    return _atomic_replace(path, retained)


def reconcile_surface_indices(
    samples: Path, checkpoint_step: int, time_step_size: float
) -> tuple[int, int]:
    """Remove uncheckpointed VTS frames and rebuild complete PVD indices."""
    pattern = re.compile(r"^(?P<name>.+)_(?P<step>\d+)\.vts$")
    grouped: dict[str, list[tuple[int, str]]] = {}
    removed = 0
    for path in sorted(samples.glob("*.vts")):
        match = pattern.match(path.name)
        if match is None:
            continue
        step = int(match.group("step"))
        if step > checkpoint_step:
            path.unlink()
            removed += 1
            continue
        grouped.setdefault(match.group("name"), []).append((step, path.name))

    entry_count = 0
    for name, frames in grouped.items():
        destination = samples / f"{name}.pvd"
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            "  <Collection>",
        ]
        lines.extend(
            f'    <DataSet timestep="{step * time_step_size:.15g}" file="{filename}"/>'
            for step, filename in sorted(frames)
        )
        lines.extend(("  </Collection>", "</VTKFile>"))
        temporary = destination.with_name(f".{destination.name}.restart-tail.tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        entry_count += len(frames)
    return removed, entry_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reference_flow",
        help="reference_flow directory",
    )
    args = parser.parse_args()
    case = args.case.resolve()
    checkpoint = case / "solution" / "checkpoint"
    if not checkpoint.exists():
        checkpoint = checkpoint.with_suffix(".npz")
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint found below {case / 'solution'}")

    step, time = checkpoint_position(checkpoint)
    metadata = json.loads(
        (case / "solution" / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    surface_files, pvd_entries = reconcile_surface_indices(
        case / "samples", step, float(metadata["time"]["fvm_time_step"])
    )
    removed = {
        "diagnostics": prune_json_lines(case / "solution" / "diagnostics.jsonl", step),
        "performance": prune_json_lines(case / "solution" / "performance.jsonl", step),
        "sample_rows": sum(
            prune_csv(path, step) for path in sorted((case / "samples").glob("*.csv"))
        ),
        "surface_files": surface_files,
    }
    print(
        json.dumps(
            {
                "checkpoint_step": step,
                "checkpoint_time": time,
                "removed": removed,
                "pvd_entries": pvd_entries,
            }
        )
    )


if __name__ == "__main__":
    main()
