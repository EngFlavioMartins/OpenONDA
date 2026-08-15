#!/usr/bin/env python3
"""Run and field-average a reproducible sequential RWM ensemble.

The members execute on CPU because Taichi's Metal backend does not accept a
random seed.  They are run sequentially, so ensemble size affects wall time
and disk use but not peak solver memory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .vortex_diagnostics import average_surface_histories
else:
    from vortex_diagnostics import average_surface_histories

TUTORIAL_DIR = Path(__file__).resolve().parent.parent
SETUP_SCRIPT = TUTORIAL_DIR / "lambossen_setup.py"


def case_name(gamma1: float, gamma2: float) -> str:
    if abs(gamma2) < 1e-12:
        physics = "vortex"
    elif gamma1 * gamma2 < 0.0:
        physics = "dipole"
    else:
        physics = "merging"
    return f"{physics}_rwm"


def average_numeric_csv(target: Path, sources: list[Path]) -> None:
    frames = [pd.read_csv(path) for path in sources]
    if not frames:
        raise ValueError("no RWM CSV histories to average")
    if any(list(frame.columns) != list(frames[0].columns) for frame in frames[1:]):
        raise ValueError(f"RWM ensemble CSV schemas differ for {target.name}")
    for coordinate in ("time", "step"):
        if coordinate in frames[0]:
            reference = frames[0][coordinate].to_numpy()
            if any(
                not np.allclose(frame[coordinate].to_numpy(), reference) for frame in frames[1:]
            ):
                raise ValueError(f"RWM ensemble {coordinate} histories differ for {target.name}")

    averaged = frames[0].copy()
    numeric_columns = averaged.select_dtypes(include=[np.number]).columns
    stacked = np.stack([frame[numeric_columns].to_numpy(float) for frame in frames])
    averaged.loc[:, numeric_columns] = np.nanmean(stacked, axis=0)
    # Time and integer step are coordinates, not stochastic observables.
    for coordinate in ("time", "step"):
        if coordinate in averaged:
            averaged[coordinate] = frames[0][coordinate]
    averaged.to_csv(target, index=False)


def build_ensemble(
    gamma1: float,
    gamma2: float,
    realizations: int,
    base_seed: int,
    output_root: Path,
    keep_members: bool,
) -> int:
    name = case_name(gamma1, gamma2)
    member_root = output_root / ".rwm_ensemble_members" / name
    completed_dirs: list[Path] = []
    completed_metadata: list[dict] = []
    failures: list[dict] = []

    for member in range(realizations):
        seed = base_seed + member
        root = member_root / f"seed_{seed}"
        command = [
            sys.executable,
            "-u",
            str(SETUP_SCRIPT),
            "--gamma1",
            str(gamma1),
            "--gamma2",
            str(gamma2),
            "--schemes",
            "rwm",
            "--processing-unit",
            "CPU",
            "--random-seed",
            str(seed),
            "--output-root",
            str(root),
        ]
        print(f"\n---- {name}: RWM ensemble member {member + 1}/{realizations}, seed={seed} ----")
        result = subprocess.run(command, check=False)  # noqa: S603
        sample_dir = root / "samples" / name
        metadata_path = sample_dir / "run_metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        )
        if result.returncode == 0 and metadata.get("completed") is True:
            completed_dirs.append(sample_dir)
            completed_metadata.append(metadata)
        else:
            failures.append({"seed": seed, "returncode": result.returncode})

    if not completed_dirs:
        print(f"[RWM ensemble] {name}: no completed members; nothing to aggregate", file=sys.stderr)
        return 1

    destination = output_root / "samples" / name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(completed_dirs[0], destination)
    averaged_surfaces = average_surface_histories(destination, completed_dirs)

    integral_sources = [directory / "flow_integrals.csv" for directory in completed_dirs]
    if all(path.is_file() for path in integral_sources):
        average_numeric_csv(destination / "flow_integrals.csv", integral_sources)

    metadata_path = destination / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    seeds = [int(item["random_seed"]) for item in completed_metadata]
    metadata.update(
        {
            "processing_unit": "CPU",
            "resolved_processing_unit": "CPU",
            "random_seed": None,
            "rwm_realizations_requested": realizations,
            "rwm_realizations": len(completed_dirs),
            "rwm_member_seeds": seeds,
            "rwm_failed_members": failures,
            "ensemble_average_stage": "sampled_eulerian_fields_before_diagnostics",
            "ensemble_averaged_surface_count": len(averaged_surfaces),
            "ensemble_member_wall_time_seconds": [
                float(item.get("wall_time_seconds", float("nan"))) for item in completed_metadata
            ],
            "wall_time_seconds": float(
                np.nansum([item.get("wall_time_seconds", np.nan) for item in completed_metadata])
            ),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not keep_members:
        shutil.rmtree(member_root)
    print(
        f"[RWM ensemble] {name}: averaged {len(completed_dirs)}/{realizations} "
        f"members across {len(averaged_surfaces)} sampled fields"
    )
    return 0 if len(completed_dirs) == realizations else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma1", type=float, default=1.0)
    parser.add_argument("--gamma2", type=float, default=0.0)
    parser.add_argument("--realizations", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=TUTORIAL_DIR)
    parser.add_argument("--keep-members", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.realizations < 1:
        raise ValueError("realizations must be at least one")
    return build_ensemble(
        args.gamma1,
        args.gamma2,
        args.realizations,
        args.base_seed,
        args.output_root.resolve(),
        args.keep_members,
    )


if __name__ == "__main__":
    raise SystemExit(main())
