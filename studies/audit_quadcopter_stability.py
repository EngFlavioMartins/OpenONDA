#!/usr/bin/env python3
"""Inspect stopped quadcopter checkpoints without starting a solver or device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np


def audit(case_directory):
    case_directory = Path(case_directory)
    frames = sorted(
        (case_directory / "solution/vpm").glob("vpm_*.h5"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    checkpoint = frames[-1]
    text = (case_directory / "solution/vpm.log").read_text()
    matches = re.findall(
        r"particle=(\d+), position=.*?strain_rate=([\d.eE+-]+) 1/s, time_step_size=([\d.eE+-]+)",
        text,
        re.DOTALL,
    )
    if not matches:
        raise ValueError("No recorded strain-limit failure identifies a target particle")
    target_index, logged_strain, logged_dt = matches[-1]
    target_index = int(target_index)
    with h5py.File(checkpoint) as saved:
        configuration = json.loads(saved["solver"].attrs["numerical_configuration"])
        if configuration["particle_kernel"] != "WINCKELMANS":
            raise ValueError("This direct diagnostic implements the recorded Winckelmans kernel")
        if configuration["stabilization"]["filament_refinement"]["interval_steps"] != 0:
            raise ValueError("Storage-index histories require the original unsplit configuration")
        position = saved["particles/position"][:].astype(float)
        strength = saved["particles/vortex_strength"][:].astype(float)
        core = saved["particles/core_radius"][:].astype(float)
        volume = saved["particles/particle_volume"][:].astype(float)
        groups = saved["particles/group_id"][:]
        eddy = saved["particles/eddy_viscosity"][:]
        step = int(saved["solver"].attrs["step"])
        time = float(saved["solver"].attrs["time"])
        lineage = {
            name: name in saved["particles"]
            for name in ("filament_reference_vortex_strength", "filament_reference_length")
        }
    displacement = position[target_index] - position
    distance = np.linalg.norm(displacement, axis=1)
    rho2 = (distance / core) ** 2
    radial_derivative = (3 * rho2 + 10.5) / (4 * np.pi * core**5 * (1 + rho2) ** 3.5)
    cross = np.cross(strength, displacement)
    strain_terms = (
        -0.5
        * radial_derivative[:, None, None]
        * (
            cross[:, :, None] * displacement[:, None, :]
            + displacement[:, :, None] * cross[:, None, :]
        )
    )
    direct_strain = float(np.max(np.abs(strain_terms.sum(axis=0)).sum(axis=1)))
    source_norms = np.max(np.abs(strain_terms).sum(axis=2), axis=1)
    dominant = int(np.argmax(source_norms))
    history = []
    for frame in (3, 30, 90, 180, 270, step - 1, step):
        path = checkpoint.parent / f"vpm_{frame:06d}.h5"
        if not path.exists():
            continue
        with h5py.File(path) as saved:
            history.append(
                {
                    "step": frame,
                    "strength": float(np.linalg.norm(saved["particles/vortex_strength"][dominant])),
                    "core_radius_m": float(saved["particles/core_radius"][dominant]),
                }
            )
    return {
        "checkpoint": str(checkpoint.relative_to(case_directory)),
        "step": step,
        "time_s": time,
        "particles": len(position),
        "lineage_datasets_present": lineage,
        "recorded_strain_rate_per_s": float(logged_strain),
        "time_step_s": float(logged_dt),
        "recorded_strain_increment": float(logged_strain) * float(logged_dt),
        "particle_only_direct_strain_per_s": direct_strain,
        "particle_only_direct_strain_increment": direct_strain * float(logged_dt),
        "target": {
            "index": target_index,
            "position_m": position[target_index].tolist(),
            "core_radius_m": core[target_index],
            "strength": float(np.linalg.norm(strength[target_index])),
            "volume_m3": volume[target_index],
            "eddy_viscosity_m2_per_s": float(eddy[target_index]),
        },
        "dominant_source": {
            "index": dominant,
            "group": int(groups[dominant]),
            "distance_m": distance[dominant],
            "core_radius_m": core[dominant],
            "strength": float(np.linalg.norm(strength[dominant])),
            "individual_strain_norm_per_s": source_norms[dominant],
            "storage_index_history": history,
        },
        "core_quantiles_m": np.quantile(core, [0, 0.01, 0.5, 0.99, 1]).tolist(),
        "nearest_nonself_distances_m": np.sort(distance[distance > 0])[:8].tolist(),
        "all_loaded_arrays_finite": all(
            np.isfinite(a).all() for a in (position, strength, core, volume, eddy)
        ),
        "qualification": "Particle-only direct diagnostic at one saved target, not a full CFD qualification or complete combined VLM gradient.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("studies/quadcopter_stability_audit.json")
    )
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.case_directory), indent=2) + "\n")


if __name__ == "__main__":
    main()
