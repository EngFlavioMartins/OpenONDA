#!/usr/bin/env python3
"""Isolate omega*volume -> velocity using an existing cylinder FVM snapshot.

This is an offline free-space quadrature experiment, not a coupled simulation.
It applies the analytic Gaussian Biot-Savart kernels without a treecode, time
advance, renewal, pruning, or panels. The 3D field is repeated across 1, 3 and
9 spans; a separate 2D kernel integrates the span exactly. Differences from
FVM include quadrature/core errors and its finite outer boundary conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gaussian_velocity(target, position, strength, sigma, *, planar=False):
    """Independent direct sum, with bounded temporary memory and float64."""
    velocity = np.zeros_like(target)
    for start in range(0, len(position), 2048):
        delta = target[:, None] - position[None, start : start + 2048]
        if planar:
            delta[..., 2] = 0.0
        radius2 = np.sum(delta**2, axis=-1)
        safe = np.where(radius2 > 0.0, radius2, 1.0)
        if planar:
            factor = -np.expm1(-radius2 / sigma**2) / (2.0 * np.pi * safe)
        else:
            q = np.sqrt(radius2) / sigma
            factor = (erf(q) - 2.0 / np.sqrt(np.pi) * q * np.exp(-q * q)) / (
                4.0 * np.pi * safe**1.5
            )
            # Analytic origin expansion avoids subtracting nearly equal terms.
            small = q < 0.01
            factor[small] = (1.0 / (3.0 * np.pi**1.5 * sigma**3)) * (
                1.0 - 0.6 * q[small] ** 2 + 3.0 / 14.0 * q[small] ** 4
            )
        velocity += np.sum(
            np.cross(strength[None, start : start + 2048], delta) * factor[..., None], axis=1
        )
    return velocity


def run(reference, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyvista as pv

    solution = CASE / "reference_flow/solution" / reference
    samples = CASE / "reference_flow/samples" / reference
    profiles = [pd.read_csv(samples / f"transverse_x{x}.csv") for x in (1, 2, 4)]
    last_common_sample = min(float(table.time.max()) for table in profiles)
    pvd = solution / f"{reference}.pvd"
    datasets = ET.parse(pvd).getroot().findall(".//DataSet")
    eligible = [
        entry for entry in datasets if float(entry.attrib["timestep"]) <= last_common_sample + 1e-8
    ]
    latest = max(eligible, key=lambda entry: float(entry.attrib["timestep"]))
    flow_time = float(latest.attrib["timestep"])
    snapshot = solution / latest.attrib["file"]
    grid = pv.read(snapshot)
    ids = np.asarray(grid.cell_data["global_cell_id"], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Snapshot includes duplicate/ghost cell IDs")
    mesh = pv.read(solution / "mesh.vtu")
    if mesh.n_cells != len(ids) or ids.min() != 0 or ids.max() != mesh.n_cells - 1:
        raise ValueError("Snapshot cell IDs do not cover the reference mesh")
    volumes = np.asarray(mesh.cell_data["cell_volume"], dtype=np.float64)[ids]
    position = np.asarray(grid.cell_centers().points, dtype=np.float64)
    strength = volumes[:, None] * np.asarray(grid.cell_data["vorticity"], dtype=np.float64)
    span = float(grid.bounds[5] - grid.bounds[4])
    target, expected = [], []
    for x, table in zip((1, 2, 4), profiles, strict=True):
        rows = table[np.isclose(table.time, flow_time, atol=1e-8, rtol=0)].sort_values("position_y")
        if len(rows) < 2:
            raise ValueError(f"Missing profile at t={flow_time:g}")
        y = rows.position_y.to_numpy(dtype=np.float64)
        target.append(np.column_stack((np.full(len(y), x), y, np.zeros(len(y)))))
        expected.append(rows[[f"velocity_{axis}" for axis in "xyz"]].to_numpy(dtype=np.float64))
    target, expected = np.concatenate(target), np.concatenate(expected)
    sigma = 1.0 / 16.0
    started = time.perf_counter()
    outputs = {}
    # Integrating omega*V/span over all spanwise cells gives 2D circulation.
    planar_strength = np.zeros_like(strength)
    planar_strength[:, 2] = strength[:, 2] / span
    outputs["2D"] = gaussian_velocity(target, position, planar_strength, sigma, planar=True) + [
        1,
        0,
        0,
    ]
    for count in (1, 3, 9):
        velocity = np.zeros_like(target)
        for shift in range(-(count // 2), count // 2 + 1):
            shifted = position.copy()
            shifted[:, 2] += shift * span
            velocity += gaussian_velocity(target, shifted, strength, sigma)
        outputs[f"3D, span {count * span:g}D"] = velocity + [1, 0, 0]
        print(f"Finished {count} spans", flush=True)
    metrics = {}
    for name, velocity in outputs.items():
        difference = velocity - expected
        metrics[name] = {
            "velocity_rms_error_over_Uinf": float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))),
            "velocity_max_error_over_Uinf": float(np.max(np.linalg.norm(difference, axis=1))),
            "rms_difference_from_2D_over_Uinf": float(
                np.sqrt(np.mean(np.sum((velocity - outputs["2D"]) ** 2, axis=1)))
            ),
        }
    output.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((target, expected, *(outputs.values())))
    names = ["x", "y", "z", "reference_u", "reference_v", "reference_w"]
    names += [f"{name}_{axis}" for name in outputs for axis in "uvw"]
    pd.DataFrame(data, columns=names).to_csv(output / "snapshot-induction.csv", index=False)
    source_files = [snapshot, solution / "mesh.vtu", solution / "fvm_metadata.json"]
    source_files += [samples / f"transverse_x{x}.csv" for x in (1, 2, 4)]
    source_files += [
        snapshot.parent / item.attrib["Source"]
        for item in ET.parse(snapshot).getroot().findall(".//Piece")
    ]
    report = {
        "schema": "openonda-coupler-snapshot-induction/1",
        "reference": reference,
        "time": flow_time,
        "cells": len(position),
        "sigma": sigma,
        "target_count": len(target),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "method": "Direct free-space Gaussian induction of saved FVM omega*V, no panels; replicated spans or exact 2D kernel.",
        "limitations": [
            "VTK parametric cell centres approximate native volume centroids.",
            "Free-space induction has different outer boundary conditions from the finite FVM reference.",
            "This is a fixed-field diagnostic, not a coupled-flow validation.",
        ],
        "sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in source_files
        ],
    }
    (output / "snapshot-induction.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    colors = ["#167d5b", "#bc5038", "#da9532", "#396bb5"]
    for column, x in enumerate((1, 2, 4)):
        at = target[:, 0] == x
        for row in (0, 1):
            axis = axes[row, column]
            axis.plot(target[at, 1], expected[at, row], "k.-", label="FVM snapshot", linewidth=1.5)
            for (name, velocity), color in zip(outputs.items(), colors, strict=True):
                axis.plot(target[at, 1], velocity[at, row], label=name, color=color, linewidth=1.2)
            axis.set(xlabel="y/D", ylabel=("u/U∞" if row == 0 else "v/U∞"))
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(f"x/D = {x}")
    axes[0, 2].legend(fontsize=8)
    fig.suptitle(
        f"Fixed reference vorticity → velocity; {reference}, t = {flow_time:.2f}\nDiagnostic only: no time advance or coupling"
    )
    fig.savefig(output / "snapshot-induction.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="fine")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    run(args.reference, args.output)
