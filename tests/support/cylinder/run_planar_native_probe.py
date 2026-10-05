"""Bounded fresh planar cylinder probe using a read-only native XY section.

The old field is never continued or imported. Only the cut-cell XY geometry
is sliced and freshly extruded into the authored unit-span periodic cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from source.coupler.backup import artifact_digest
from source.solvers.fvm.io.mesh_storage import load_native_mesh, save_native_mesh
from source.solvers.fvm.mesh.cartesian.extrusion import extrude_mesh_section
from tests.coupler.test_cylinder_planar_native import _advance, _physical_case


def run(directory, source_mesh, *, coordinate=0.02, h=0.04, end_time=0.08, exchanges=2):
    directory.mkdir(parents=True, exist_ok=False)
    _flow, _particles, _exchange, builder, _seed = _physical_case(h=h, end_time=end_time)
    started = time.perf_counter()
    old_mesh = load_native_mesh(source_mesh)
    mesh = extrude_mesh_section(
        old_mesh,
        coordinate=coordinate,
        levels=builder.levels,
        domain=builder.domain,
        surfaces=builder.source.surfaces,
    )
    native = save_native_mesh(mesh, directory / "one_layer_mesh.npz")
    preparation_seconds = time.perf_counter() - started
    result = _advance(
        directory / "case",
        limit=exchanges,
        start_from="initial",
        mesh_path=native,
        h=h,
        end_time=end_time,
    )
    report = {
        "scope": "bounded fresh CPU probe; not developed shedding or production GPU speed qualification",
        "source_mesh": str(source_mesh.resolve()),
        "source_mesh_sha256": artifact_digest(source_mesh),
        "source_mesh_cells": old_mesh["n_cells"],
        "mesh_section_coordinate": coordinate,
        "mesh_preparation_seconds": preparation_seconds,
        "mesh_provenance": "native source XY geometry; new unit-span extrusion and fresh initial fields",
        "cell_count": result["cells"],
        "h": h,
        "span": 1.0,
        "z_layers": 1,
        "particle_z_rows": 1,
        "particle_count": len(result["positions"]),
        "particle_volume": h**2,
        "covered_flow_time": [0.0, result["manifest"]["time"]],
        "accepted_exchanges": result["accepted"],
        "committed_checkpoint_time": result["manifest"]["time"],
        "exchanges": result["exchanges"],
        "forces": result["forces"],
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("cell_count", "particle_count", "accepted_exchanges")}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("source_mesh", type=Path)
    parser.add_argument("--coordinate", type=float, default=0.02)
    parser.add_argument("--h", type=float, default=0.04)
    parser.add_argument("--end-time", type=float, default=0.08)
    parser.add_argument("--exchanges", type=int, default=2)
    arguments = parser.parse_args()
    run(
        arguments.directory,
        arguments.source_mesh,
        coordinate=arguments.coordinate,
        h=arguments.h,
        end_time=arguments.end_time,
        exchanges=arguments.exchanges,
    )
