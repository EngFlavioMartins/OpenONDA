#!/usr/bin/env python3
"""Build and publish one reference-flow cylinder mesh without starting FVM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.solvers.fvm.io.mesh_storage import save_native_mesh
from source.solvers.fvm.io.vtk_exporter import VTKExporter
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.validation import (
    validate_cell_area_closure,
    validate_geometry,
    validate_single_fluid_component,
    validate_topology,
)

try:
    from .canonical_surface import prepare_canonical_surfaces
    from .setup import CYLINDER_STL, grid_mesh, solver_setup
except ImportError:  # Direct execution from the reference_flow directory.
    from canonical_surface import prepare_canonical_surfaces
    from setup import CYLINDER_STL, grid_mesh, solver_setup


CASES = {
    "very_coarse": 1.0 / 12.0,
    "coarse": 0.025,
    "medium": 0.0125,
    "fine": 0.00625,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _mesh_identity(mesh: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ("vertex_position", "owners", "neighbours", "cell_levels", "cell_sizes"):
        value = mesh.get(name)
        if value is None:
            continue
        array = value if hasattr(value, "tobytes") else repr(value).encode()
        digest.update(name.encode("ascii"))
        digest.update(array.tobytes() if hasattr(array, "tobytes") else array)
    for face in mesh["faces"]:
        digest.update(bytes(str(tuple(int(point) for point in face)), "ascii"))
    return digest.hexdigest()


def _build(case: str, destination: Path) -> None:
    dx = CASES[case]
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing mesh directory {destination}; choose a new output path"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    canonical = prepare_canonical_surfaces(CYLINDER_STL, destination.parent / "canonical_inputs")
    mesher = grid_mesh(dx)
    print(f"[mesh] case={case} stage=build dx={dx:.17g}", flush=True)
    mesh = mesher.build()
    elapsed = time.perf_counter() - started
    print(
        f"[mesh] case={case} stage=validation cells={mesh['n_cells']} "
        f"faces={mesh['n_faces']} elapsed={elapsed:.3f}s",
        flush=True,
    )
    topology = validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    quality = dict(validate_geometry(mesh, geometry))
    quality.update(validate_cell_area_closure(mesh, geometry))
    quality.update(validate_single_fluid_component(mesh))
    identity = _mesh_identity(mesh)

    with tempfile.TemporaryDirectory(prefix=f".{case}.mesh.", dir=destination.parent) as temp_name:
        temporary = Path(temp_name)
        save_native_mesh(mesh, temporary / "mesh.npz")
        fields = {
            "cell_volume": geometry["cell_volume"],
            "cell_size": mesh.get("cell_sizes"),
            "refinement_level": mesh.get("cell_levels"),
        }
        VTKExporter(mesh, solver_setup(case, dx).output).export(
            str(temporary / "mesh.vtu"),
            {name: value for name, value in fields.items() if value is not None},
        )
        report = {
            "case": case,
            "requested": {"dx": dx, "background": 8.0 * dx, "cylinder": dx},
            "canonical_inputs": _jsonable(canonical),
            "mesh_generation": _jsonable(mesh.get("mesh_generation", {})),
            "counts": {
                "cells": int(mesh["n_cells"]),
                "faces": int(mesh["n_faces"]),
                "interior_faces": int(mesh["n_interior_faces"]),
                "points": int(mesh["n_points"]),
            },
            "mesh_identity": identity,
            "topology": _jsonable(topology),
            "quality": _jsonable(quality),
            "elapsed_seconds": elapsed,
        }
        (temporary / "mesh_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "format": "openonda.cylinder_mesh_manifest.v1",
            "case": case,
            "source_stl": str(CYLINDER_STL),
            "source_stl_sha256": _sha256(CYLINDER_STL),
            "code_files": {
                "mesher": _sha256(ROOT / "source/solvers/fvm/mesh/cartesian/mesher.py"),
                "setup": _sha256(Path(__file__).with_name("setup.py")),
                "canonical_surface": _sha256(Path(__file__).with_name("canonical_surface.py")),
            },
            "mesh_identity": identity,
            "requested_dx": dx,
            "output_sha256": {
                "mesh.npz": _sha256(temporary / "mesh.npz"),
                "mesh.vtu": _sha256(temporary / "mesh.vtu"),
                "mesh_report.json": _sha256(temporary / "mesh_report.json"),
            },
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        (temporary / "mesh_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        destination.mkdir()
        for source in temporary.iterdir():
            source.replace(destination / source.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    output = arguments.output_dir or Path(__file__).resolve().parent / "solution" / arguments.case
    _build(arguments.case, output.resolve())


if __name__ == "__main__":
    main()
