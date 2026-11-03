#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hash the Cartesian mesher's output on small cases, to prove a refactor bit for bit.

Every array and value of the returned mesh, octree and diagnostics is hashed
separately, so a difference names the field that changed. Run once before a
change and once after it::

    python tests/mesh_parity/output_hashes.py --stage templateGeneration --output before.json
    python tests/mesh_parity/output_hashes.py --stage templateGeneration --compare before.json

``--stage final`` runs the complete build, including the wall constraint and
the final validation. The cases cover box and patch refinement, automatic
refinement, exact cell sizes, a surface crossing the domain, and extrusion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

from openonda.fvm.mesh.progress import mesher_log_session
import openonda.fvm.mesher as msh

ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = ROOT / "tutorials/coupled_fvm_vpm"
CYLINDER = TUTORIALS / "01_cylinder_shedding_flow/assets/cylinder_long.stl"
PATCHES = {"xmin": "inlet", "xmax": "outlet", "ymin": "ymin", "ymax": "ymax"}
PATCHES |= {"zmin": "zmin", "zmax": "zmax"}


def _cylinder_box() -> msh.CartesianMesher:
    """Box and patch refinement around the cylinder, 56k template cells."""
    return msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=(-4.0, 8.0, -4.0, 4.0, -1.6, 1.6), patches=msh.BoxPatches(**PATCHES)
        ),
        surfaces=(msh.STLSurface(CYLINDER, patch="body"),),
        max_cell_size=0.8,
        refinements=(msh.BoxRefinement("near_body", (-1.5, 4.0, -1.5, 1.5, -1.2, 1.2), 0.2),),
        patch_refinements=(msh.PatchRefinement("body", 0.1),),
    )


def _cylinder_automatic() -> msh.CartesianMesher:
    """The parity case ``cylinder_box_refinement``, with automatic refinement."""
    patches = {"xmin": "inlet", "xmax": "outlet", "ymin": "walls", "ymax": "walls"}
    patches |= {"zmin": "front_back", "zmax": "front_back"}
    return msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=(-3.0, 5.0, -3.0, 3.0, -7.0, 7.0), patches=msh.BoxPatches(**patches)
        ),
        surfaces=(msh.STLSurface(CYLINDER, patch="body"),),
        max_cell_size=1.0,
        boundary_cell_size=0.5,
        min_cell_size=0.25,
        refinements=(msh.BoxRefinement("near_body", (-1.5, 2.0, -1.5, 1.5, -6.5, 6.5), 0.25),),
    )


def _wing(directory: Path) -> msh.CartesianMesher:
    """Case 03's wing and refinement boxes at four times its cell size, exact sizes."""
    path = TUTORIALS / "03_naca4412_flow/reference_flow/assets/naca_surface.py"
    spec = importlib.util.spec_from_file_location("naca_surface", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stl = directory / "wing.stl"
    module.write_wing_stl(stl, "4412", 1.0, 5.0, 161, 1, 10.0)
    cell = 0.16
    return msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=(-10.24, 30.72, -12.8, 12.8, -12.8, 12.8), patches=msh.BoxPatches(**PATCHES)
        ),
        surfaces=(msh.STLSurface(stl, patch="wing"),),
        max_cell_size=16 * cell,
        cell_size_anchor=cell,
        refinements=(
            msh.BoxRefinement("nearBody", (-1.6, 2.72, -1.28, 1.24, -3.6, 3.6), cell),
            msh.BoxRefinement("nearWake", (-2.0, 6.0, -1.84, 1.84, -4.0, 4.0), 2 * cell),
            msh.BoxRefinement("wake", (-2.4, 14.0, -3.2, 3.2, -4.8, 4.8), 4 * cell),
        ),
        patch_refinements=(msh.PatchRefinement("wing", cell),),
    )


def _cylinder_slab() -> msh.ExtrudedCartesianMesher:
    """Case 01's reference slab at cell size 0.16: the cylinder crosses the domain."""
    cell, slab = 0.16, 0.5
    request = 4.0 * cell / 3.0
    return msh.ExtrudedCartesianMesher(
        source=msh.CartesianMesher(
            domain=msh.BoxDomain(
                bounds=(-16.0, 30.0, -16.0, 16.0, -slab, slab), patches=msh.BoxPatches(**PATCHES)
            ),
            surfaces=(msh.STLSurface(CYLINDER, patch="cylinder"),),
            max_cell_size=16 * cell,
            refinements=(
                msh.BoxRefinement("coupledDomain", (-2.4, 4.4, -2.4, 2.4, -slab, slab), request),
                msh.BoxRefinement("wake", (-3.0, 30.0, -3.0, 3.0, -slab, slab), 2 * request),
            ),
            patch_refinements=(msh.PatchRefinement("cylinder", request),),
            surface_may_cross_domain_boundary=True,
        ),
        domain=msh.BoxDomain(
            bounds=(-16.0, 30.0, -16.0, 16.0, -0.5, 0.5), patches=msh.BoxPatches(**PATCHES)
        ),
        levels=(-0.5, 0.5),
    )


CASES = {
    "cylinder_box": lambda _directory: _cylinder_box(),
    "cylinder_automatic": lambda _directory: _cylinder_automatic(),
    "wing": _wing,
    "cylinder_slab": lambda _directory: _cylinder_slab(),
}


def _digest(value: Any, digest: Any) -> None:
    """Feed a mesh value into ``digest`` with its type, shape and exact bits."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(f"array:{array.dtype.str}:{array.shape}".encode())
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(f"dict:{len(value)}".encode())
        for key in sorted(value, key=str):
            digest.update(str(key).encode())
            _digest(value[key], digest)
    elif isinstance(value, list | tuple | set | frozenset):
        items = sorted(value, key=repr) if isinstance(value, set | frozenset) else value
        digest.update(f"{type(value).__name__}:{len(items)}".encode())
        for item in items:
            _digest(item, digest)
    elif isinstance(value, float | np.floating):
        digest.update(b"float:" + np.float64(value).tobytes())
    else:
        digest.update(f"{type(value).__name__}:{value!r}".encode())


def mesh_hashes(mesh: dict[str, Any]) -> dict[str, str]:
    """Return one SHA-256 per top-level mesh entry."""
    result = {}
    for key in sorted(mesh):
        digest = hashlib.sha256()
        _digest(mesh[key], digest)
        result[key] = digest.hexdigest()
    return result


def run(
    stage: str, names: list[str], logs: Path | None, output: Path | None = None
) -> dict[str, Any]:
    """Build every named case to ``stage`` and hash the result, saved after each case.

    A build that raises is an outcome too: its error is hashed, with the
    generated mesh that ``on_generated`` received before the final checks.
    """
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as directory:
        for name in names:
            mesher = CASES[name](Path(directory))
            generated: list[dict[str, Any]] = []
            log = Path(directory) / f"{name}.log" if logs is None else logs / f"{name}.log"
            start = time.process_time()
            try:
                with mesher_log_session(log):
                    if stage == "final":
                        mesh = mesher.build(on_generated=generated.append)
                    else:
                        source = (
                            mesher.source
                            if isinstance(mesher, msh.ExtrudedCartesianMesher)
                            else mesher
                        )
                        mesh = source.build(stop_after=stage)
                hashes = mesh_hashes(mesh)
                outcome = f"{mesh['n_cells']} cells"
            except Exception as error:  # noqa: BLE001 - a failure is a result to compare
                hashes = mesh_hashes(generated[-1]) if generated else {}
                message = f"{type(error).__name__}: {error}"
                hashes["error"] = hashlib.sha256(message.encode()).hexdigest()
                outcome = message
            seconds = time.process_time() - start
            results[name] = {"cpu_seconds": seconds, "outcome": outcome, "hashes": hashes}
            print(f"{name}: {outcome}, {seconds:.1f} s CPU", flush=True)
            if output is not None:
                output.write_text(json.dumps(results, indent=1, sort_keys=True))
    return results


def compare(results: dict[str, Any], reference: dict[str, Any]) -> int:
    """Print every field that differs from ``reference``; return the count."""
    differences = 0
    for name, result in results.items():
        expected = reference.get(name)
        if expected is None:
            print(f"{name}: not in the reference")
            differences += 1
            continue
        for key in sorted(set(result["hashes"]) | set(expected["hashes"])):
            if result["hashes"].get(key) != expected["hashes"].get(key):
                print(f"{name}: {key} differs")
                differences += 1
        speedup = expected["cpu_seconds"] / max(result["cpu_seconds"], 1.0e-9)
        print(
            f"{name}: {expected['cpu_seconds']:.1f} s -> {result['cpu_seconds']:.1f} s CPU ({speedup:.1f}x)"
        )
    print("identical" if differences == 0 else f"{differences} differing fields")
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage", default="templateGeneration", help="workflow checkpoint, or 'final'"
    )
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--logs", type=Path, help="directory for one mesher log per case")
    parser.add_argument("--output", type=Path, help="write the hashes to this JSON file")
    parser.add_argument("--compare", type=Path, help="compare against hashes written by --output")
    arguments = parser.parse_args()
    if arguments.output is None and arguments.compare is None:
        parser.error("give --output, --compare or both")
    results = run(arguments.stage, arguments.cases, arguments.logs, arguments.output)
    if arguments.compare is None:
        return 0
    return int(compare(results, json.loads(arguments.compare.read_text())) > 0)


if __name__ == "__main__":
    sys.exit(main())
