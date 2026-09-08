#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the durable Cartesian-mesher acceptance matrix.

The runner intentionally owns no meshing algorithm.  It constructs public API
inputs, records the first failing stage and writes compact JSON evidence so a
repair loop can be rerun without relying on files in ``/private/tmp``.

Examples::

    python tests/mesh_parity/run_acceptance.py --suite quick \
        --output artifacts/mesher-acceptance
    python tests/mesh_parity/run_acceptance.py --suite release \
        --output artifacts/mesher-acceptance
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

import openonda.fvm.mesher as msh
from source.solvers.fvm.io.mesh_storage import save_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.validation import (
    validate_cell_area_closure,
    validate_geometry,
    validate_single_fluid_component,
    validate_topology,
)


def _load_fixture_builder():
    """Load the repository fixture module without depending on ``tests`` imports."""
    path = ROOT / "tests" / "fvm" / "cartesian_acceptance_fixtures.py"
    spec = importlib.util.spec_from_file_location("_openonda_acceptance_fixtures", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load acceptance fixtures from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.make_acceptance_fixtures


make_acceptance_fixtures = _load_fixture_builder()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *arguments), cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _diff_hash() -> str:
    """Hash the mesher-scoped dirty identity without invoking LFS filters."""
    digest = hashlib.sha256()
    try:
        status = subprocess.check_output(
            ("git", "status", "--porcelain=v1", "--untracked-files=all", "-z"),
            cwd=ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    for record in status.split(b"\0"):
        if len(record) < 4:
            continue
        raw_path = record[3:]
        path_text = os.fsdecode(raw_path)
        in_scope = path_text in {"CFMESH_MESHER_COMPLETION_PLAN.md", "pyproject.toml"} or any(
            path_text.startswith(prefix)
            for prefix in (
                "source/solvers/fvm/",
                "openonda/fvm/",
                "tests/fvm/",
                "tests/mesh_parity/",
                "tutorials/fvm/cartesian_mesher/",
            )
        )
        if not in_scope:
            continue
        digest.update(record[:3])
        digest.update(raw_path)
        path = ROOT / os.fsdecode(raw_path)
        if path.is_file():
            digest.update(raw_path)
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _domain() -> msh.BoxDomain:
    return msh.BoxDomain(
        bounds=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5),
        patches=msh.BoxPatches(
            xmin="inlet",
            xmax="outlet",
            ymin="farfield_ymin",
            ymax="farfield_ymax",
            zmin="farfield_zmin",
            zmax="farfield_zmax",
        ),
    )


def _case_controls(fixture_name: str, mode: str, surfaces: tuple[Any, ...]) -> dict[str, Any]:
    box = msh.BoxRefinement(
        "acceptance_box",
        (-1.0, 1.0, -0.75, 0.75, -0.75, 0.75),
        0.25,
    )
    if fixture_name == "two_disjoint_bodies":
        patch_requests = tuple(msh.PatchRefinement(surface.patch, 0.25) for surface in surfaces)
    else:
        patch_requests = (msh.PatchRefinement(surfaces[0].patch, 0.25),)
    return {
        "refinements": (box,) if mode in {"box", "box_plus_patch"} else (),
        "patch_refinements": patch_requests if mode in {"patch", "box_plus_patch"} else (),
    }


def _build_case(
    output: Path,
    fixture_name: str,
    mode: str,
    fixture_paths: tuple[Path, ...],
) -> dict[str, Any]:
    started = time.perf_counter()
    patches = ("body_a", "body_b") if fixture_name == "two_disjoint_bodies" else (fixture_name,)
    surfaces = tuple(
        msh.STLSurface(path, patch=patch)
        for path, patch in zip(fixture_paths, patches, strict=True)
    )
    controls = _case_controls(fixture_name, mode, surfaces)
    config_snapshot = {
        "fixture": fixture_name,
        "mode": mode,
        "domain": _domain().bounds,
        "max_cell_size": 0.5,
        "boundary_cell_size": 0.25,
        "min_cell_size": 0.125,
        "surfaces": [{"patch": surface.patch, "sha256": surface.sha256} for surface in surfaces],
        "refinements": [
            {"name": item.name, "bounds": item.bounds, "cell_size": item.cell_size}
            for item in controls["refinements"]
        ],
        "patch_refinements": [
            {"patch": item.patch, "cell_size": item.cell_size}
            for item in controls["patch_refinements"]
        ],
    }
    result: dict[str, Any] = {
        "name": f"{fixture_name}:{mode}",
        "status": "fail",
        "first_failing_stage": "input",
        "configuration_sha256": hashlib.sha256(
            json.dumps(config_snapshot, sort_keys=True).encode()
        ).hexdigest(),
        "configuration": config_snapshot,
        "input_sha256": [_sha256(path) for path in fixture_paths],
        "image_review": "not_inspected",
    }
    try:
        result["first_failing_stage"] = "build"
        mesher = msh.CartesianMesher(
            domain=_domain(),
            surfaces=surfaces,
            max_cell_size=0.5,
            boundary_cell_size=0.25,
            min_cell_size=0.125,
            **controls,
        )
        mesh = mesher.build()
        result["first_failing_stage"] = "topology"
        topology = validate_topology(mesh)
        geometry = compute_mesh_geometry(mesh, compute_lsq=False)
        result["first_failing_stage"] = "geometry"
        quality = dict(validate_geometry(mesh, geometry))
        quality.update(validate_cell_area_closure(mesh, geometry))
        quality.update(validate_single_fluid_component(mesh))
        evidence = output / "meshes" / fixture_name / mode
        save_native_mesh(mesh, evidence / "mesh.npz")
        result.update(
            {
                "status": "pass",
                "first_failing_stage": None,
                "n_cells": int(mesh["n_cells"]),
                "n_faces": int(mesh["n_faces"]),
                "n_points": int(mesh["n_points"]),
                "topology": topology,
                "quality": quality,
                "report": mesher.report.as_dict() if mesher.report is not None else None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - the manifest must record diagnostics.
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
        1 if sys.platform == "darwin" else 1024
    )
    return result


def run(suite: str, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    previous_manifest: dict[str, Any] = {}
    previous_path = output / "manifest.json"
    if previous_path.exists():
        try:
            previous_manifest = json.loads(previous_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_manifest = {}
    fixture_dir = output / "inputs"
    fixtures = make_acceptance_fixtures(fixture_dir)
    if suite == "quick":
        matrix = (("ellipsoid", ("default", "box_plus_patch")), ("rotated_box", ("patch",)))
    else:
        matrix = tuple(
            (name, ("default", "box", "patch", "box_plus_patch"))
            for name in (
                "rotated_box",
                "ellipsoid",
                "torus",
                "finite_naca_wing",
                "two_disjoint_bodies",
            )
        )
    results = [
        _build_case(output, name, mode, fixtures[name].paths)
        for name, modes in matrix
        for mode in modes
    ]
    # Preserve an explicit visual-review acknowledgement only when the exact
    # code/input/configuration identity is being rerun and the reviewed image
    # files still exist.  Any source or fixture change naturally resets the
    # mandatory release gate to ``not_inspected``.
    previous_results = {
        str(item.get("name")): item for item in previous_manifest.get("results", [])
    }
    same_identity = (
        previous_manifest.get("revision") == _git_value("rev-parse", "HEAD")
        and previous_manifest.get("dirty_diff_sha256") == _diff_hash()
    )
    if same_identity:
        for result in results:
            previous = previous_results.get(str(result["name"]))
            image_paths = tuple(previous.get("image_paths", ())) if previous else ()
            if (
                previous is not None
                and previous.get("image_review") == "inspected"
                and previous.get("configuration_sha256") == result.get("configuration_sha256")
                and previous.get("input_sha256") == result.get("input_sha256")
                and image_paths
                and all(Path(path).exists() for path in image_paths)
            ):
                result["image_review"] = "inspected"
                result["image_paths"] = list(image_paths)
    manifest = {
        "schema": "openonda.cartesian-mesher.acceptance.v1",
        "suite": suite,
        "revision": _git_value("rev-parse", "HEAD"),
        "dirty_diff_sha256": _diff_hash(),
        "python": sys.version,
        "platform": sys.platform,
        "required_cases": len(results),
        "image_review_policy": "release requires inspected images; this runner records status",
        "results": results,
        "generated_at_epoch": time.time(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    failures = [
        item
        for item in results
        if item.get("status") != "pass"
        or (suite == "release" and item.get("image_review") != "inspected")
    ]
    print(json.dumps({"suite": suite, "cases": len(results), "failures": len(failures)}))
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("quick", "release"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    return run(arguments.suite, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
