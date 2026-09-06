#!/usr/bin/env python3
"""Run the pinned native cfMesh cylinder oracle in an isolated evidence tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import shutil
import subprocess
import time

from canonical_surface import DOMAIN, prepare_canonical_surfaces

ROOT = Path(__file__).resolve().parents[4]
SOURCE_STL = ROOT / "tutorials/coupled_fvm_vpm/cylinder_shedding_flow/assets/cylinder_long.stl"
EXECUTABLE = Path(
    "/Users/flaviomartins/OpenFOAM/flaviomartins-v2412/platforms/"
    "darwin64ClangDPInt32Opt/bin/cartesianMesh"
)
LAUNCHER = Path("/Applications/OpenFOAM-v2412.app/Contents/Resources/etc/openfoam")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _foam_header(name: str, location: str) -> str:
    return (
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        "    class       dictionary;\n"
        f'    location    "{location}";\n'
        f"    object      {name};\n"
        "}\n\n"
    )


def _mesh_dict() -> str:
    return (
        _foam_header("meshDict", "system")
        + """surfaceFile "constant/triSurface/native_geometry.stl";
maxCellSize 0.2;
boundaryCellSize 0.2;

localRefinement
{
    cylinder
    {
        cellSize 0.025;
    }
}

objectRefinements
{
    near_body
    {
        type box;
        cellSize 0.05000000000005;
        centre (2 0 0);
        lengthX 8;
        lengthY 4;
        lengthZ 1.2;
    }
    wake
    {
        type box;
        cellSize 0.1000000000001;
        centre (4 0 0);
        lengthX 16;
        lengthY 8;
        lengthZ 1.2;
    }
}

renameBoundary
{
    newPatchNames
    {
        inlet { newName inlet; type patch; }
        outlet { newName outlet; type patch; }
        ymin { newName ymin; type patch; }
        ymax { newName ymax; type patch; }
        zmin { newName zmin; type patch; }
        zmax { newName zmax; type patch; }
        cylinder { newName cylinder; type wall; }
    }
}
"""
    )


def _control_dict() -> str:
    return (
        _foam_header("controlDict", "system")
        + """application cartesianMesh;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
writeFormat ascii;
writePrecision 15;
timeFormat general;
timePrecision 15;
runTimeModifiable false;
"""
    )


def _run_one(case: Path, *, repeat: int) -> dict[str, object]:
    tri_surface = case / "constant" / "triSurface"
    tri_surface.mkdir(parents=True, exist_ok=True)
    canonical = prepare_canonical_surfaces(SOURCE_STL, tri_surface)
    system = case / "system"
    system.mkdir(parents=True, exist_ok=True)
    (system / "meshDict").write_text(_mesh_dict(), encoding="ascii")
    (system / "controlDict").write_text(_control_dict(), encoding="ascii")
    log = case / "cartesianMesh.log"
    environment = os.environ.copy()
    environment["FOAM_CASE"] = str(case)
    environment["OMP_NUM_THREADS"] = "1"
    command = [str(LAUNCHER), str(EXECUTABLE)]
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=case,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=1_200,
        )
    elapsed = time.perf_counter() - started
    poly_mesh = case / "constant" / "polyMesh"
    checkmesh = shutil.which("checkMesh")
    sibling_checkmesh = EXECUTABLE.with_name("checkMesh")
    if checkmesh is None and sibling_checkmesh.is_file():
        checkmesh = str(sibling_checkmesh)
    if checkmesh is None:
        checkmesh_record: dict[str, object] = {
            "status": "unavailable",
            "reason": "No checkMesh executable was present in PATH or beside cartesianMesh",
        }
    else:
        checkmesh_log = case / "checkMesh.log"
        with checkmesh_log.open("w", encoding="utf-8") as stream:
            checkmesh_run = subprocess.run(
                [str(LAUNCHER), checkmesh, "-case", str(case), "-allTopology", "-allGeometry"],
                cwd=case,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        text = checkmesh_log.read_text(encoding="utf-8", errors="replace")
        from tests.mesh_parity.native_check import parse_checkmesh_output

        checkmesh_record = parse_checkmesh_output(text, exit_code=checkmesh_run.returncode)
    metadata: dict[str, object] = {
        "repeat": repeat,
        "command": command,
        "return_code": result.returncode,
        "elapsed_seconds": elapsed,
        "environment": {"FOAM_CASE": str(case), "OMP_NUM_THREADS": "1"},
        "source_stl_sha256": _sha256(SOURCE_STL),
        "canonical_inputs": {key: str(value) for key, value in canonical.items()},
        "executable": {"path": str(EXECUTABLE), "sha256": _sha256(EXECUTABLE)},
        "launcher": {"path": str(LAUNCHER), "sha256": _sha256(LAUNCHER)},
        "source_revision": "3ff8555514827646c34cacfe5f0f691e49cdbc96",
        "domain": DOMAIN,
        "controls": {"maxCellSize": 0.2, "boundaryCellSize": 0.2, "cylinder": 0.025},
        "log": str(log),
        "mesh_present": (poly_mesh / "points").is_file(),
        "checkMesh": checkmesh_record,
    }
    try:
        metadata["linked_libraries"] = subprocess.run(
            ["otool", "-L", str(EXECUTABLE)], capture_output=True, text=True, check=False
        ).stdout
    except OSError as error:
        metadata["linked_libraries"] = f"unavailable: {error}"
    (case / "native_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    if result.returncode != 0 or not metadata["mesh_present"]:
        raise RuntimeError(f"Native cartesianMesh failed; inspect {log}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "mesh_evidence" / "native-d40-20260906",
    )
    parser.add_argument("--repeats", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite native evidence: {arguments.output}")
    arguments.output.mkdir(parents=True, exist_ok=True)
    records = []
    for repeat in range(1, arguments.repeats + 1):
        records.append(_run_one(arguments.output / f"repeat-{repeat}", repeat=repeat))
    (arguments.output / "run_summary.json").write_text(
        json.dumps(records, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
