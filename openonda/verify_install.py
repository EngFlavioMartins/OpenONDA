"""Prove that an OpenONDA installation works outside its source checkout."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile

import gmsh
import numpy as np
import taichi as ti

import openonda
import openonda.coupler
from openonda.fvm import (
    BoundaryConfig,
    ExecutionConfig,
    FVMSetup,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    TimeConfig,
    TransportConfig,
    coupling_box_mesh,
    setup_fvm_solver,
)
import openonda.vpm


def _verify_package_location(require_site_packages: bool) -> Path:
    package_path = Path(openonda.__file__).resolve()
    if require_site_packages and "site-packages" not in package_path.parts:
        raise RuntimeError(f"OpenONDA was not imported from site-packages: {package_path}")
    return package_path


def _verify_gmsh() -> str:
    gmsh.initialize()
    try:
        gmsh.model.add("openonda_install_verification")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        gmsh.model.occ.synchronize()
        if len(gmsh.model.getEntities(3)) != 1:
            raise RuntimeError("Gmsh failed to construct the verification volume")
        return str(gmsh.__version__)
    finally:
        gmsh.finalize()


def _verify_taichi() -> tuple[str, str]:
    ti.reset()
    try:
        ti.init(arch=ti.cpu, offline_cache=False)
        architecture = str(ti.lang.impl.current_cfg().arch)
        version = ti.__version__
        version_text = (
            ".".join(str(value) for value in version)
            if isinstance(version, tuple)
            else str(version)
        )
        return version_text, architecture
    finally:
        ti.reset()


def _verify_native_fvm() -> dict[str, float | int]:
    mesh = coupling_box_mesh(
        (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        0.5,
        hole_box=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
        wall_patch_name="body",
    )
    setup = FVMSetup(
        case_name="installedWheelSmoke",
        execution=ExecutionConfig(operator_backend="numpy"),
        time=TimeConfig.transient(dt=0.01, duration=0.01, write_interval=100),
        schemes=SchemesConfig(convection_scheme="upwind"),
        linear=LinearSolverConfig(linear_solver="spsolve"),
        pimple=PimpleControl(n_correctors=1, n_outer_correctors=1),
        transport=TransportConfig(density=1.0, nu=0.01),
        boundaries=[
            BoundaryConfig.inlet("numericalBoundary", [1.0, 0.0, 0.0]),
            BoundaryConfig.wall("body"),
        ],
        initial_U=[1.0, 0.0, 0.0],
        initial_p=0.0,
    )

    with (
        tempfile.TemporaryDirectory(prefix="openonda-installed-fvm-") as case_dir,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        solver = setup_fvm_solver(setup, case_dir=case_dir, mesh=mesh)
        try:
            solver.auto_write = False
            solver.evolve(0.01)
            velocity = np.asarray(solver.U[: mesh["n_elements"]], dtype=float)
            pressure = np.asarray(solver.p[: mesh["n_elements"]], dtype=float)
            diagnostics = solver.last_diagnostics
        finally:
            solver.close()

    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(pressure)):
        raise RuntimeError("Native FVM installation smoke produced non-finite fields")
    if diagnostics is None or diagnostics.nonfinite_count:
        raise RuntimeError("Native FVM installation smoke did not produce healthy diagnostics")
    if not diagnostics.linear_solves or not all(
        result.converged for result in diagnostics.linear_solves
    ):
        raise RuntimeError("Native FVM installation smoke had an unconverged linear solve")
    return {
        "cells": int(mesh["n_elements"]),
        "cfl_max": float(diagnostics.cfl_max),
        "continuity_max": float(diagnostics.continuity_max),
        "linear_solves": len(diagnostics.linear_solves),
        "velocity_max": float(np.max(np.linalg.norm(velocity, axis=1))),
        "pressure_max_abs": float(np.max(np.abs(pressure))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-site-packages",
        action="store_true",
        help="fail if OpenONDA resolves to an editable/source checkout",
    )
    args = parser.parse_args()

    report = {
        "openonda_version": openonda.__version__,
        "package_path": str(_verify_package_location(args.require_site_packages)),
        "gmsh_version": _verify_gmsh(),
    }
    taichi_version, taichi_arch = _verify_taichi()
    report.update(
        {
            "taichi_version": taichi_version,
            "taichi_arch": taichi_arch,
            "native_fvm": _verify_native_fvm(),
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
