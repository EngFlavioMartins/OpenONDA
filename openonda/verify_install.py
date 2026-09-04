"""Prove that an OpenONDA installation works outside its source checkout."""

from __future__ import annotations

import argparse
import contextlib
from importlib import resources
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile

import gmsh
import numpy as np
import taichi as ti

import openonda
import openonda.coupler
from openonda.fvm import (
    BoundaryConfig,
    ComputeConfig,
    DiscretizationConfig,
    FVMSetup,
    LinearSolverConfig,
    PimpleControl,
    RunSchedule,
    TimeConfig,
    TransportConfig,
    create_fvm_solver,
)
import openonda.fvm.mesher as msh
from openonda.tutorials import TUTORIALS, materialize_tutorial
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
        with tempfile.TemporaryDirectory(prefix="openonda-taichi-cache-") as cache:
            ti.init(
                arch=ti.cpu,
                offline_cache=False,
                offline_cache_file_path=cache,
            )
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
    mesh = msh.coupling_box_mesh(
        (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        0.5,
        hole_box=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
        wall_patch_name="body",
    )
    setup = FVMSetup(
        case_name="installedWheelSmoke",
        execution=ComputeConfig(operator_backend="numpy"),
        time=TimeConfig(
            time_step_size=0.01,
            end_time=0.01,
            output_schedule=RunSchedule(every_n_steps=100),
        ),
        schemes=DiscretizationConfig(convection_scheme="upwind"),
        linear=LinearSolverConfig(linear_solver="spsolve"),
        pimple=PimpleControl(n_correctors=1, n_outer_correctors=1),
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.01),
        boundaries=[
            BoundaryConfig.inlet("numericalBoundary", [1.0, 0.0, 0.0]),
            BoundaryConfig.wall("body"),
        ],
        initial_velocity=[1.0, 0.0, 0.0],
        initial_kinematic_pressure=0.0,
    )

    with (
        tempfile.TemporaryDirectory(prefix="openonda-installed-fvm-") as case_dir,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        solver = create_fvm_solver(setup, case_dir=case_dir, mesh=mesh)
        try:
            solver.auto_write = False
            solver.advance()
            velocity = np.asarray(solver.velocity[: mesh["n_cells"]], dtype=float)
            pressure = np.asarray(solver.kinematic_pressure[: mesh["n_cells"]], dtype=float)
            diagnostics = solver.last_diagnostics
        finally:
            solver.close()

    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(pressure)):
        raise RuntimeError("Native FVM installation smoke produced non-finite fields")
    if diagnostics is None or diagnostics.n_nonfinite_values:
        raise RuntimeError("Native FVM installation smoke did not produce healthy diagnostics")
    if not diagnostics.linear_solves or not all(
        result.converged for result in diagnostics.linear_solves
    ):
        raise RuntimeError("Native FVM installation smoke had an unconverged linear solve")
    return {
        "n_cells": int(mesh["n_cells"]),
        "max_courant_number": float(diagnostics.max_courant_number),
        "max_continuity_error": float(diagnostics.max_continuity_error),
        "n_linear_solves": len(diagnostics.linear_solves),
        "max_velocity_magnitude": float(np.max(np.linalg.norm(velocity, axis=1))),
        "max_absolute_kinematic_pressure": float(np.max(np.abs(pressure))),
    }


def _verify_distribution_resources() -> dict[str, object]:
    """Verify typing, tutorial, and plotting resources from the installation."""
    if not (resources.files("openonda") / "py.typed").is_file():
        raise RuntimeError("The installed distribution is missing openonda/py.typed")

    tutorial_root = resources.files("tutorials")
    if not isinstance(tutorial_root, Path):
        raise RuntimeError("OpenONDA tutorials require an unpacked installation")
    path_markers = ("/" + "Users/", "/" + "home/")
    for source in tutorial_root.rglob("*"):
        if source.suffix not in {".py", ".sh"}:
            continue
        text = source.read_text(encoding="utf-8")
        if any(marker in text for marker in path_markers):
            raise RuntimeError(f"Installed tutorial contains a machine-specific path: {source}")

    with tempfile.TemporaryDirectory(prefix="openonda-installed-resources-") as directory:
        workspace = Path(directory) / "workspace"
        cache = Path(directory) / "cache"
        cache.mkdir()
        previous_matplotlib_cache = os.environ.get("MPLCONFIGDIR")
        previous_xdg_cache = os.environ.get("XDG_CACHE_HOME")
        os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
        os.environ["XDG_CACHE_HOME"] = str(cache)

        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        case_path = materialize_tutorial("fvm/taylor_green", workspace)
        required = (
            case_path / "setup.py",
            case_path / "allrun.sh",
            case_path / "allplot.sh",
            workspace / "docs/themes/matplotlib_setup.py",
            workspace / "docs/themes/DejaVuSerif.ttf",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Installed tutorial resources are incomplete: {missing}")

        theme_path = workspace / "docs/themes/matplotlib_setup.py"
        spec = importlib.util.spec_from_file_location("_openonda_installed_theme", theme_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load the installed plotting theme: {theme_path}")
        theme = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(theme)
        theme.set_style()
        figure, axes = plt.subplots(figsize=theme.figure_size("single_short"))
        axes.plot([0.0, 1.0], [0.0, 1.0])
        axes.set_xlabel(r"$x$")
        figure_path = workspace / "plot-smoke.png"
        theme.save_fig(figure, figure_path, dpi=72)
        if not figure_path.is_file() or figure_path.stat().st_size == 0:
            raise RuntimeError("Matplotlib installation smoke did not create a figure")

        if previous_matplotlib_cache is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_matplotlib_cache
        if previous_xdg_cache is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = previous_xdg_cache

    return {
        "tutorial_count": len(TUTORIALS),
        "typed_package": True,
        "matplotlib_backend": str(matplotlib.get_backend()),
        "latex_rendering": bool(matplotlib.rcParams["text.usetex"]),
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
        "distribution": _verify_distribution_resources(),
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
