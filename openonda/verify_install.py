"""Prove that an OpenONDA installation works outside its source checkout."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
from importlib import resources
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numba
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
from source.solution_layout import vpm_backup_files


def _verify_package_location(require_site_packages: bool) -> Path:
    package_path = Path(openonda.__file__).resolve()
    if require_site_packages and "site-packages" not in package_path.parts:
        raise RuntimeError(f"OpenONDA was not imported from site-packages: {package_path}")
    return package_path


def _verify_gmsh() -> str:
    import gmsh

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


def _verify_cartesian_mesher() -> dict[str, str | int]:
    """Exercise the installed, compiled octree using packaged STL geometry."""
    from source.solvers.fvm.mesh.cartesian.cfmesh_octree import _balance_selection_kernel

    surface = msh.STLSurface(
        Path(resources.files("tutorials"))
        / "coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow/assets/cylinder_long.stl",
        patch="body",
    )
    mesh = msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=(-2.0, 2.0, -2.0, 2.0, -1.0, 1.0),
            patches=msh.BoxPatches("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
        ),
        surfaces=(surface,),
        max_cell_size=0.5,
        patch_refinements=(msh.PatchRefinement("body", 0.125),),
        surface_may_cross_domain_boundary=True,
    ).build(stop_after="templateGeneration")
    if not getattr(_balance_selection_kernel, "nopython_signatures", ()):
        raise RuntimeError(
            "Cartesian octree acceleration is inactive; check that NUMBA_DISABLE_JIT is unset"
        )
    if mesh["n_cells"] <= 0 or not np.all(np.isfinite(mesh["vertex_position"])):
        raise RuntimeError("Cartesian mesher installation smoke produced an invalid template")
    return {
        "balancing_backend": "numba",
        "n_cells": int(mesh["n_cells"]),
        "n_faces": int(mesh["n_faces"]),
    }


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
        linear=LinearSolverConfig(linear_solver="bicgstab", pressure_solver="amg"),
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


def _verify_native_vpm() -> dict[str, object]:
    """Advance particles, write native output, and resume from an installed wheel."""
    import h5py
    import pyvista as pv

    from openonda import vpm

    with (
        tempfile.TemporaryDirectory(prefix="openonda-installed-vpm-") as directory,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        case = vpm.VPMCase(
            directory=Path(directory) / "original",
            numerics=vpm.Numerics(
                compute_device="CPU",
                max_n_particles=8,
                max_evaluation_points=8,
                time_step_size=0.01,
                viscous=vpm.ViscousConfig.cs(kinematic_viscosity=0.01, particle_spacing=0.2),
                verbose=False,
            ),
            backup=vpm.Backup(interval_steps=1),
            run=vpm.RunPlan(steps=1, initial_samples=False),
        )
        solver = vpm.VPMSolver(case)
        try:
            solver.add_vortex_particles(
                position=np.array([[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]),
                velocity=np.zeros((2, 3)),
                vortex_strength=np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]),
                core_radius=np.full(2, 0.2),
                particle_volume=np.full(2, 0.2**3),
                kinematic_viscosity=np.full(2, 0.01),
            )
            solver.run()
            if solver.run_status != "completed":
                raise RuntimeError(f"VPM installation smoke stopped: {solver.run_status}")
        finally:
            solver.close()

        solution = Path(case.directory) / "solution"
        checkpoints = vpm_backup_files(solution)
        if len(checkpoints) != 1:
            raise RuntimeError(f"VPM installation smoke expected one checkpoint: {checkpoints}")
        checkpoint = checkpoints[0]
        with h5py.File(checkpoint, "r") as archive:
            positions = archive["particles/position"][:]
            step = int(archive["solver"].attrs["step"])
            time = float(archive["solver"].attrs["time"])
        if step != 1 or not np.isclose(time, 0.01) or not np.isfinite(positions).all():
            raise RuntimeError("VPM installation smoke saved an invalid state")
        reader = pv.get_reader(solution / "vpm.pvd")
        if reader.read()[0].n_points != 2:
            raise RuntimeError("VPM installation smoke has invalid ParaView output")

        resumed = vpm.VPMSolver(replace(case, directory=Path(directory) / "resumed"))
        try:
            resumed.load_backup(checkpoint)
            if resumed.step != step or resumed.time != time:
                raise RuntimeError("VPM restart did not preserve the accepted clock")
            np.testing.assert_array_equal(resumed.particle_position, positions)
            resumed.advance()
            if resumed.step != 2 or not np.isfinite(resumed.particle_position).all():
                raise RuntimeError("VPM restarted step produced an invalid state")
        finally:
            resumed.close()
    return {"n_particles": 2, "steps": 1, "restart_step": 2, "backend": "CPU"}


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
            Path(resources.files("openonda")) / "_resources/DejaVuSerif.ttf",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Installed tutorial resources are incomplete: {missing}")

        from openonda import plotting as theme

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


def _verify_direct_tutorial_scripts() -> int:
    """Exercise normal Python file commands in copied cases outside the checkout."""
    scripts = {
        "vpm/lamb_oseen_vortex": (
            "setup.py",
            "assets/rwm_ensemble.py",
            "assets/postprocess.py",
            "assets/plot_merging_snapshots.py",
        ),
        "vpm/vortex_ring": ("setup.py", "assets/postprocess.py"),
        "vpm/vortex_interactions": (
            "setup.py",
            "assets/plot_core_sections.py",
            "assets/plot_core_trajectories.py",
        ),
        "coupled_fvm_vpm/cube_flow": ("assets/validate_results.py",),
    }
    checked = 0
    with tempfile.TemporaryDirectory(prefix="openonda-direct-scripts-") as directory:
        workspace = Path(directory) / "case with spaces"
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["MPLCONFIGDIR"] = str(Path(directory) / "matplotlib")
        for tutorial, filenames in scripts.items():
            case = materialize_tutorial(tutorial, workspace)
            for filename in filenames:
                result = subprocess.run(
                    [sys.executable, "-I", str(case / filename), "--help"],
                    cwd=directory,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode or "usage:" not in result.stdout:
                    raise RuntimeError(
                        f"Direct tutorial command failed: {tutorial}/{filename} --help\n"
                        f"{result.stdout}\n{result.stderr}"
                    )
                checked += 1
    return checked


def main() -> int:
    """Run the command-line installation verification suite.

    The check validates that the imported package, distribution resources, direct
    tutorial entry points, Cartesian meshing, FVM iterative solves, and VPM
    stepping/output/restart are usable from an installed environment. It removes
    ``PYTHONPATH`` while exercising tutorial scripts so a source checkout cannot
    mask packaging errors.

    Returns
    -------
    int
        Process exit status.  ``0`` is returned after every requested check
        succeeds; failed checks raise their underlying exception before this
        function returns.

    Raises
    ------
    RuntimeError
        If a package, tutorial, meshing, runtime, or native-extension check fails.
    ImportError
        If an optional dependency required by a requested check is unavailable.

    Notes
    -----
    The ``--require-site-packages`` option rejects editable/source-checkout
    imports.  The ``--with-meshing`` option additionally initializes Gmsh and
    checks a one-unit cube.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-site-packages",
        action="store_true",
        help="fail if OpenONDA resolves to an editable/source checkout",
    )
    parser.add_argument(
        "--with-meshing", action="store_true", help="also exercise optional Gmsh geometry"
    )
    args = parser.parse_args()

    # Initialize Numba before FVM runtime setup, as in a mixed-solver process.
    numba.get_num_threads()
    report = {
        "openonda_version": openonda.__version__,
        "package_path": str(_verify_package_location(args.require_site_packages)),
        "distribution": _verify_distribution_resources(),
        "direct_tutorial_scripts": _verify_direct_tutorial_scripts(),
        "cartesian_mesher": _verify_cartesian_mesher(),
    }
    if args.with_meshing:
        report["gmsh_version"] = _verify_gmsh()
    taichi_version, taichi_arch = _verify_taichi()
    report.update(
        {
            "taichi_version": taichi_version,
            "taichi_arch": taichi_arch,
            "native_fvm": _verify_native_fvm(),
            "native_vpm": _verify_native_vpm(),
        }
    )
    numba.config.reload_config()
    report["numba"] = {"version": numba.__version__, "active_threads": numba.get_num_threads()}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
