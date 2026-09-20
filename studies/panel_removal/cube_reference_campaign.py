"""Execution and resource admission for the cube grid-convergence assessment.

The tutorial declares the physical setup. This study owns queued-job identity,
native mesh comparability and restart selection for the multi-grid assessment.
"""

from collections.abc import Callable
import json
import os
from pathlib import Path
import time

import numpy as np

import openonda.fvm as fvm
from source.solvers.fvm.io.logging import Logging


def _experiment_driver(command, directory: Path, repository: Path):
    """Recognize only our coupled drivers with this exact output destination."""
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return None
    if "--output" not in command:
        return None
    index = command.index("--output") + 1
    if index >= len(command) or Path(command[index]).resolve() != directory.resolve():
        return None
    for name in ("run_cube", "run_cylinder"):
        module = f"studies.panel_removal.{name}"
        script = str(repository / "studies/panel_removal" / f"{name}.py")
        if script in command or any(
            command[i : i + 2] == ["-m", module] for i in range(len(command) - 1)
        ):
            return name
    return None


def _recorded_experiment_is_active(record: Path, repository: Path, process_factory=None):
    try:
        launch = json.loads(record.read_text())
        pid = launch["pid"]
        driver = _experiment_driver(launch.get("command"), record.parent, repository)
        if not driver or not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    import psutil

    try:
        process = (process_factory or psutil.Process)(pid)
        if not process.is_running() or process.status() in (
            psutil.STATUS_ZOMBIE,
            psutil.STATUS_DEAD,
        ):
            return False
        # Command identity alone is insufficient when a PID is reused for a
        # later invocation of the same case. Records are written at launch.
        if abs(process.create_time() - record.stat().st_mtime) > 60:
            return False
        return _experiment_driver(process.cmdline(), record.parent, repository) == driver
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied as error:
        raise RuntimeError(
            f"Cannot verify coupled-process identity for PID {pid}; "
            "inspect resources or explicitly set CUBE_WAIT_FOR_COUPLED=0."
        ) from error


def wait_for_coupled_experiments(campaign: bool, dx: float, *, output_root: Path) -> None:
    """Queue larger meshes until this study's known coupled jobs release memory."""
    if not campaign or dx >= 0.1 or os.environ.get("CUBE_WAIT_FOR_COUPLED", "1") == "0":
        return
    repository = Path(__file__).resolve().parents[2]
    runs = repository / "studies/panel_removal/runs"
    previous = None
    logger = None
    last_notice = float("-inf")
    while True:
        active = tuple(
            str(record.parent)
            for record in sorted(runs.glob("*/process.json"))
            if _recorded_experiment_is_active(record, repository)
        )
        if not active:
            if previous:
                logger.section(
                    "CAMPAIGN ADMISSION", [("status", "coupled jobs finished")], flush=True
                )
                logger.close(status="admitted")
            return
        now = time.monotonic()
        if active != previous or now - last_notice >= 300:
            if logger is None:
                logger = Logging(
                    output_root, solution_dir=output_root / "logs", filename="admission.log"
                )
            logger.section(
                "CAMPAIGN ADMISSION",
                [
                    ("status", "queued before meshing"),
                    ("wall spacing", dx, "m"),
                    ("active experiments", ", ".join(active)),
                    ("check interval", 30, "s"),
                ],
                flush=True,
            )
            last_notice = now
        previous = active
        time.sleep(30)


def validate_campaign_mesh(solver, output_root: Path, name: str, dx: float) -> None:
    """Check the recorded global mesh before spending time on a spatial level."""
    failure = None
    if solver.parallel.is_root:
        try:
            with np.load(output_root / "solution" / name / "fvm" / "mesh.npz") as mesh:
                points = mesh["vertex_position"]
                bounds = np.column_stack((points.min(axis=0), points.max(axis=0))).ravel()
                generation = json.loads(str(mesh["metadata"]))["mesh_generation"]
                h = generation["resolved_surface_patch_sizes"]["cube"]
                if not np.isclose(h, dx, rtol=1e-10, atol=1e-12):
                    raise ValueError(f"realized cube spacing {h} differs from campaign h={dx}")
                if not np.allclose(
                    bounds, [-6.48, 12.96, -6.48, 6.48, -6.48, 6.48], rtol=0, atol=1e-10
                ):
                    raise ValueError(f"campaign domain changed: {bounds.tolist()}")
                solver.logger.section(
                    "CAMPAIGN MESH",
                    [
                        ("wall spacing", h, "m"),
                        ("cells", len(mesh["cell_sizes"])),
                        ("bounds", bounds.tolist(), "m"),
                    ],
                    flush=True,
                )
        except Exception as error:
            failure = str(error)
    failure = solver.parallel.bcast(failure, root=0)
    if failure:
        raise ValueError(f"Campaign mesh qualification failed: {failure}")


def campaign_mesh(
    output_root: Path, directory_name: str, mesh: Path | None, restart_from: Path | None
) -> Path | None:
    """Select the checkpoint mesh or identical spatial fine mesh for a dt control."""
    native_mesh = mesh
    if restart_from is not None:
        native_mesh = output_root / "solution" / directory_name / "fvm" / "mesh.npz"
    elif (
        native_mesh is None
        and directory_name == "time_h0045_dt_half"
        and output_root.name == "temporal"
    ):
        native_mesh = output_root.parent / "solution/grid_h0045/fvm/mesh.npz"
    if native_mesh is not None:
        native_mesh = Path(native_mesh).resolve()
        if not native_mesh.is_file():
            raise FileNotFoundError(
                f"Required native mesh is missing: {native_mesh}. "
                "Run the spatial fine stage before a fresh temporal control; "
                "a restart requires its own saved mesh."
            )
    return native_mesh


def run_campaign_level(
    create_solver: Callable[..., fvm.FVMSolver],
    *,
    case_dir: Path,
    directory_name: str,
    dx: float,
    campaign: bool,
    output_root: Path | None,
    cores: int,
    end_time: float,
    max_dt: float,
    courant: float,
    lean: bool,
    restart_from: Path | None,
    mesh: Path | None,
    output_interval: float | None = None,
    backup_interval: float | None = None,
) -> None:
    """Run one geometric or temporal level through the tutorial's physical factory.

    Paths identify one assessment output root; dx and max_dt are metres and
    seconds. This routine waits for matching coupled experiments, validates the
    saved global mesh, restores a supplied checkpoint and runs the requested
    horizon. It does not change numerical controls or remove existing outputs.
    ``output_interval`` and ``backup_interval`` are independent optional
    intervals in seconds, passed directly to the physical case configuration.
    """
    output_root = (case_dir if output_root is None else output_root).resolve()
    mesh = campaign_mesh(output_root, directory_name, mesh, restart_from)
    wait_for_coupled_experiments(campaign, dx, output_root=output_root)
    if lean:
        os.environ.setdefault("FVM_PROFILE", "0")
    with create_solver(
        directory_name,
        dx,
        campaign=campaign,
        output_root=output_root,
        cores=cores,
        end_time=end_time,
        max_dt=max_dt,
        courant=courant,
        lean=lean,
        output_interval=output_interval,
        backup_interval=backup_interval,
        restart_from=restart_from,
        mesh=mesh,
    ) as solver:
        validate_campaign_mesh(solver, output_root, directory_name, dx)
        if restart_from is not None:
            solver.load_state(restart_from)
        solver.run()
        fvm.update_grid_study(solver, dx, profiles=("centreline", "offaxis_y075"))
