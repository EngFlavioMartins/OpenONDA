"""Fully meshed conformal FVM reference for the Re=150 cylinder benchmark.

This case contains no immersed-boundary model. A long watertight-cylinder STL
crosses the two spanwise slip planes, so the solved four-diameter segment has
a physical circular side-wall patch but no artificial cap forces.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np

import openonda.fvm as fvm
from openonda.runtime import RunConfig

SOURCE_DIR = Path(__file__).resolve().parent
CASE_SOURCE_DIR = SOURCE_DIR.parent
sys.path.insert(0, str(CASE_SOURCE_DIR))

import benchmark_config as cfg  # noqa: E402
from seed_perturbation import build_cylinder_initial_state  # noqa: E402

GRID = cfg.selected_grid()
DOMAIN_NAME = cfg.selected_domain_name()
FVM_DOMAIN = cfg.selected_reference_domain()
FVM_TIME_STEP_SIZE = cfg.fvm_time_step(GRID)
END_TIME = cfg.end_time()
SEED_AMPLITUDE = cfg.seed_amplitude()
DT_SCALE = cfg.positive_environment_float("OPENONDA_DT_SCALE", 1.0)
RUN_ID = cfg.reference_run_id(GRID, DOMAIN_NAME, DT_SCALE)
OUTER_CORRECTORS = int(os.environ.get("OPENONDA_OUTER_CORRECTORS", "2"))
NONORTHOGONAL_CORRECTORS = int(
    os.environ.get("OPENONDA_NONORTHOGONAL_CORRECTORS", "1")
)
if OUTER_CORRECTORS < 1 or NONORTHOGONAL_CORRECTORS < 0:
    raise ValueError("PIMPLE corrector counts must be non-negative, with at least one outer loop")

CASE_DIR = SOURCE_DIR


def _acquire_run_directory(run_dir: Path, *, restarting: bool):
    """Hold a single-writer lock and reject accidental history appends."""
    run_dir.mkdir(parents=True, exist_ok=True)
    handle = (run_dir / ".openonda_run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit(f"Another solver already owns output directory {run_dir}") from None
    markers = (
        run_dir / "solution" / "benchmark_metadata.json",
        run_dir / "solution" / "diagnostics.jsonl",
        run_dir / "samples" / "forces_history.csv",
    )
    existing = [path for path in markers if path.exists()]
    if existing and not restarting:
        handle.close()
        names = ", ".join(str(path.relative_to(run_dir)) for path in existing)
        raise SystemExit(
            f"Refusing to append to existing reference output {run_dir} ({names}). "
            "Run reference_flow/allclean.sh before starting a replacement calculation."
        )
    return handle


DEFAULT_FVM_CORES = 1 if GRID.name == "smoke" else 4
FVM_CORES = int(os.environ.get("OPENONDA_FVM_CORES", str(DEFAULT_FVM_CORES)))
if FVM_CORES < 1:
    raise ValueError("OPENONDA_FVM_CORES must be at least one")
_restart_value = os.environ.get("OPENONDA_RESTART_FROM", "").strip()
RESTART_FROM = Path(_restart_value).expanduser().resolve() if _restart_value else None
RESTART_ALLOW_CONFIG_CHANGE = os.environ.get(
    "OPENONDA_RESTART_ALLOW_CONFIG_CHANGE", "0"
) == "1"
RunConfig(cpu_cores=FVM_CORES, parallel_mode="mpi").ensure_runtime(__file__)
if FVM_CORES > 1:
    from mpi4py import MPI

    MPI_RANK = MPI.COMM_WORLD.Get_rank()
else:
    MPI_RANK = 0
IS_ROOT = MPI_RANK == 0
_RUN_LOCK = (
    _acquire_run_directory(CASE_DIR, restarting=RESTART_FROM is not None)
    if IS_ROOT
    else None
)

FVM_MESH = cfg.build_mesh(FVM_DOMAIN, GRID) if IS_ROOT else None

FORCE_INTERVAL_STEPS = cfg.physical_sample_steps(FVM_TIME_STEP_SIZE)
LINE_INTERVAL_STEPS = cfg.physical_sample_steps(FVM_TIME_STEP_SIZE, 0.2)
SLICE_INTERVAL_STEPS = cfg.physical_sample_steps(FVM_TIME_STEP_SIZE, 0.5)
FIELD_BACKUP_STEPS = cfg.physical_sample_steps(
    FVM_TIME_STEP_SIZE, cfg.field_output_interval()
)
FORCE_SCHEDULE = fvm.SamplingSchedule(every_n_steps=FORCE_INTERVAL_STEPS)
LINE_SCHEDULE = fvm.SamplingSchedule(every_n_steps=LINE_INTERVAL_STEPS)
SLICE_SCHEDULE = fvm.SamplingSchedule(every_n_steps=SLICE_INTERVAL_STEPS)

# Sampler spacing follows the wake resolution but never exceeds D/8. The
# sampler-first design keeps a 60-unit run compact while retaining enough
# points for phase-conditioned wake profiles.
SAMPLE_SPACING = min(0.125, GRID.near_wake)
WAKE_X_BOUNDS = (-2.0, 12.0)
WAKE_Y_BOUNDS = (-3.0, 3.0)


def _line_x(x: float, name: str) -> fvm.LineSampler:
    return fvm.LineSampler(
        start=[x, WAKE_Y_BOUNDS[0], 0.0],
        end=[x, WAKE_Y_BOUNDS[1], 0.0],
        spacing=SAMPLE_SPACING,
        file_name=name,
        schedule=LINE_SCHEDULE,
    )


SAMPLERS = (
    fvm.ForceSampler(
        patch_names=["cylinder"],
        reference_velocity=cfg.FREESTREAM_SPEED,
        reference_area=cfg.REFERENCE_AREA,
        reference_length=cfg.REFERENCE_LENGTH,
        moment_centre=[0.0, 0.0, 0.0],
        file_name="forces_history",
        schedule=FORCE_SCHEDULE,
    ),
    fvm.LineSampler(
        start=[1.5, 0.0, 0.0],
        end=[1.5, 0.0, 0.0],
        n_points=1,
        file_name="midspan_probe",
        schedule=FORCE_SCHEDULE,
    ),
    fvm.LineSampler(
        start=[WAKE_X_BOUNDS[0], 0.0, 0.0],
        end=[WAKE_X_BOUNDS[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="centreline",
        schedule=LINE_SCHEDULE,
    ),
    _line_x(1.0, "transverse_x1"),
    _line_x(2.0, "transverse_x2"),
    _line_x(4.0, "transverse_x4"),
    fvm.LineSampler(
        start=[1.5, 0.0, -1.75],
        end=[1.5, 0.0, 1.75],
        spacing=SAMPLE_SPACING,
        file_name="spanwise_line",
        schedule=LINE_SCHEDULE,
    ),
    fvm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[*WAKE_X_BOUNDS, *WAKE_Y_BOUNDS],
        spacing=SAMPLE_SPACING,
        file_name="slice_z0",
        schedule=SLICE_SCHEDULE,
        body_bounds=cfg.CYLINDER_STL_BOUNDS,
    ),
)

FVM_SETUP = fvm.FVMSetup(
    case_name=f"cylinder_reference_{RUN_ID}",
    cores=FVM_CORES,
    mesh=fvm.MeshQualityConfig(
        max_non_orthogonality_deg=60.0,
        max_skewness=0.5,
        max_aspect_ratio=80.0,
    ),
    execution=fvm.ComputeConfig(operator_backend="numba"),
    output=fvm.OutputConfig(
        format="vtk_xml",
        data_location="cell",
        encoding="appended",
        compression="lz4",
        precision="f32",
        asynchronous=False,
        ghost_layers=1,
    ),
    time=fvm.TimeConfig(
        time_step_size=FVM_TIME_STEP_SIZE,
        start_time=0.0,
        end_time=END_TIME,
        output_interval_steps=FIELD_BACKUP_STEPS,
        output_interval_time=None,
        adjust_time_step=False,
    ),
    schemes=fvm.DiscretizationConfig(
        convection_scheme="limitedLinear",
        gradient_scheme="lsq",
        time_scheme="backward",
    ),
    linear=fvm.LinearSolverConfig(
        linear_solver="bicgstab",
        pressure_solver="amg",
        pressure_tolerance=1e-7,
        pressure_relative_tolerance=0.005,
        pressure_final_relative_tolerance=0.0,
        momentum_tolerance=1e-6,
        momentum_relative_tolerance=0.05,
        momentum_final_relative_tolerance=0.0,
        momentum_max_iterations=2000,
        ilu_drop_tolerance=1e-4,
        ilu_fill_factor=10.0,
        ilu_reuse_tolerance=0.05,
    ),
    pimple=fvm.PimpleControl(
        n_correctors=2,
        n_outer_correctors=OUTER_CORRECTORS,
        n_orthogonal_correctors=NONORTHOGONAL_CORRECTORS,
        velocity_relaxation=0.7,
        pressure_relaxation=0.3,
    ),
    samplers=SAMPLERS,
    transport=fvm.TransportConfig(
        density=cfg.DENSITY,
        kinematic_viscosity=cfg.KINEMATIC_VISCOSITY,
    ),
    turbulence=fvm.TurbulenceConfig.none(),
    boundaries=[
        fvm.BoundaryConfig.inlet("inlet", list(cfg.FREESTREAM_VELOCITY)),
        fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
        fvm.BoundaryConfig.slip("ymin"),
        fvm.BoundaryConfig.slip("ymax"),
        fvm.BoundaryConfig.slip("zmin"),
        fvm.BoundaryConfig.slip("zmax"),
        fvm.BoundaryConfig.wall("cylinder"),
    ],
    initial_velocity=list(cfg.INITIAL_VELOCITY),
    initial_kinematic_pressure=0.0,
)


def _available_memory_gib() -> float:
    fields = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as stream:
        for line in stream:
            key, value = line.split(":", 1)
            fields[key] = float(value.split()[0]) * 1024.0
    return fields["MemAvailable"] / 1024.0**3


def _resource_gate() -> None:
    minimum = cfg.positive_environment_float(
        "OPENONDA_MIN_AVAILABLE_GIB", cfg.minimum_available_memory_gib(GRID)
    )
    available = _available_memory_gib()
    if available < minimum:
        raise RuntimeError(
            f"Resource gate: {available:.2f} GiB RAM available, {minimum:.2f} GiB required "
            f"for grid {GRID.name}. Close other runs or explicitly lower "
            "OPENONDA_MIN_AVAILABLE_GIB after reviewing the risk."
        )


def _apply_initial_state(fvm_solver) -> None:
    n_cells = fvm_solver.mesh_data["n_cells"]
    centres = np.asarray(fvm_solver.geo_data["cell_centre"][:n_cells], dtype=np.float64)
    velocity, kinematic_pressure = build_cylinder_initial_state(
        centres,
        freestream_velocity=cfg.FREESTREAM_VELOCITY,
        diameter=cfg.DIAMETER,
        seed_amplitude=SEED_AMPLITUDE,
    )
    fvm_solver.set_initial_state(velocity, kinematic_pressure)
    if IS_ROOT:
        print(
            f"  divergence-free cylinder start; seed eps={SEED_AMPLITUDE:.3e}; "
            f"max|u|/Uinf={np.linalg.norm(velocity, axis=1).max() / cfg.FREESTREAM_SPEED:.3f}"
        )


def _write_metadata(mesh_data: dict) -> None:
    solution = CASE_DIR / "solution"
    solution.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "openonda-cylinder-reference/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "case_source": str(SOURCE_DIR),
        "run_directory": str(CASE_DIR),
        "physics": {
            "reynolds_number": cfg.REYNOLDS,
            "diameter": cfg.DIAMETER,
            "cylinder_length": cfg.CYLINDER_LENGTH,
            "freestream_velocity": list(cfg.FREESTREAM_VELOCITY),
            "density": cfg.DENSITY,
            "kinematic_viscosity": cfg.KINEMATIC_VISCOSITY,
            "end_time": END_TIME,
            "seed_amplitude": SEED_AMPLITUDE,
        },
        "mesh": {
            "grid": GRID.name,
            "domain": DOMAIN_NAME,
            "requested_domain": list(FVM_DOMAIN),
            "effective_domain": list(FVM_MESH.effective_domain),
            "surface_file": str(cfg.CYLINDER_STL),
            "surface_sha256": FVM_MESH.surface.sha256,
            "surface_triangles": len(FVM_MESH.surface.triangles),
            "surface_cell_size": GRID.surface,
            "shear_layer_cell_size": GRID.shear_layer,
            "near_wake_cell_size": GRID.near_wake,
            "downstream_wake_cell_size": GRID.downstream_wake,
            "background_cell_size": GRID.background,
            "first_cell_height": GRID.first_cell_height,
            "wall_layers": GRID.wall_layers,
            "layer_growth_ratio": GRID.layer_growth,
            "transition_layers": GRID.transition_layers,
            "spanwise_cell_size": cfg.SPANWISE_CELL_SIZE,
            "spanwise_cells": int(round(cfg.CYLINDER_LENGTH / cfg.SPANWISE_CELL_SIZE)),
            "cell_order": "morton",
            "cell_count": int(mesh_data["n_cells"]),
            "target_cell_count": GRID.target_cells,
        },
        "execution": {
            "mpi_ranks": FVM_CORES,
            "operator_backend": FVM_SETUP.execution.operator_backend,
            "linear_backend": "petsc-partitioned" if FVM_CORES > 1 else "serial",
            "outer_correctors": OUTER_CORRECTORS,
            "nonorthogonal_correctors": NONORTHOGONAL_CORRECTORS,
        },
        "time": {
            "fvm_time_step": FVM_TIME_STEP_SIZE,
            "dt_scale": DT_SCALE,
            "force_interval": FORCE_INTERVAL_STEPS * FVM_TIME_STEP_SIZE,
            "line_interval": LINE_INTERVAL_STEPS * FVM_TIME_STEP_SIZE,
            "slice_interval": SLICE_INTERVAL_STEPS * FVM_TIME_STEP_SIZE,
            "field_output_interval": FIELD_BACKUP_STEPS * FVM_TIME_STEP_SIZE,
        },
    }
    (solution / "benchmark_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")


def _mesh_cache_path() -> Path:
    return CASE_DIR / "solution" / "reference_mesh.pkl"


def _load_or_build_mesh() -> dict:
    """Build once, then reuse the exact global mesh for checkpoint restarts."""
    if FVM_MESH is None:
        raise RuntimeError("Root rank is missing the global mesh generator")
    cache = _mesh_cache_path()
    if RESTART_FROM is not None and cache.is_file():
        with cache.open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("grid") != GRID.name or payload.get("domain") != DOMAIN_NAME:
            raise ValueError(f"Mesh cache {cache} does not match this grid/domain")
        print(f"  loaded cached mesh: {cache}")
        return payload["mesh"]

    mesh_data = FVM_MESH.build()
    temporary = cache.with_suffix(".pkl.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(
            {"schema": 1, "grid": GRID.name, "domain": DOMAIN_NAME, "mesh": mesh_data},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary.replace(cache)
    return mesh_data


def _checkpoint_path() -> Path:
    """Return the canonical restart artifact for this execution mode."""
    stem = CASE_DIR / "solution" / "checkpoint"
    return stem if FVM_CORES > 1 else stem.with_suffix(".npz")


def _save_checkpoint(solver) -> Path:
    """Atomically replace the latest restart while retaining field backups."""
    destination = _checkpoint_path()
    solver.save_state(destination)
    if FVM_CORES > 1 and IS_ROOT:
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        retained = set(manifest["files"])
        for old_rank_file in destination.glob("rank-*.npz"):
            if old_rank_file.name not in retained:
                old_rank_file.unlink()
    return destination


def main() -> None:
    _resource_gate()
    if IS_ROOT:
        print("\n===== BODY-FITTED FVM REFERENCE =====")
        print(
            f"  run={RUN_ID}, Re={cfg.REYNOLDS:g}, AR={cfg.CYLINDER_LENGTH / cfg.DIAMETER:g}, "
            f"grid={GRID.name}, dt={FVM_TIME_STEP_SIZE:g}, end={END_TIME:g}, "
            f"MPI ranks={FVM_CORES}"
        )
        print(f"  output={CASE_DIR}")
        mesh_data = _load_or_build_mesh()
        n_cells = int(mesh_data["n_cells"])
        print(f"  mesh cells={n_cells:,} (target cap {GRID.target_cells:,})")
        if n_cells > GRID.target_cells:
            raise MemoryError(
                f"Grid {GRID.name} built {n_cells:,} cells, above its {GRID.target_cells:,} cap"
            )
        if n_cells > 1_000_000:
            raise MemoryError(
                "Cylinder mesh exceeds the FVM solver's verified one-million-cell range"
            )
        if RESTART_FROM is None:
            _write_metadata(mesh_data)
    else:
        mesh_data = None
    solver = fvm.create_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=mesh_data)
    del mesh_data
    solver.write_run_manifest()
    if RESTART_FROM is None:
        _apply_initial_state(solver)
        solver.write_vtk()
    else:
        solver.load_state(
            RESTART_FROM, allow_config_change=RESTART_ALLOW_CONFIG_CHANGE
        )
        if IS_ROOT:
            print(f"  restarted from {RESTART_FROM} at step={solver.step}, t={solver.time:g}")

    configured_steps = int(round((END_TIME - FVM_SETUP.time.start_time) / FVM_TIME_STEP_SIZE))
    if not np.isclose(
        configured_steps * FVM_TIME_STEP_SIZE,
        END_TIME - FVM_SETUP.time.start_time,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Reference end time must be an integer number of FVM steps")
    remaining_steps = configured_steps - solver.step
    if remaining_steps < 0:
        raise ValueError(
            f"Checkpoint step {solver.step} is beyond configured end step {configured_steps}"
        )
    max_steps = int(os.environ.get("OPENONDA_MAX_STEPS", "0"))
    run_steps = min(remaining_steps, max_steps) if max_steps > 0 else remaining_steps
    maximum_cfl = cfg.positive_environment_float("OPENONDA_MAX_ALLOWED_CFL", 1.5)
    completed_steps = 0
    try:
        for local_step in range(1, run_steps + 1):
            solver.advance()
            completed_steps = local_step
            diagnostics = solver.last_diagnostics
            if diagnostics is not None and diagnostics.max_courant_number > maximum_cfl:
                raise RuntimeError(
                    f"CFL gate: step {solver.step} reached {diagnostics.max_courant_number:.3f}, "
                    f"above OPENONDA_MAX_ALLOWED_CFL={maximum_cfl:.3f}"
                )
            if solver.step % FIELD_BACKUP_STEPS == 0:
                _save_checkpoint(solver)
        if run_steps < remaining_steps and IS_ROOT:
            print(f"  bounded stop after {run_steps} FVM steps")
        if solver.step % FIELD_BACKUP_STEPS != 0:
            # Always leave an inspectable terminal field, including smoke and
            # developer-bounded runs shorter than the regular backup cadence.
            solver.write_vtk()
            _save_checkpoint(solver)
        stop_time = solver.time
        stop_step = solver.step
    finally:
        solver.close()
    if IS_ROOT:
        print(
            f"Reference stopped at t={stop_time:.12g}, step={stop_step} "
            f"({completed_steps} steps this invocation)"
        )


if __name__ == "__main__":
    main()
