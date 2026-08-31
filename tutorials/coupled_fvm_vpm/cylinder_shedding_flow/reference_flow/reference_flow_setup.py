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
import time as wall_clock

import numpy as np

import openonda.fvm as fvm
from openonda.runtime import RunConfig

SOURCE_DIR = Path(__file__).resolve().parent
CASE_SOURCE_DIR = SOURCE_DIR.parent
sys.path.insert(0, str(CASE_SOURCE_DIR))

import benchmark_config as cfg  # noqa: E402
from seed_perturbation import (  # noqa: E402
    build_cylinder_initial_state,
    build_transferred_initial_state,
)
from source.solvers.fvm.fields.spanwise_projection import (  # noqa: E402
    SpanwiseInvariantProjector,
    build_spanwise_projection_layout,
)

GRID = cfg.selected_grid()
DOMAIN_NAME = cfg.selected_domain_name()
FVM_DOMAIN = cfg.selected_reference_domain()
FVM_TIME_STEP_SIZE = cfg.fvm_time_step(GRID)
END_TIME = cfg.end_time()
SEED_AMPLITUDE = cfg.seed_amplitude()
DT_SCALE = cfg.positive_environment_float("OPENONDA_DT_SCALE", 1.0)
RUN_ID = cfg.reference_run_id(GRID, DOMAIN_NAME, DT_SCALE)
OUTER_CORRECTORS = int(os.environ.get("OPENONDA_OUTER_CORRECTORS", "2"))
NONORTHOGONAL_CORRECTORS = int(os.environ.get("OPENONDA_NONORTHOGONAL_CORRECTORS", "1"))
LOG_INTERVAL_STEPS = int(os.environ.get("OPENONDA_LOG_INTERVAL_STEPS", "25"))
PRESSURE_TOLERANCE = cfg.positive_environment_float("OPENONDA_PRESSURE_TOLERANCE", 1.0e-7)
MOMENTUM_TOLERANCE = cfg.positive_environment_float("OPENONDA_MOMENTUM_TOLERANCE", 1.0e-6)
TIME_SCHEME = os.environ.get("OPENONDA_TIME_SCHEME", "backward").strip()
if TIME_SCHEME not in {"euler_implicit", "backward"}:
    raise ValueError("OPENONDA_TIME_SCHEME must be either 'euler_implicit' or 'backward'")
_ddt_corr_value = os.environ.get("OPENONDA_DDT_CORR", "1").strip()
if _ddt_corr_value not in {"0", "1"}:
    raise ValueError("OPENONDA_DDT_CORR must be either '0' or '1'")
DDT_CORR = _ddt_corr_value == "1"
_initial_field_value = os.environ.get("OPENONDA_INITIAL_FIELD", "").strip()
INITIAL_FIELD = Path(_initial_field_value).expanduser().resolve() if _initial_field_value else None
if INITIAL_FIELD is not None and not INITIAL_FIELD.is_file():
    raise FileNotFoundError(f"OPENONDA_INITIAL_FIELD does not exist: {INITIAL_FIELD}")
if OUTER_CORRECTORS < 1 or NONORTHOGONAL_CORRECTORS < 0:
    raise ValueError("PIMPLE corrector counts must be non-negative, with at least one outer loop")
if LOG_INTERVAL_STEPS < 1:
    raise ValueError("OPENONDA_LOG_INTERVAL_STEPS must be at least one")

_case_directory_value = os.environ.get("OPENONDA_REFERENCE_CASE_DIR", "").strip()
CASE_DIR = (
    Path(_case_directory_value).expanduser().resolve() if _case_directory_value else SOURCE_DIR
)
if CASE_DIR in {Path("/"), Path.home().resolve()}:
    raise ValueError("OPENONDA_REFERENCE_CASE_DIR must identify a dedicated case directory")


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


DEFAULT_FVM_CORES = 1 if GRID.name == "smoke" else 6
FVM_CORES = int(os.environ.get("OPENONDA_FVM_CORES", str(DEFAULT_FVM_CORES)))
if FVM_CORES < 1:
    raise ValueError("OPENONDA_FVM_CORES must be at least one")
_restart_value = os.environ.get("OPENONDA_RESTART_FROM", "").strip()
RESTART_FROM = Path(_restart_value).expanduser().resolve() if _restart_value else None
RESTART_ALLOW_CONFIG_CHANGE = os.environ.get("OPENONDA_RESTART_ALLOW_CONFIG_CHANGE", "0") == "1"
REUSE_MESH_CACHE = os.environ.get("OPENONDA_REUSE_MESH_CACHE", "0") == "1"
ENFORCE_SPANWISE_INVARIANCE = (
    os.environ.get("OPENONDA_ENFORCE_SPANWISE_INVARIANCE", "0").strip() == "1"
)
RunConfig(cpu_cores=FVM_CORES, parallel_mode="mpi").ensure_runtime(__file__)
if FVM_CORES > 1:
    from mpi4py import MPI

    MPI_COMM = MPI.COMM_WORLD
    MPI_RANK = MPI_COMM.Get_rank()
else:
    MPI_COMM = None
    MPI_RANK = 0
IS_ROOT = MPI_RANK == 0
_rank_cpu_value = os.environ.get("OPENONDA_RANK_CPUS", "").strip()
RANK_CPUS = tuple(int(value) for value in _rank_cpu_value.split(",") if value.strip())
if RANK_CPUS:
    if len(RANK_CPUS) != FVM_CORES or len(set(RANK_CPUS)) != FVM_CORES:
        raise ValueError("OPENONDA_RANK_CPUS must list one unique Linux CPU per MPI rank")
    os.sched_setaffinity(0, {RANK_CPUS[MPI_RANK]})
_RUN_LOCK = (
    _acquire_run_directory(CASE_DIR, restarting=RESTART_FROM is not None) if IS_ROOT else None
)

FVM_MESH = cfg.build_mesh(FVM_DOMAIN, GRID) if IS_ROOT else None

FORCE_INTERVAL_STEPS = cfg.physical_sample_steps(FVM_TIME_STEP_SIZE)
LINE_INTERVAL_STEPS = cfg.physical_sample_steps(FVM_TIME_STEP_SIZE, 0.2)
SLICE_INTERVAL_STEPS = cfg.physical_sample_steps(FVM_TIME_STEP_SIZE, 0.5)
FIELD_BACKUP_STEPS = cfg.physical_sample_steps(FVM_TIME_STEP_SIZE, cfg.field_output_interval())
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
        k=12,
        reconstruction="affine",
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
        k=12,
        reconstruction="affine",
        file_name="midspan_probe",
        schedule=FORCE_SCHEDULE,
    ),
    fvm.LineSampler(
        start=[WAKE_X_BOUNDS[0], 0.0, 0.0],
        end=[WAKE_X_BOUNDS[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        k=12,
        reconstruction="affine",
        file_name="centreline",
        schedule=LINE_SCHEDULE,
    ),
    _line_x(1.0, "transverse_x1"),
    _line_x(2.0, "transverse_x2"),
    _line_x(4.0, "transverse_x4"),
    fvm.LineSampler(
        start=[1.5, 0.0, cfg.CYLINDER_Z_BOUNDS[0] + 0.5 * cfg.SPANWISE_CELL_SIZE],
        end=[1.5, 0.0, cfg.CYLINDER_Z_BOUNDS[1] - 0.5 * cfg.SPANWISE_CELL_SIZE],
        # Probe once at every extruded slab centre. Oversampling this line at
        # the much smaller x-y wake spacing changes the 3-D reconstruction
        # stencil with z and can manufacture apparent spanwise modulation.
        spacing=cfg.SPANWISE_CELL_SIZE,
        k=12,
        reconstruction="affine",
        file_name="spanwise_line",
        schedule=LINE_SCHEDULE,
    ),
    fvm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[*WAKE_X_BOUNDS, *WAKE_Y_BOUNDS],
        spacing=SAMPLE_SPACING,
        k=12,
        reconstruction="affine",
        file_name="slice_z0",
        schedule=SLICE_SCHEDULE,
        body_bounds=cfg.CYLINDER_STL_BOUNDS,
        body_geometry="cylinder_z",
    ),
)

FVM_SETUP = fvm.FVMSetup(
    case_name=f"cylinder_reference_{RUN_ID}",
    cores=FVM_CORES,
    mesh=fvm.MeshQualityConfig(
        max_non_orthogonality_deg=60.0,
        max_skewness=0.5,
        # High spanwise/normal aspect ratio is intentional in the first
        # body-fitted wall layer; non-orthogonality and skewness stay tight.
        max_aspect_ratio=150.0,
    ),
    execution=fvm.ComputeConfig(operator_backend="numba"),
    logging=fvm.LoggingConfig(interval_steps=LOG_INTERVAL_STEPS),
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
        time_scheme=TIME_SCHEME,
    ),
    linear=fvm.LinearSolverConfig(
        linear_solver="bicgstab",
        pressure_solver="amg",
        pressure_tolerance=PRESSURE_TOLERANCE,
        pressure_relative_tolerance=0.005,
        pressure_final_relative_tolerance=0.0,
        momentum_tolerance=MOMENTUM_TOLERANCE,
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
        ddt_corr=DDT_CORR,
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


def _package_temperature_c() -> float | None:
    """Return the preferred Linux CPU temperature, broadcast to every rank."""
    temperature = None
    if IS_ROOT:
        temperatures: dict[str, list[float]] = {"TCPU": [], "x86_pkg_temp": []}
        for type_path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/type")):
            zone_type = type_path.read_text(encoding="utf-8").strip()
            if zone_type not in temperatures:
                continue
            value_path = type_path.with_name("temp")
            if value_path.is_file():
                temperatures[zone_type].append(
                    float(value_path.read_text(encoding="utf-8").strip()) / 1000.0
                )
        preferred = temperatures["TCPU"] or temperatures["x86_pkg_temp"]
        if preferred:
            temperature = max(preferred)
    if MPI_COMM is not None:
        temperature = MPI_COMM.bcast(temperature, root=0)
    return temperature


def _wait_for_package_temperature(target_c: float) -> float | None:
    """Idle all MPI ranks until the CPU package cools to ``target_c``."""
    while True:
        temperature = _package_temperature_c()
        if temperature is None or temperature <= target_c:
            return temperature
        if IS_ROOT:
            reading = "unavailable" if temperature is None else f"{temperature:.1f} C"
            print(f"  thermal pause: CPU package {reading}; resume at {target_c:.1f} C")
        wall_clock.sleep(5.0)


def _apply_initial_state(fvm_solver) -> None:
    n_cells = fvm_solver.mesh_data["n_cells"]
    centres = np.asarray(fvm_solver.geo_data["cell_centre"][:n_cells], dtype=np.float64)
    if INITIAL_FIELD is None:
        velocity, kinematic_pressure = build_cylinder_initial_state(
            centres,
            freestream_velocity=cfg.FREESTREAM_VELOCITY,
            diameter=cfg.DIAMETER,
            seed_amplitude=SEED_AMPLITUDE,
        )
    else:
        velocity, kinematic_pressure = build_transferred_initial_state(centres, INITIAL_FIELD)
    fvm_solver.set_initial_state(velocity, kinematic_pressure)
    if IS_ROOT:
        if INITIAL_FIELD is None:
            print(
                f"  divergence-free cylinder start; seed eps={SEED_AMPLITUDE:.3e}; "
                f"max|u|/Uinf={np.linalg.norm(velocity, axis=1).max() / cfg.FREESTREAM_SPEED:.3f}"
            )
        else:
            print(
                f"  spanwise-mean field transfer from {INITIAL_FIELD}; "
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
            "initial_field": None if INITIAL_FIELD is None else str(INITIAL_FIELD),
            "restart_from": None if RESTART_FROM is None else str(RESTART_FROM),
            "initial_state_method": (
                "checkpoint_restart"
                if RESTART_FROM is not None
                else (
                    "analytic_divergence_free_seed"
                    if INITIAL_FIELD is None
                    else "spanwise_mean_affine_field_transfer"
                )
            ),
        },
        "mesh": {
            "grid": GRID.name,
            "domain": DOMAIN_NAME,
            "geometry_treatment": "body-fitted_conformal_surface_mesh",
            "immersed_boundary": False,
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
            "spanwise_invariance": {
                "enforced": ENFORCE_SPANWISE_INVARIANCE,
                "method": (
                    "conservative_cell_and_face_flux_stack_average"
                    if ENFORCE_SPANWISE_INVARIANCE
                    else "none"
                ),
                "physical_basis": "Re=150 below the three-dimensional Mode-A wake instability",
            },
            "cell_order": "morton",
            "cell_count": int(mesh_data["n_cells"]),
            "target_cell_count": GRID.target_cells,
        },
        "forces": {
            "method": "pressure_and_viscous_traction_on_cylinder_wall_patch",
            "patch": "cylinder",
            "immersed_boundary_forcing": False,
        },
        "execution": {
            "mpi_ranks": FVM_CORES,
            "rank_cpus": list(RANK_CPUS),
            "operator_backend": FVM_SETUP.execution.operator_backend,
            "linear_backend": "petsc-partitioned" if FVM_CORES > 1 else "serial",
            "outer_correctors": OUTER_CORRECTORS,
            "nonorthogonal_correctors": NONORTHOGONAL_CORRECTORS,
            "log_interval_steps": LOG_INTERVAL_STEPS,
            "pressure_tolerance": PRESSURE_TOLERANCE,
            "momentum_tolerance": MOMENTUM_TOLERANCE,
            "time_scheme": TIME_SCHEME,
            "ddt_corr": DDT_CORR,
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
    cache.parent.mkdir(parents=True, exist_ok=True)
    if (RESTART_FROM is not None or REUSE_MESH_CACHE) and cache.is_file():
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
    maximum_cpu_temperature = cfg.positive_environment_float("OPENONDA_MAX_CPU_TEMP_C", 90.0)
    thermal_resume_temperature = cfg.positive_environment_float(
        "OPENONDA_RESUME_CPU_TEMP_C", maximum_cpu_temperature - 25.0
    )
    if thermal_resume_temperature >= maximum_cpu_temperature:
        raise ValueError("OPENONDA_RESUME_CPU_TEMP_C must be below OPENONDA_MAX_CPU_TEMP_C")
    thermal_action = os.environ.get("OPENONDA_THERMAL_ACTION", "pause").strip().lower()
    if thermal_action not in {"pause", "stop"}:
        raise ValueError("OPENONDA_THERMAL_ACTION must be 'pause' or 'stop'")
    thermal_check_steps = int(os.environ.get("OPENONDA_THERMAL_CHECK_STEPS", "25"))
    if thermal_check_steps < 1:
        raise ValueError("OPENONDA_THERMAL_CHECK_STEPS must be at least one")
    starting_temperature = _package_temperature_c()
    if starting_temperature is not None:
        if thermal_action == "stop" and starting_temperature >= maximum_cpu_temperature:
            raise RuntimeError(
                f"Thermal start gate: CPU package is {starting_temperature:.1f} C, "
                f"at or above OPENONDA_MAX_CPU_TEMP_C={maximum_cpu_temperature:.1f} C"
            )
        if thermal_action == "pause" and starting_temperature > thermal_resume_temperature:
            _wait_for_package_temperature(thermal_resume_temperature)
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
        if (
            RESTART_FROM is None
            or not (CASE_DIR / "solution" / "benchmark_metadata.json").is_file()
        ):
            _write_metadata(mesh_data)
    else:
        mesh_data = None
    projection_layout = (
        build_spanwise_projection_layout(mesh_data, FVM_CORES)
        if IS_ROOT and ENFORCE_SPANWISE_INVARIANCE
        else None
    )
    solver = fvm.create_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=mesh_data)
    del mesh_data
    projector = None
    if ENFORCE_SPANWISE_INVARIANCE:
        projector = SpanwiseInvariantProjector(solver, projection_layout)
        solver.set_post_solve_state_callback(projector)
    solver.write_run_manifest()
    if RESTART_FROM is None:
        _apply_initial_state(solver)
        solver.write_vtk()
    else:
        solver.load_state(RESTART_FROM, allow_config_change=RESTART_ALLOW_CONFIG_CHANGE)
        if projector is not None:
            projector(solver)
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
    projection_limits = {
        "velocity": cfg.positive_environment_float(
            "OPENONDA_MAX_PROJECTION_VELOCITY", 2.0e-3
        ),
        "kinematic_pressure": cfg.positive_environment_float(
            "OPENONDA_MAX_PROJECTION_PRESSURE", 1.0e-1
        ),
        "volumetric_face_flux": cfg.positive_environment_float(
            "OPENONDA_MAX_PROJECTION_FACE_FLUX", 2.0e-5
        ),
    }
    completed_steps = 0
    thermal_stop = False
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
            if diagnostics is not None and diagnostics.state_projection:
                for name, limit in projection_limits.items():
                    value = diagnostics.state_projection.get(name, 0.0)
                    if value > limit:
                        raise RuntimeError(
                            f"Projection gate: step {solver.step} removed {name}={value:.6g}, "
                            f"above the audited limit {limit:.6g}"
                        )
            if solver.step % FIELD_BACKUP_STEPS == 0:
                _save_checkpoint(solver)
            if local_step % thermal_check_steps == 0:
                package_temperature = _package_temperature_c()
                temperature_limited = (
                    package_temperature is not None
                    and package_temperature >= maximum_cpu_temperature
                )
                if temperature_limited:
                    if solver.step % FIELD_BACKUP_STEPS != 0:
                        _save_checkpoint(solver)
                    reason = f"CPU {package_temperature:.1f} C"
                    if thermal_action == "stop":
                        thermal_stop = True
                        if IS_ROOT:
                            print(f"  thermal checkpoint stop: {reason}")
                        break
                    if IS_ROOT:
                        print(f"  thermal checkpoint pause: {reason}")
                    _wait_for_package_temperature(thermal_resume_temperature)
        if (run_steps < remaining_steps or thermal_stop) and IS_ROOT:
            print(f"  bounded stop after {completed_steps} FVM steps")
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
