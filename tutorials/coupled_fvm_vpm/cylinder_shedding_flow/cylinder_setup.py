"""Cut-cell FVM--VPM replacement of the Re=150 cylinder reference.

The FVM owns the same long watertight-cylinder STL and local Cartesian lattice
as the fully meshed reference. The STL crosses the two spanwise slip planes,
so the solved four-diameter segment has a circular side wall and no cap-force
contamination. There is no immersed-boundary model; forces come from the
physical ``cylinder`` wall patch. The VPM panel body is used only to include
the cylinder's potential-flow influence in the VPM boundary condition seen by
the truncated FVM domain.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent
CASE_DIR = SOURCE_DIR


def _acquire_fresh_run_directory(run_dir: Path):
    """Hold a single-writer lock and reject sample-appending reruns."""
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
        run_dir / "solution" / "coupler_diagnostics.jsonl",
        run_dir / "samples" / "forces_history.csv",
    )
    existing = [path for path in markers if path.exists()]
    if existing:
        handle.close()
        names = ", ".join(str(path.relative_to(run_dir)) for path in existing)
        raise SystemExit(
            f"Refusing to append to existing coupled output {run_dir} ({names}). "
            "Run the root allclean.sh before starting a replacement calculation."
        )
    return handle


_RUN_LOCK = _acquire_fresh_run_directory(CASE_DIR)
_TAICHI_CACHE = CASE_DIR / ".taichi_cache"
_TAICHI_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TI_OFFLINE_CACHE_FILE_PATH", str(_TAICHI_CACHE))

import numpy as np  # noqa: E402

import openonda.coupler as coupling  # noqa: E402
import openonda.fvm as fvm  # noqa: E402
import openonda.vpm as vpm  # noqa: E402

import benchmark_config as cfg  # noqa: E402
from seed_perturbation import build_cylinder_initial_state  # noqa: E402

GRID = cfg.selected_grid()
END_TIME = cfg.end_time()
SEED_AMPLITUDE = cfg.seed_amplitude()
FVM_TIME_STEP_SIZE = cfg.fvm_time_step(GRID)
VPM_TIME_STEP_SIZE = 0.1 if GRID.name in {"smoke", "g0"} else 0.05
VPM_SUBSTEPS = int(round(VPM_TIME_STEP_SIZE / FVM_TIME_STEP_SIZE))
if not np.isclose(VPM_SUBSTEPS * FVM_TIME_STEP_SIZE, VPM_TIME_STEP_SIZE, rtol=0.0, atol=1.0e-12):
    raise ValueError("VPM time step must be an integer multiple of the FVM time step")

if int(os.environ.get("OPENONDA_FVM_CORES", "1")) != 1:
    raise ValueError("The cylinder comparison is intentionally serial; set OPENONDA_FVM_CORES=1")

FVM_BOX = cfg.COUPLED_FVM_BOX
FVM_MESH = cfg.build_mesh(FVM_BOX, GRID, merge_outer_patch="numericalBoundary")
FVM_SAMPLE_STEPS = cfg.physical_sample_steps(FVM_TIME_STEP_SIZE)
VPM_SAMPLE_STEPS = cfg.physical_sample_steps(VPM_TIME_STEP_SIZE)
FVM_LINE_STEPS = cfg.physical_sample_steps(FVM_TIME_STEP_SIZE, 0.2)
VPM_LINE_STEPS = cfg.physical_sample_steps(VPM_TIME_STEP_SIZE, 0.2)
FVM_SLICE_STEPS = cfg.physical_sample_steps(FVM_TIME_STEP_SIZE, 0.5)
VPM_SLICE_STEPS = cfg.physical_sample_steps(VPM_TIME_STEP_SIZE, 0.5)
FVM_BACKUP_STEPS = cfg.physical_sample_steps(FVM_TIME_STEP_SIZE, cfg.field_output_interval())
VPM_BACKUP_STEPS = cfg.physical_sample_steps(VPM_TIME_STEP_SIZE, cfg.field_output_interval())

FVM_FORCE_SCHEDULE = fvm.SamplingSchedule(every_n_steps=FVM_SAMPLE_STEPS)
FVM_LINE_SCHEDULE = fvm.SamplingSchedule(every_n_steps=FVM_LINE_STEPS)
FVM_SLICE_SCHEDULE = fvm.SamplingSchedule(every_n_steps=FVM_SLICE_STEPS)
VPM_FORCE_SCHEDULE = vpm.SamplingSchedule(every_n_steps=VPM_SAMPLE_STEPS)
VPM_LINE_SCHEDULE = vpm.SamplingSchedule(every_n_steps=VPM_LINE_STEPS)
VPM_SLICE_SCHEDULE = vpm.SamplingSchedule(every_n_steps=VPM_SLICE_STEPS)

VPM_PARTICLE_SPACING = 0.125 if GRID.name in {"smoke", "g0"} else 1.0 / 16.0
VPM_CORE_RADIUS_RATIO = 1.0
PARTICLE_LIMIT = 200_000 if GRID.name == "smoke" else 750_000
GBD_VORTICITY_FLOOR = 0.01
GBD_ALPHA = cfg.KINEMATIC_VISCOSITY * VPM_TIME_STEP_SIZE / VPM_PARTICLE_SPACING**2
if GBD_ALPHA >= 1.0 / 6.0:
    raise ValueError(f"GBD diffusion stability violated: alpha={GBD_ALPHA:.6g} >= 1/6")

SAMPLE_SPACING = min(0.125, GRID.near_wake, VPM_PARTICLE_SPACING)
WAKE_X_BOUNDS = (-2.0, 12.0)
WAKE_Y_BOUNDS = (-3.0, 3.0)


def _fvm_line_x(x: float, name: str) -> fvm.LineSampler:
    return fvm.LineSampler(
        start=[x, WAKE_Y_BOUNDS[0], 0.0],
        end=[x, WAKE_Y_BOUNDS[1], 0.0],
        spacing=SAMPLE_SPACING,
        k=12,
        reconstruction="affine",
        file_name=f"fvm_{name}",
        schedule=FVM_LINE_SCHEDULE,
    )


def _vpm_line_x(x: float, name: str) -> vpm.LineSampler:
    return vpm.LineSampler(
        start=[x, WAKE_Y_BOUNDS[0], 0.0],
        end=[x, WAKE_Y_BOUNDS[1], 0.0],
        spacing=SAMPLE_SPACING,
        file_name=f"vpm_{name}",
        schedule=VPM_LINE_SCHEDULE,
    )


FVM_SAMPLERS = (
    fvm.ForceSampler(
        patch_names=["cylinder"],
        reference_velocity=cfg.FREESTREAM_SPEED,
        reference_area=cfg.REFERENCE_AREA,
        reference_length=cfg.REFERENCE_LENGTH,
        moment_centre=[0.0, 0.0, 0.0],
        file_name="forces_history",
        schedule=FVM_FORCE_SCHEDULE,
    ),
    fvm.LineSampler(
        start=[1.5, 0.0, 0.0],
        end=[1.5, 0.0, 0.0],
        n_points=1,
        k=12,
        reconstruction="affine",
        file_name="fvm_midspan_probe",
        schedule=FVM_FORCE_SCHEDULE,
    ),
    fvm.LineSampler(
        start=[WAKE_X_BOUNDS[0], 0.0, 0.0],
        end=[FVM_BOX[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        k=12,
        reconstruction="affine",
        file_name="fvm_centreline",
        schedule=FVM_LINE_SCHEDULE,
    ),
    _fvm_line_x(1.0, "transverse_x1"),
    _fvm_line_x(2.0, "transverse_x2"),
    _fvm_line_x(4.0, "transverse_x4"),
    fvm.LineSampler(
        start=[1.5, 0.0, cfg.CYLINDER_Z_BOUNDS[0] + 0.5 * cfg.SPANWISE_CELL_SIZE],
        end=[1.5, 0.0, cfg.CYLINDER_Z_BOUNDS[1] - 0.5 * cfg.SPANWISE_CELL_SIZE],
        spacing=cfg.SPANWISE_CELL_SIZE,
        k=12,
        reconstruction="affine",
        file_name="fvm_spanwise_line",
        schedule=FVM_LINE_SCHEDULE,
    ),
    fvm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[FVM_BOX[0], FVM_BOX[1], FVM_BOX[2], FVM_BOX[3]],
        spacing=SAMPLE_SPACING,
        k=12,
        reconstruction="affine",
        file_name="fvm_slice_z0",
        schedule=FVM_SLICE_SCHEDULE,
        body_bounds=FVM_MESH.surface_bounds,
        body_geometry="cylinder_z",
    ),
)

FVM_SETUP = fvm.FVMSetup(
    case_name=f"cylinder_coupled_{GRID.name}",
    cores=1,
    execution=fvm.ComputeConfig(operator_backend="numba"),
    output=fvm.OutputConfig(
        format="vtk_xml",
        data_location="cell",
        encoding="appended",
        compression="lz4",
        precision="f32",
        asynchronous=False,
        ghost_layers=0,
    ),
    time=fvm.TimeConfig(
        time_step_size=FVM_TIME_STEP_SIZE,
        start_time=0.0,
        end_time=END_TIME,
        output_interval_steps=FVM_BACKUP_STEPS,
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
        n_outer_correctors=2,
        n_orthogonal_correctors=1,
        velocity_relaxation=0.7,
        pressure_relaxation=0.3,
    ),
    samplers=FVM_SAMPLERS,
    transport=fvm.TransportConfig(
        density=cfg.DENSITY,
        kinematic_viscosity=cfg.KINEMATIC_VISCOSITY,
    ),
    turbulence=fvm.TurbulenceConfig.none(),
    boundaries=[
        fvm.BoundaryConfig(
            name="numericalBoundary",
            velocity_type="fixedValue",
            velocity_value=list(cfg.FREESTREAM_VELOCITY),
            pressure_type="fixedFluxPressure",
        ),
        fvm.BoundaryConfig.slip("zmin"),
        fvm.BoundaryConfig.slip("zmax"),
        fvm.BoundaryConfig.wall("cylinder"),
    ],
    initial_velocity=list(cfg.INITIAL_VELOCITY),
    initial_kinematic_pressure=0.0,
)

COUPLER_SETUP = coupling.CouplerSetup(
    freestream_velocity=list(cfg.FREESTREAM_VELOCITY),
    transfer_method="buffered_m4_renewal",
    transfer_region_bounds=cfg.TRANSFER_REGION_BOX,
    checkpoint_interval_steps=VPM_BACKUP_STEPS,
    boundary_condition_mode="vorticity_mixed",
    fvm_consistency_width=0.25,
    eta_blend_width=6.0 * VPM_PARTICLE_SPACING,
    vpm_only_width=0.0,
    transfer_vorticity_cutoff=0.05,
    transfer_boundary_prune_multiplier=10.0,
    transfer_amplification_cap=1.8,
    transfer_diagnostic_interval_steps=max(1, VPM_SAMPLE_STEPS),
)

VPM_SAMPLERS = (
    vpm.LineSampler(
        start=[1.5, 0.0, 0.0],
        end=[1.5, 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_midspan_probe",
        schedule=VPM_FORCE_SCHEDULE,
    ),
    vpm.LineSampler(
        start=[WAKE_X_BOUNDS[0], 0.0, 0.0],
        end=[WAKE_X_BOUNDS[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_centreline",
        schedule=VPM_LINE_SCHEDULE,
    ),
    _vpm_line_x(1.0, "transverse_x1"),
    _vpm_line_x(2.0, "transverse_x2"),
    _vpm_line_x(4.0, "transverse_x4"),
    vpm.LineSampler(
        start=[1.5, 0.0, -3.0],
        end=[1.5, 0.0, 3.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_spanwise_line",
        schedule=VPM_LINE_SCHEDULE,
    ),
    vpm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[*WAKE_X_BOUNDS, *WAKE_Y_BOUNDS],
        spacing=SAMPLE_SPACING,
        file_name="vpm_slice_z0",
        include_derivatives=False,
        schedule=VPM_SLICE_SCHEDULE,
    ),
)

VPM_PANEL_SOLVER = vpm.PanelSolver(
    max_n_panels=2048,
    float_dtype="f32",
    linear_solver="SCIPY",
    boundary_condition_type="NEUMANN",
    density=cfg.DENSITY,
    freestream_velocity=np.asarray(cfg.FREESTREAM_VELOCITY),
    coupling_scope="vpm_boundary_condition",
)

VPM_SETUP = vpm.VPMSetup(
    time_step_size=VPM_TIME_STEP_SIZE,
    time_integration="COUPLED",
    coupled_max_strain_increment=None,
    coupled_max_advection_fraction=None,
    freestream_velocity=list(cfg.FREESTREAM_VELOCITY),
    viscous=vpm.ViscousConfig.gbd(
        particle_spacing=VPM_PARTICLE_SPACING,
        padding=5.0,
        kinematic_viscosity=cfg.KINEMATIC_VISCOSITY,
        threshold_mode="absolute",
        threshold=GBD_VORTICITY_FLOOR * VPM_PARTICLE_SPACING**3,
        max_nodes=PARTICLE_LIMIT,
        core_radius_ratio=VPM_CORE_RADIUS_RATIO,
    ),
    stretching=vpm.StretchingConfig.transposed(scheme="RK2"),
    advection=vpm.AdvectionConfig(scheme="RK2"),
    turbulence=vpm.TurbulenceConfig.inviscid(),
    velocity=vpm.VelocityConfig.treecode(theta=0.3, multipole_order=2),
    stabilization=vpm.StabilizationConfig.bounded_domain(cfg.VPM_DOMAIN),
    particle_kernel="GAUSSIAN",
    precision="f32",
    compute_device="AUTO",
    max_n_particles=PARTICLE_LIMIT,
    max_evaluation_points=PARTICLE_LIMIT,
    domain_bounds=list(cfg.VPM_DOMAIN),
    log_mode="file",
    logging_interval_steps=max(1, VPM_SAMPLE_STEPS),
    timing_interval_steps=max(1, VPM_SAMPLE_STEPS),
    write_precision="f32",
    checkpoint_store_velocity_gradient=False,
    checkpoint_interval_steps=0,
    checkpoint_directory=str(CASE_DIR / "solution"),
    export_flow_integrals=False,
    samplers=VPM_SAMPLERS,
    panel_solver=VPM_PANEL_SOLVER,
    bodies=(
        vpm.PanelBodySetup(
            stl=str(cfg.CYLINDER_STL),
            uid="cylinder",
            reference_area=cfg.REFERENCE_AREA,
        ),
    ),
)


def _available_memory_gib() -> float:
    values = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as stream:
        for line in stream:
            key, value = line.split(":", 1)
            values[key] = float(value.split()[0]) * 1024.0
    return values["MemAvailable"] / 1024.0**3


def _resource_gate() -> None:
    minimum = cfg.positive_environment_float(
        "OPENONDA_MIN_AVAILABLE_GIB", cfg.minimum_available_memory_gib(GRID)
    )
    available = _available_memory_gib()
    if available < minimum:
        raise RuntimeError(
            f"Resource gate: {available:.2f} GiB RAM available, {minimum:.2f} GiB required"
        )


def _apply_initial_state(solver) -> None:
    n_cells = solver.mesh_data["n_cells"]
    centres = np.asarray(solver.geo_data["cell_centre"][:n_cells], dtype=np.float64)
    velocity, kinematic_pressure = build_cylinder_initial_state(
        centres,
        freestream_velocity=cfg.FREESTREAM_VELOCITY,
        diameter=cfg.DIAMETER,
        seed_amplitude=SEED_AMPLITUDE,
    )
    solver.set_initial_state(velocity, kinematic_pressure)


def _write_metadata(mesh_data: dict) -> None:
    solution = CASE_DIR / "solution"
    solution.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "openonda-cylinder-coupled/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_source": str(SOURCE_DIR),
        "run_directory": str(CASE_DIR),
        "physics": {
            "reynolds_number": cfg.REYNOLDS,
            "diameter": cfg.DIAMETER,
            "cylinder_length": cfg.CYLINDER_LENGTH,
            "freestream_velocity": list(cfg.FREESTREAM_VELOCITY),
            "kinematic_viscosity": cfg.KINEMATIC_VISCOSITY,
            "end_time": END_TIME,
            "seed_amplitude": SEED_AMPLITUDE,
        },
        "fvm": {
            "grid": GRID.name,
            "domain": list(FVM_BOX),
            "cell_count": int(mesh_data["n_cells"]),
            "time_step": FVM_TIME_STEP_SIZE,
            "background_cell_size": GRID.background,
            "surface_cell_size": GRID.surface,
            "shear_layer_cell_size": GRID.shear_layer,
            "near_wake_cell_size": GRID.near_wake,
            "downstream_wake_cell_size": GRID.downstream_wake,
            "substeps_per_coupling_step": VPM_SUBSTEPS,
            "field_output_interval": FVM_BACKUP_STEPS * FVM_TIME_STEP_SIZE,
            "surface_sha256": FVM_MESH.surface.sha256,
        },
        "vpm": {
            "domain": list(cfg.VPM_DOMAIN),
            "particle_spacing": VPM_PARTICLE_SPACING,
            "time_step": VPM_TIME_STEP_SIZE,
            "particle_limit": PARTICLE_LIMIT,
            "checkpoint_interval": VPM_BACKUP_STEPS * VPM_TIME_STEP_SIZE,
            "gbd_alpha": GBD_ALPHA,
            "panel_coupling_scope": "vpm_boundary_condition",
        },
        "coupler": {
            "transfer_method": COUPLER_SETUP.transfer_method,
            "boundary_condition_mode": COUPLER_SETUP.boundary_condition_mode,
            "transfer_region": list(cfg.TRANSFER_REGION_BOX),
        },
    }
    (solution / "benchmark_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    _resource_gate()
    print("\n===== CUT-CELL COUPLED CYLINDER =====")
    print(
        f"  Re={cfg.REYNOLDS:g}, grid={GRID.name}, FVM dt={FVM_TIME_STEP_SIZE:g}, "
        f"VPM dt={VPM_TIME_STEP_SIZE:g}, hp={VPM_PARTICLE_SPACING:g}, alpha={GBD_ALPHA:.4f}"
    )
    print(f"  output={CASE_DIR}")
    mesh_data = FVM_MESH.build()
    if int(mesh_data["n_cells"]) > min(GRID.target_cells, 1_000_000):
        raise MemoryError(f"Coupled FVM mesh is unexpectedly large: {mesh_data['n_cells']:,} cells")
    _write_metadata(mesh_data)
    fvm_solver = fvm.create_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=mesh_data)
    _apply_initial_state(fvm_solver)
    fvm_solver.write_vtk()
    vpm_solver = vpm.create_vpm_solver(VPM_SETUP, case_dir=CASE_DIR)
    coupled_solver = coupling.create_coupler(fvm_solver, vpm_solver, COUPLER_SETUP)
    max_steps = int(os.environ.get("OPENONDA_MAX_COUPLING_STEPS", "0")) or None
    completed_step = coupled_solver.run(
        max_coupling_steps=max_steps,
        checkpoint_at_stop=True,
    )
    if fvm_solver.step % FVM_BACKUP_STEPS != 0:
        # The full coupled checkpoint is restartable; this additional terminal
        # VTU keeps short/bounded runs immediately inspectable in ParaView.
        fvm_solver.write_vtk()
    return completed_step


if __name__ == "__main__":
    main()
