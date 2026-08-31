#!/usr/bin/env python3
"""Lamb--Oseen vortex benchmark: single vortex, vortex dipole, and merging pair.

The Lamb--Oseen vortex is an exact solution of the two-dimensional
Navier-Stokes equations for an isolated vortex, so the computed flow can be
compared directly with the analytic profile. This script runs the chosen
benchmark case (single vortex, vortex dipole, or merging vortex pair) with the
chosen diffusion schemes and stores the snapshots under ``solution`` while the
sampled z=L/4 velocity/vorticity fields and flow integrals are written under
``samples``.

Example:
    python lamb_oseen_setup.py --circulation1 +1 --circulation2 -1 --viscous-scheme CS \
        --case-name dipole_cs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import openonda.vpm as vpm

TUTORIAL_DIR = Path(__file__).resolve().parent

# ---- Physics (Lamb--Oseen benchmark) ------------------------------------
CIRCULATION_REYNOLDS_NUMBER = 530.0  # Re_Γ = |Γ|/ν — sets the vortex Reynolds number
SEPARATION = 1.0  # distance between the two vortex centres [m]
BETA_RMAX = 1.12  # r(u_θ,max)/a — velocity-peak radius / Gaussian core radius
CORE_RADIUS = 0.125  # a₀ — initial velocity-peak radius [m] (defines the analytic profile)
GAUSSIAN_CORE_RADIUS = CORE_RADIUS / BETA_RMAX  # a₀ — Gaussian 1/e vorticity radius [m]
COLUMN_LENGTH = 40.0 * CORE_RADIUS  # finite vortex column length along z [m]

# ---- Numerical setup shared by every viscous scheme ----------------------
SPACING = 0.47 * CORE_RADIUS  # in-plane particle spacing (controls resolution)
COLUMN_SPACING = 0.45 * CORE_RADIUS  # axial layer spacing along the vortex column
PARTICLE_RADIUS = 1.50 * SPACING  # vortex particle core radius (1.5× spacing)
FIELD_SPACING = 0.16 * CORE_RADIUS  # sampling field resolution for surface output
TIME_STEP_SIZE = 0.291 / 9.0  # Δt [s]
TOTAL_TIME = 103.0 * 0.291  # total simulation time [s]
SAMPLE_INTERVAL_TIME = 2.0 * 0.291  # time between field samples [s]
MERGING_SAMPLE_INTERVAL_STEPS = 6  # resolve the rapid final collapse of the two vorticity peaks
BACKUP_INTERVAL_TIME = 10.0 * 0.291  # time between snapshots [s]
TREECODE_THETA = 0.30  # treecode accuracy parameter (higher = faster, less accurate)
TREECODE_MULTIPOLE_ORDER = 3  # treecode multipole expansion order
ADVECTION_SCHEME = "RK2"  # Runge-Kutta 2nd-order particle advection
DVH_RD_RATIO = 4  # DVH support radius / particle spacing ratio
DVH_PADDING = 5.0  # DVH diffusion grid: extra cells around domain [in cell units]
GBD_PADDING = 5.0  # GBD diffusion grid: extra cells around domain [in cell units]
DVH_THRESHOLD = 1.0e-4  # DVH: strength budget threshold for particle regeneration
GBD_THRESHOLD = 5.0e-5  # GBD: strength budget threshold for particle regeneration
DVH_MAX_NODES = 300_000  # DVH: max octree nodes for diffusion grid
GBD_MAX_NODES = 300_000  # GBD: max octree nodes for diffusion grid
INITIAL_STRENGTH_CUTOFF = 0.01  # discard particles with Γ < 1% of peak (reduces particle count)
CORE_RADIUS_RATIO = 1.2  # DVH/GBD core radius ratio for regeneration
VISCOUS_SCHEMES = ("CS", "RWM", "DVH", "GBD")


def viscous_config(
    scheme: str,
    kinematic_viscosity: float,
    spacing: float,
) -> vpm.ViscousConfig:
    if scheme == "cs":
        return vpm.ViscousConfig.cs(
            kinematic_viscosity=kinematic_viscosity,
            particle_spacing=spacing,
        )
    if scheme == "rwm":
        return vpm.ViscousConfig.rwm(
            kinematic_viscosity=kinematic_viscosity,
            particle_spacing=spacing,
        )
    if scheme == "dvh":
        return vpm.ViscousConfig.dvh(
            particle_spacing=spacing,
            padding=DVH_PADDING,
            kinematic_viscosity=kinematic_viscosity,
            dvh_support_radius_ratio=DVH_RD_RATIO,
            threshold=DVH_THRESHOLD,
            threshold_mode="budget",
            max_nodes=DVH_MAX_NODES,
            core_radius_ratio=CORE_RADIUS_RATIO,
        )
    return vpm.ViscousConfig.gbd(
        particle_spacing=spacing,
        padding=GBD_PADDING,
        kinematic_viscosity=kinematic_viscosity,
        threshold=GBD_THRESHOLD,
        threshold_mode="budget",
        max_nodes=GBD_MAX_NODES,
        core_radius_ratio=CORE_RADIUS_RATIO,
    )


def normalize_retained_circulation(
    particle_vortex_strength: np.ndarray,
    keep: np.ndarray,
    requested_circulation_per_length: float,
    column_length: float,
) -> tuple[np.ndarray, float, float]:
    """Preserve the requested circulation after truncating weak particles."""
    retained = particle_vortex_strength[keep].copy()
    raw_per_length = float(retained[:, 2].sum() / column_length)
    scale = requested_circulation_per_length / raw_per_length
    retained *= scale
    return retained, raw_per_length, scale


def run_case(
    physics: str,
    scheme: str,
    circulations: tuple[float, ...],
    case_name: str,
    compute_device: str = "AUTO",
    random_seed: int = 42,
    ensemble_member: int | None = None,
) -> None:
    if ensemble_member is not None and scheme != "rwm":
        raise ValueError("--ensemble-member is only valid with --viscous-scheme RWM")
    if ensemble_member is not None and ensemble_member < 0:
        raise ValueError("--ensemble-member must be non-negative")
    is_rwm_ensemble = scheme == "rwm" and ensemble_member is not None
    # ---- Derived physical quantities ----
    spacing = SPACING
    column_spacing = COLUMN_SPACING
    particle_core_radius = 1.5 * spacing
    field_spacing = FIELD_SPACING
    circulation = abs(circulations[0])
    kinematic_viscosity = circulation / CIRCULATION_REYNOLDS_NUMBER  # ν = |Γ|/Re_Γ
    # ---- Time stepping ----
    viscous = viscous_config(scheme, kinematic_viscosity, spacing)
    sample_steps = round(SAMPLE_INTERVAL_TIME / TIME_STEP_SIZE)
    backup_steps = round(BACKUP_INTERVAL_TIME / TIME_STEP_SIZE)
    field_interval_steps = MERGING_SAMPLE_INTERVAL_STEPS if physics == "merging" else sample_steps
    if is_rwm_ensemble:
        # The projected two-dimensional estimator is reconstructed from these
        # compact particle backups.  Online sampling of a single z-plane
        # would store a noisy realization and is intentionally disabled.
        backup_steps = field_interval_steps

    # ---- Initial vortex geometry ----
    y_positions = (0.0,) if physics == "vortex" else (SEPARATION / 2, -SEPARATION / 2)
    initial_half_width = max(abs(y) for y in y_positions) + 7.0 * CORE_RADIUS
    column_half_length = COLUMN_LENGTH / 2.0

    # ---- Domain sizing (must contain vortex at t=end_time) ----
    # Lamb-Oseen core spreads as a(t)² = a₀² + 4νt; final_core_radius is
    # the velocity-peak radius at end_time — the domain must be large enough
    # to hold the fully-diffused vortex without truncation.
    final_core_radius = BETA_RMAX * np.sqrt(
        GAUSSIAN_CORE_RADIUS**2 + 4.0 * kinematic_viscosity * TOTAL_TIME
    )
    padding = (
        0.0 if physics == "vortex" else 4.0 * final_core_radius
    )  # extra room for dipole/merging spread
    lateral_half_width = (
        initial_half_width if physics == "vortex" else max(abs(y) for y in y_positions) + padding
    )
    field_padding = 0.0 if physics == "vortex" else 3.0 * final_core_radius
    field_lateral_half_width = (
        initial_half_width
        if physics == "vortex"
        else max(abs(y) for y in y_positions) + field_padding
    )
    axial_half_length = column_half_length + padding
    downstream_length = (
        8.0 * SEPARATION if physics == "dipole" else 0.0
    )  # dipole advects downstream

    domain_bounds = [
        -lateral_half_width,
        lateral_half_width + downstream_length,
        -lateral_half_width,
        lateral_half_width,
        -axial_half_length,
        axial_half_length,
    ]
    field_bounds = [
        -field_lateral_half_width,
        field_lateral_half_width + downstream_length,
        -field_lateral_half_width,
        field_lateral_half_width,
    ]
    initial_bounds = [
        -initial_half_width,
        initial_half_width,
        -initial_half_width,
        initial_half_width,
        -column_half_length,
        column_half_length,
    ]
    # ---- Particle distribution and field sampler ----
    distribution = vpm.create_triangular_prism_distribution(
        bounds=(
            (initial_bounds[0], initial_bounds[1]),
            (initial_bounds[2], initial_bounds[3]),
            (initial_bounds[4], initial_bounds[5]),
        ),
        spacing=spacing,
        axial_spacing=column_spacing,
        core_radius_ratio=particle_core_radius / spacing,
    )

    case_dir = TUTORIAL_DIR.resolve()
    solution_dir = case_dir / "solution"

    sample_plane_fraction = 0.25  # sample at z = L/4
    if is_rwm_ensemble:
        scheduled_samplers = []
    else:
        field_sampler = vpm.SurfaceSampler(
            point=[0, 0, sample_plane_fraction * COLUMN_LENGTH],
            normal=[0, 0, 1],
            bounds=field_bounds,
            spacing=field_spacing,
            file_name=f"{case_name}_zq",
            include_derivatives=False,
            schedule=vpm.SamplingSchedule(every_n_steps=field_interval_steps),
        )
        scheduled_samplers = [field_sampler]

    member_name = f"member_{ensemble_member:03d}" if is_rwm_ensemble else None
    sample_directory = (
        str(Path("rwm_ensemble") / case_name / member_name)
        if member_name is not None
        else case_name
    )
    solution_directory = (
        solution_dir / "rwm_ensemble" / case_name / member_name
        if member_name is not None
        else solution_dir / case_name
    )

    solver = vpm.VPMSolver(
        setup=vpm.VPMSetup.viscous_flow_simulation(
            time_step_size=TIME_STEP_SIZE,
            viscous=viscous,
            advection=vpm.AdvectionConfig(scheme=ADVECTION_SCHEME),
            velocity=vpm.VelocityConfig.treecode(
                theta=TREECODE_THETA,
                multipole_order=TREECODE_MULTIPOLE_ORDER,
                sort_particle_targets=True,
            ),
            backup=vpm.Backup(
                interval_steps=backup_steps,
                directory=str(solution_directory),
                log_directory=str(solution_directory),
            ),
            samplers=vpm.Samplers(
                vpm.FlowIntegralsSampler(
                    schedule=vpm.SamplingSchedule(every_n_steps=field_interval_steps)
                ),
                *scheduled_samplers,
                directory=sample_directory,
            ),
            write_precision="f32",
            domain_bounds=domain_bounds,
            compute_device=compute_device,
            random_seed=random_seed,
        ),
        case_dir=case_dir,
    )
    # DVH may pin the requested advection step to an integer subdivision of
    # its physical diffusion interval. The run horizon follows that applied
    # step, not the value requested before the solver was constructed.
    n_steps = round(TOTAL_TIME / solver.time_step_size)

    # ---- Metadata and vortex particle seeding ----
    samples_dir = case_dir / "samples" / sample_directory
    samples_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "running",
        "completed": False,
        "case": physics,
        "scheme": scheme,
        "circulations": circulations,
        "kinematic_viscosity": kinematic_viscosity,
        "core_radius": GAUSSIAN_CORE_RADIUS,
        "velocity_peak_radius": CORE_RADIUS,
        "vortex_separation": SEPARATION,
        "column_half_length": column_half_length,
        "end_time": TOTAL_TIME,
        "time_step_size": solver.time_step_size,
        "field_sample_interval_steps": (field_interval_steps),
        "random_seed": random_seed,
        "ensemble_member": ensemble_member,
        "raw_output_estimator": (
            "particle_backup_for_column_projection"
            if is_rwm_ensemble
            else "instantaneous_surface_field"
        ),
    }
    metadata_path = samples_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for group_id, (circulation, y_position) in enumerate(
        zip(circulations, y_positions, strict=True)
    ):
        particles = vpm.initialize_vortex_filament(
            distribution,
            kinematic_viscosity=kinematic_viscosity,
            centre=(0.0, y_position, 0.0),
            direction=(0.0, 0.0, 1.0),
            circulation=circulation,
            vortex_core_radius=GAUSSIAN_CORE_RADIUS,
            compensate_particle_core=True,
        )
        vortex_strength_magnitude = np.linalg.norm(particles.vortex_strength, axis=1)
        keep = (
            vortex_strength_magnitude >= INITIAL_STRENGTH_CUTOFF * vortex_strength_magnitude.max()
        )
        retained_vortex_strength, _, _ = normalize_retained_circulation(
            particles.vortex_strength,
            keep,
            circulation,
            COLUMN_LENGTH,
        )
        retained = particles.select(keep)
        solver_arguments = retained.solver_kwargs()
        solver_arguments["vortex_strength"] = retained_vortex_strength
        group_id = np.full(np.count_nonzero(keep), group_id, dtype=np.int32)
        solver.add_vortex_particles(
            **solver_arguments,
            group_id=group_id,
        )

    metadata["initial_n_particles_total"] = len(solver.particles)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # ---- Time integration ----
    try:
        if is_rwm_ensemble:
            if str(solver.compute_device).upper() == "METAL":
                raise RuntimeError(
                    "Seeded RWM ensembles are not supported on METAL because Taichi "
                    "1.7 does not accept random_seed for that backend; use CPU, CUDA, or VULKAN."
                )
            solver.save_backup()
        else:
            solver.record_diagnostics(refresh_fields=True)
        for _ in range(n_steps):
            solver.advance()
        if is_rwm_ensemble:
            if solver.step % backup_steps != 0:
                solver.save_backup()
            if solver.step % field_interval_steps != 0:
                # The final backup refreshes the particle velocity field;
                # record the matching final-time integral state as well.
                solver.record_diagnostics(refresh_fields=False)
        else:
            if solver.step % sample_steps != 0:
                solver.record_diagnostics(refresh_fields=False)
            if physics == "merging" and solver.step % MERGING_SAMPLE_INTERVAL_STEPS != 0:
                solver.record_diagnostics(refresh_fields=False)
    except BaseException:
        metadata["status"] = "failed"
        metadata["completed"] = False
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    finally:
        solver.reset_gpu()

    metadata["status"] = "complete"
    metadata["completed"] = True
    metadata["final_time"] = n_steps * solver.time_step_size
    metadata["compute_device"] = solver.compute_device
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circulation1", type=float, required=True)
    parser.add_argument("--circulation2", type=float, default=0.0)
    parser.add_argument("--viscous-scheme", choices=VISCOUS_SCHEMES, required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--compute-device", default="AUTO")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--ensemble-member",
        type=int,
        default=None,
        help="zero-based RWM ensemble member; writes raw data under rwm_ensemble/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if abs(args.circulation2) < 1e-12:
        physics, circulations = "vortex", (args.circulation1,)
    elif args.circulation1 * args.circulation2 < 0.0:
        physics, circulations = "dipole", (args.circulation1, args.circulation2)
    else:
        physics, circulations = "merging", (args.circulation1, args.circulation2)
    run_case(
        physics,
        args.viscous_scheme.lower(),
        circulations,
        args.case_name,
        args.compute_device,
        args.random_seed,
        args.ensemble_member,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
