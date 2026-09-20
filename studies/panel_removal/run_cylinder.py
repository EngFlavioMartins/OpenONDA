"""Isolated panel-free, infinite-span cylinder experiment; launch after cube gate.

FVM resolves the body and formation region, with the same planar meshing recipe
and equations as reference_flow. The exact 2D VPM carries the unbounded wake.
No reference or production tutorial output is modified.
"""

import argparse
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path

import numpy as np

import openonda.coupler as coupling
import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.vpm as vpm

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def configuration(
    output,
    *,
    cores=2,
    end_time=100.0,
    dx=0.04,
    particle_spacing=0.05,
    coupling_dt=0.04,
    downstream=6.5,
    half_height=3.5,
    device="AUTO",
    transfer_cutoff=0.05,
    gbd_vorticity_floor=0.01,
    snapshot_interval=0.24,
):
    """Declare the Re=150 infinite-span cylinder comparison without starting it.

    Parameters
    ----------
    output : pathlib.Path
        Isolated output directory for the experiment.
    cores : int, default=2
        MPI process count.
    end_time : float, default=100.0
        Requested horizon in s, on a coupling-step boundary.
    dx : float, default=0.04
        Requested in-plane mesh upper bound in m; nominal wall size is 0.75*dx.
    particle_spacing : float, default=0.05
        Planar particle/lattice spacing h in m.
    coupling_dt : float, default=0.04
        Coupling interval in s; an integer multiple of the 0.004 s FVM step.
    downstream, half_height : float, defaults=6.5, 3.5
        FVM outlet x and positive lateral extent in m. The domain must
        contain the reference profiles at x/D=4 and y/D=±3.
    device : str, default='AUTO'
        VPM Taichi device selection.
    transfer_cutoff, gbd_vorticity_floor : float, defaults=0.05, 0.01
        Finite vorticity floors in s⁻¹ with 0<=GBD<=transfer. Absolute
        strength cutoffs are each floor times h²*span, in m³/s.

    snapshot_interval : float, default=0.24
        Coupled checkpoint and retained FVM/VPM frame interval in s. It must
        be a positive multiple of coupling_dt; 0.24 s is six baseline steps.

    Returns
    -------
    tuple
        FVM setup, VPM case, coupling settings and mesh builder. Constructors
        retain configuration only; the coupled factory owns mesh/device creation.

    Raises
    ------
    ValueError
        If the domain omits required profiles, scales/cutoffs are invalid,
        or the horizon and coupling cadence do not align with the FVM step.

    Notes
    -----
    The represented span is 1 m with four FVM layers and one VPM plane.
    The native FVM wall supplies solid exclusion; no panels are attached.
    """
    if downstream < 5 or half_height < 3.5:
        raise ValueError("The baseline experiment must enclose x=4 and y=±3 reference profiles")
    if dx <= 0 or particle_spacing <= 0 or coupling_dt <= 0 or end_time <= 0:
        raise ValueError("Spacings, time step and horizon must be positive")
    if not (np.isfinite(transfer_cutoff) and np.isfinite(gbd_vorticity_floor)) or not (
        0 <= gbd_vorticity_floor <= transfer_cutoff
    ):
        raise ValueError("Require finite 0 <= GBD vorticity floor <= transfer cutoff")
    if not np.isclose(coupling_dt / 0.004, round(coupling_dt / 0.004)):
        raise ValueError("coupling_dt must be an integer multiple of the .004s FVM step")
    if not np.isclose(end_time / coupling_dt, round(end_time / coupling_dt)):
        raise ValueError("end_time must fall on a coupled step")
    output_steps = snapshot_interval / coupling_dt
    if (
        not np.isfinite(output_steps)
        or output_steps < 1
        or not np.isclose(output_steps, round(output_steps), rtol=0, atol=1e-10)
    ):
        raise ValueError("snapshot_interval must be a positive multiple of coupling_dt")
    case = importlib.import_module("tutorials.coupled_fvm_vpm.01_cylinder_shedding_flow.setup")
    box = (-2.5, downstream, -half_height, half_height, -0.5, 0.5)
    transfer = (-2.0, downstream - 0.5, -half_height + 0.5, half_height - 0.5, -0.375, 0.375)
    source_box = (*box[:4], -16 * dx, 16 * dx)
    patches = msh.BoxPatches(
        xmin="numericalBoundary",
        xmax="numericalBoundary",
        ymin="numericalBoundary",
        ymax="numericalBoundary",
        zmin="zmin",
        zmax="zmax",
    )
    mesh = msh.ExtrudedCartesianMesher(
        source=msh.CartesianMesher(
            domain=msh.BoxDomain(bounds=source_box, patches=patches),
            surfaces=(msh.STLSurface(CASE / "assets/cylinder_long.stl", patch="cylinder"),),
            max_cell_size=12 * dx,
            refinements=tuple(
                msh.BoxRefinement(name=name, bounds=(*xy, *source_box[4:]), cell_size=size)
                for name, xy, size in (
                    ("nearBody", (-1.5, 1.5, -1.5, 1.5), dx),
                    ("nearWake", (-2.0, 6.0, -2.0, 2.0), 2 * dx),
                    ("wake", (-2.5, min(12.0, downstream), -2.5, 2.5), 4 * dx),
                )
            ),
            patch_refinements=(msh.PatchRefinement("cylinder", dx),),
            surface_may_cross_domain_boundary=True,
        ),
        domain=msh.BoxDomain(bounds=box, patches=patches),
        levels=tuple(np.linspace(-0.5, 0.5, 5)),
    )
    # Sampling at accepted coupling boundaries avoids interface rollback records.
    schedule = fvm.RunSchedule(every_time=coupling_dt)
    lines = fvm.RunSchedule(every_time=5 * coupling_dt)
    samplers = [
        fvm.ForceSampler(
            patch_names=["cylinder"],
            reference_velocity=1.0,
            reference_area=1.0,
            reference_length=1.0,
            file_name="forces_history",
            schedule=schedule,
        )
    ]
    specifications = [
        ("fvm_centreline", [-2.0, 0.0, 0.0], [min(6.0, downstream - 0.25), 0.0, 0.0], None),
        ("midspan_probe", [1.5, 0.0, 0.0], [1.5, 0.0, 0.0], 1),
        ("span_probe", [1.5, 0.0, -0.45], [1.5, 0.0, 0.45], 9),
        *[(f"fvm_transverse_x{x:g}", [x, -3.0, 0.0], [x, 3.0, 0.0], None) for x in (1.0, 2.0, 4.0)],
    ]
    for name, start, end, count in specifications:
        sampling = {"spacing": 0.08} if count is None else {"n_points": count}
        samplers.append(
            fvm.LineSampler(
                start=start,
                end=end,
                # A span diagnostic must sample one raw XY stack. A 3D
                # affine stencil changes its XY support with z on this mesh.
                k=1 if name == "span_probe" else 12,
                reconstruction="idw" if name == "span_probe" else "affine",
                file_name=name,
                schedule=lines,
                **sampling,
            )
        )
    fvm_setup = replace(
        case.FVM_SETUP,
        case_name="panel_free_planar_cylinder",
        cores=cores,
        mesh=fvm.MeshQualityConfig(
            max_non_orthogonality_deg=70.0, max_skewness=1.0, max_lsq_condition=9.0
        ),
        output=fvm.OutputConfig(compression="lz4", asynchronous=False, ghost_layers=0),
        time=fvm.TimeConfig(
            time_step_size=0.004,
            end_time=end_time,
            output_schedule=fvm.RunSchedule(every_time=snapshot_interval),
        ),
        logging=fvm.LoggingConfig(schedule=fvm.RunSchedule(every_time=0.2)),
        samplers=tuple(samplers),
        acceptance=fvm.RunAcceptanceLimits(
            max_continuity_error_warning=1.0e-4,
            max_continuity_error_abort=1.0e-2,
            max_equation_residual_warning=1.0e-4,
            max_equation_residual_abort=1.0e-2,
            max_courant_number_warning=0.9,
            max_courant_number_abort=1.5,
            max_velocity_magnitude_warning=3.0,
            max_velocity_magnitude_abort=5.0,
        ),
    )
    bounds = (-8.0, 24.0, -10.0, 10.0, -0.5, 0.5)
    h = particle_spacing
    numerics = replace(
        case.VPM_CASE.numerics,
        time_step_size=coupling_dt,
        induction=vpm.PlanarInduction(span=1.0, plane_z=0.0),
        panel_solver=None,
        bodies=(),
        compute_device=device,
        domain_bounds=bounds,
        stabilization=vpm.StabilizationConfig.bounded_domain(bounds),
        max_n_particles=100_000,
        max_evaluation_points=100_000,
        viscous=vpm.ViscousConfig.gbd(
            particle_spacing=h,
            gbd_grid_spacing=h,
            padding=5.0,
            kinematic_viscosity=1 / 150.0,
            threshold_mode="absolute",
            threshold=gbd_vorticity_floor * h * h * 1.0,
            max_nodes=100_000,
            core_radius_ratio=1.0,
        ),
    )
    vpm_samplers = [
        vpm.LineSampler(
            start=[0.6, 0.0, 0.0],
            end=[12.0, 0.0, 0.0],
            spacing=0.08,
            file_name="vpm_centreline",
            include_derivatives=False,
            schedule=vpm.EverySteps(5),
        )
    ]
    vpm_samplers.extend(
        vpm.LineSampler(
            start=[x, -3.0, 0.0],
            end=[x, 3.0, 0.0],
            spacing=0.08,
            file_name=f"vpm_transverse_x{x:g}",
            include_derivatives=False,
            schedule=vpm.EverySteps(5),
        )
        for x in (1.0, 2.0, 4.0)
    )
    vpm_case = replace(
        case.VPM_CASE,
        directory=output,
        numerics=numerics,
        samplers=vpm.Samplers(samples=tuple(vpm_samplers)),
        run=vpm.RunPlan(steps=round(end_time / coupling_dt)),
    )
    settings = replace(
        case.COUPLER_SETUP,
        interface_iterations=6,
        transfer_region_bounds=transfer,
        backup_interval_steps=round(output_steps),
        eta_blend_width=6 * h,
        vpm_only_width=2 * h,
        transfer_diagnostic_interval_steps=max(1, round(1 / coupling_dt)),
        transfer_vorticity_cutoff=transfer_cutoff,
    )
    manifest = {
        "geometry": "span-invariant cylinder Re150",
        "panel": False,
        "snapshot_interval": snapshot_interval,
        "span": 1.0,
        "span_layers": 4,
        "fvm_box": box,
        "transfer_box": transfer,
        "requested_dx": dx,
        "nominal_body_lattice": mesh.source.effective_cell_size(dx),
        "particle_spacing": h,
        "particle_strength_measure": "omega_z*h^2*span",
        "vorticity_floor": gbd_vorticity_floor,
        "absolute_strength_floor": gbd_vorticity_floor * h * h,
        "gbd_vorticity_floor": gbd_vorticity_floor,
        "transfer_vorticity_cutoff": transfer_cutoff,
        "transfer_absolute_strength_cutoff": transfer_cutoff * h * h,
        "fvm_dt": 0.004,
        "coupling_dt": coupling_dt,
        "end_time": end_time,
        "span_projection": False,
        "qualification": "Fresh trajectory; require common mature-time force/profile statistics, span invariance, interface and coupling-step sensitivity before acceptance.",
    }
    return fvm_setup, vpm_case, settings, mesh, manifest


def require_cube_gate(path):
    """Admit a physical run only after the verified four-second cube A/B gate."""
    if not path.is_file():
        raise RuntimeError(f"Cube admission gate is not available: {path}")
    gate = json.loads(path.read_text())
    criteria = gate.get("criteria", {})
    if not (
        gate.get("gate_passed") is True
        and criteria.get("verified_panel_free_experiment") is True
        and criteria.get("four_seconds_reached") is True
        and float(gate.get("latest_time", 0.0)) >= 4.0
    ):
        raise RuntimeError(
            "Cylinder execution requires a passing, verified panel-free cube comparison through4s"
        )
    return gate


def validate_native_mesh(path, manifest):
    """Do not label a replay as matched when its cached geometry differs."""
    with np.load(path) as native:
        generation = json.loads(str(native["metadata"]))["mesh_generation"]
        vertices = native["vertex_position"]
        bounds = np.column_stack((vertices.min(axis=0), vertices.max(axis=0))).ravel()
        spacing = generation["resolved_surface_patch_sizes"]["cylinder"]
        levels = generation["extrusion_levels"]
    if not np.allclose(bounds, manifest["fvm_box"], rtol=0, atol=1e-10):
        raise ValueError("Native replay mesh domain differs from the experiment")
    if not np.isclose(spacing, manifest["nominal_body_lattice"], rtol=0, atol=1e-10):
        raise ValueError("Native replay wall spacing differs from the experiment")
    expected = np.linspace(-manifest["span"] / 2, manifest["span"] / 2, manifest["span_layers"] + 1)
    if len(levels) != len(expected) or not np.allclose(levels, expected, rtol=0, atol=1e-10):
        raise ValueError("Native replay span/layers differ from the experiment")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--end-time", type=float, default=100.0)
    parser.add_argument(
        "--snapshot-interval",
        type=float,
        default=0.24,
        help="Coupled checkpoint and retained FVM/VPM frame interval in seconds",
    )
    parser.add_argument("--dx", type=float, default=0.04)
    parser.add_argument("--particle-spacing", type=float, default=0.05)
    parser.add_argument("--coupling-dt", type=float, default=0.04)
    parser.add_argument("--downstream", type=float, default=6.5)
    parser.add_argument("--half-height", type=float, default=3.5)
    parser.add_argument("--device", default="AUTO")
    parser.add_argument("--transfer-cutoff", type=float, default=0.05)
    parser.add_argument("--gbd-vorticity-floor", type=float, default=0.01)
    parser.add_argument(
        "--mesh", type=Path, help="Reuse an existing native mesh for an exact-mesh fresh replay"
    )
    parser.add_argument(
        "--max-coupling-steps",
        type=int,
        help="Stop and checkpoint after this many steps, retaining the configured horizon for restart",
    )
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--gate", type=Path, default=Path(__file__).with_name("cube_gate.json"))
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Validate and print configuration without meshing or solving",
    )
    args = parser.parse_args()
    if args.max_coupling_steps is not None and args.max_coupling_steps <= 0:
        parser.error("--max-coupling-steps must be positive")
    output = args.output.resolve()
    if output == CASE or output.is_relative_to(CASE / "reference_flow"):
        raise ValueError("Use a separate study directory, not tutorial/reference outputs")
    fv, vp, settings, mesh, manifest = configuration(
        output,
        cores=args.cores,
        end_time=args.end_time,
        dx=args.dx,
        particle_spacing=args.particle_spacing,
        coupling_dt=args.coupling_dt,
        downstream=args.downstream,
        half_height=args.half_height,
        device=args.device,
        transfer_cutoff=args.transfer_cutoff,
        gbd_vorticity_floor=args.gbd_vorticity_floor,
        snapshot_interval=args.snapshot_interval,
    )
    if args.describe:
        print(json.dumps(manifest, indent=2))
        return
    gate = require_cube_gate(args.gate)
    os.environ.setdefault("FVM_PROFILE", "0")
    manifest["cube_admission_gate"] = {
        "path": str(args.gate.resolve()),
        "latest_time": gate["latest_time"],
        "criteria": gate["criteria"],
    }
    cached = output / "solution/fvm/mesh.npz"
    if args.restart:
        if not cached.is_file():
            raise FileNotFoundError(f"Restart requires the original native mesh: {cached}")
        mesh = cached
    elif args.mesh is not None:
        mesh = args.mesh.resolve()
        if not mesh.is_file():
            raise FileNotFoundError(f"Native replay mesh is missing: {mesh}")
        manifest["native_mesh_source"] = str(mesh)
    if isinstance(mesh, Path):
        validate_native_mesh(mesh, manifest)
    with coupling.create_coupler(
        fv, vp, settings, mesh=mesh, case_dir=output, require_empty_output=not args.restart
    ) as solver:
        solver.initialize()
        if solver._is_master:
            transfer = solver.vorticity_transfer
            if not transfer._solid_bodies and transfer._body_bounds is None:
                raise RuntimeError("Panel-free cylinder must retain its native wall solid mask")
            if transfer._planar_induction.solid_at is None:
                raise RuntimeError("Planar diffusion must receive the same native-wall mask")
            output.joinpath("experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
        solver.run(
            restart_from=output / "solution/backups" if args.restart else None,
            max_coupling_steps=args.max_coupling_steps,
            backup_at_stop=True,
        )


if __name__ == "__main__":
    main()
