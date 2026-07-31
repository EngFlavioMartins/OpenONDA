"""Native FVM backend for the FVM–VPM coupler."""

from __future__ import annotations

import contextlib
import io
import os

import numpy as np

from source.solvers.FVM.mesh.rectilinear import (  # noqa: F401
    coupling_box_mesh,
    wall_refined_axis,
)


def box_surface_markers(
    center,
    side_lengths,
    spacing: float,
) -> np.ndarray:
    """Return half-offset surface markers for all six faces of a box."""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    s = np.asarray(side_lengths, dtype=np.float64).reshape(-1)
    if s.size == 1:
        s = np.repeat(s, 3)
    if s.size != 3 or (s <= 0).any():
        raise ValueError(
            f"side_lengths must be a positive scalar or 3-vector, got {side_lengths!r}"
        )

    def _face_grid(axis: int, sign: float) -> np.ndarray:
        t1, t2 = [a for a in range(3) if a != axis]
        n1 = max(1, int(round(s[t1] / spacing)))
        n2 = max(1, int(round(s[t2] / spacing)))
        u = c[t1] - s[t1] / 2 + (np.arange(n1) + 0.5) * (s[t1] / n1)
        v = c[t2] - s[t2] / 2 + (np.arange(n2) + 0.5) * (s[t2] / n2)
        U, V = np.meshgrid(u, v, indexing="ij")
        pts = np.empty((n1 * n2, 3), dtype=np.float64)
        pts[:, axis] = c[axis] + sign * s[axis] / 2
        pts[:, t1] = U.ravel()
        pts[:, t2] = V.ravel()
        return pts

    return np.vstack([_face_grid(ax, sg) for ax in range(3) for sg in (-1.0, 1.0)])


def coupling_patch_boundaries(
    patch_name: str,
    u_inf,
) -> list:
    """Supported velocity and pressure conditions for the coupling patch."""
    from source.solvers.FVM import BoundaryConfig

    u_inf = [float(v) for v in u_inf]
    return [
        BoundaryConfig(
            name=patch_name,
            type_U="fixedValue",
            value_U=u_inf,
            type_p="fixedFluxPressure",
        )
    ]


def wall_patch_bounds(mesh_data: dict, wall_patch_name: str) -> np.ndarray:
    """Axis-aligned bounds (x0, x1, y0, y1, z0, z1) of a wall patch's faces."""
    (wall,) = [b for b in mesh_data["boundary"] if b["name"] == wall_patch_name]
    point_ids = np.unique(
        np.concatenate(
            [
                np.asarray(mesh_data["faces"][f]).ravel()
                for f in range(wall["startFace"], wall["startFace"] + wall["nFaces"])
            ]
        )
    )
    pts = np.asarray(mesh_data["points"])[point_ids]
    return np.array(
        [
            pts[:, 0].min(),
            pts[:, 0].max(),
            pts[:, 1].min(),
            pts[:, 1].max(),
            pts[:, 2].min(),
            pts[:, 2].max(),
        ]
    )


def build_fvm_backend(
    *,
    mesh_data: dict,
    case_dir: str,
    dt: float,
    t_end: float,
    nu: float,
    u_inf,
    rho: float = 1.0,
    patch_name: str = "numericalBoundary",
    wall_patch_name: str | None = None,
    initial_U=None,
    schemes=None,
    linear=None,
    pimple=None,
    forces=None,
    execution=None,
    write_interval_time: float | None = None,
    quiet: bool = False,
):
    """Build a complete native FVM solver for coupled use."""
    from source.solvers.FVM import (
        BoundaryConfig,
        ExecutionConfig,
        ForcesConfig,
        FVMSetup,
        LinearSolverConfig,
        PimpleControl,
        SchemesConfig,
        TimeConfig,
        TransportConfig,
    )
    from source.solvers.FVM.core.solver import Solver

    u_inf = [float(v) for v in u_inf]
    execution = execution or ExecutionConfig()
    schemes = schemes or SchemesConfig(
        convection_scheme="LUST",
        gradient_scheme="lsq",
        time_scheme="backward",
    )
    linear = linear or LinearSolverConfig(
        linear_solver="bicgstab",
        pressure_solver="amg",
        pressure_tol=1e-8,
        momentum_maxiter=2000,
        ilu_drop_tol=1e-4,
        ilu_fill_factor=10.0,
        ilu_reuse_tol=0.05,
    )
    pimple = pimple or PimpleControl(n_correctors=2, n_outer_correctors=2)
    time_cfg = TimeConfig(
        delta_t=float(dt),
        end_time=float(t_end),
        write_interval=10**9,
        write_interval_time=write_interval_time,
        adjust_timestep=False,
    )

    boundaries = coupling_patch_boundaries(patch_name, u_inf)
    if wall_patch_name is not None:
        boundaries.append(BoundaryConfig.wall(wall_patch_name))
        if forces is None:
            body = wall_patch_bounds(mesh_data, wall_patch_name)
            forces = ForcesConfig(
                force_patches=[wall_patch_name],
                ref_velocity=float(np.linalg.norm(u_inf)) or 1.0,
                ref_area=float((body[3] - body[2]) * (body[5] - body[4])),
                ref_length=float(body[1] - body[0]),
                force_log_interval=1,
            )
    forces = forces or ForcesConfig()

    fvm_config = FVMSetup(
        case_name=f"coupled_{patch_name}",
        execution=execution,
        time=time_cfg,
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        forces=forces,
        transport=TransportConfig(density=float(rho), nu=float(nu)),
        boundaries=boundaries,
        initial_U=u_inf if initial_U is None else [float(v) for v in initial_U],
    )

    root = os.path.abspath(case_dir)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            solver = Solver(fvm_config, case_dir=root, mesh_data=mesh_data)
    else:
        solver = Solver(fvm_config, case_dir=root, mesh_data=mesh_data)

    solver.auto_write = write_interval_time is not None
    return solver
