"""Build the native FVM backend used by the FVM–VPM coupler.

The generated mesh is a uniform hexahedral box with one coupling patch.
Execution is serial by default; ``ExecutionConfig.petsc_replicated()`` enables
collective PETSc linear solves with replicated NumPy assembly.
"""

from __future__ import annotations

import contextlib
import io
import os

import numpy as np


def _cells_along(lo: float, hi: float, spacing: float, axis: str) -> int:
    """Number of cells along one axis; the box must be an integer number of
    cells wide (the hand-off lattice and the fringe band assume a uniform,
    box-conforming grid)."""
    length = float(hi) - float(lo)
    if length <= 0.0:
        raise ValueError(f"fvm_box has non-positive extent along {axis}: {length}")
    n = length / float(spacing)
    n_int = int(round(n))
    if n_int < 1 or abs(n - n_int) > 1e-6 * max(n, 1.0):
        raise ValueError(
            f"fvm_box extent along {axis} ({length:g}) is not an integer "
            f"multiple of grid_spacing ({spacing:g}): {n:g} cells."
        )
    return n_int


def coupling_box_mesh(
    fvm_box: tuple[float, float, float, float, float, float],
    spacing: float,
    patch_name: str = "numericalBoundary",
) -> dict:
    """Return a uniform hex mesh whose six sides form one boundary patch."""
    x0, x1, y0, y1, z0, z1 = (float(v) for v in fvm_box)
    nx = _cells_along(x0, x1, spacing, "x")
    ny = _cells_along(y0, y1, spacing, "y")
    nz = _cells_along(z0, z1, spacing, "z")
    dx, dy, dz = (x1 - x0) / nx, (y1 - y0) / ny, (z1 - z0) / nz
    npx, npy = nx + 1, ny + 1

    # --- points -----------------------------------------------------------
    xs = x0 + dx * np.arange(nx + 1)
    ys = y0 + dy * np.arange(ny + 1)
    zs = z0 + dz * np.arange(nz + 1)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)

    def pid(i, j, k):
        return i + npx * (j + npy * k)

    def cid(i, j, k):
        return i + nx * (j + ny * k)

    def oriented_block(quads, desired_dir):
        """Flip the whole (M, 4) family if its uniform winding opposes desired_dir."""
        quads = np.ascontiguousarray(quads, dtype=np.int32)
        if quads.shape[0]:
            p = points[quads[0]]
            if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), desired_dir) < 0:
                quads = np.ascontiguousarray(quads[:, ::-1])
        return quads

    def x_family(i_arr):
        i, k, j = np.meshgrid(i_arr, np.arange(nz), np.arange(ny), indexing="ij")
        i, k, j = i.ravel(), k.ravel(), j.ravel()
        quads = np.stack(
            [pid(i, j, k), pid(i, j + 1, k), pid(i, j + 1, k + 1), pid(i, j, k + 1)], axis=1
        )
        return quads, i, j, k

    def y_family(j_arr):
        j, k, i = np.meshgrid(j_arr, np.arange(nz), np.arange(nx), indexing="ij")
        i, k, j = i.ravel(), k.ravel(), j.ravel()
        quads = np.stack(
            [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j, k + 1), pid(i, j, k + 1)], axis=1
        )
        return quads, i, j, k

    def z_family(k_arr):
        k, j, i = np.meshgrid(k_arr, np.arange(ny), np.arange(nx), indexing="ij")
        i, k, j = i.ravel(), k.ravel(), j.ravel()
        quads = np.stack(
            [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k)], axis=1
        )
        return quads, i, j, k

    ex, ey, ez = np.eye(3)

    # Interior families (owner = lower cell, neighbour = upper cell).
    qx, ix, jx, kx = x_family(np.arange(1, nx))
    qy, iy, jy, ky = y_family(np.arange(1, ny))
    qz, iz, jz, kz = z_family(np.arange(1, nz))
    interior_quads = np.vstack(
        [oriented_block(qx, ex), oriented_block(qy, ey), oriented_block(qz, ez)]
    )
    interior_owners = np.concatenate(
        [cid(ix - 1, jx, kx), cid(iy, jy - 1, ky), cid(iz, jz, kz - 1)]
    )
    interior_neighbours = np.concatenate([cid(ix, jx, kx), cid(iy, jy, ky), cid(iz, jz, kz)])

    # Boundary families in the fixed patch order xmin, xmax, ymin, ymax, zmin,
    # zmax, merged into the single coupling patch (owner = adjacent cell).
    q_x0, _, jb, kb = x_family(np.array([0]))
    o_x0 = cid(np.zeros_like(jb), jb, kb)
    q_x1, _, jb, kb = x_family(np.array([nx]))
    o_x1 = cid(np.full_like(jb, nx - 1), jb, kb)
    q_y0, ib, _, kb = y_family(np.array([0]))
    o_y0 = cid(ib, np.zeros_like(ib), kb)
    q_y1, ib, _, kb = y_family(np.array([ny]))
    o_y1 = cid(ib, np.full_like(ib, ny - 1), kb)
    q_z0, ib, jb, _ = z_family(np.array([0]))
    o_z0 = cid(ib, jb, np.zeros_like(ib))
    q_z1, ib, jb, _ = z_family(np.array([nz]))
    o_z1 = cid(ib, jb, np.full_like(ib, nz - 1))

    boundary_quads = np.vstack(
        [
            oriented_block(q_x0, -ex),
            oriented_block(q_x1, ex),
            oriented_block(q_y0, -ey),
            oriented_block(q_y1, ey),
            oriented_block(q_z0, -ez),
            oriented_block(q_z1, ez),
        ]
    )
    boundary_owners = np.concatenate([o_x0, o_x1, o_y0, o_y1, o_z0, o_z1])

    all_quads = np.vstack([interior_quads, boundary_quads])
    owners = np.concatenate([interior_owners, boundary_owners])
    n_interior = interior_quads.shape[0]

    boundary = [
        {
            "name": patch_name,
            "startFace": n_interior,
            "nFaces": boundary_quads.shape[0],
            "type": "patch",
        }
    ]

    return {
        "points": points,
        "faces": list(all_quads),
        "owners": np.asarray(owners, dtype=np.int32),
        "neighbours": np.asarray(interior_neighbours, dtype=np.int32),
        "boundary": boundary,
        "n_elements": nx * ny * nz,
        "n_faces": all_quads.shape[0],
        "n_interior_faces": n_interior,
        "n_points": points.shape[0],
    }


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


def build_fvm_backend(
    coupler_setup,
    *,
    solver_params=None,
    execution=None,
    write_interval_time: float | None = None,
    case_dir: str | None = None,
    quiet: bool = False,
):
    """Build a native FVM solver from a :class:`CouplerSetup`.

    ``write_interval_time=None`` disables automatic FVM output. Adaptive time
    stepping is disabled because the coupler requires an integer subcycle
    ratio.
    """
    from source.solvers.FVM import (
        BoundaryConfig,
        ExecutionConfig,
        FVMConfig,
        SolverParams,
        TimeConfig,
        TransportConfig,
    )
    from source.solvers.FVM.core.solver import Solver

    cfg = coupler_setup
    if cfg.backend != "fvm":
        raise ValueError(
            "build_fvm_backend expects CouplerSetup(backend='fvm'); got "
            f"backend={cfg.backend!r}. The 'ofw' backend is "
            "built from an OpenFOAM case via fvm_solver(case_dir)."
        )

    mesh_data = coupling_box_mesh(cfg.fvm_box, cfg.grid_spacing, cfg.patch_name)

    u_inf = [float(v) for v in cfg.u_inf]
    execution = execution or ExecutionConfig()
    if solver_params is None:
        solver_params = SolverParams.pimple(
            n_correctors=2,
            linear_solver=("bicgstab" if execution.linear_backend == "petsc" else "spsolve"),
            convection_scheme="central",
            gradient_scheme="lsq",
        )

    time_cfg = TimeConfig(
        delta_t=float(cfg.dt),
        end_time=float(cfg.t_end),
        write_interval=10**9,  # step-based writing off; time-based below
        write_interval_time=write_interval_time,
        adjust_timestep=False,  # coupler owns dt (integer sub-cycle ratio)
    )

    fvm_config = FVMConfig(
        case_name=f"coupled_{cfg.patch_name}",
        execution=execution,
        time=time_cfg,
        solver=solver_params,
        transport=TransportConfig(density=float(cfg.rho), nu=float(cfg.nu)),
        boundaries=[BoundaryConfig.freestream(cfg.patch_name, u_inf)],
        initial_U=u_inf,
    )

    root = os.path.abspath(case_dir if case_dir is not None else cfg.case_dir)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            solver = Solver(fvm_config, case_dir=root, mesh_data=mesh_data)
    else:
        solver = Solver(fvm_config, case_dir=root, mesh_data=mesh_data)

    solver.auto_write = write_interval_time is not None
    return solver
