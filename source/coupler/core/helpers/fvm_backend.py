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


def _plane_index(
    value: float,
    origin: float,
    spacing: float,
    n: int,
    what: str,
    *,
    allow_boundary: bool = False,
) -> int:
    """Grid-plane index of a body face; the face must lie ON a mesh plane."""
    r = (float(value) - float(origin)) / float(spacing)
    idx = int(round(r))
    if abs(r - idx) > 1e-6 * max(abs(r), 1.0):
        raise ValueError(
            f"{what} at {value:g} does not lie on a mesh plane (spacing "
            f"{spacing:g} from origin {origin:g})"
        )
    lower = 0 if allow_boundary else 1
    upper = n if allow_boundary else n - 1
    if not lower <= idx <= upper:
        raise ValueError(f"{what} at {value:g} must map to a grid plane in [{lower}, {upper}]")
    return idx


def coupling_box_mesh(
    fvm_box: tuple[float, float, float, float, float, float],
    spacing: float,
    patch_name: str = "numericalBoundary",
    hole_box: tuple[float, float, float, float, float, float] | None = None,
    wall_patch_name: str = "cube",
) -> dict:
    """Return a uniform hex mesh whose six sides form one coupling patch.

    With ``hole_box``, the cells inside it are removed (body-fitted, like the
    OpenFOAM/cfMesh case): the exposed faces become a second boundary patch
    ``wall_patch_name`` of type ``wall``, where the no-slip condition is
    applied.  The hole faces must lie exactly on mesh planes.  Points inside
    the hole are kept but unreferenced (harmless to the solver and to VTK).
    """
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

    # ── Optional body: carve the hole and expose its faces as a wall patch ──
    wall_quads = np.empty((0, 4), dtype=np.int32)
    wall_owners = np.empty(0, dtype=np.int64)
    keep = np.ones(nx * ny * nz, dtype=bool)
    if hole_box is not None:
        hx0, hx1, hy0, hy1, hz0, hz1 = (float(v) for v in hole_box)
        i0 = _plane_index(hx0, x0, dx, nx, f"{wall_patch_name} x-min face", allow_boundary=True)
        i1 = _plane_index(hx1, x0, dx, nx, f"{wall_patch_name} x-max face", allow_boundary=True)
        j0 = _plane_index(hy0, y0, dy, ny, f"{wall_patch_name} y-min face", allow_boundary=True)
        j1 = _plane_index(hy1, y0, dy, ny, f"{wall_patch_name} y-max face", allow_boundary=True)
        k0 = _plane_index(hz0, z0, dz, nz, f"{wall_patch_name} z-min face", allow_boundary=True)
        k1 = _plane_index(hz1, z0, dz, nz, f"{wall_patch_name} z-max face", allow_boundary=True)
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            raise ValueError(f"hole_box {hole_box} has empty extent on the grid.")

        ci, cj, ck = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        in_hole = (ci >= i0) & (ci < i1) & (cj >= j0) & (cj < j1) & (ck >= k0) & (ck < k1)
        keep[cid(ci.ravel(), cj.ravel(), ck.ravel())] = ~in_hole.ravel()

        own_kept = keep[interior_owners]
        nb_kept = keep[interior_neighbours]
        both = own_kept & nb_kept
        # Wall faces sit between a fluid cell and a removed cell.  Interior
        # winding points from owner (lower cell) to neighbour (upper cell), so
        # it already points out of the fluid when the owner survives; when the
        # neighbour survives it becomes the owner and the ring is reversed.
        low_side = own_kept & ~nb_kept
        up_side = ~own_kept & nb_kept
        wall_quads = np.vstack([interior_quads[low_side], interior_quads[up_side][:, ::-1]]).astype(
            np.int32
        )
        wall_owners = np.concatenate([interior_owners[low_side], interior_neighbours[up_side]])
        interior_quads = interior_quads[both]
        interior_owners = interior_owners[both]
        interior_neighbours = interior_neighbours[both]
        outer_fluid = keep[boundary_owners]
        boundary_quads = boundary_quads[outer_fluid]
        boundary_owners = boundary_owners[outer_fluid]

    all_quads = np.vstack([interior_quads, boundary_quads, wall_quads])
    owners = np.concatenate([interior_owners, boundary_owners, wall_owners])
    n_interior = interior_quads.shape[0]

    # Compact cell numbering after the carve (identity when there is no hole).
    new_id = np.cumsum(keep) - 1
    owners = new_id[owners]
    neighbours = new_id[interior_neighbours]

    boundary = [
        {
            "name": patch_name,
            "startFace": n_interior,
            "nFaces": boundary_quads.shape[0],
            "type": "patch",
        }
    ]
    if hole_box is not None:
        boundary.append(
            {
                "name": wall_patch_name,
                "startFace": n_interior + boundary_quads.shape[0],
                "nFaces": wall_quads.shape[0],
                "type": "wall",
            }
        )

    return {
        "points": points,
        "faces": list(all_quads),
        "owners": np.asarray(owners, dtype=np.int32),
        "neighbours": np.asarray(neighbours, dtype=np.int32),
        "boundary": boundary,
        "n_elements": int(keep.sum()),
        "n_faces": all_quads.shape[0],
        "n_interior_faces": n_interior,
        "n_points": points.shape[0],
    }


def _body_hole_box(cfg) -> tuple[float, ...] | None:
    """Axis-aligned hole bounds from CouplerSetup's OFW-style body spec.

    Reads ``cfg.surface`` — e.g. ``{"cube": {"side_length": 1.0, "center":
    [0, 0, 0]}}`` — and returns ``(x0, x1, y0, y1, z0, z1)``.  Returns ``None``
    (no body; plain box) when ``wall_patch_name`` or ``surface`` is unset.
    Only box-shaped bodies are supported by the in-memory mesh generator;
    other shapes need the OFW backend (cfMesh) or a Gmsh mesh.
    """
    if not cfg.wall_patch_name or not cfg.surface:
        return None
    spec = None
    for key in (cfg.wall_patch_name, "cube", "box"):
        if key in cfg.surface:
            spec = cfg.surface[key]
            break
    if spec is None:
        raise ValueError(
            f"surface={cfg.surface!r} has no entry for wall patch "
            f"{cfg.wall_patch_name!r} (or 'cube'/'box')."
        )
    if "side_length" not in spec:
        raise ValueError(
            f"Body spec {spec!r} is not box-shaped ('side_length' missing). "
            "The native mesh generator only carves axis-aligned boxes; use "
            "the OFW backend (cfMesh) for general geometry."
        )
    side = np.asarray(spec["side_length"], dtype=float).reshape(-1)
    if side.size == 1:
        side = np.repeat(side, 3)
    center = np.asarray(spec.get("center", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
    lo = center - side / 2.0
    hi = center + side / 2.0
    return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


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

    # Body-fitted body from the same spec the OFW case uses: CouplerSetup's
    # ``surface`` (e.g. {"cube": {"side_length": 1.0, "center": [0,0,0]}}) and
    # ``wall_patch_name``.  The body is carved out of the mesh and its faces
    # form a no-slip wall patch, matching the OpenFOAM topology — cells inside
    # the body do not exist, so the coupler can never inject fictitious
    # interior vorticity into the VPM.
    hole_box = _body_hole_box(cfg)

    mesh_data = coupling_box_mesh(
        cfg.fvm_box,
        cfg.grid_spacing,
        cfg.patch_name,
        hole_box=hole_box,
        wall_patch_name=cfg.wall_patch_name or "cube",
    )

    u_inf = [float(v) for v in cfg.u_inf]
    execution = execution or ExecutionConfig()
    if solver_params is None:
        # bicgstab momentum + AMG (pyamg) pressure: ~60× faster per
        # solve_pimple than spsolve at 30³ cells.  Convergence is
        # residual-verified in linear_interface, so degenerate solves
        # (zero RHS, converged initial guess) no longer abort the run.
        solver_params = SolverParams.pimple(
            n_correctors=2,
            linear_solver="bicgstab",
            convection_scheme="central",
            gradient_scheme="lsq",
        )
        # Route the Poisson solve to pyamg-preconditioned CG explicitly:
        # pressure_solver=None would inherit bicgstab+ILU, whose per-solve ILU
        # factorization grows superlinearly with mesh size (measured 24 s per
        # pressure solve at 208k cells vs ~0.5 s with the cached AMG hierarchy).
        solver_params.pressure_solver = "amg"
        # Loose ILU: the transient momentum matrix is diagonally dominant, so
        # a cheap factorization preconditions it just as well (measured 15%
        # faster per step, lower memory, identical continuity).
        solver_params.ilu_drop_tol = 1e-3
        solver_params.ilu_fill_factor = 3.0

    time_cfg = TimeConfig(
        delta_t=float(cfg.dt),
        end_time=float(cfg.t_end),
        write_interval=10**9,  # step-based writing off; time-based below
        write_interval_time=write_interval_time,
        adjust_timestep=False,  # coupler owns dt (integer sub-cycle ratio)
    )

    # Coupling patch, selected by the coupler's donor_bc_mode:
    #
    # * dirichlet / mixed — Dirichlet velocity (the coupler overwrites the
    #   value each sub-step) + momentum-compatible fixed-flux pressure on ALL
    #   faces.  The donor trace is projected to zero net flux, so the
    #   all-Neumann pressure problem is compatible and the solver pins the
    #   level at a reference cell (pRefCell equivalent).
    #
    # * characteristic — donor velocity applied on INFLOW faces only, with
    #   convective (owner-extrapolated) outflow and the matching per-face
    #   freestream pressure (zero-gradient inflow / fixed p outflow).  The
    #   all-face Dirichlet cut couples the donor's Biot–Savart self-image to
    #   the box's own wake vorticity with loop gain ≥ 1 (measured secular
    #   face-deficit growth 0.9 → −3 U∞ and blow-up by t≈2, against a
    #   monolith truth of ≈0.89); letting the outflow state come from the
    #   FVM's own transport breaks that loop.
    if cfg.donor_bc_mode == "characteristic":
        boundaries = [
            BoundaryConfig(
                name=cfg.patch_name,
                type_U="freestream",
                value_U=u_inf,
                type_p="freestream",
                value_p=0.0,
            )
        ]
    else:
        boundaries = [
            BoundaryConfig(
                name=cfg.patch_name,
                type_U="fixedValue",
                value_U=u_inf,
                type_p="fixedFluxPressure",
            )
        ]
    if hole_box is not None:
        wall = cfg.wall_patch_name or "cube"
        boundaries.append(BoundaryConfig.wall(wall))
        # Wall-patch force integration (Cd/Cl in solution/forces_history.csv),
        # replacing the OFW case's OpenFOAM force function object.
        if solver_params.force_patches is None:
            solver_params.force_patches = [wall]
            side = np.asarray(hole_box, dtype=float)
            solver_params.ref_velocity = float(np.linalg.norm(u_inf)) or 1.0
            solver_params.ref_area = float((side[3] - side[2]) * (side[5] - side[4]))
            solver_params.ref_length = float(side[1] - side[0])
            solver_params.force_log_interval = 1

    fvm_config = FVMConfig(
        case_name=f"coupled_{cfg.patch_name}",
        execution=execution,
        time=time_cfg,
        solver=solver_params,
        transport=TransportConfig(density=float(cfg.rho), nu=float(cfg.nu)),
        boundaries=boundaries,
        initial_U=u_inf if cfg.initial_U is None else [float(v) for v in cfg.initial_U],
    )

    root = os.path.abspath(case_dir if case_dir is not None else cfg.case_dir)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            solver = Solver(fvm_config, case_dir=root, mesh_data=mesh_data)
    else:
        solver = Solver(fvm_config, case_dir=root, mesh_data=mesh_data)

    solver.auto_write = write_interval_time is not None
    return solver
