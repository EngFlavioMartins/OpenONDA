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

# Mesh generation lives with the FVM solver; re-exported here because the
# coupler and its tests historically imported it from this module.
from source.solvers.FVM.mesh.rectilinear import (  # noqa: F401
    coupling_box_mesh,
    wall_refined_axis,
)


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
    schemes=None,
    linear=None,
    pimple=None,
    forces=None,
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
        ForcesConfig,
        FVMConfig,
        LinearSolverConfig,
        PimpleControl,
        SchemesConfig,
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

    # Optional boundary-layer refinement toward the body faces (matches the
    # OFW/cfMesh case's near-cube cellSize).  Grades from wall_refinement_size
    # at the body to grid_spacing at the coupling faces, identically on every
    # axis; the reference case reuses wall_refined_axis over the shared region.
    nodes = None
    wr = getattr(cfg, "wall_refinement_size", None)
    if wr is not None and hole_box is not None:
        box = cfg.fvm_box
        ratio = float(getattr(cfg, "wall_refinement_ratio", 1.25))
        nodes = (
            wall_refined_axis(
                box[0], box[1], hole_box[0], hole_box[1], wr, cfg.grid_spacing, ratio
            ),
            wall_refined_axis(
                box[2], box[3], hole_box[2], hole_box[3], wr, cfg.grid_spacing, ratio
            ),
            wall_refined_axis(
                box[4], box[5], hole_box[4], hole_box[5], wr, cfg.grid_spacing, ratio
            ),
        )

    mesh_data = coupling_box_mesh(
        cfg.fvm_box,
        cfg.grid_spacing,
        cfg.patch_name,
        hole_box=hole_box,
        wall_patch_name=cfg.wall_patch_name or "cube",
        nodes=nodes,
    )

    u_inf = [float(v) for v in cfg.u_inf]
    execution = execution or ExecutionConfig()
    schemes = schemes or SchemesConfig(convection_scheme="central", gradient_scheme="lsq")
    linear = linear or LinearSolverConfig(
        linear_solver="bicgstab",
        pressure_solver="amg",
        ilu_drop_tol=1e-3,
        ilu_fill_factor=3.0,
    )
    pimple = pimple or PimpleControl(n_correctors=2)
    forces = forces or ForcesConfig()

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
        if forces.force_patches is None:
            forces.force_patches = [wall]
            side = np.asarray(hole_box, dtype=float)
            forces.ref_velocity = float(np.linalg.norm(u_inf)) or 1.0
            forces.ref_area = float((side[3] - side[2]) * (side[5] - side[4]))
            forces.ref_length = float(side[1] - side[0])
            forces.force_log_interval = 1

    fvm_config = FVMConfig(
        case_name=f"coupled_{cfg.patch_name}",
        execution=execution,
        time=time_cfg,
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        forces=forces,
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
