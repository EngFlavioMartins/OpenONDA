#!/usr/bin/env python3
"""Oracle boundary-condition replay for the coupled cubeFlow box.

Drives the production coupled-geometry FVM box with the EXACT boundary velocity
that ``referenceFlow`` recorded on the six faces of that box, instead of with a
velocity supplied by the VPM.  Everything else -- mesh, schemes, PIMPLE
controls, turbulence model, time step, samplers -- is the production coupled
configuration.

The point is to separate two things that a coupled run confounds:

    BC FORMULATION   is "merged Dirichlet U + fixedFluxPressure p on every
                     face" able to carry the solution at all?
    BC DATA          is what the VPM hands to that formulation accurate enough?

A mesh-matched lockstep study already answered the first at reduced resolution:
given exact data the formulation reproduces the truth to 0.59% in Cd, while a
~12% under-supplied displacement field costs +5.5% mean / +6.3% max.  The
production coupled run sits at +9.2% at t=2.4, so roughly a third of the error
was still unattributed.  This script closes that at PRODUCTION resolution with
REAL reference data rather than an emulated deficit.

It reads ``referenceFlow/samples/couplingFace_*.vts``, whose sampling cadence
(0.05 s) is deliberately the coupler's own VPM boundary-update interval, so the
replay sees exactly the temporal resolution the real coupling sees.

Usage (after referenceFlow has run far enough)::

    python scripts/experiments/cube_oracle_bc.py --t-end 2.5
    python scripts/experiments/cube_oracle_bc.py --check-only   # validate trace

``--flux-authority`` additionally prescribes the conservative face flux through
``set_external_face_flux_boundary_condition`` rather than letting the flux be
rebuilt from the interpolated velocity.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CUBE = ROOT / "tutorials/coupled_FVM_VPM/cubeFlow"
REF_SAMPLES = CUBE / "referenceFlow" / "samples"
PATCH = "numericalBoundary"

# --- production coupled configuration -------------------------------------
# Duplicated from cubeFlow_setup.py on purpose.  Importing a tutorial setup
# module runs its module-level bootstrap and can re-exec the real case under
# MPI; these few constants are cheaper to restate than that risk.  They are
# asserted against the tutorial in tests/coupler/test_cube_benchmark_parity.py.
CUBE_SIDE = 1.0
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
RHO = 1.0
NU = 1.0e-3
SMAGORINSKY_CK = 0.094
SMAGORINSKY_CE = 1.048
FVM_TIME_STEP_SIZE = 0.01
FVM_BOX = (-1.5, 3.5, -1.5, 1.5, -1.5, 1.5)
FVM_WAKE_BOX = (-1.25, 3.2, -1.25, 1.25, -1.25, 1.25)
FVM_CELL_SIZE = 0.0625
FVM_WAKE_CELL_SIZE = 0.03125
SURFACE_CELL_SIZE = 0.015625
SAMPLE_SPACING = 0.04
OFFAXIS_Y = 0.75 * CUBE_SIDE

FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
# Which box bound each face sits on, and the two in-plane axes, matching the
# SurfaceSampler convention (x-plane -> y,z; y-plane -> x,z; z-plane -> x,y).
FACE_SPEC = {
    "xmin": (0, 0, (1, 2)),
    "xmax": (0, 1, (1, 2)),
    "ymin": (1, 2, (0, 2)),
    "ymax": (1, 3, (0, 2)),
    "zmin": (2, 4, (0, 1)),
    "zmax": (2, 5, (0, 1)),
}


def _rank() -> int:
    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.Get_rank())
    except Exception:
        return 0


class FaceTrace:
    """The reference velocity on the six coupling-box faces, in time.

    Each face sampler writes a regular in-plane grid, so the spatial lookup is
    an exact bilinear interpolation rather than a scattered-point fit.  Times
    come from each sampler's ``.pvd`` index rather than from the file number:
    the archived reference ran at dt=0.0125 while the setup declares 0.01, and
    assuming a step-to-time factor is how that trap bites.
    """

    def __init__(self, sample_dir: Path):
        from scipy.interpolate import RegularGridInterpolator

        self.times: dict[str, np.ndarray] = {}
        self.interp: dict[str, list] = {}
        self.files: dict[str, list[Path]] = {}
        for face in FACES:
            pvd = sample_dir / f"couplingFace_{face}.pvd"
            if not pvd.exists():
                raise FileNotFoundError(
                    f"{pvd} not found -- has referenceFlow run with the "
                    "couplingFace samplers enabled?"
                )
            times, files = _read_pvd(pvd)
            self.times[face] = times
            self.interp[face] = [None] * len(files)
            self.files[face] = [sample_dir / f for f in files]
        self._rgi = RegularGridInterpolator
        n = min(len(v) for v in self.times.values())
        self.t_max = min(float(self.times[f][n - 1]) for f in FACES)
        print(
            f"[trace] {n} samples per face, t in "
            f"[{min(float(self.times[f][0]) for f in FACES):.3f}, {self.t_max:.3f}]",
            flush=True,
        )

    def _plane(self, face: str, index: int):
        """Bilinear interpolators (one per velocity component) for one snapshot."""
        cached = self.interp[face][index]
        if cached is not None:
            return cached
        import pyvista as pv

        grid = pv.read(self.files[face][index])
        ni, nj, _ = grid.dimensions
        pts = np.asarray(grid.points, dtype=float)
        vel = np.asarray(grid["Velocity"], dtype=float).reshape(-1, 3)
        a1, a2 = FACE_SPEC[face][2]
        # The writer lays points out with the first in-plane index fastest, and
        # ravels field data with order="F" to match (see SurfaceSampler.save_vts
        # and the OpenONDASurfaceOrdering marker).  Reading both flat and
        # reshaping order="F" keeps them consistent.
        c1 = pts[:, a1].reshape(ni, nj, order="F")[:, 0]
        c2 = pts[:, a2].reshape(ni, nj, order="F")[0, :]
        comps = [
            self._rgi(
                (c1, c2),
                vel[:, k].reshape(ni, nj, order="F"),
                bounds_error=False,
                fill_value=None,
            )
            for k in range(3)
        ]
        self.interp[face][index] = comps
        return comps

    def velocity(self, face: str, points: np.ndarray, t: float) -> np.ndarray:
        """Reference velocity at *points* on *face*, linear in time."""
        times = self.times[face]
        a1, a2 = FACE_SPEC[face][2]
        query = np.column_stack([points[:, a1], points[:, a2]])
        j = int(np.searchsorted(times, t))
        j = min(max(j, 1), len(times) - 1)
        t0, t1 = float(times[j - 1]), float(times[j])
        w = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        w = float(np.clip(w, 0.0, 1.0))
        lo = np.column_stack([f(query) for f in self._plane(face, j - 1)])
        hi = np.column_stack([f(query) for f in self._plane(face, j)])
        return (1.0 - w) * lo + w * hi


def _read_pvd(path: Path):
    root = ET.parse(path).getroot()
    entries = [
        (float(ds.attrib["timestep"]), ds.attrib["file"])
        for ds in root.iter("DataSet")
        if "timestep" in ds.attrib
    ]
    entries.sort()
    return np.array([e[0] for e in entries]), [e[1] for e in entries]


def route_faces(centres: np.ndarray) -> dict[str, np.ndarray]:
    """Assign each boundary-face centre to one box face (corners claimed once)."""
    claimed = np.zeros(len(centres), dtype=bool)
    out: dict[str, np.ndarray] = {}
    for face in FACES:
        axis, bound, _ = FACE_SPEC[face]
        hit = (~claimed) & (np.abs(centres[:, axis] - FVM_BOX[bound]) < 1.0e-6)
        out[face] = np.flatnonzero(hit)
        claimed |= hit
    if not claimed.all():
        raise RuntimeError(f"{int((~claimed).sum())} boundary faces matched no box face")
    return out


def project_to_solenoidal(u, normals, areas):
    """Uniform normal shift enforcing the discrete compatibility condition.

    The same minimum-disturbance correction the coupler applies, so the replay
    exercises the production path rather than a cleaner one.
    """
    total = float(np.sum(areas))
    eps = float(np.dot(np.einsum("ij,ij->i", u, normals), areas))
    return u - (eps / total) * normals


def build_setup(case_dir: Path, t_end: float, cores: int, cell_size: float):
    from openonda.fvm import (
        AdaptiveCartesianMesher,
        BoundaryConfig,
        BoxRefinement,
        ComputeConfig,
        DiscretizationConfig,
        ForceSampler,
        FVMSetup,
        LinearSolverConfig,
        LineSampler,
        OutputConfig,
        PimpleControl,
        SamplingSchedule,
        TimeConfig,
        TransportConfig,
        TurbulenceConfig,
    )

    schedule = SamplingSchedule(every_time=0.05)
    samplers = (
        ForceSampler(
            patch_names=["cube"],
            ref_velocity=1.0,
            ref_area=CUBE_SIDE**2,
            ref_length=CUBE_SIDE,
            moment_centre=[0.0, 0.0, 0.0],
            schedule=schedule,
        ),
        LineSampler(
            start=[FVM_BOX[0], OFFAXIS_Y, 0.0],
            end=[FVM_BOX[1], OFFAXIS_Y, 0.0],
            spacing=SAMPLE_SPACING,
            file_name="offaxis_y075",
            schedule=schedule,
        ),
        LineSampler(
            start=[FVM_BOX[0], 0.0, 0.0],
            end=[FVM_BOX[1], 0.0, 0.0],
            spacing=SAMPLE_SPACING,
            file_name="centerline",
            schedule=schedule,
        ),
    )
    mesh = AdaptiveCartesianMesher(
        domain=FVM_BOX,
        max_cell_size=cell_size,
        surface_file=CUBE / "assets" / "cube.stl",
        wall_patch_name="cube",
        surface_cell_size=cell_size / 4.0,
        refinements=(BoxRefinement(FVM_WAKE_BOX, cell_size / 2.0, "wakeBox"),),
        merge_outer_patch=PATCH,
    )
    setup = FVMSetup(
        case_name="oracle_bc",
        cores=cores,
        execution=ComputeConfig(operator_backend="numba"),
        output=OutputConfig(
            format="vtk_xml",
            data_location="cell",
            encoding="appended",
            compression="lz4",
            precision="float32",
            asynchronous=True,
            ghost_layers=0,
        ),
        time=TimeConfig(
            time_step_size=FVM_TIME_STEP_SIZE,
            start_time=0.0,
            end_time=t_end,
            output_interval_steps=10**9,
            output_interval_time=1.0e9,
            adjust_timestep=False,
        ),
        schemes=DiscretizationConfig(
            convection_scheme="linearUpwind",
            gradient_scheme="gauss",
            time_scheme="backward",
        ),
        linear=LinearSolverConfig(
            linear_solver="bicgstab",
            pressure_solver="amg",
            pressure_tolerance=1e-6,
            pressure_relative_tolerance=0.01,
            pressure_final_relative_tolerance=0.0,
            momentum_tolerance=1e-6,
            momentum_relative_tolerance=0.1,
            momentum_final_relative_tolerance=0.0,
            momentum_max_iterations=2000,
            ilu_drop_tolerance=1e-4,
            ilu_fill_factor=10.0,
            ilu_reuse_tolerance=0.05,
        ),
        pimple=PimpleControl(
            n_correctors=2,
            n_outer_correctors=2,
            n_orthogonal_correctors=1,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        samplers=samplers,
        transport=TransportConfig(density=RHO, kinematic_viscosity=NU),
        turbulence=TurbulenceConfig.equilibrium_smagorinsky(c_k=SMAGORINSKY_CK, c_e=SMAGORINSKY_CE),
        boundaries=[
            BoundaryConfig(
                name=PATCH,
                velocity_type="fixedValue",
                velocity_value=list(FREESTREAM_VELOCITY),
                pressure_type="fixedFluxPressure",
            ),
            BoundaryConfig.wall("cube"),
        ],
        initial_velocity=list(FREESTREAM_VELOCITY),
        initial_kinematic_pressure=0.0,
    )
    return setup, mesh


def check_only(trace: FaceTrace) -> None:
    """Round-trip the trace at its own sample points and report the residual."""
    import pyvista as pv

    worst = 0.0
    for face in FACES:
        grid = pv.read(trace.files[face][min(2, len(trace.files[face]) - 1)])
        pts = np.asarray(grid.points, dtype=float)
        exact = np.asarray(grid["Velocity"], dtype=float).reshape(-1, 3)
        t = float(trace.times[face][min(2, len(trace.times[face]) - 1)])
        got = trace.velocity(face, pts, t)
        err = float(np.abs(got - exact).max())
        worst = max(worst, err)
        print(f"  {face}: {len(pts):6d} pts  max|interp-exact| = {err:.3e}  (t={t:.3f})")
    print(f"[check] worst round-trip error {worst:.3e} (should be ~1e-12)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-end", type=float, default=2.5)
    ap.add_argument("--cores", type=int, default=4)
    # The mesher snaps the root to a size dividing all three box extents, which
    # for 5x3x3 means exactly 1/k.  The reference box (15x10x10) snaps to 5/88,
    # which is not of that form, so no cell size here reproduces the reference
    # mesh: 0.06 -> 1/17 (+3.5%), 0.0556 -> 1/18 (-2.2%).  Run both to bracket
    # it and interpolate the mesh contribution out of the comparison.
    ap.add_argument("--cell-size", type=float, default=FVM_CELL_SIZE)
    ap.add_argument("--flux-authority", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--case-dir", type=Path, default=CUBE / "oracleFlow")
    args = ap.parse_args()

    rank = _rank()
    if args.check_only:
        check_only(FaceTrace(REF_SAMPLES))
        return

    from openonda.fvm import create_fvm_solver

    trace = FaceTrace(REF_SAMPLES) if rank == 0 else None
    if rank == 0 and trace.t_max + 1e-9 < args.t_end:
        raise SystemExit(
            f"referenceFlow has only reached t={trace.t_max:.2f}; "
            f"asked for {args.t_end:.2f}. Let it run further or lower --t-end."
        )

    args.case_dir.mkdir(parents=True, exist_ok=True)
    setup, mesh = build_setup(args.case_dir, args.t_end, args.cores, args.cell_size)
    solver = create_fvm_solver(setup, case_dir=args.case_dir, mesh=mesh)

    centres = solver.get_boundary_face_center_coordinates(PATCH)
    normals = solver.get_boundary_face_normals(PATCH)
    areas = solver.get_boundary_face_areas(PATCH)
    routing = route_faces(centres) if rank == 0 else {}
    if rank == 0:
        print(
            f"[bc] {len(centres)} coupling faces, area={areas.sum():.3f}, "
            + " ".join(f"{f}={len(routing[f])}" for f in FACES),
            flush=True,
        )

    n_steps = int(round(args.t_end / FVM_TIME_STEP_SIZE))
    empty = np.zeros((0, 3), dtype=np.float64)
    for step in range(1, n_steps + 1):
        t = step * FVM_TIME_STEP_SIZE
        if rank == 0:
            u_bc = np.empty_like(centres)
            for face, idx in routing.items():
                if len(idx):
                    u_bc[idx] = trace.velocity(face, centres[idx], t)
            u_bc = project_to_solenoidal(
                np.ascontiguousarray(u_bc, dtype=np.float64), normals, areas
            )
        else:
            u_bc = empty
        solver.set_dirichlet_velocity_boundary_condition_vec(u_bc, PATCH)
        if args.flux_authority:
            # Prescribe the conservative face flux directly instead of letting
            # it be rebuilt from the interpolated velocity.
            flux = np.einsum("ij,ij->i", u_bc, normals) * areas if rank == 0 else np.zeros(0)
            solver.set_external_face_flux_boundary_condition(flux, PATCH)
        solver.solve_pimple()
        solver.advance_time()
        if rank == 0 and step % 25 == 0:
            print(f"[step {step:4d}/{n_steps}] t={t:5.2f}", flush=True)

    solver.close()
    if rank == 0:
        print(f"[done] samples in {args.case_dir / 'samples'}", flush=True)


if __name__ == "__main__":
    main()
