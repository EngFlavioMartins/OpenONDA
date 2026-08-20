#!/usr/bin/env python3
"""Numerical-equivalence baseline capture/compare for the nomenclature refactor.

Runs a small set of deterministic CPU cases and stores the *actual solver
state* (field arrays, particle state) as ``.npz`` snapshots, keyed by a case
name.  The same script runs against the pre-refactor API and the canonical API
via a tiny compatibility shim, so the two snapshots can be diffed array-by-array.

Usage
-----
    python scripts/nomenclature/numerical_baseline.py capture --outdir <dir>
    python scripts/nomenclature/numerical_baseline.py compare --old <dir> --new <dir>

Cases
-----
* ``fvm_taylor_green``  2D periodic Taylor-Green, central/BDF2, 10 steps.
* ``fvm_lid_cavity``    cubic lid-driven cavity, SIMPLE, 5 steps.
* ``vpm_two_particle``  two Gaussian blobs, advection only, 20 steps (CPU).
* ``coupled_smoke``     two coupled FVM-VPM steps on a uniform freestream.

The snapshots hold full arrays, not log text, and are written with
deterministic float64 output.  GPU backends are never used.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np

import openonda.coupler as coupler_api
import openonda.fvm as fvm_api
import openonda.vpm as vpm_api


# ---------------------------------------------------------------------------
# API compatibility shim: pre-refactor (349d4b8) vs canonical names.
# ---------------------------------------------------------------------------
def _resolve_fvm():
    if hasattr(fvm_api, "FVMSolver"):
        return fvm_api.FVMSolver, fvm_api.create_fvm_solver, "time_step_size"
    return fvm_api.Solver, fvm_api.setup_fvm_solver, "delta_t"


def _resolve_vpm():
    if hasattr(vpm_api, "VPMSolver"):
        return vpm_api.VPMSolver, vpm_api.create_vpm_solver
    return vpm_api.Solver, vpm_api.setup_vpm_solver


def _resolve_coupler():
    if hasattr(coupler_api, "create_coupler"):
        return coupler_api.create_coupler, coupler_api.FVMVPMCoupler
    return coupler_api.setup_coupler, coupler_api.FVMVPMCoupler


_FVM_SOLVER, _FVM_FACTORY, _TIME_KW = _resolve_fvm()
_VPM_SOLVER, _VPM_FACTORY = _resolve_vpm()
_COUPLER_FACTORY, _COUPLER_CLS = _resolve_coupler()


def _fvmsetup(**kwargs):
    return fvm_api.FVMSetup(**kwargs)


def _phi_from_U(solver) -> np.ndarray:
    from source.solvers.FVM.assemble import convection

    return convection.compute_volumetric_face_flux(solver.U, solver.mesh_data, solver.geo_data)


def _timeconfig(**kwargs):
    return fvm_api.TimeConfig(**{_TIME_KW: kwargs.pop("time_step_size"), **kwargs})


def _structured_box(nx, ny, nz, lx=1.0, ly=1.0, lz=1.0):
    from source.solvers.FVM.mesh.cartesian import structured_box as _sb

    return _sb(nx, ny, nz, lx=lx, ly=ly, lz=lz)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def case_fvm_taylor_green():
    TWO_PI = 2.0 * np.pi
    nu = 0.1
    nsteps = 10
    mesh = _structured_box(16, 16, 1, lx=TWO_PI, ly=TWO_PI, lz=TWO_PI)
    cfg = _fvmsetup(
        case_name="baseline-tgv",
        time=_timeconfig(time_step_size=0.005, end_time=0.005 * nsteps, write_interval=10**9),
        schemes=fvm_api.SchemesConfig(convection_scheme="central", time_scheme="backward"),
        linear=fvm_api.LinearSolverConfig(linear_solver="spsolve"),
        pimple=fvm_api.PimpleControl(n_correctors=2, n_outer_correctors=1),
        transport=fvm_api.TransportConfig(density=1.0, nu=nu),
        boundaries=[
            fvm_api.BoundaryConfig.cyclic("xmin", "xmax"),
            fvm_api.BoundaryConfig.cyclic("xmax", "xmin"),
            fvm_api.BoundaryConfig.cyclic("ymin", "ymax"),
            fvm_api.BoundaryConfig.cyclic("ymax", "ymin"),
            fvm_api.BoundaryConfig.empty("zmin"),
            fvm_api.BoundaryConfig.empty("zmax"),
        ],
        initial_velocity=[0.0, 0.0, 0.0],
    )
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        solver = _FVM_SOLVER(cfg, case_dir=d, mesh_data=mesh)
        solver.auto_write = False
        x = mesh["points"][: mesh["n_elements"], 0]
        y = mesh["points"][: mesh["n_elements"], 1]
        solver.U[: mesh["n_elements"]] = np.column_stack(
            [np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y), np.zeros_like(x)]
        )
        solver.U_old[:] = solver.U
        solver.U_old_old[:] = solver.U
        solver.phi = _phi_from_U(solver)
        ke = []
        for _ in range(nsteps):
            solver.advance()
            owned = solver.U[: mesh["n_elements"]]
            ke.append(float(0.5 * np.mean(np.sum(owned**2, axis=1))))
        n = mesh["n_elements"]
        return {
            "U": solver.U[:n],
            "p": solver.p[:n],
            "phi": solver.phi,
            "ke": np.asarray(ke, dtype=np.float64),
            "time": np.float64(solver.time),
            "step": np.int64(solver.step),
        }


def case_fvm_lid_cavity():
    mesh = _structured_box(8, 8, 8)
    cfg = _fvmsetup(
        case_name="baseline-lid",
        time=_timeconfig(time_step_size=0.01, end_time=0.05, write_interval=10**9),
        schemes=fvm_api.SchemesConfig(convection_scheme="limitedLinear"),
        linear=fvm_api.LinearSolverConfig(linear_solver="spsolve"),
        pimple=fvm_api.PimpleControl(algorithm="SIMPLE", alpha_u=0.7, alpha_p=0.3),
        transport=fvm_api.TransportConfig(density=1.0, nu=1.0e-3),
        boundaries=[
            fvm_api.BoundaryConfig.wall("xmin"),
            fvm_api.BoundaryConfig.wall("xmax"),
            fvm_api.BoundaryConfig.wall("ymin"),
            fvm_api.BoundaryConfig(
                "ymax", type_velocity="fixedValue", value_velocity=[1.0, 0.0, 0.0]
            ),
            fvm_api.BoundaryConfig.wall("zmin"),
            fvm_api.BoundaryConfig.wall("zmax"),
        ],
        initial_velocity=[0.0, 0.0, 0.0],
    )
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        solver = _FVM_SOLVER(cfg, case_dir=d, mesh_data=mesh)
        solver.auto_write = False
        for _ in range(5):
            solver.advance()
        n = mesh["n_elements"]
        return {
            "U": solver.U[:n],
            "p": solver.p[:n],
            "phi": solver.phi,
            "time": np.float64(solver.time),
            "step": np.int64(solver.step),
        }


def case_vpm_two_particle():
    sigma = 0.15
    volume = (4.0 / 3.0) * np.pi * sigma**3
    setup = vpm_api.VPMSetup(
        time_step_size=0.01,
        processing_unit="CPU",
        max_particles=1000,
        particles_kernel="GAUSSIAN",
        stretching=vpm_api.StretchingConfig.disabled(),
        viscous=vpm_api.ViscousConfig(scheme="NONE"),
        advection=vpm_api.AdvectionConfig(scheme="NONE"),
        max_targets=64,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = _VPM_SOLVER(setup=setup)
        solver.add_vortex_particles(
            position=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            velocity=np.zeros((2, 3)),
            circulation=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            radius=np.full(2, sigma),
            volume=np.full(2, volume),
            viscosity=np.zeros(2),
        )
        for _ in range(20):
            solver.advance()
        particles = solver.particles
        return {
            "position": particles.position.to_numpy(),
            "circulation": particles.circulation.to_numpy(),
            "velocity": particles.velocity.to_numpy(),
            "vorticity": particles.vorticity.to_numpy(),
            "radius": particles.radius.to_numpy(),
            "n_particles": np.int64(particles.number_of_particles),
            "time": np.float64(solver.time),
            "step": np.int64(solver.step),
        }


def case_coupled_smoke():
    FVM_DT = 0.05
    VPM_DT = 0.15
    H = 0.25
    setup = coupler_api.CouplerSetup(
        freestream_velocity=[1.0, 0.0, 0.0],
        vpm_particle_spacing=H,
        overlap_zone_ramp_width=2 * H,
        overlap_zone_dead_zone_width=H,
    )
    from source.solvers.FVM.mesh.rectilinear import coupling_box_mesh

    vpm_setup = vpm_api.VPMSetup(
        time_step_size=VPM_DT,
        processing_unit="CPU",
        max_particles=10000,
        vpm_domain_bounds=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        freestream_velocity=[1.0, 0.0, 0.0],
    )
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        vpm = _VPM_SOLVER(setup=vpm_setup)
        fvm_cfg = _fvmsetup(
            case_name="baseline-coupled",
            time=_timeconfig(time_step_size=FVM_DT, end_time=2 * VPM_DT),
            transport=fvm_api.TransportConfig(nu=0.01),
            boundaries=[
                fvm_api.BoundaryConfig(
                    name="numericalBoundary",
                    type_velocity="fixedValue",
                    value_velocity=setup.freestream_velocity,
                    type_p="fixedFluxPressure",
                )
            ],
            initial_velocity=setup.freestream_velocity,
        )
        fvm = _FVM_SOLVER(
            fvm_cfg,
            case_dir=d,
            mesh_data=coupling_box_mesh((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), H),
        )
        if _COUPLER_FACTORY is coupler_api.create_coupler:
            coupled = coupler_api.create_coupler(fvm, vpm, setup)
        else:
            coupled = coupler_api.setup_coupler(vpm, fvm, setup)
        coupled.run()
        particles = vpm.particles
        n = int(vpm.particles.number_of_particles)
        return {
            "U": fvm.U[: fvm.mesh_data["n_elements"]],
            "p": fvm.p[: fvm.mesh_data["n_elements"]],
            "phi": fvm.phi,
            "vpm_position": particles.position.to_numpy()[:n],
            "vpm_circulation": particles.circulation.to_numpy()[:n],
            "vpm_n": np.int64(n),
            "fvm_step": np.int64(fvm.step),
            "vpm_step": np.int64(vpm.step),
            "fvm_time": np.float64(fvm.time),
            "vpm_time": np.float64(vpm.time),
        }


_CASES = {
    "fvm_taylor_green": case_fvm_taylor_green,
    "fvm_lid_cavity": case_fvm_lid_cavity,
    "vpm_two_particle": case_vpm_two_particle,
    "coupled_smoke": case_coupled_smoke,
}


def _capture(outdir: Path, cases: list[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in cases:
        runner = _CASES[name]
        print(f"[baseline] capturing {name} ...", flush=True)
        data = runner()
        case_dir = outdir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        import hashlib

        sha = {}
        for key, value in data.items():
            array = np.asarray(value)
            path = case_dir / f"{key}.npy"
            np.save(path, array)
            sha[key] = hashlib.sha256(array.tobytes()).hexdigest()[:16]
        manifest[name] = sha
        print(f"[baseline] {name}: {json.dumps(sha)}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[baseline] wrote {len(cases)} cases to {outdir}")


def _compare(old_dir: Path, new_dir: Path) -> int:
    old_manifest = json.loads((old_dir / "manifest.json").read_text())
    new_manifest = json.loads((new_dir / "manifest.json").read_text())
    failures = 0
    for name in sorted(old_manifest):
        if name not in new_manifest:
            print(f"[compare] MISSING case in new: {name}")
            failures += 1
            continue
        for key in sorted(old_manifest[name]):
            if key not in new_manifest[name]:
                print(f"[compare] MISSING array {name}/{key}")
                failures += 1
                continue
            old = np.load(old_dir / name / f"{key}.npy")
            new = np.load(new_dir / name / f"{key}.npy")
            if old.shape != new.shape:
                print(f"[compare] SHAPE MISMATCH {name}/{key}: {old.shape} vs {new.shape}")
                failures += 1
                continue
            if old.dtype.kind in "iu" or old.dtype == np.bool_:
                ok = bool(np.array_equal(old, new))
                tol = ""
            else:
                max_diff = float(np.max(np.abs(old - new))) if old.size else 0.0
                scale = float(np.max(np.abs(old))) if old.size else 1.0
                ok = max_diff <= 1.0e-11 * max(scale, 1.0)
                tol = f" max_diff={max_diff:.3e}"
            status = "ok" if ok else "FAIL"
            if not ok:
                failures += 1
            print(f"[compare] {name:20s} {key:16s} {status}{tol}")
    print(f"[compare] {'PASS' if failures == 0 else f'{failures} FAILURES'}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture")
    cap.add_argument("--outdir", type=Path, required=True)
    cap.add_argument("--cases", nargs="*", default=list(_CASES))

    cmp = sub.add_parser("compare")
    cmp.add_argument("--old", type=Path, required=True)
    cmp.add_argument("--new", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "capture":
        _capture(args.outdir, args.cases)
    else:
        raise SystemExit(_compare(args.old, args.new))


if __name__ == "__main__":
    main()
