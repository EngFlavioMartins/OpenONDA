"""Bounded cube benchmark; reuse the tutorial mesh and write only to --output."""

from __future__ import annotations

import argparse
import cProfile
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter

from openonda.runtime import RunConfig
from openonda.tutorial_runner import load_case_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    case_dir = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/02_cube_flow"
    case = load_case_module(case_dir)
    RunConfig(cpu_cores=case.FVM_SETUP.cores, parallel_mode="mpi").ensure_runtime(sys.argv[0])
    output = args.output.resolve()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        output.mkdir(parents=True, exist_ok=False)
    comm.Barrier()
    if args.reference:
        reference = load_case_module(case_dir / "reference_flow")
        original_factory = reference.fvm.create_fvm_solver

        class SavedMesh:
            def build(self):
                from source.solvers.fvm.io.mesh_storage import load_native_mesh

                return load_native_mesh(case_dir / "reference_flow/solution/fine/mesh.npz")

        def isolated_factory(setup, **kwargs):
            setup = replace(
                setup, time=replace(setup.time, end_time=args.steps * case.VPM_TIME_STEP_SIZE)
            )
            kwargs.update(
                case_dir=output,
                solution_dir=output / "solution",
                samples_dir=output / "samples",
                mesh=SavedMesh(),
            )
            return original_factory(setup, **kwargs)

        reference.fvm.create_fvm_solver = isolated_factory
        try:
            fvm = reference.create_solver("fine", case.REFERENCE_FINE_DX)
        finally:
            reference.fvm.create_fvm_solver = original_factory
        try:
            profile = cProfile.Profile() if args.profile else None
            if profile is not None:
                profile.enable()
            try:
                fvm.run()
            finally:
                if profile is not None:
                    profile.disable()
                    profile.dump_stats(str(output / f"profile-rank-{comm.Get_rank()}.pstats"))
        finally:
            fvm.close()
        return
    mesh = case.msh.CachedMesh(case.FVM_MESH, case_dir / "constant/mesh.npz")
    fvm = case.fvm.create_fvm_solver(case.FVM_SETUP, case_dir=output, mesh=mesh)
    qualities = comm.gather(fvm.mesh_quality, root=0)
    if comm.Get_rank() == 0:
        (output / "mesh-quality.json").write_text(json.dumps(qualities, indent=2) + "\n")
    vpm = None
    profile = cProfile.Profile() if args.profile else None
    original_solve = fvm.solve_pimple

    def measured_solve(*positional, **keywords):
        fvm.profiler.begin_step(
            step=fvm.step + 1,
            time=fvm.time + fvm.time_step_size,
            time_step_size=fvm.time_step_size,
        )
        start = perf_counter()
        try:
            return original_solve(*positional, **keywords)
        finally:
            fvm.profiler.finish_step(
                perf_counter() - start,
                getattr(fvm.algorithm, "last_linear_results", ()),
            )

    fvm.solve_pimple = measured_solve
    try:
        if fvm.parallel.is_root:
            vpm = case.vpm.VPMSolver(replace(case.VPM_CASE, directory=output))
        coupler = case.coupling.create_coupler(fvm, vpm, case.COUPLER_SETUP)
        if profile is not None:
            profile.enable()
        coupler.run(max_coupling_steps=args.steps, backup_at_stop=True)
    finally:
        if profile is not None:
            profile.disable()
            profile.dump_stats(str(output / f"profile-rank-{comm.Get_rank()}.pstats"))
        fvm.close()
        if vpm is not None:
            vpm.close()


if __name__ == "__main__":
    main()
