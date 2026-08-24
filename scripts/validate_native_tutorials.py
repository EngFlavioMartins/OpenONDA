#!/usr/bin/env python3
"""Run a lightweight installed-API FVM--VPM tutorial validation.

The parent process creates an isolated case directory and launches this file
again from that directory.  Consequently imports resolve through the installed
``openonda`` package, and every solver artifact is easy to audit and discard.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _worker(case_dir: Path, compute_device: str) -> None:
    import numpy as np

    import openonda.coupler as coupling
    import openonda.fvm as fvm
    import openonda.vpm as vpm

    fvm_time_step_size = 0.05
    vpm_time_step_size = 0.15
    spacing = 0.125
    freestream = [1.0, 0.0, 0.0]

    coupler_setup = coupling.CouplerSetup(
        freestream_velocity=freestream,
        vpm_particle_spacing=spacing,
        authority_ramp_width=2.0 * spacing,
        vpm_only_width=spacing,
        checkpoint_interval_steps=1,
    )
    line = fvm.LineSampler(
        start=[-0.25, 0.0, 0.0],
        end=[0.25, 0.0, 0.0],
        n_points=3,
        file_name="centreline",
        schedule=fvm.SamplingSchedule(every_n_steps=1),
    )
    fvm_setup = fvm.FVMSetup(
        case_name="native_tutorial_validation",
        time=fvm.TimeConfig(
            time_step_size=fvm_time_step_size,
            end_time=2.0 * vpm_time_step_size,
            output_interval_steps=10**9,
        ),
        transport=fvm.TransportConfig(kinematic_viscosity=0.01),
        boundaries=[
            fvm.BoundaryConfig(
                name="numericalBoundary",
                velocity_type="fixedValue",
                velocity_value=freestream,
                pressure_type="fixedFluxPressure",
            )
        ],
        samplers=(line,),
        initial_velocity=freestream,
    )
    mesh = fvm.coupling_box_mesh(
        (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
        spacing,
    )

    def make_fvm():
        solver = fvm.create_fvm_solver(fvm_setup, case_dir=case_dir, mesh=mesh)
        solver.auto_write = False
        return solver

    def make_vpm():
        setup = vpm.VPMSetup(
            time_step_size=vpm_time_step_size,
            compute_device=compute_device,
            max_n_particles=50_000,
            domain_bounds=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            freestream_velocity=freestream,
            checkpoint_directory=str(case_dir / "solution"),
        )
        return vpm.create_vpm_solver(setup, case_dir=case_dir)

    fvm_solver = make_fvm()
    vpm_solver = make_vpm()
    coupled = coupling.create_coupler(fvm_solver, vpm_solver, coupler_setup)
    coupled.run()

    velocity = np.asarray(fvm_solver.get_velocity_field())
    if not np.isfinite(velocity).all():
        raise RuntimeError("coupled tutorial produced non-finite FVM velocity")
    if not np.allclose(velocity.mean(axis=0), freestream, atol=1.0e-6):
        raise RuntimeError("uniform freestream was not preserved")
    if (fvm_solver.step, vpm_solver.step) != (6, 2):
        raise RuntimeError(
            f"unexpected subcycling result: FVM={fvm_solver.step}, VPM={vpm_solver.step}"
        )

    solution = case_dir / "solution"
    samples = case_dir / "samples"
    required = (
        solution / "fvm.log",
        solution / "coupler.log",
        samples / "centreline.csv",
        solution / "checkpoints" / "manifest.json",
    )
    missing = [str(path.relative_to(case_dir)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing tutorial outputs: {', '.join(missing)}")

    checkpoint = solution / "checkpoints"
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    expected_artifacts = {
        "fvm": "fvm_000002.npz",
        "vpm": "vpm_000002.h5",
        "vpm_xdmf": "vpm_000002.xdmf",
        "vpm_boundary_condition": "vpm_boundary_condition_000002.npz",
    }
    if manifest.get("kind") != "openonda.coupled_checkpoint":
        raise RuntimeError(f"unexpected checkpoint kind: {manifest.get('kind')!r}")
    if manifest.get("artifacts") != expected_artifacts:
        raise RuntimeError(f"unexpected checkpoint filenames: {manifest.get('artifacts')!r}")
    if set(manifest.get("artifact_sha256", {})) != set(expected_artifacts):
        raise RuntimeError("checkpoint manifest has incomplete artifact hashes")
    if not all((checkpoint / name).is_file() for name in expected_artifacts.values()):
        raise RuntimeError("checkpoint manifest references a missing artifact")
    if list(checkpoint.glob("*_000001*")):
        raise RuntimeError("latest-only checkpoint retention left a stale generation")

    expected_u = fvm_solver.velocity.copy()
    expected_p = fvm_solver.kinematic_pressure.copy()
    expected_flux = fvm_solver.volumetric_face_flux.copy()
    restored = coupling.create_coupler(make_fvm(), make_vpm(), coupler_setup)
    restored.initialize()
    restored_step = restored.load_state(checkpoint)
    if restored_step != 2:
        raise RuntimeError(f"restart restored coupled step {restored_step}, expected 2")
    np.testing.assert_allclose(restored.fvm_solver.velocity, expected_u, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        restored.fvm_solver.kinematic_pressure, expected_p, rtol=0.0, atol=1.0e-13
    )
    np.testing.assert_allclose(
        restored.fvm_solver.volumetric_face_flux, expected_flux, rtol=0.0, atol=1.0e-13
    )

    if (case_dir / "constant").exists() or (case_dir / "system").exists():
        raise RuntimeError("legacy external-solver case artifacts were created")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--compute-device",
        choices=("AUTO", "CPU", "METAL", "VULKAN", "CUDA"),
        default="CPU",
        help="Taichi backend for the VPM half of the validation (default: CPU)",
    )
    args = parser.parse_args(argv)
    if args.worker is not None:
        _worker(args.worker.resolve(), args.compute_device)
        return 0

    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="openonda-native-tutorial-") as temporary:
        case_dir = Path(temporary).resolve()
        environment = os.environ.copy()
        environment["OPENONDA_COMPUTE_DEVICE"] = args.compute_device
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--worker",
                str(case_dir),
                "--compute-device",
                args.compute_device,
            ],
            cwd=case_dir,
            env=environment,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"Native tutorial validation failed in isolated case {case_dir}",
                file=sys.stderr,
            )
            return result.returncode
    print(f"Native installed-API FVM--VPM tutorial validation passed ({args.compute_device})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
