"""Two CPU coupling steps in uniform flow, with an exact constant solution."""

from pathlib import Path
import json

import numpy as np
from openonda import coupler, fvm, vpm

CASE_DIR = Path(__file__).resolve().parent
SPACING = 0.125
VELOCITY = (1.0, 0.0, 0.0)


def main():
    mesh = fvm.mesher.coupling_box_mesh((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), SPACING)
    flow = fvm.FVMSolver(
        fvm.FVMCase(
            name="uniform_flow",
            directory=CASE_DIR,
            mesh=mesh,
            numerics=fvm.Numerics(
                transport=fvm.TransportConfig(kinematic_viscosity=0.01),
            ),
            run=fvm.RunPlan(time_step_size=0.05, end_time=0.3),
            boundaries=(
                fvm.BoundaryConfig(
                    name="numericalBoundary",
                    velocity_type="fixedValue",
                    velocity_value=VELOCITY,
                    pressure_type="fixedFluxPressure",
                ),
            ),
            initial_conditions=fvm.InitialFields(velocity=VELOCITY),
        )
    )
    particles = vpm.VPMSolver(
        vpm.VPMCase(
            directory=CASE_DIR,
            numerics=vpm.Numerics(
                compute_device="CPU",
                max_n_particles=50_000,
                time_step_size=0.15,
                freestream_velocity=VELOCITY,
                domain_bounds=(-1, 1, -1, 1, -1, 1),
                viscous=vpm.ViscousConfig.cs(kinematic_viscosity=0.01, particle_spacing=SPACING),
            ),
        )
    )
    hybrid = coupler.create_coupler(
        flow,
        particles,
        coupler.CouplerSetup(
            freestream_velocity=VELOCITY,
            eta_blend_width=0.0,
            backup_interval_steps=2,
        ),
    )
    try:
        hybrid.run()
        velocity = np.asarray(flow.get_velocity_field())
        error = float(np.max(np.abs(velocity - VELOCITY)))
        if not np.isfinite(velocity).all() or error > 1e-6:
            raise RuntimeError(f"Uniform-flow velocity error is too large: {error}")
        if flow.step != 6 or particles.step != 2:
            raise RuntimeError("Coupling did not commit the requested substeps")
        report = {
            "fvm_steps": flow.step,
            "vpm_steps": particles.step,
            "time": flow.time,
            "max_velocity_error": error,
            "particles": particles.particles.n_particles_total,
        }
        (CASE_DIR / "solution/verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        flow.close()
        particles.close()


if __name__ == "__main__":
    main()
