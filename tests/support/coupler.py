"""Uniform-stream FVM and VPM cases for coupled-solver tests."""

from pathlib import Path

from openonda import fvm, vpm
from openonda.fvm.mesh.rectilinear import coupling_box_mesh

FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
KINEMATIC_VISCOSITY = 0.01


def uniform_stream_cases(
    directory: str | Path,
    *,
    fvm_time_step_size: float,
    vpm_time_step_size: float | None = None,
    end_time: float,
    particle_spacing: float = 0.25,
    cores: int = 1,
    fvm_output: fvm.Output | None = None,
    fvm_samplers: fvm.Samplers | None = None,
    vpm_output: vpm.Output | None = None,
    max_n_particles: int = 4000,
    fvm_time_step_control: fvm.MaximumCourantTimeStep | None = None,
) -> tuple[fvm.FVMCase, vpm.VPMCase]:
    """Return a unit FVM box in a uniform stream and its particle case.

    The FVM box spans [-0.5, 0.5]^3 with cubic cells of ``particle_spacing``;
    the particle domain spans [-1, 1]^3.
    """
    h = particle_spacing
    flow = fvm.FVMCase(
        name="uniform_stream",
        directory=directory,
        numerics=fvm.Numerics(
            time_step_size=fvm_time_step_size,
            time_step_control=fvm_time_step_control,
            transport=fvm.TransportConfig(kinematic_viscosity=KINEMATIC_VISCOSITY),
        ),
        initial_conditions=fvm.InitialFields(velocity=FREESTREAM_VELOCITY),
        run=fvm.RunPlan(end_time=end_time),
        output=fvm_output or fvm.Output(),
        samplers=fvm_samplers or fvm.Samplers(),
        mesh=lambda: coupling_box_mesh((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), h),
        boundaries=(
            fvm.BoundaryConfig(
                name="numericalBoundary",
                velocity_value=FREESTREAM_VELOCITY,
                pressure_type="fixed_flux_pressure",
            ),
        ),
        cores=cores,
    )
    particles = vpm.VPMCase(
        name="uniform_stream",
        directory=directory,
        numerics=vpm.Numerics(
            time_step_size=(
                fvm_time_step_size if vpm_time_step_size is None else vpm_time_step_size
            ),
            compute_device="cpu",
            max_n_particles=max_n_particles,
            domain_bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            freestream_velocity=FREESTREAM_VELOCITY,
            induction=vpm.DirectInduction(),
            viscous=vpm.ViscousConfig.gbd(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
                particle_spacing=h,
                threshold=1e-5,
                threshold_mode="absolute",
            ),
        ),
        run=vpm.RunPlan(end_time=end_time),
        output=vpm_output or vpm.Output(),
    )
    return flow, particles
