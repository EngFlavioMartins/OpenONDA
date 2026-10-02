"""Shared physical sampling and FVM controls for cylinder flow."""

from dataclasses import replace
import math

import openonda.fvm as fvm
import openonda.vpm as vpm
from openonda.cylinder_case import align_cylinder_sampling

SPAN = 0.96
DT = 0.008
EXCHANGE = 0.04
END = 100.0
FORCES = 0.04
PROFILES = 0.2
SLICES = 0.4
BACKUPS = 1.0
VOLUMES = 4.0


def steps(interval, dt=DT):
    count = round(interval / dt)
    if count < 1 or abs(count * dt - interval) > 1e-10:
        raise ValueError("Every output must land on a physical accepted time")
    return count


def phase_lines(outer=False):
    """Identical coordinates in the reference and corresponding coupled region."""
    lo, hi, count = (1.6, 8.0, 65) if outer else (0.6, 1.4, 9)
    region = "wake" if outer else "near"
    return [(f"phase_{region}_{side}", [lo, y, 0.], [hi, y, 0.], count)
            for side, y in (("upper", .6), ("lower", -.6))]


def field_lines(reference, *, span=SPAN):
    lines = [("centreline", [.6, 0., 0.], [1.4, 0., 0.], 11)]
    for label, z in (("lower", -span/4), ("middle", 0.), ("upper", span/4)):
        lines.append((f"span_{label}", [1., -1.2, z], [1., 1.2, z], 31))
    if reference:
        for x in (2, 4):
            lines.append((f"transverse_x{x}", [x, -2., 0.], [x, 2., 0.], 51))
    return lines


def _sampling(period, *, end=END, exchange_dt=EXCHANGE, fvm_time_step=DT):
    """Resolve physical periods onto the actual accepted exchange clock."""
    steps(exchange_dt, fvm_time_step)
    steps(end, exchange_dt)
    return align_cylinder_sampling(end_time=end, exchange_dt=exchange_dt,
        fvm_time_step=fvm_time_step, sample_period=period,
        slice_period=SLICES, output_period=VOLUMES)


def fvm_samplers(reference, *, span=SPAN, end=END, exchange_dt=EXCHANGE,
                 fvm_time_step=DT):
    clocks = dict(end=end, exchange_dt=exchange_dt, fvm_time_step=fvm_time_step)
    force, profile = _sampling(FORCES, **clocks), _sampling(PROFILES, **clocks)
    fast = fvm.RunSchedule(every_n_steps=force.fvm_sample_steps)
    slow = fvm.RunSchedule(every_n_steps=profile.fvm_sample_steps)
    samplers = [fvm.ForceSampler(patch_names=["cylinder"], reference_velocity=1.,
        reference_area=span, reference_length=1., file_name="forces_history", schedule=fast)]
    for name, start, end, count in phase_lines() + (phase_lines(True) if reference else []):
        samplers.append(fvm.LineSampler(start=start, end=end, n_points=count, k=12,
            reconstruction="affine", file_name=name, schedule=fast))
    for name, start, end, count in field_lines(reference, span=span):
        samplers.append(fvm.LineSampler(start=start, end=end, n_points=count, k=12,
            reconstruction="affine", file_name=name, schedule=slow))
    samplers.append(fvm.SurfaceSampler(point=[0., 0., 0.], normal=[0., 0., 1.],
        bounds=[-1.6, 8. if reference else 1.6, -2. if reference else -1.6,
                2. if reference else 1.6], spacing=.1, file_name="midspan",
        schedule=fvm.RunSchedule(every_n_steps=profile.fvm_slice_steps),
        body_bounds=[-.5, .5, -.5, .5, -6., 6.], body_geometry="cylinder_z"))
    return tuple(samplers)


def configure_fvm(setup, reference, end=END, *, span=SPAN, exchange_dt=EXCHANGE):
    """Apply the shared FVM controls and physical observation schedules."""
    dt = setup.time.time_step_size
    clocks = dict(end=end, exchange_dt=exchange_dt, fvm_time_step=dt)
    sampling = _sampling(PROFILES, **clocks)
    return replace(setup, samplers=fvm_samplers(reference, span=span, **clocks),
        schemes=fvm.DiscretizationConfig(convection_scheme="limitedLinear",
                                        gradient_scheme="lsq", time_scheme="backward"),
        pimple=fvm.PimpleControl(n_outer_correctors=2, n_correctors=2,
                                velocity_relaxation=.7, pressure_relaxation=.3),
        time=fvm.TimeConfig(time_step_size=dt, end_time=end,
            output_schedule=fvm.RunSchedule(every_n_steps=sampling.fvm_output_steps)),
        backup=fvm.BackupConfig(schedule=(fvm.RunSchedule(
            every_n_steps=_sampling(BACKUPS, **clocks).fvm_sample_steps)
                                         if reference else None), write_at_end=reference))


def vpm_samplers(*, end=END, exchange_dt=EXCHANGE, fvm_time_step=DT):
    """Particle-side observations shared by the ordinary tutorial entry point."""
    clocks = dict(end=end, exchange_dt=exchange_dt, fvm_time_step=fvm_time_step)
    force, profile = _sampling(FORCES, **clocks), _sampling(PROFILES, **clocks)
    substeps = steps(exchange_dt, fvm_time_step)
    samples = []
    for name, start, finish, count in phase_lines() + phase_lines(True):
        samples.append(vpm.LineSampler(start=start, end=finish,
            spacing=math.dist(start, finish)/(count-1)*(1+1e-12), file_name="vpm_" + name,
            schedule=vpm.EverySteps(force.sample_steps)))
    for x in (2, 4):
        samples.append(vpm.LineSampler(start=[x, -2., 0.], end=[x, 2., 0.],
            spacing=.08*(1+1e-12), file_name=f"vpm_transverse_x{x}",
            schedule=vpm.EverySteps(profile.sample_steps)))
    samples.append(vpm.SurfaceSampler(point=[0., 0., 0.], normal=[0., 0., 1.],
        bounds=[1.6, 8., -2., 2.], spacing=.1, file_name="vpm_midspan",
        include_derivatives=False, schedule=vpm.EverySteps(profile.fvm_slice_steps//substeps)))
    return tuple(samples)
