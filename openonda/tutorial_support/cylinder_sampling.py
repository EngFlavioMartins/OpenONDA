"""Physical observation schedules for the coupled and reference cylinder cases."""

import math

from openonda.cylinder_case import align_cylinder_sampling
import openonda.fvm as fvm
import openonda.vpm as vpm

FORCES = 0.04
PROFILES = 0.2
SLICES = 0.4
BACKUPS = 1.0
VOLUMES = 4.0


def steps(interval, dt):
    count = round(interval / dt)
    if count < 1 or abs(count * dt - interval) > 1e-10:
        raise ValueError("Every output must land on a physical accepted time")
    return count


def phase_lines(outer=False):
    """Identical coordinates in the reference and corresponding coupled region."""
    lo, hi, count = (1.6, 8.0, 65) if outer else (0.6, 1.4, 9)
    region = "wake" if outer else "near"
    return [
        (f"phase_{region}_{side}", [lo, y, 0.0], [hi, y, 0.0], count)
        for side, y in (("upper", 0.6), ("lower", -0.6))
    ]


def field_lines(reference, *, span):
    lines = [("centreline", [0.6, 0.0, 0.0], [1.4, 0.0, 0.0], 11)]
    for label, z in (("lower", -span / 4), ("middle", 0.0), ("upper", span / 4)):
        lines.append((f"span_{label}", [1.0, -1.2, z], [1.0, 1.2, z], 31))
    if reference:
        for x in (2, 4):
            lines.append((f"transverse_x{x}", [x, -2.0, 0.0], [x, 2.0, 0.0], 51))
    return lines


def sampling_plan(period, *, end, exchange_dt, fvm_time_step):
    """Resolve physical periods onto the actual accepted exchange clock."""
    steps(exchange_dt, fvm_time_step)
    steps(end, exchange_dt)
    return align_cylinder_sampling(
        end_time=end,
        exchange_dt=exchange_dt,
        fvm_time_step=fvm_time_step,
        sample_period=period,
        slice_period=SLICES,
        output_period=VOLUMES,
    )


def fvm_samplers(reference, *, span, freestream_speed, diameter, end, exchange_dt, fvm_time_step):
    clocks = {"end": end, "exchange_dt": exchange_dt, "fvm_time_step": fvm_time_step}
    force, profile = sampling_plan(FORCES, **clocks), sampling_plan(PROFILES, **clocks)
    fast = fvm.RunSchedule(every_n_steps=force.fvm_sample_steps)
    slow = fvm.RunSchedule(every_n_steps=profile.fvm_sample_steps)
    samplers = [
        fvm.ForceSampler(
            patch_names=["cylinder"],
            reference_velocity=freestream_speed,
            reference_area=diameter * span,
            reference_length=diameter,
            file_name="forces_history",
            schedule=fast,
        )
    ]
    for name, start, end, count in phase_lines() + (phase_lines(True) if reference else []):
        samplers.append(
            fvm.LineSampler(
                start=start,
                end=end,
                n_points=count,
                k=12,
                reconstruction="affine",
                file_name=name,
                schedule=fast,
            )
        )
    for name, start, end, count in field_lines(reference, span=span):
        samplers.append(
            fvm.LineSampler(
                start=start,
                end=end,
                n_points=count,
                k=12,
                reconstruction="affine",
                file_name=name,
                schedule=slow,
            )
        )
    samplers.append(
        fvm.SurfaceSampler(
            point=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            bounds=[
                -1.6,
                8.0 if reference else 1.6,
                -2.0 if reference else -1.6,
                2.0 if reference else 1.6,
            ],
            spacing=0.1,
            file_name="midspan",
            schedule=fvm.RunSchedule(every_n_steps=profile.fvm_slice_steps),
            body_bounds=[-0.5, 0.5, -0.5, 0.5, -6.0, 6.0],
            body_geometry="cylinder_z",
        )
    )
    return tuple(samplers)


def vpm_samplers(*, end, exchange_dt, fvm_time_step):
    """Particle-side observations shared by the ordinary tutorial entry point."""
    clocks = {"end": end, "exchange_dt": exchange_dt, "fvm_time_step": fvm_time_step}
    force, profile = sampling_plan(FORCES, **clocks), sampling_plan(PROFILES, **clocks)
    substeps = steps(exchange_dt, fvm_time_step)
    samples = []
    for name, start, finish, count in phase_lines() + phase_lines(True):
        samples.append(
            vpm.LineSampler(
                start=start,
                end=finish,
                spacing=math.dist(start, finish) / (count - 1) * (1 + 1e-12),
                file_name="vpm_" + name,
                schedule=vpm.EverySteps(force.sample_steps),
            )
        )
    for x in (2, 4):
        samples.append(
            vpm.LineSampler(
                start=[x, -2.0, 0.0],
                end=[x, 2.0, 0.0],
                spacing=0.08 * (1 + 1e-12),
                file_name=f"vpm_transverse_x{x}",
                schedule=vpm.EverySteps(profile.sample_steps),
            )
        )
    samples.append(
        vpm.SurfaceSampler(
            point=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            bounds=[1.6, 8.0, -2.0, 2.0],
            spacing=0.1,
            file_name="vpm_midspan",
            include_derivatives=False,
            schedule=vpm.EverySteps(profile.fvm_slice_steps // substeps),
        )
    )
    return tuple(samples)
