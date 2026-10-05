"""Validation and integer schedules for explicitly supplied case inputs."""

import argparse
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class CylinderSampling:
    """Integer schedules shared by the FVM and VPM exchange clocks."""

    sample_steps: int
    fvm_sample_steps: int
    fvm_slice_steps: int
    fvm_output_steps: int


def align_cylinder_sampling(
    *, end_time, exchange_dt, fvm_time_step, sample_period, slice_period, output_period
) -> CylinderSampling:
    """Round physical sampling cadence to exchange steps, preferring finer ties.

    Short runs clamp the interval to their length. Periodic sampling need not
    land on an off-cadence destination; final-only schedules remain separate.
    """
    total_exchanges = round(end_time / exchange_dt)
    substeps = round(exchange_dt / fvm_time_step)
    desired_steps = sample_period / exchange_dt
    sample_steps = max(1, min(total_exchanges, math.ceil(desired_steps - 0.5)))
    return CylinderSampling(
        sample_steps=sample_steps,
        fvm_sample_steps=sample_steps * substeps,
        fvm_slice_steps=max(1, math.ceil(slice_period / exchange_dt)) * substeps,
        fvm_output_steps=max(1, math.ceil(output_period / exchange_dt)) * substeps,
    )


def steps(interval, dt):
    count = round(interval / dt)
    if count < 1 or abs(count * dt - interval) > 1e-10:
        raise ValueError("Every output must land on a physical accepted time")
    return count


def positive_coupling_steps(value: str) -> int:
    """Require a positive accepted-exchange limit."""
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def validate_inputs(
    hxy, span, dz, spacing_ratio, core_ratio, exchange_dt, end, release, blend, cores, particles
):
    if min(hxy, span, dz, spacing_ratio, core_ratio, exchange_dt, end) <= 0:
        raise ValueError("Mesh, particle and time spacings must be positive")
    if not 0 < release < blend:
        raise ValueError("The release width must lie inside the blend width")
    if cores < 1 or particles < 1:
        raise ValueError("Cores and particle limit must be positive")


def validate_authority(edge, blend_width, radius, clearance):
    if edge - blend_width <= radius + clearance:
        raise ValueError("The blend region must leave the cylinder in full FVM authority")
