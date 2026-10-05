"""Local accepted-clock startup and continuation for this tutorial."""

from dataclasses import replace
import json
from numbers import Integral

import numpy as np


def startup_velocity(time, duration, transition_duration, startup, steady) -> tuple[float, ...]:
    """Hold the startup flow, then remove it with a C2 quintic taper.

    The transition occupies the final ``transition_duration`` seconds of
    startup. Its first and second time derivatives vanish at both ends, so
    removing the crossflow does not introduce an impulsive acceleration.
    Caller-side schedule validation keeps this hot-path function small.
    """
    if time >= duration:
        return tuple(float(value) for value in steady)
    if time <= duration - transition_duration:
        return tuple(float(value) for value in startup)
    fraction = (float(time) - (duration - transition_duration)) / transition_duration
    blend = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
    return tuple(
        float(first + blend * (last - first)) for first, last in zip(startup, steady, strict=True)
    )


def cylinder_initial_velocity(
    positions: np.ndarray,
    span: float,
    *,
    amplitude: float,
    radius: float,
    centre: float,
    freestream_velocity,
) -> np.ndarray:
    """Return a reproducible divergence-free 3D perturbation of unit inflow.

    Coordinates and span are in metres for the D=1, U=1 cylinder case. The
    perturbation is curl(0, A_y, A_z), with compact XY support around x=.65,
    sinusoidal A_y vanishing at the slip planes and span-independent A_z.
    The latter breaks transverse reflection symmetry to seed shedding without
    relying on mesh asymmetry or roundoff. Thus w=0 and the normal derivative
    of tangential velocity vanishes on both planes. Amplitude is
    the dimensionless vector-potential coefficient; this is an initial test
    disturbance, not a sustained forcing or a prescribed turbulent state.
    """
    if not np.isfinite(span) or span <= 0 or not np.isfinite(amplitude):
        raise ValueError("span must be positive and perturbation amplitude finite")
    position = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    x, y, z = position.T
    radius_squared = radius**2
    support = np.maximum(1 - ((x - centre) ** 2 + y**2) / radius_squared, 0)
    wave_number = np.pi / span
    phase = wave_number * (z + 0.5 * span)
    velocity = np.tile(np.asarray(freestream_velocity, dtype=np.float64), (len(position), 1))
    velocity[:, 0] += (
        -amplitude * wave_number * support**4 * np.cos(phase)
        - 8 * amplitude * y / radius_squared * support**3
    )
    velocity[:, 1] += 8 * amplitude * (x - centre) / radius_squared * support**3
    velocity[:, 2] += -8 * amplitude * (x - centre) / radius_squared * support**3 * np.sin(phase)
    return velocity


def initialize_cylinder_perturbation(
    solver, span: float, *, perturbation, freestream_velocity
) -> None:
    """Install the same small 3D initial disturbance on each local FVM mesh."""
    count = solver.mesh_data["n_cells"]
    velocity = cylinder_initial_velocity(
        solver.geo_data["cell_centre"][:count],
        span,
        **perturbation,
        freestream_velocity=freestream_velocity,
    )
    solver.set_initial_velocity(velocity)
