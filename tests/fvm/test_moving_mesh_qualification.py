"""Qualification of the moving FVM mesh against oscillatory Stokes-layer theory."""

from __future__ import annotations

import math

import numpy as np

from openonda import fvm
from tests.support.fvm_mesh import cylinder_o_grid

# Cylinder oscillating in still fluid, x(t) = A sin(omega t)
DIAMETER = 1.0  # m
PERIOD = 1.0  # s
STOKES_NUMBER = 300.0  # beta = D^2 / (nu T)
KEULEGAN_CARPENTER_NUMBER = 0.5  # KC = U_max T / D


def _wang_coefficients() -> tuple[float, float]:
    """Added-mass and drag coefficients of Wang (1968), J. Fluid Mech. 32, 55-68.

    For KC << 1 and beta >> 1 the force on a cylinder oscillating in still
    fluid has the inertia coefficient ``Cm - 1`` and the drag coefficient of
    the attached oscillatory Stokes layer.
    """
    stokes = math.pi * STOKES_NUMBER
    added_mass = 1.0 + 4.0 * stokes**-0.5 + stokes**-1.5
    drag = (
        3.0
        * math.pi**3
        / (2.0 * KEULEGAN_CARPENTER_NUMBER)
        * (stokes**-0.5 + stokes**-1.0 - 0.25 * stokes**-1.5)
    )
    return added_mass, drag


def _force_coefficients(tmp_path, steps_per_period: int) -> tuple[float, float]:
    """Return Fourier-averaged added-mass and drag coefficients of the last two cycles."""
    viscosity = DIAMETER**2 / (STOKES_NUMBER * PERIOD)
    amplitude = KEULEGAN_CARPENTER_NUMBER * DIAMETER / (2.0 * math.pi)
    omega = 2.0 * math.pi / PERIOD
    speed = amplitude * omega
    mesh = cylinder_o_grid(DIAMETER, 30.0 * DIAMETER, 128, 60, 0.004 * DIAMETER)
    case = fvm.FVMCase(
        name="oscillating_cylinder",
        directory=tmp_path / f"steps_{steps_per_period}",
        mesh=mesh,
        numerics=fvm.Numerics(
            transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=viscosity),
            schemes=fvm.DiscretizationConfig(time_scheme="backward"),
        ),
        # The far field moves with the cylinder through still fluid. A fixed
        # zero velocity there keeps one boundary condition through each flow
        # reversal; at 15 D it changes the added mass by 2 (a/R)^2 = 0.2 %.
        boundaries=(
            fvm.BoundaryConfig.inlet("farfield", [0.0, 0.0, 0.0]),
            fvm.BoundaryConfig.wall("cylinder"),
            fvm.BoundaryConfig.empty("front"),
            fvm.BoundaryConfig.empty("back"),
        ),
        run=fvm.RunPlan(
            end_time=5.0 * PERIOD,
            time_step_size=PERIOD / steps_per_period,
            initial_output=False,
            final_output=False,
        ),
        samplers=fvm.Samplers(
            (
                fvm.ForceSampler(
                    ["cylinder"],
                    reference_velocity=speed,
                    reference_area=DIAMETER,
                    reference_length=DIAMETER,
                    schedule=fvm.RunSchedule(every_n_steps=1),
                ),
            )
        ),
        mesh_motion=fvm.MeshMotion(
            fvm.HeavingMotion(amplitude, 1.0 / PERIOD, direction=(1.0, 0.0, 0.0))
        ),
    )
    with fvm.FVMSolver(case) as solver:
        solver.auto_write = False
        solver.run(start_from="initial")
    history = np.genfromtxt(
        tmp_path / f"steps_{steps_per_period}" / "samples" / "forces_history.csv",
        delimiter=",",
        names=True,
        usecols=("time", "total_force_x"),
    )
    last = history["time"] > 3.0 * PERIOD + 1e-9
    time, force = history["time"][last], history["total_force_x"][last]
    step = PERIOD / steps_per_period
    # F = rho (pi D^2/4) Ca U omega sin(omega t) - 1/2 rho D Cd |U| U, U = U_max cos(omega t)
    added_mass = (
        np.sum(force * np.sin(omega * time))
        * step
        / 2.0
        / (math.pi * DIAMETER**2 / 4.0 * speed * math.pi)
    )
    drag = (
        -3.0
        * math.pi
        * np.sum(force * np.cos(omega * time))
        * step
        / 2.0
        / (2.0 * DIAMETER * speed**2 * PERIOD)
    )
    return float(added_mass), float(drag)


def test_oscillating_cylinder_reproduces_stokes_layer_forces(tmp_path, record_property):
    """Claim: a rigidly moving mesh carries the added mass and Stokes-layer drag.

    The whole O-grid oscillates with the cylinder. At beta = 300 and KC = 0.5
    the Fourier-averaged inertia and drag coefficients reproduce Wang's
    asymptotic theory, and the inertia error falls as the time step halves.
    """
    wang_added_mass, wang_drag = _wang_coefficients()
    coarse = _force_coefficients(tmp_path, 100)
    fine = _force_coefficients(tmp_path, 200)
    added_mass_errors = [abs(result[0] / wang_added_mass - 1.0) for result in (coarse, fine)]
    drag_errors = [abs(result[1] / wang_drag - 1.0) for result in (coarse, fine)]
    record_property("wang_added_mass", wang_added_mass)
    record_property("wang_drag", wang_drag)
    record_property("added_mass_t100", coarse[0])
    record_property("added_mass_t200", fine[0])
    record_property("drag_t100", coarse[1])
    record_property("drag_t200", fine[1])
    assert added_mass_errors[1] < added_mass_errors[0] < 0.015
    assert added_mass_errors[1] < 0.01
    assert max(drag_errors) < 0.03
