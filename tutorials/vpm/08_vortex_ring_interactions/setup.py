#!/usr/bin/env python3
"""Leapfrogging and colliding vortex rings at Re_Gamma = 2e4: DNS against two LES closures.

Examples (from this case directory)::

    python setup.py collision_les_cusp
    python setup.py leapfrog_dns
"""

import argparse
from pathlib import Path

import numpy as np
import openonda.vpm as vpm

from assets.initial_conditions import ToroidalDistribution, VortexRing, WidnallDisturbance
from assets.ring_diagnostics import AzimuthalModeSampler, RingDiagnosticsSampler


# Rings: R_0 = 1 m and Gamma_0 = 1 m²/s, so t Gamma_0 / R_0^2 is the time in s
RING_RADIUS = 1.0  # [m]
RING_CIRCULATION = 1.0  # [m²/s]
REYNOLDS_NUMBER = 2.0e4  # Gamma_0 / nu
KINEMATIC_VISCOSITY = RING_CIRCULATION / REYNOLDS_NUMBER  # [m²/s]
CORE_RADIUS = 0.1  # [m], Gaussian e-folding radius a_0 of every ring

# Widnall disturbance of each ring centreline, radial and axial, with its own phases. Above
# DISTURBANCE_ROLLOFF_MODE the amplitude falls as (m_r/m)^2, so the centreline bends with a
# radius at least 1.7 a_0; an equal amplitude up to m = 48 folds the cores, which then create energy.
DISTURBANCE_AMPLITUDE = 0.005  # relative to R_0, per mode up to DISTURBANCE_ROLLOFF_MODE
DISTURBANCE_MODES = 48
DISTURBANCE_ROLLOFF_MODE = 8

# Particles: the particle numerics of the coupled FVM-VPM cases
PARTICLE_SPACING = 0.025  # [m], h = R_0 / 40
CORE_RADIUS_RATIO = 1.05  # sigma / h
VORTICITY_FLOOR = 0.02  # [1/s], regenerated nodes below it are dropped
TOROIDAL_TAIL_FRACTION = 1.0e-3  # initial lattice ends where the Gaussian falls to this
MAX_N_PARTICLES = 2_500_000
DIVERGENCE_LIMIT = 0.5  # |div w| / |grad w|, weighted by |w|: O(1) once vortex lines are lost

# Head-on collision: opposite rings 2.5 R_0 apart meet at x = 0 and stretch radially
COLLISION_SEPARATION = 2.5  # [m]
COLLISION_DOMAIN = (-1.5, 1.5, -3.25, 3.25, -3.25, 3.25)  # [m]
COLLISION_TIME_STEP = 0.0125  # [s]; stretched cores reach about 3 times their initial vorticity
COLLISION_END_TIME = 30.0  # [s]

# Leapfrogging: equal rings R_0 apart, followed by a frame moving with the pair
LEAPFROG_SEPARATION = 1.0  # [m]
LEAPFROG_FRAME_SPEED = 0.26  # [m/s], mean speed of the pair; the free stream is its opposite
LEAPFROG_DOMAIN = (-4.0, 3.0, -2.25, 2.25, -2.25, 2.25)  # [m]
LEAPFROG_FRINGE = 1.0  # [m], strengths fade out before the upstream face x = -4 m
LEAPFROG_TIME_STEP = 0.02  # [s]
LEAPFROG_END_TIME = 40.0  # [s]

# Output
SAMPLE_STEPS = 10
BACKUP_INTERVAL = 0.5  # [s], also the flow integrals, which the backups need anyway

TUTORIAL_DIR = Path(__file__).parent
MODELS = ("dns", "les", "les_cusp")
CASES = tuple(f"{flow}_{model}" for flow in ("collision", "leapfrog") for model in MODELS)


def vortex_ring(x, circulation, group):
    """Gaussian ring centred at (x, 0, 0) with its own Widnall disturbance."""
    centre = (x, 0.0, 0.0)
    disturbance = WidnallDisturbance(
        DISTURBANCE_AMPLITUDE,
        number_of_modes=DISTURBANCE_MODES,
        rolloff_mode=DISTURBANCE_ROLLOFF_MODE,
        seed=2 * group + 7,
        direction="both",
    )
    particle_core_radius = CORE_RADIUS_RATIO * PARTICLE_SPACING
    tube_radius = np.sqrt(CORE_RADIUS**2 - particle_core_radius**2) * np.sqrt(
        -np.log(TOROIDAL_TAIL_FRACTION)
    )
    return VortexRing(
        centre=centre,
        radius=RING_RADIUS,
        circulation=circulation,
        vortex_core_radius=CORE_RADIUS,
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        disturbance=disturbance,
        distribution=ToroidalDistribution(
            centre=centre,
            ring_radius=RING_RADIUS,
            tube_radius=tube_radius,
            spacing=PARTICLE_SPACING,
            core_radius_ratio=CORE_RADIUS_RATIO,
            disturbance=disturbance,
        ),
        group_id=group,
    )


def build_case(name):
    flow, model = name.split("_", 1)
    if flow == "collision":
        # A positive ring moves towards +x: ring 0 from the left, ring 1 from the right
        x = 0.5 * COLLISION_SEPARATION
        rings = (
            vortex_ring(-x, RING_CIRCULATION, 0),
            vortex_ring(x, -RING_CIRCULATION, 1),
        )
        domain, time_step, end_time = COLLISION_DOMAIN, COLLISION_TIME_STEP, COLLISION_END_TIME
        freestream = (0.0, 0.0, 0.0)
        stabilization = vpm.StabilizationConfig.bounded_domain(domain)
    else:
        # Ring 0 starts behind ring 1 and passes through it first
        x = 0.5 * LEAPFROG_SEPARATION
        rings = (
            vortex_ring(-x, RING_CIRCULATION, 0),
            vortex_ring(x, RING_CIRCULATION, 1),
        )
        domain, time_step, end_time = LEAPFROG_DOMAIN, LEAPFROG_TIME_STEP, LEAPFROG_END_TIME
        freestream = (-LEAPFROG_FRAME_SPEED, 0.0, 0.0)
        stabilization = vpm.StabilizationConfig.bounded_domain(
            domain, outflow_fringe_depth=(LEAPFROG_FRINGE, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
    # les: the equilibrium Smagorinsky model of the coupled cases, with the selective eddy
    # viscosity it switches on. les_cusp: the spectral eddy viscosity with the Kraichnan
    # cusp on the GBD lattice. dns: molecular viscosity only.
    turbulence = {
        "dns": vpm.TurbulenceConfig.dns(),
        "les": vpm.TurbulenceConfig.equilibrium_smagorinsky(),
        "les_cusp": vpm.TurbulenceConfig.spectral_cusp(),
    }[model]
    return vpm.VPMCase(
        name=name,
        directory=TUTORIAL_DIR,
        variant=name,
        initial_conditions=rings,
        numerics=vpm.Numerics(
            time_step_size=time_step,
            freestream_velocity=freestream,
            integrator=vpm.RK2(),
            induction=vpm.FMMInduction(),
            # Grid-based diffusion regenerates the particles on the lattice every step
            viscous=vpm.ViscousConfig.gbd(
                particle_spacing=PARTICLE_SPACING,
                padding=5.0,
                kinematic_viscosity=KINEMATIC_VISCOSITY,
                threshold_mode="absolute",
                threshold=VORTICITY_FLOOR * PARTICLE_SPACING**3,
                core_radius_ratio=CORE_RADIUS_RATIO,
            ),
            turbulence=turbulence,
            stabilization=stabilization,
            domain_bounds=domain,
            max_n_particles=MAX_N_PARTICLES,
            random_seed=42,
            # A run stops when the step no longer resolves the velocity gradient or the
            # particles have lost the vortex-line structure: its lifetime. GBD holds the
            # divergence error near 0.13 from the first steps, so the limit sits above it.
            state_limits=vpm.ParticleStateLimits(
                lagrangian_cfl=vpm.LagrangianCFLLimit(maximum=1.0),
                divergence=vpm.DivergenceLimit(maximum=DIVERGENCE_LIMIT),
            ),
        ),
        run=vpm.RunPlan(end_time=end_time, state_limit_action="stop"),
        samplers=vpm.Samplers(
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EveryTime(BACKUP_INTERVAL)),
                RingDiagnosticsSampler(schedule=vpm.EverySteps(SAMPLE_STEPS)),
                AzimuthalModeSampler(schedule=vpm.EverySteps(SAMPLE_STEPS)),
            ),
        ),
        backup=vpm.Backup(schedule=vpm.EveryTime(BACKUP_INTERVAL)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default="collision_les_cusp", choices=CASES)
    vpm.VPMSolver(build_case(parser.parse_args().case)).run()
