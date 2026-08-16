"""Fully meshed FVM reference for the matched Re=100 cylinder benchmark."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from openonda.fvm import (
    BoundaryConfig,
    ExecutionConfig,
    FVMSetup,
    IBMForceSampler,
    ImmersedBody,
    LinearSolverConfig,
    LineSampler,
    OutputSetup,
    PimpleControl,
    SamplingSchedule,
    SchemesConfig,
    SurfaceSampler,
    TimeConfig,
    TransportConfig,
    coupling_box_mesh,
    setup_fvm_solver,
)

CASE_DIR = Path(__file__).resolve().parent
SMOKE = os.environ.get("OPENONDA_SMOKE", "0") == "1"

DIAMETER = 1.0
RHO = 1.0
REYNOLDS = 100.0
U_INF = (1.0, 0.0, 0.0)
INITIAL_U = (1.0, 0.01, 0.0)
NU = float(np.linalg.norm(U_INF)) * DIAMETER / REYNOLDS

SPACING = float(os.environ.get("OPENONDA_SPACING", "0.20" if SMOKE else "0.10"))
DT = float(os.environ.get("OPENONDA_FVM_DT", "0.025"))
T_END = float(os.environ.get("OPENONDA_T_END", "0.20" if SMOKE else "20.0"))
# Direct-forcing IBM interpolation is currently rank-local. Until marker
# support is exchanged across partitions, immersed-body cases must be serial.
CORES = int(os.environ.get("OPENONDA_FVM_CORES", "1"))
DOMAIN = (-4.0, 10.4, -4.0, 4.0, -1.2, 1.2)
SPAN = DOMAIN[5] - DOMAIN[4]

FORCE_INTERVAL = float(os.environ.get("OPENONDA_FORCE_INTERVAL", "0.10"))
DIAGNOSTIC_INTERVAL = float(os.environ.get("OPENONDA_DIAGNOSTIC_INTERVAL", "1.0"))
VOLUME_INTERVAL = float(os.environ.get("OPENONDA_VOLUME_INTERVAL", "10.0"))
SAMPLE_SPACING = float(os.environ.get("OPENONDA_SAMPLE_SPACING", "0.10"))


def _period(name: str, interval: float) -> int:
    ratio = interval / DT
    period = int(round(ratio))
    if interval <= 0.0 or period < 1 or not np.isclose(ratio, period, atol=1e-10):
        raise ValueError(f"{name}={interval:g} must be a positive integer multiple of dt={DT:g}")
    return period


FORCE_SCHEDULE = SamplingSchedule(every_n_steps=_period("force interval", FORCE_INTERVAL))
FIELD_SCHEDULE = SamplingSchedule(every_n_steps=_period("diagnostic interval", DIAGNOSTIC_INTERVAL))

MESH = coupling_box_mesh(
    DOMAIN,
    SPACING,
    patch_name="numericalBoundary",
)
CYLINDER = ImmersedBody.extruded_cylinder_z(
    centre=[0.0, 0.0, 0.0],
    diameter=DIAMETER,
    z_bounds=[DOMAIN[4], DOMAIN[5]],
    h=SPACING,
    alpha=1.5,
    name="cylinder",
    caps=False,
)

SAMPLERS = (
    IBMForceSampler(
        ref_velocity=float(np.linalg.norm(U_INF)),
        ref_area=DIAMETER * SPAN,
        schedule=FORCE_SCHEDULE,
    ),
    LineSampler(
        start=[DOMAIN[0], 0.0, 0.0],
        end=[DOMAIN[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="centerline",
        schedule=FIELD_SCHEDULE,
    ),
    LineSampler(
        start=[DOMAIN[0], 0.75, 0.0],
        end=[DOMAIN[1], 0.75, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="offaxis_y075",
        schedule=FIELD_SCHEDULE,
    ),
    LineSampler(
        start=[1.0, DOMAIN[2], 0.0],
        end=[1.0, DOMAIN[3], 0.0],
        spacing=SAMPLE_SPACING,
        file_name="section_x100",
        schedule=FIELD_SCHEDULE,
    ),
    LineSampler(
        start=[2.0, DOMAIN[2], 0.0],
        end=[2.0, DOMAIN[3], 0.0],
        spacing=SAMPLE_SPACING,
        file_name="section_x200",
        schedule=FIELD_SCHEDULE,
    ),
    SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[-1.6, 6.0, -2.0, 2.0],
        spacing=SAMPLE_SPACING,
        file_name="slice_z0",
        schedule=FIELD_SCHEDULE,
    ),
)

SETUP = FVMSetup(
    case_name="reference_cylinderFlow",
    cores=CORES,
    execution=ExecutionConfig(operator_backend="numba"),
    output=OutputSetup(
        compression="lz4",
        precision="float32",
        asynchronous=False,
        ghost_layers=0,
    ),
    time=TimeConfig(
        delta_t=DT,
        end_time=T_END,
        write_interval=10**9,
        write_interval_time=VOLUME_INTERVAL,
    ),
    schemes=SchemesConfig(
        convection_scheme="limitedLinear",
        gradient_scheme="gauss",
        time_scheme="backward",
    ),
    linear=LinearSolverConfig(
        pressure_solver="amg",
        pressure_tol=1e-6,
        pressure_rel_tol=0.01,
        momentum_tol=1e-6,
        momentum_rel_tol=0.1,
        momentum_maxiter=2000,
    ),
    pimple=PimpleControl(
        n_correctors=2,
        n_outer_correctors=2,
        alpha_u=0.7,
        alpha_p=0.3,
        ibm_forcing_loops=2,
    ),
    samplers=SAMPLERS,
    transport=TransportConfig(density=RHO, nu=NU),
    turbulence=None,
    boundaries=[
        BoundaryConfig(
            name="numericalBoundary",
            type_U="fixedValue",
            value_U=list(U_INF),
            type_p="fixedFluxPressure",
        ),
    ],
    initial_U=list(INITIAL_U),
    initial_p=0.0,
)


def main() -> None:
    print(f"Reference cylinder: Re={REYNOLDS:g}, h={SPACING:g}, dt={DT:g}, t_end={T_END:g}")
    solver = setup_fvm_solver(SETUP, case_dir=CASE_DIR, mesh=MESH)
    solver.set_immersed_bodies(CYLINDER, h=SPACING)
    solver.write_vtk()
    while solver.flow_time < SETUP.time.end_time:
        solver.evolve()
    solver.close()


if __name__ == "__main__":
    main()
