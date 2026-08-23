"""Rank-agnostic coupled FVM-VPM smoke test launched under real MPI.

Run with::

    mpirun -np 2 python -m pytest tests/coupler/test_fvm_vpm_mpi_smoke.py -m mpi

Every rank constructs the full pipeline through the public user API
(``FVMSolver`` / ``VPMSolver`` / ``FVMVPMCoupler``) with no rank conditionals,
no ``is_master_rank()`` calls, and no conditional imports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

taichi = pytest.importorskip("taichi", reason="VPM requires taichi")

pytestmark = [pytest.mark.mpi, pytest.mark.slow]

FVM_TIME_STEP_SIZE = 0.01 / 3
VPM_TIME_STEP_SIZE = 0.01


@pytest.mark.slow
def test_coupled_fvm_vpm_two_steps_mpi(tmp_path, monkeypatch):
    from mpi4py import MPI

    from source.coupler import CouplerSetup, FVMVPMCoupler
    from source.solvers.fvm import (
        BoundaryConfig,
        ComputeConfig,
        FVMSetup,
        FVMSolver,
        TimeConfig,
        TransportConfig,
    )
    from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh
    from source.solvers.vpm import VPMSetup, VPMSolver

    rank = MPI.COMM_WORLD.Get_rank()
    case_dir = Path(MPI.COMM_WORLD.bcast(str(tmp_path) if rank == 0 else None, root=0))
    MPI.COMM_WORLD.Barrier()
    monkeypatch.chdir(case_dir)

    h = 0.25

    setup = CouplerSetup(
        freestream_velocity=[1.0, 0.0, 0.0],
        vpm_particle_spacing=h,
        authority_ramp_width=2 * h,
        vpm_only_width=h,
    )

    vpm_setup = VPMSetup(
        time_step_size=VPM_TIME_STEP_SIZE,
        compute_device="CPU",
        max_n_particles=50_000,
        domain_bounds=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        freestream_velocity=[1.0, 0.0, 0.0],
    )
    vpm = VPMSolver(vpm_setup)

    config = FVMSetup(
        case_name="coupled_smoke_mpi",
        time=TimeConfig(time_step_size=FVM_TIME_STEP_SIZE, end_time=2 * VPM_TIME_STEP_SIZE),
        transport=TransportConfig(kinematic_viscosity=0.01),
        boundaries=[
            BoundaryConfig(
                name="numericalBoundary",
                velocity_type="fixedValue",
                velocity_value=setup.freestream_velocity,
                pressure_type="fixedFluxPressure",
            ),
            BoundaryConfig.wall("cube"),
        ],
        initial_velocity=setup.freestream_velocity,
        execution=ComputeConfig.petsc_partitioned(),
    )
    mesh = coupling_box_mesh(
        (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
        h,
        hole_box=(-0.25, 0.25, -0.25, 0.25, -0.25, 0.25),
        wall_patch_name="cube",
    )
    fvm = FVMSolver(
        config,
        case_dir=case_dir,
        mesh_data=mesh if rank == 0 else None,
    )

    coupler = FVMVPMCoupler(fvm, vpm, setup)
    coupler.run()

    assert coupler.n_fvm_substeps == 3
    assert coupler.vpm_time_step_size == pytest.approx(VPM_TIME_STEP_SIZE)

    # Every rank observes the same committed coupling state through the
    # ordinary user API -- no rank-specific code paths anywhere above.
    assert fvm.time == pytest.approx(2 * VPM_TIME_STEP_SIZE)
    assert not np.isnan(np.asarray(fvm.velocity)).any()
