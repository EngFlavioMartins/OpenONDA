"""Small actual FVM/RK/planar-GBD/renewal integration, without cylinder flow."""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize("device", ["CPU", "METAL"])
def test_planar_coupler_carries_an_exterior_dipole(tmp_path, monkeypatch, device):
    if device == "METAL" and os.environ.get("OPENONDA_TEST_METAL") != "1":
        pytest.skip("Set OPENONDA_TEST_METAL=1 for native Metal qualification")
    monkeypatch.chdir(tmp_path)
    from openonda import vpm
    from source.coupler import CouplerSetup, FVMVPMCoupler
    from source.solvers.fvm import BoundaryConfig, FVMSetup, FVMSolver, TimeConfig, TransportConfig
    from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh

    h = 0.125
    vpm_solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            numerics=vpm.Numerics(
                induction=vpm.PlanarInduction(span=1),
                compute_device=device,
                time_step_size=0.01,
                freestream_velocity=(1, 0, 0),
                max_n_particles=5000,
                max_evaluation_points=5000,
                viscous=vpm.ViscousConfig.gbd(
                    particle_spacing=h,
                    gbd_grid_spacing=h,
                    kinematic_viscosity=0.01,
                    threshold_mode="absolute",
                    threshold=1e-9,
                    core_radius_ratio=1,
                ),
                verbose=False,
            ),
            run=vpm.RunPlan(steps=2, final_backup=False, initial_samples=False),
        )
    )
    fvm = FVMSolver(
        FVMSetup(
            case_name="planar_smoke",
            time=TimeConfig(time_step_size=0.01, end_time=0.02),
            transport=TransportConfig(kinematic_viscosity=0.01),
            boundaries=[
                BoundaryConfig(
                    name="numericalBoundary",
                    velocity_type="fixedValue",
                    velocity_value=[1, 0, 0],
                    pressure_type="fixedFluxPressure",
                ),
                BoundaryConfig.slip("zmin"),
                BoundaryConfig.slip("zmax"),
            ],
            initial_velocity=[1, 0, 0],
        ),
        case_dir=tmp_path,
        mesh_data=coupling_box_mesh(
            (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), 0.25, separate_outer=("zmin", "zmax")
        ),
    )
    driver = FVMVPMCoupler(
        fvm,
        vpm_solver,
        CouplerSetup(
            freestream_velocity=[1, 0, 0],
            transfer_method="buffered_m4_renewal",
            transfer_region_bounds=(-0.375, 0.375, -0.375, 0.375, -0.375, 0.375),
            eta_blend_width=0.25,
            vpm_only_width=0,
            transfer_vorticity_cutoff=0.001,
            boundary_condition_mode="vorticity_mixed",
            interface_iterations=1,
            backup_interval_steps=0,
        ),
    )
    try:
        driver.initialize()
        positions = np.array([[0.8, -0.25, 0], [0.8, 0.25, 0]])
        vpm_solver.add_vortex_particles(
            position=positions,
            velocity=np.tile([1.0, 0, 0], (2, 1)),
            vortex_strength=np.array([[0, 0, 0.001], [0, 0, -0.001]]),
            core_radius=np.full(2, h),
            particle_volume=np.full(2, h * h),
            kinematic_viscosity=np.full(2, 0.01),
        )
        assert driver.run(max_coupling_steps=2, backup_at_stop=False) == 2
        assert vpm_solver.particles.n_particles_total > 0
        assert np.isfinite(fvm.get_velocity_field()).all()
        assert np.isfinite(vpm_solver.particle_velocity).all()
        np.testing.assert_array_equal(vpm_solver.particle_position[:, 2], 0)
        np.testing.assert_allclose(vpm_solver.particle_volume, h * h)
        records = [
            json.loads(line)
            for line in (tmp_path / "solution/coupler_diagnostics.jsonl").read_text().splitlines()
        ]
        assert len(records) == 2
        for record in records:
            span = record["planar_spanwise_consistency"]
            assert np.isfinite(list(span.values())).all()
            assert span["span_variation_max"] < 1e-3 * span["velocity_scale"]
            assert span["span_velocity_max"] < 1e-3 * span["velocity_scale"]
    finally:
        driver.close()


@pytest.mark.integration
def test_planar_coupling_two_rank_global_fields_and_shutdown(tmp_path):
    """The real distributed donor gather, root-only VPM and close must finish."""
    if find_spec("mpi4py") is None or find_spec("petsc4py") is None:
        pytest.skip("MPI and PETSc are required")
    if not (Path(sys.executable).with_name("mpiexec").is_file() or shutil.which("mpiexec")):
        pytest.skip("mpiexec is required")
    script = Path(__file__).with_name("_planar_mpi_smoke.py")
    output = tmp_path / "two-rank-planar"
    environment = os.environ.copy()
    environment.pop("_OPENONDA_MPI_CHILD", None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    environment["TI_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path / "taichi-cache")
    result = subprocess.run(
        [sys.executable, str(script), str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-8000:]
    assert result.stdout.count("PLANAR_MPI_QUALIFIED ") == 1
    for rank in range(2):
        assert json.loads((output / f"rank-{rank}.json").read_text()) == {
            "rank": rank,
            "steps": 2,
            "time": 0.02,
            "closed": True,
        }
    report = json.loads((output / "qualification.json").read_text())
    assert report["ranks"] == 2
    assert report["global_cells"] == 128
    assert report["particle_volume"] == 0.125**2 * 2
    assert len(report["span_diagnostics"]) == 2
