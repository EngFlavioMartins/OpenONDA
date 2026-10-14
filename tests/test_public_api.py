"""Public solver interfaces remain importable and construct valid cases."""

from openonda import coupler, fvm, vpm


def test_public_exports_and_case_construction(tmp_path):
    required = {
        fvm: {"FVMCase", "FVMSolver", "Numerics", "RunPlan", "EverySteps", "mesher"},
        vpm: {"VPMCase", "VPMSolver", "Numerics", "RunPlan", "EverySteps"},
        coupler: {"CoupledSolver", "CouplerConfig"},
        fvm.mesher: {"CartesianMesher", "ExtrudedCartesianMesher", "STLSurface"},
    }
    for module, names in required.items():
        assert names <= set(module.__all__)
        for name in module.__all__:
            assert getattr(module, name) is not None
    assert fvm.EverySteps is vpm.EverySteps
    mesh = fvm.mesher.coupling_box_mesh((0, 1, 0, 1, 0, 1), 0.5)
    flow = fvm.FVMCase(name="api", directory=tmp_path, run=fvm.RunPlan(end_time=0.1), mesh=mesh)
    particles = vpm.VPMCase(
        name="api",
        directory=tmp_path,
        numerics=vpm.Numerics(compute_device="cpu", time_step_size=0.05),
        run=vpm.RunPlan(end_time=0.1),
    )
    assert flow.solution_directory == tmp_path / "solution"
    assert particles.n_steps == 2
