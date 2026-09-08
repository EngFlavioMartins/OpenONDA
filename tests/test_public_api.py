"""Public solver interfaces remain importable and construct valid cases."""

from openonda import coupler, fvm, vpm


def test_public_exports_and_case_construction(tmp_path):
    required = {
        fvm: {"FVMCase", "FVMSolver", "Numerics", "RunPlan", "mesher"},
        vpm: {"VPMCase", "VPMSolver", "Numerics", "RunPlan"},
        coupler: {"CouplerSetup", "FVMVPMCoupler", "create_coupler"},
        fvm.mesher: {"CartesianMesher", "ExtrudedCartesianMesher", "STLSurface"},
    }
    for module, names in required.items():
        assert names <= set(module.__all__)
        assert len(module.__all__) == len(set(module.__all__))
        for name in module.__all__:
            assert getattr(module, name) is not None
    mesh = fvm.mesher.structured_box(2, 2, 2)
    flow = fvm.FVMCase(name="api", mesh=mesh, directory=tmp_path)
    particles = vpm.VPMCase(numerics=vpm.Numerics(compute_device="CPU"), directory=tmp_path)
    assert flow.to_setup().case_name == "api"
    assert flow.directory == particles.directory == tmp_path
