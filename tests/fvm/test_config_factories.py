from source.solvers.FVM.config.types import (
    BoundaryConfig,
    FVMConfig,
    MeshConfig,
    SolverParams,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
)


class TestConfigFactories:
    def test_boundary_config_inlet(self):
        bc = BoundaryConfig.inlet("inlet", [1.0, 0.0, 0.0])
        assert bc.name == "inlet"
        assert bc.type_U == "fixedValue"
        assert bc.value_U == [1.0, 0.0, 0.0]
        assert bc.type_p == "zeroGradient"

    def test_boundary_config_outlet(self):
        bc = BoundaryConfig.outlet("outlet", p=0.0)
        assert bc.name == "outlet"
        assert bc.type_p == "fixedValue"
        assert bc.value_p == 0.0

    def test_boundary_config_wall(self):
        bc = BoundaryConfig.wall("wall")
        assert bc.type_U == "fixedValue"
        assert bc.value_U == [0.0, 0.0, 0.0]
        assert bc.type_p == "zeroGradient"

    def test_boundary_config_empty(self):
        bc = BoundaryConfig.empty("empty")
        assert bc.type_U == "empty"
        assert bc.type_p == "empty"

    def test_time_config_transient(self):
        tc = TimeConfig.transient(dt=0.1, duration=20.0)
        assert tc.delta_t == 0.1
        assert tc.start_time == 0.0
        assert tc.end_time == 20.0

    def test_time_config_steady(self):
        tc = TimeConfig.steady(max_iter=1000)
        assert tc.delta_t == 1  # default dt for steady
        assert tc.end_time == 1000
        assert tc.start_time == 0

    def test_solver_params_pimple(self):
        sp = SolverParams.pimple(n_correctors=2, n_outer=1)
        assert sp.algorithm == "PIMPLE"
        assert sp.n_correctors == 2
        assert sp.alpha_u == 1.0
        assert sp.alpha_p == 1.0

    def test_solver_params_simple(self):
        sp = SolverParams.simple(alpha_u=0.7, alpha_p=0.3)
        assert sp.algorithm == "SIMPLE"
        assert sp.alpha_u == 0.7
        assert sp.alpha_p == 0.3

    def test_transport_config_air(self):
        tc = TransportConfig.air()
        assert tc.density == 1.225
        assert tc.nu == 1.5e-5

    def test_transport_config_water(self):
        tc = TransportConfig.water()
        assert tc.density == 1000.0
        assert tc.nu == 1e-6

    def test_turbulence_config_smagorinsky(self):
        tc = TurbulenceConfig.smagorinsky(Cs=0.17)
        assert tc.model == "Smagorinsky"
        assert tc.Cs == 0.17

    def test_fvm_config_roundtrip_json(self, tmp_path):
        config = FVMConfig(
            case_name="test_case",
            mesh=MeshConfig.block_mesh(),
            time=TimeConfig.transient(dt=0.1, duration=10.0),
            solver=SolverParams.pimple(),
            transport=TransportConfig.air(),
            boundaries=[BoundaryConfig.inlet("in", [1, 0, 0])],
        )
        path = tmp_path / "test_config.json"
        config.save(path)
        loaded = FVMConfig.load(path)
        assert loaded.case_name == "test_case"
        assert loaded.time.delta_t == 0.1
        assert loaded.solver.algorithm == "PIMPLE"
        assert loaded.boundaries[0].name == "in"

    def test_fvm_config_roundtrip_preserves_every_solver_setting(self, tmp_path):
        solver = SolverParams(
            reuse_ilu=False,
            yplus_patches=["wall"],
            force_patches=["body"],
            ref_velocity=12.0,
            ref_area=3.0,
            ref_length=2.0,
            force_log_interval=7,
            moment_centre=[1.0, 2.0, 3.0],
            ibm_forcing_loops=9,
            ibm_second_solve=False,
        )
        config = FVMConfig(case_name="complete", solver=solver)
        path = tmp_path / "complete.json"
        config.save(path)

        assert FVMConfig.load(path) == config
