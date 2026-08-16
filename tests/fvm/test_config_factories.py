import pytest

from source.solvers.FVM.config.types import (
    BoundaryConfig,
    ExecutionConfig,
    FVMSetup,
    LinearSolverConfig,
    MeshConfig,
    OutputSetup,
    PimpleControl,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
)
from source.solvers.FVM.factory import _runtime_setup
from source.solvers.FVM.sampling.base import SamplingSchedule
from source.solvers.FVM.sampling.forces import ForceSampler, YPlusSampler


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
        assert bc.mesh_type == "wall"

    def test_boundary_config_empty(self):
        bc = BoundaryConfig.empty("empty")
        assert bc.type_U == "empty"
        assert bc.type_p == "empty"
        assert bc.mesh_type == "empty"

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

    def test_pimple_defaults(self):
        pimple = PimpleControl(n_correctors=2, n_outer_correctors=1)
        assert pimple.algorithm == "PIMPLE"
        assert pimple.n_correctors == 2
        assert pimple.alpha_u == 1.0
        assert pimple.alpha_p == 1.0

    def test_simple_control(self):
        pimple = PimpleControl(algorithm="SIMPLE", alpha_u=0.7, alpha_p=0.3)
        assert pimple.algorithm == "SIMPLE"
        assert pimple.alpha_u == 0.7
        assert pimple.alpha_p == 0.3

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

    def test_turbulence_config_equilibrium_smagorinsky(self):
        tc = TurbulenceConfig.equilibrium_smagorinsky()
        assert tc.model == "EquilibriumSmagorinsky"
        assert tc.Ck == pytest.approx(0.094)
        assert tc.Ce == pytest.approx(1.048)
        assert tc.Cs == pytest.approx(tc.Ck**0.75 / tc.Ce**0.25)

    def test_fvm_config_roundtrip_json(self, tmp_path):
        config = FVMSetup(
            case_name="test_case",
            cores=3,
            mesh=MeshConfig(max_aspect_ratio=100.0),
            time=TimeConfig.transient(dt=0.1, duration=10.0),
            transport=TransportConfig.air(),
            boundaries=[BoundaryConfig.inlet("in", [1, 0, 0])],
        )
        path = tmp_path / "test_config.json"
        config.save(path)
        loaded = FVMSetup.load(path)
        assert loaded.case_name == "test_case"
        assert loaded.cores == 3
        assert loaded.time.delta_t == 0.1
        assert loaded.pimple.algorithm == "PIMPLE"
        assert loaded.boundaries[0].name == "in"

    def test_fvm_config_roundtrip_preserves_every_solver_setting(self, tmp_path):
        # Grouped configs: linear solvers, PIMPLE/IBM controls,
        # forces (functionObjects/forces) round-trip through JSON intact.
        config = FVMSetup(
            case_name="complete",
            linear=LinearSolverConfig(reuse_ilu=False),
            output=OutputSetup(
                compression="none",
                asynchronous=False,
                ghost_layers=0,
            ),
            pimple=PimpleControl(ibm_forcing_loops=9, ibm_second_solve=False),
            samplers=[
                YPlusSampler(
                    patch_names=["wall"],
                    schedule=SamplingSchedule(every_n_steps=7),
                ),
                ForceSampler(
                    patch_names=["body"],
                    ref_velocity=12.0,
                    ref_area=3.0,
                    ref_length=2.0,
                    moment_centre=[1.0, 2.0, 3.0],
                    schedule=SamplingSchedule(every_n_steps=7),
                ),
            ],
        )
        path = tmp_path / "complete.json"
        config.save(path)

        assert FVMSetup.load(path) == config

    def test_fvm_setup_validates_cores(self):
        with pytest.raises(ValueError, match="at least one"):
            FVMSetup(case_name="invalid", cores=0)
        with pytest.raises(TypeError, match="integer"):
            FVMSetup(case_name="invalid", cores=True)

    def test_cores_select_partitioned_petsc_internally(self):
        user_setup = FVMSetup(case_name="parallel", cores=4)
        runtime_setup = _runtime_setup(user_setup)

        assert user_setup.execution.parallel_mode == "serial"
        assert runtime_setup.execution.parallel_mode == "petsc_partitioned"
        assert runtime_setup.execution.linear_backend == "petsc"
        assert user_setup.output.asynchronous
        assert not runtime_setup.output.asynchronous

    def test_cores_preserve_explicit_replicated_petsc_mode(self):
        user_setup = FVMSetup(
            case_name="parallel_ibm",
            cores=4,
            execution=ExecutionConfig.petsc_replicated(),
        )
        runtime_setup = _runtime_setup(user_setup)

        assert runtime_setup.execution.parallel_mode == "petsc_replicated"
        assert runtime_setup.execution.linear_backend == "petsc"
        assert not runtime_setup.output.asynchronous

    def test_output_setup_is_cell_centred_appended_lz4_by_default(self):
        output = OutputSetup()

        assert output.data_location == "cell"
        assert output.encoding == "appended"
        assert output.compression == "lz4"
        assert output.precision == "float64"
        assert output.asynchronous
        assert output.ghost_layers == 1

    @pytest.mark.parametrize(
        ("keyword", "value"),
        (
            ("data_location", "point"),
            ("encoding", "ascii"),
            ("compression", "unsupported"),
            ("precision", "float16"),
            ("ghost_layers", 2),
        ),
    )
    def test_output_setup_rejects_unqualified_modes(self, keyword, value):
        with pytest.raises((TypeError, ValueError)):
            OutputSetup(**{keyword: value})

    def test_physics_first_name_is_canonical(self):
        from source.solvers.FVM import FVMSetup

        setup = FVMSetup(case_name="named")
        assert type(setup).__name__ == "FVMSetup"
