"""VPM tutorial setups build their physical cases through the public API."""

import pytest

from tests._tutorial_helpers import copy_tutorial, load_case_module, load_tutorial_module


def _load_lamb_oseen_setup():
    return load_tutorial_module("vpm/lamb_oseen_vortex")


@pytest.mark.parametrize("physics", ["vortex", "dipole", "merging"])
def test_lamb_oseen_workspace_reserves_full_time_diffusion(physics, monkeypatch, tmp_path):
    setup = _load_lamb_oseen_setup()
    monkeypatch.setattr(setup, "TUTORIAL_DIR", tmp_path)
    captured = []

    class CaseCapturedError(Exception):
        pass

    def capture(case):
        captured.append(case)
        raise CaseCapturedError

    monkeypatch.setattr(setup.vpm, "VPMSolver", capture)
    with pytest.raises(CaseCapturedError):
        setup.run_case(physics, "DVH", surfaces=False)
    case = captured[0]
    viscosity = case.numerics.viscous.kinematic_viscosity
    heat_margin = 3.6 * (4.0 * viscosity * setup.TOTAL_TIME) ** 0.5
    xmin, xmax, ymin, ymax, zmin, zmax = case.numerics.domain_bounds
    for condition in case.initial_conditions:
        position = condition.build().position
        for axis, (lower, upper) in enumerate(((xmin, xmax), (ymin, ymax), (zmin, zmax))):
            assert lower <= position[:, axis].min() - heat_margin
            assert upper >= position[:, axis].max() + heat_margin


@pytest.mark.parametrize(
    "backend, stretching",
    [
        ("DIRECT", "transposed"),
        ("DIRECT", "mixed"),
        ("FMM", "mixed"),
        ("TREECODE", "direct"),
    ],
)
def test_lamb_oseen_selects_backend_and_stretching_independently(monkeypatch, backend, stretching):
    setup = _load_lamb_oseen_setup()
    monkeypatch.setitem(setup.COMPUTE_METHOD, "CS", backend)
    monkeypatch.setattr(setup, "STRETCHING_SCHEME", stretching)
    runtime = setup.induction_config("CS").build()
    assert runtime.method == backend
    assert runtime.stretching_scheme.lower() == stretching


def test_all_vpm_tutorials_construct_cases_with_the_public_api(tmp_path, monkeypatch):
    class CaseCapturedError(Exception):
        def __init__(self, case):
            self.case = case

    def capture(case):
        raise CaseCapturedError(case)

    builders = {
        "delta_wing": lambda setup: setup.vpm.VPMSolver(setup.build_case()),
        "quadcopter": lambda setup: setup.run(),
        "flat_plate": lambda setup: setup.run("static", 8),
        "rotor_flow": lambda setup: setup.run(),
        "lamb_oseen_vortex": lambda setup: setup.run_case("vortex", "CS", surfaces=False),
        "vortex_ring": lambda setup: setup.run_case("dns_mixed", n_steps=1),
        "vortex_interactions": lambda setup: setup.build_case("baseline", n_steps=1),
    }
    for name, build in builders.items():
        case_dir = copy_tutorial(f"vpm/{name}", tmp_path / name)
        setup = load_case_module(case_dir)
        with monkeypatch.context() as patch:
            # Exercise real case/geometry construction without starting a
            # long GPU parameter study or writing into installed tutorial resources.
            patch.setattr(setup.vpm, "VPMSolver", capture)
            try:
                case = build(setup)
            except CaseCapturedError as captured:
                case = captured.case
        assert case.directory == case_dir
        configured = case.numerics.induction
        runtime = configured.build()
        assert runtime.method == configured.method
        assert runtime.stretching_scheme == configured.stretching_scheme
