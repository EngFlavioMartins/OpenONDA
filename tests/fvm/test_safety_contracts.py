"""Regression tests for fail-fast FVM configuration and field input."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from source.solvers.FVM import DynamicMeshConfig, FVMConfig, Solver
from source.solvers.FVM.core.solver import _load_pressure_field, _load_velocity_field
from source.solvers.FVM.fields.diagnostics import _should_compute_yplus
from source.solvers.FVM.fields.field_io import (
    parse_boundary_field,
    parse_internal_field,
    write_foam_field,
)


def _write_case_file(path: Path, field_class: str) -> None:
    path.write_text(
        f"""
FoamFile
{{
    class {field_class};
}}
internalField uniform {"(1 0 0)" if field_class == "volVectorField" else "0"};
boundaryField
{{
}}
"""
    )


def test_from_case_uses_case_fields_and_rejects_unknown_overrides(tmp_path, monkeypatch):
    (tmp_path / "system").mkdir()
    (tmp_path / "constant").mkdir()
    (tmp_path / "0").mkdir()
    (tmp_path / "system" / "controlDict").write_text(
        "deltaT 0.1; startTime 0; endTime 1; writeInterval 2;"
    )
    (tmp_path / "system" / "fvSolution").write_text(
        """
solvers
{
    U { solver PBiCGStab; tolerance 1e-8; }
    p { solver GAMG; tolerance 1e-10; }
}
PIMPLE
{
    nOuterCorrectors 3;
    nCorrectors 2;
    nNonOrthogonalCorrectors 1;
}
"""
    )
    (tmp_path / "system" / "fvSchemes").write_text(
        """
ddtSchemes { default backward; }
gradSchemes { default leastSquares; }
divSchemes { div(phi,U) bounded Gauss limitedLinear 1; }
"""
    )
    (tmp_path / "constant" / "transportProperties").write_text("rho 1.0; nu [0 2 -1 0 0 0 0] 1e-5;")
    _write_case_file(tmp_path / "0" / "U", "volVectorField")
    _write_case_file(tmp_path / "0" / "p", "volScalarField")

    def capture_init(self, config, case_dir=None, mesh_data=None):
        self.config = config
        self.case_dir = case_dir

    monkeypatch.setattr(Solver, "__init__", capture_init)
    solver = Solver.from_case(str(tmp_path))
    assert solver.config.initial_U is None
    assert solver.config.initial_p is None
    assert solver.config.transport.nu == pytest.approx(1e-5)
    assert solver.config.solver.momentum_solver == "bicgstab"
    assert solver.config.solver.pressure_solver == "amg"
    assert solver.config.solver.n_outer_correctors == 3
    assert solver.config.solver.time_scheme == "backward"
    assert solver.config.solver.gradient_scheme == "lsq"
    assert solver.config.solver.convection_scheme == "limitedLinear"

    with pytest.raises(TypeError, match="Unknown FVMConfig override"):
        Solver.from_case(str(tmp_path), presure_solver="cg")


def test_from_case_requires_authoritative_files(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="controlDict.*fvSolution.*fvSchemes.*transportProperties.*0/U.*0/p",
    ):
        Solver.from_case(str(tmp_path))


def test_case_field_loaders_propagate_read_errors(monkeypatch, tmp_path):
    config = FVMConfig(case_name="case", initial_U=None, initial_p=None)

    def fail(*args, **kwargs):
        raise ValueError("malformed field")

    from source.solvers.FVM.fields import field_io

    monkeypatch.setattr(field_io, "read_field", fail)
    with pytest.raises(ValueError, match="malformed field"):
        _load_velocity_field(config, str(tmp_path), 2, {})
    with pytest.raises(ValueError, match="malformed field"):
        _load_pressure_field(config, str(tmp_path), 2, {})


def test_nonuniform_boundary_values_are_parsed_without_banner(tmp_path):
    path = tmp_path / "U"
    path.write_text(
        """
boundaryField
{
    inlet
    {
        type fixedValue;
        value nonuniform List<vector>
        2
        (
            (1 0 0)
            (2 0 0)
        );
    }
}
"""
    )
    parsed = parse_boundary_field(
        path,
        "volVectorField",
        [{"name": "inlet", "startFace": 0, "nFaces": 2}],
        n_elements=1,
        n_interior_faces=0,
    )
    assert np.array_equal(parsed["boundary_values"], [[1, 0, 0], [2, 0, 0]])
    assert parsed["boundary_patches"][0]["value"] == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]


def test_fixed_value_without_value_is_rejected(tmp_path):
    path = tmp_path / "p"
    path.write_text("boundaryField { outlet { type fixedValue; } }")
    with pytest.raises(ValueError, match="without a value"):
        parse_boundary_field(
            path,
            "volScalarField",
            [{"name": "outlet", "startFace": 0, "nFaces": 1}],
            n_elements=1,
            n_interior_faces=0,
        )


def test_internal_field_count_is_validated(tmp_path):
    path = tmp_path / "p"
    path.write_text("internalField nonuniform List<scalar> 3 (1 2);")
    with pytest.raises(ValueError, match="declares 3 values; expected 2"):
        parse_internal_field(path, "volScalarField", 2)


def test_field_writer_does_not_invent_missing_boundary_conditions(tmp_path, hand_built_3d_mesh):
    mesh = deepcopy(hand_built_3d_mesh)
    for patch in mesh["boundary"]:
        for key in tuple(patch):
            if key.startswith("bc_type"):
                patch.pop(key)
    n_total = mesh["n_elements"] + mesh["n_faces"] - mesh["n_interior_faces"]
    with pytest.raises(ValueError, match="has no condition for field 'U'"):
        write_foam_field(
            tmp_path / "U",
            mesh,
            {
                "name": "U",
                "type": "volVectorField",
                "phi": np.zeros((n_total, 3)),
            },
        )


def test_dynamic_mesh_is_explicitly_unsupported(hand_built_3d_mesh, tmp_path):
    config = FVMConfig(
        case_name="moving",
        dynamic_mesh=DynamicMeshConfig.rigid(velocity=[1.0, 0.0, 0.0]),
    )
    with pytest.raises(NotImplementedError, match="ALE mesh-flux"):
        Solver(config, case_dir=str(tmp_path), mesh_data=hand_built_3d_mesh)


def test_turbulence_failure_does_not_switch_to_laminar():
    class BrokenModel:
        def compute_nut(self, *args):
            raise RuntimeError("LES failed")

    solver = Solver.__new__(Solver)
    solver.turbulence = BrokenModel()
    solver.U = np.zeros((1, 3))
    solver.mesh_data = {}
    solver.geo_data = {}
    solver.config = FVMConfig(case_name="les")
    with pytest.raises(RuntimeError, match="LES failed"):
        solver._effective_viscosity()


def test_fixed_value_inlet_is_not_auto_selected_as_wall():
    inlet = {"name": "inlet", "type": "patch", "bc_type_U": "fixedValue"}
    wall = {"name": "body", "type": "wall", "bc_type_U": "fixedValue"}
    assert not _should_compute_yplus(inlet, None)
    assert _should_compute_yplus(wall, None)
