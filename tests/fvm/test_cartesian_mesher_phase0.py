# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 0 contract tests for the planned general Cartesian mesher.

These tests freeze the public and acceptance-matrix contract for the staged
implementation. They remain deliberately independent of tutorial setup code
and are not weakened to accommodate a particular geometry.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.fvm.cartesian_acceptance_fixtures import make_acceptance_fixtures

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TARGET_PUBLIC_NAMES = {
    "CartesianMesher",
    "BoxDomain",
    "BoxPatches",
    "STLSurface",
    "BoxRefinement",
    "SphereRefinement",
    "FeatureRefinement",
    "BoundaryLayers",
}
REQUIRED_PIPELINE_MODULES = {
    "config.py",
    "surface.py",
    "features.py",
    "size_field.py",
    "octree.py",
    "extraction.py",
    "surface_recovery.py",
    "boundary_layers.py",
    "optimisation.py",
    "native_mesh.py",
    "report.py",
    "mesher.py",
}
VALID_FIXTURE_NAMES = (
    "rotated_box",
    "ellipsoid",
    "torus",
    "finite_naca_wing",
    "two_disjoint_bodies",
)
INVALID_FIXTURE_NAMES = (
    "open_edge",
    "non_manifold_edge",
    "inverted_component",
    "degenerate_triangle",
)


def test_target_public_cartesian_api_is_exposed():
    """The target names must be exported through ``openonda.fvm``."""
    import openonda.fvm as fvm

    missing = sorted(name for name in TARGET_PUBLIC_NAMES if not hasattr(fvm, name))
    assert not missing, f"target Cartesian mesher exports are missing: {missing}"


def test_target_cartesian_pipeline_is_split_into_required_modules():
    """The production package must have one independently testable stage per concern."""
    package = REPOSITORY_ROOT / "source" / "solvers" / "fvm" / "mesh" / "cartesian"
    missing = sorted(name for name in REQUIRED_PIPELINE_MODULES if not (package / name).is_file())
    assert not missing, f"Cartesian mesher pipeline modules are missing: {missing}"


def test_cartesian_mesher_production_package_has_no_forbidden_geometry_names():
    """The future package must not contain geometry-recognition special cases."""
    package = REPOSITORY_ROOT / "source" / "solvers" / "fvm" / "mesh" / "cartesian"
    assert package.is_dir(), f"target Cartesian mesher package is missing: {package}"
    forbidden = ("cylinder", "airfoil", "cube", "plate", "sphere", "tutorial")
    allowed = {"SphereRefinement"}
    offenders: list[str] = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name | ast.Attribute | ast.Constant):
                value = (
                    node.id
                    if isinstance(node, ast.Name)
                    else node.attr
                    if isinstance(node, ast.Attribute)
                    else node.value
                )
                if (
                    isinstance(value, str)
                    and value not in allowed
                    and any(term in value.lower() for term in forbidden)
                ):
                    offenders.append(f"{path.name}:{getattr(node, 'lineno', '?')}:{value}")
    assert not offenders, "forbidden geometry-specific production identifiers: " + ", ".join(
        offenders
    )


@pytest.mark.parametrize("fixture_name", VALID_FIXTURE_NAMES)
def test_geometry_independence_acceptance_matrix(tmp_path, fixture_name):
    """Every valid fixture must use the same target construction and build path."""
    import openonda.fvm as fvm

    fixtures = make_acceptance_fixtures(tmp_path)
    fixture = fixtures[fixture_name]
    patches = ("body_a", "body_b") if fixture_name == "two_disjoint_bodies" else (fixture_name,)
    surfaces = tuple(
        fvm.STLSurface(path, patch=patch)
        for path, patch in zip(fixture.paths, patches, strict=True)
    )
    mesher = fvm.CartesianMesher(
        domain=fvm.BoxDomain(
            bounds=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5),
            patches=fvm.BoxPatches(
                xmin="inlet",
                xmax="outlet",
                ymin="farfield",
                ymax="farfield",
                zmin="front",
                zmax="back",
            ),
        ),
        surfaces=surfaces,
        max_cell_size=0.50,
        boundary_cell_size=0.25,
        min_cell_size=0.125,
        features=fvm.FeatureRefinement(angle=35.0, cell_size=0.125),
    )
    mesh = mesher.build()
    assert {patch["name"] for patch in mesh["boundary"]} >= {"inlet", "outlet", *patches}


@pytest.mark.parametrize("fixture_name", INVALID_FIXTURE_NAMES)
def test_invalid_surface_fixtures_fail_diagnostically(tmp_path, fixture_name):
    """Broken topology must fail explicitly during target surface construction."""
    import openonda.fvm as fvm

    fixture = make_acceptance_fixtures(tmp_path)[fixture_name]
    with pytest.raises((ValueError, RuntimeError)):
        fvm.STLSurface(fixture.paths[0], patch="broken")


def test_target_configuration_objects_are_immutable():
    """Declarative intent objects cannot be mutated after construction."""
    import dataclasses

    import openonda.fvm as fvm

    patches = fvm.BoxPatches(
        xmin="inlet", xmax="outlet", ymin="farfield", ymax="farfield", zmin="front", zmax="back"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        patches.xmin = "changed"


def test_target_cartesian_mesher_has_native_build_contract():
    """The public object must expose both equivalent build entry points and a report."""
    import openonda.fvm as fvm

    assert hasattr(fvm.CartesianMesher, "build")
    assert callable(fvm.CartesianMesher)
    assert hasattr(fvm.CartesianMesher, "report")
