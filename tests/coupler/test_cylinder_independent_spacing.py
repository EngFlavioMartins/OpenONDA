"""Independent physical cylinder factors agree with the native case builder."""

import importlib.util
import itertools
import json
from pathlib import Path
import sys

import pytest

from openonda.tutorial_runner import load_case_module

CASE = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def _study():
    path = Path(__file__).resolve().parents[2] / "tests/support/cylinder/run_sensitivity.py"
    spec = importlib.util.spec_from_file_location("independent_cylinder_sensitivity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "requested",
    [
        {},
        {"particle_spacing_ratio": 1.25},
        {"particle_spacing_ratio": 1.5},
        {"particle_spacing_ratio": 1.5, "core_radius_ratio": 0.8},
        {"particle_spacing_ratio": 1.5, "blend_width_ratio": 7.0},
        {"particle_spacing_ratio": 1.5, "release_width_ratio": 3.0},
        {"particle_spacing_ratio": 1.5, "span": 0.48},
        {"particle_spacing_ratio": 1.5, "span": 1.92},
        {"particle_spacing_ratio": 1.25, "exchange_dt": 0.02},
    ],
)
def test_native_builder_preserves_independent_physical_lengths(requested):
    study = _study()
    module = load_case_module(CASE)
    baseline, _ = study._study_overrides({})
    base_fvm, base_vpm, _, base_mesh = module.build_case(end_time=100.0, overrides=baseline)
    values, resolved = study._study_overrides(requested)
    flow, particles, policy, mesh = module.build_case(end_time=100.0, overrides=values)
    viscous = particles.numerics.viscous
    hp = viscous.particle_spacing
    assert hp == pytest.approx(resolved["particle_spacing"])
    assert viscous.core_radius_ratio * hp == pytest.approx(resolved["core_radius"])
    assert policy.eta_blend_width == pytest.approx(resolved["blend_width"])
    assert policy.vpm_only_width == pytest.approx(resolved["release_width"])
    assert viscous.core_radius_ratio == pytest.approx(resolved["sigma_over_hp"])
    assert round(resolved["span"] / hp) == resolved["particle_span_layers"]
    assert resolved["particle_span_layers"] >= 6
    assert flow.time.time_step_size == base_fvm.time.time_step_size
    assert particles.numerics.max_n_particles == base_vpm.numerics.max_n_particles
    assert particles.numerics.time_step_size == pytest.approx(requested.get("exchange_dt", 0.04))
    if "span" not in requested:
        assert mesh.domain.bounds == base_mesh.domain.bounds
        assert mesh.levels == base_mesh.levels
        assert mesh.source.max_cell_size == base_mesh.source.max_cell_size
        assert mesh.source.cell_size_anchor == base_mesh.source.cell_size_anchor
        assert mesh.source.patch_refinements == base_mesh.source.patch_refinements
    for name, default in (
        ("core_radius_ratio", 1.0),
        ("blend_width_ratio", 6.0),
        ("release_width_ratio", 2.0),
    ):
        target = {
            "core_radius_ratio": "core_radius",
            "blend_width_ratio": "blend_width",
            "release_width_ratio": "release_width",
        }[name]
        assert resolved[target] == pytest.approx(0.08 * requested.get(name, default))


def test_every_single_factor_and_interaction_resolves_native_geometry():
    study = _study()
    module = load_case_module(CASE)
    factors = [(name, value) for name, values in study.FACTORS.items() for value in values]
    requests = [{name: value} for name, value in factors]
    requests.extend(
        {left: x, right: y}
        for (left, x), (right, y) in itertools.combinations(factors, 2)
        if left != right
    )
    for requested in requests:
        values, resolved = study._study_overrides(requested)
        _, particles, policy, _ = module.build_case(end_time=100.0, overrides=values)
        hp = particles.numerics.viscous.particle_spacing
        assert hp == pytest.approx(resolved["particle_spacing"])
        assert particles.numerics.viscous.core_radius_ratio * hp == pytest.approx(
            resolved["core_radius"]
        )
        assert policy.eta_blend_width == pytest.approx(resolved["blend_width"])
        assert policy.vpm_only_width == pytest.approx(resolved["release_width"])


def test_legacy_confounded_report_is_preserved(tmp_path, monkeypatch):
    study = _study()
    report = tmp_path / "sensitivity.json"
    original = json.dumps({"schema": "openonda-cylinder-sensitivity/2", "runs": [{"label": "old"}]})
    report.write_text(original)
    monkeypatch.setattr(sys, "argv", ["run_sensitivity.py", "--run-dir", str(tmp_path), "--resume"])
    with pytest.raises(ValueError, match="confounded"):
        study.main()
    assert report.read_text() == original
