"""Reference cases own their assets and preserve refinement/output contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from openonda.tutorial_runner import load_case_module
from openonda.tutorials import materialize_tutorial


@pytest.mark.parametrize("name", ["cube_flow", "cylinder_shedding_flow"])
def test_reference_case_owns_geometry_and_refines_requested_sizes(tmp_path, monkeypatch, name):
    case = materialize_tutorial(f"coupled_fvm_vpm/{name}/reference_flow", tmp_path)
    # Move the case away from every parent tutorial directory.
    standalone = tmp_path / "standalone"
    case.rename(standalone)
    module = load_case_module(standalone)
    captures = []

    def capture(config, **kwargs):
        captures.append((config, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(module.fvm, "create_fvm_solver", capture)
    for spacing in (0.125, 0.0625):
        module.create_solver(f"spacing-{spacing}", spacing)
    sizes = []
    for config, arguments in captures:
        mesh = arguments["mesh"]
        source = getattr(mesh, "source", mesh)
        assert source.surfaces
        for surface in source.surfaces:
            assert surface.path.is_file()
            assert surface.path.is_relative_to(standalone)
        assert arguments["solution_dir"].is_relative_to(standalone)
        assert arguments["samples_dir"].is_relative_to(standalone)
        assert config.boundaries
        sizes.append([r.cell_size for r in source.patch_refinements])
    np.testing.assert_allclose(np.array(sizes[0]) / sizes[1], 2)
