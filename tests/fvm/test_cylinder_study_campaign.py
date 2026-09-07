"""Resolution-family checks without a separate campaign implementation."""

from types import SimpleNamespace

import numpy as np

from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow import setup


def mesh_for(monkeypatch, dx):
    captured = {}

    def create(_setup, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(setup.fvm, "create_fvm_solver", create)
    setup.create_solver(f"d{dx:g}", dx)
    return captured["mesh"]


def test_recommended_family_refines_every_requested_spacing_by_two(monkeypatch):
    grids = [mesh_for(monkeypatch, dx) for dx in (0.125, 0.0625, 0.03125)]
    requested = []
    for mesh in grids:
        requested.append(
            [
                mesh.source.max_cell_size,
                mesh.source.boundary_cell_size,
                mesh.source.patch_refinements[0].cell_size,
                *(item.cell_size for item in mesh.source.refinements),
                mesh.levels[1] - mesh.levels[0],
            ]
        )
    requested = np.asarray(requested)
    np.testing.assert_allclose(requested[:-1] / requested[1:], 2.0)
    assert [len(mesh.levels) - 1 for mesh in grids] == [2, 4, 8]


def test_wake_refinement_is_compact_and_downstream(monkeypatch):
    mesh = mesh_for(monkeypatch, 0.125)
    refinements = {item.name: item for item in mesh.source.refinements}
    assert refinements["nearBody"].bounds[:4] == (-1.0, 2.0, -1.0, 1.0)
    assert refinements["nearWake"].bounds[:4] == (0.0, 6.0, -1.0, 1.0)
    assert refinements["wake"].bounds[:4] == (0.0, 12.0, -1.5, 1.5)
