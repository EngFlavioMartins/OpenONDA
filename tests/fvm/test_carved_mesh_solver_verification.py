"""Standalone-FVM verification on the carved-cube topology (AGENT_PLAN M4).

Fills the verification gaps not covered by the existing suite (restart
equivalence, MMS operator orders, Taylor–Green, lid cavity are already
certified elsewhere): discrete mass conservation, wall-BC enforcement, and
run-to-run determinism, all on the production carved-cube mesh family.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from source.solvers.FVM.fields.diagnostics import compute_continuity_error

from .test_force_first_principles import _external_flow_solver


def _run(tmp_path, n_steps=5, spacing=0.5):
    solver = _external_flow_solver(tmp_path, spacing=spacing, n_steps=n_steps)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n_steps):
            solver.evolve()
    return solver


def test_mass_conservation_on_carved_mesh(tmp_path):
    """After the pressure correction every cell's net face flux vanishes to
    the pressure-solve tolerance (1e-10 on the residual → per-cell flux
    residuals bounded well below 1e-6 of the through-flux scale), and the
    global boundary flux balances exactly (what enters leaves)."""
    solver = _run(tmp_path)
    mesh, geo = solver.mesh_data, solver.geo_data
    div = compute_continuity_error(np.asarray(solver.phi), mesh, geo)
    volumes = geo["element_volumes"][: mesh["n_elements"]]
    local = np.abs(div) / volumes  # 1/s — local divergence
    assert local.max() < 1e-6, f"max cell divergence {local.max():.3e} 1/s"
    # Global: sum of boundary fluxes = sum of per-cell residuals ≈ 0.
    n_int = mesh["n_interior_faces"]
    net_boundary = float(np.asarray(solver.phi)[n_int:].sum())
    assert abs(net_boundary) < 1e-8, f"net boundary flux {net_boundary:.3e} m³/s"


def test_wall_bc_enforcement_on_carved_cube(tmp_path):
    """No-slip wall: ghost velocity is exactly zero and the wall face flux is
    exactly zero after every step (Dirichlet enforcement is exact, not
    iterative)."""
    solver = _run(tmp_path)
    mesh = solver.mesh_data
    (wall,) = [b for b in mesh["boundary"] if b["name"] == "cube"]
    faces = np.arange(wall["startFace"], wall["startFace"] + wall["nFaces"])
    ghost = mesh["n_elements"] + (faces - mesh["n_interior_faces"])
    assert np.all(np.asarray(solver.U)[ghost] == 0.0), "wall ghost velocity not zero"
    assert np.all(np.asarray(solver.phi)[faces] == 0.0), "wall face flux not zero"


def test_deterministic_for_fixed_configuration(tmp_path):
    """Two runs of the identical serial configuration produce bit-identical
    fields and forces (no hidden nondeterminism in assembly, AMG, or ILU)."""
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    assert np.array_equal(np.asarray(a.U), np.asarray(b.U)), "velocity fields differ"
    assert np.array_equal(np.asarray(a.p), np.asarray(b.p)), "pressure fields differ"
    fa, fb = a.last_forces["cube"], b.last_forces["cube"]
    assert np.array_equal(fa["Ftot"], fb["Ftot"]), f"forces differ: {fa['Ftot']} vs {fb['Ftot']}"
