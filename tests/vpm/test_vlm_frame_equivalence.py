"""Standalone-VLM frame-equivalence tests.

The same steady rectangular wing must be represented two equivalent ways and
give the same physical solution (up to machine precision for this linear, rigid
rotation-covariant formulation):

  A. body/wind-frame representation — the wing geometry is inclined by α and
     the relative flow is along the reference streamwise direction (+x);
  B. inflow/inverse-rotation representation — the wing sits flat and the
     freestream is transformed by the same rigid rotation about the span axis,
     i.e. freestream = (cos α, 0, sin α).

Because the plate is rotated about the span axis, the two meshes share the
same spanwise stations, so normalized circulation Γ(y)/Γ_root and the
integrated circulation (∝ CL) must agree directly.

The far-wake model is the explicit standalone rule: trailing legs lie in the
wing's local tangent plane (the freestream component normal to the surface is
discarded).  This rule is rotation-covariant by construction and is what the
frame equivalence below certifies.
"""

import math

import numpy as np
import pytest

from source.solvers.vpm.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.vpm.boundary_elements.vlm.geometry.aircraft import (
    Aircraft,
    Wing,
    WingSegment,
)
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver

CHORD = 1.0
HALF_SPAN = 5.0
NC, NS = 4, 28


@pytest.fixture(scope="module", autouse=True)
def _taichi_cpu():
    """Taichi must be initialised before any VLMLattice is created."""
    import taichi as ti

    ti.init(arch=ti.cpu, default_fp=ti.f64, random_seed=0)


def _rot_y(alpha: float) -> np.ndarray:
    c, s = math.cos(alpha), math.sin(alpha)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _build_plate(alpha_body: float) -> VLMSolver:
    """Rectangular wing (symmetry=0) with its chord axis rotated about +y."""
    R = _rot_y(alpha_body)
    vertex_position = {
        "a": R @ np.array([0.0, 0.0, 0.0]),
        "b": R @ np.array([0.0, HALF_SPAN, 0.0]),
        "c": R @ np.array([CHORD, HALF_SPAN, 0.0]),
        "d": R @ np.array([CHORD, 0.0, 0.0]),
    }
    wing = Wing(uid="main_wing", symmetry=0)
    wing.add_segment(
        WingSegment(
            uid="segment_0",
            vertex_position=vertex_position,
            n_chordwise_panels=NC,
            n_spanwise_panels=NS,
        )
    )
    aircraft = Aircraft(uid="plate")
    aircraft.add_wing(wing)
    return VLMSolver(VLMSetup(surfaces=(VLMSurfaceSetup(aircraft),), max_n_panels=4096))


def _solve_stations(vlm: VLMSolver, V_rel: np.ndarray):
    """Standalone steady solve; return (station_y, spanwise circulation)."""
    vlm.generate_mesh()
    n = vlm.lattice.n_panels
    circulation = vlm.solve(
        external_velocity=np.tile(V_rel, (n, 1)), time_step_size=None, coupled=False
    )
    vortex = vlm.lattice.vortex_point_position.to_numpy()[:n]
    y_mid = 0.5 * (vortex[:, 1, 1] + vortex[:, 2, 1])
    grouped = {}
    for k in range(n):
        grouped.setdefault(round(y_mid[k], 6), []).append(circulation[k])
    stations = sorted(grouped)
    return np.array(stations), np.array([sum(grouped[y]) for y in stations])


@pytest.mark.verification
@pytest.mark.parametrize("angle_of_attack_degrees", [2, 5, 8])
def test_moving_static_frame_equivalence(angle_of_attack_degrees: int):
    """A (pitched plate, streamwise flow) and B (flat plate, rotated inflow)
    must be the same physical problem for the tangent-plane far-wake model."""
    alpha = math.radians(angle_of_attack_degrees)
    v_wind = np.array([1.0, 0.0, 0.0])
    v_body = np.array([math.cos(alpha), 0.0, math.sin(alpha)])

    _, gA = _solve_stations(_build_plate(alpha_body=alpha), v_wind)
    _, gB = _solve_stations(_build_plate(alpha_body=0.0), v_body)

    gAn = gA / np.abs(gA[np.argmax(np.abs(gA))])
    gBn = gB / np.abs(gB[np.argmax(np.abs(gB))])

    err = np.abs(gAn - gBn)
    assert err.max() < 1e-12, f"frame A/B normalized Γ disagree by {err.max():.3e}"
    assert np.sqrt((err**2).mean()) < 1e-12

    integrated_ratio = np.sum(gA) / np.sum(gB)
    assert integrated_ratio == pytest.approx(1.0, rel=1e-12), (
        f"integrated circulation (∝CL) differs between frames: {integrated_ratio}"
    )
