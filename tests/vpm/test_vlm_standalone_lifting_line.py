"""
Standalone VLM spanwise loading vs Prandtl lifting-line certification.

The flat-plate tutorial (tutorials/VPM/flatPlate, wind-frame static case at
alpha = 8 deg, chordwise = 8, spanwise = 14) solves the VLM standalone and
plots the spanwise circulation against ``theoretical_model.liftingline_circulation``.

This certification test closes the loop on the "almost-constant spanwise
loading" regression: the semi-infinite trailing legs of the standalone horseshoe
must lie in the wing plane so the finite wing keeps a lifting-line tip taper and
the whole distribution tracks the Prandtl model.

The comparison uses the *actual* VLM geometry:

  - every spanwise station is built from the bound-leg midpoints of the lattice
    (sum of |gamma| over the chordwise panels in that station column);
  - the strip edges are the real panel corner y-coordinates (geometric mesh,
    NOT an assumed uniform spacing), so the mesh-dependent spacing and the
    tip cells are handled exactly;
  - the lifting-line model is cell-averaged over each real strip (Gauss-Legendre
    quadrature on [edge_min, edge_max]) rather than sampled at the station
    midpoint, so a wide or tip-bunched cell is not pinned to a single interior
    value of a strongly-varying distribution.

Checks (values certified on the tutorial configuration):
  - interior |y| < 0.8 * half-span L2 error (normalised by root gamma) <= 5%;
  - interior max error <= 8%;
  - full-span L2 error <= 7%;
  - integrated CL within 6% of the lifting-line CL;
  - the outer-station / root-gamma ratio decreases with NS (tip taper) and it
    must already be below the pre-fix plateau (~0.41 at NS=28) at NS=28;
  - gamma(cl) symmetry: the two mirrored halves agree.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from source.solvers.VPM.boundary_elements.vlm.config import VLMMeshSetup, VLMSetup, VLMSurfaceSetup
from source.solvers.VPM.boundary_elements.vlm.geometry.aircraft import (
    Aircraft,
    Wing,
    WingSegment,
)
from source.solvers.VPM.boundary_elements.vlm.solver.vlm_solver import VLMSolver

HALF_SPAN = 5.0
CHORD = 1.0
FULL_SPAN = 2.0 * HALF_SPAN
REF_AREA = FULL_SPAN * CHORD
U_INF = 10.0
ALPHA_DEG = 8.0
ALPHA = math.radians(ALPHA_DEG)
REF_VELOCITY = U_INF * np.array([math.cos(ALPHA), 0.0, math.sin(ALPHA)])
LL_N_TERMS = 120
NS_SWEEP = (8, 14, 28, 56)  # 14 = tutorial resolution

_THEORY = (
    Path(__file__).resolve().parents[2]
    / "tutorials"
    / "VPM"
    / "flatPlate"
    / "assets"
    / "theoretical_model.py"
)


@pytest.fixture(scope="module", autouse=True)
def _taichi_cpu():
    """Taichi must be initialised before any VLMLattice is created."""
    import taichi as ti

    ti.init(arch=ti.cpu, default_fp=ti.f64, random_seed=0)


@pytest.fixture(scope="module")
def _lifting_line():
    """Import the tutorial's Prandtl lifting-line model once per module."""
    spec = importlib.util.spec_from_file_location("flatplate_theoretical_model", _THEORY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flat_plate_aircraft(uid="plate", n_chord=8, n_span=14):
    """Rectangular plate, root at y=0, tip at y=+HALF_SPAN, mirrored (symmetry=2).

    Matches the flatPlate tutorial: chord 1, full span 10, geometric spanwise
    mesh bunched at the tips.
    """
    wing = Wing(uid=f"{uid}_wing", symmetry=2)
    wing.add_segment(
        WingSegment(
            uid="segment_0",
            vertices={
                "a": np.array([0.0, 0.0, 0.0]),
                "b": np.array([0.0, HALF_SPAN, 0.0]),
                "c": np.array([CHORD, HALF_SPAN, 0.0]),
                "d": np.array([CHORD, 0.0, 0.0]),
            },
            panels_chord=n_chord,
            panels_span=n_span,
        )
    )
    aircraft = Aircraft(uid=uid)
    aircraft.add_wing(wing)
    return aircraft


def _solve_standalone(aircraft, mesh: str = "geom") -> VLMSolver:
    """One standalone (uncoupled) steady solve on the tutorial's geometric mesh.

    ``mesh == "geom"`` uses the tutorial's tip-refined mesh
    (``VLMMeshSetup.geometric(ratio=4.0, region="end")``); ``"uniform"`` uses the
    default uniform spacing (coarse outer cells, no tip bunching)."""
    mesh_setup = (
        VLMMeshSetup.geometric(ratio=4.0, region="end") if mesh == "geom" else VLMMeshSetup()
    )
    vlm = VLMSolver(
        VLMSetup(
            surfaces=(VLMSurfaceSetup(aircraft),),
            mesh=mesh_setup,
            max_panels=2048,
            linear_solver="SCIPY",
        )
    )
    vlm.generate_mesh()
    n_p = vlm.lattice.num_panels
    vlm.solve(external_velocity=np.tile(REF_VELOCITY, (n_p, 1)), time_step_size=None, coupled=False)
    return vlm


def _spanwise_loading(vlm):
    """Per-station |total circulation| on the actual geometric strips.

    Returns (y, gamma, dy, edge_min, edge_max) arrays where each station is a
    chordwise panel column at a bound-leg midpoint y, gamma is the sum of the
    column's panel circulations, and dy is the width of the station's real cell
    derived from the panel corner y-coordinates.
    """
    n = vlm.lattice.num_panels
    gamma = np.abs(vlm.lattice.circulation.to_numpy()[:n])
    vortex = vlm.lattice.vortex_points.to_numpy()[:n]
    corners = vlm.lattice.corners.to_numpy()[:n]

    y_mid = 0.5 * (vortex[:, 1, 1] + vortex[:, 2, 1])
    grouped: dict[float, list[float]] = {}
    for k in range(n):
        grouped.setdefault(round(y_mid[k], 6), []).append(gamma[k])

    edge_ys = np.unique(np.round(corners[:, :, 1].ravel(), 6))
    stations = sorted(grouped)
    y = []  # noqa: F841 (kept for symmetry with the mirrored station pass below)
    totals = np.array([sum(grouped[s]) for s in stations])
    ys = np.array(stations)

    dy = []
    edge_min = []
    edge_max = []
    for s in stations:
        below = edge_ys[edge_ys <= s]
        above = edge_ys[edge_ys >= s]
        lo = below.max()
        hi = above.min()
        dy.append(hi - lo)
        edge_min.append(lo)
        edge_max.append(hi)
    return ys, totals, np.array(dy), np.array(edge_min), np.array(edge_max)


def _lifting_line_at(model, ys: np.ndarray):
    """Prandtl lifting-line circulation evaluated at the VLM station midpoints."""
    df = model.liftingline_circulation(
        ys, FULL_SPAN, CHORD, ALPHA, U_inf=U_INF, a0=2.0 * np.pi, n_terms=LL_N_TERMS
    )
    return np.asarray(df["Gamma"])


def _lifting_line_cell_averaged(
    model, edge_min: np.ndarray, edge_max: np.ndarray, n_points: int = 32
):
    """Cell-averaged Prandtl circulation over each real strip [edge_min, edge_max].

    Each VLM strip is a finite-width cell, so the lifting-line model is averaged
    over that cell with Gauss-Legendre quadrature instead of being sampled at the
    station midpoint.  ``edge_min``/``edge_max`` come from the actual panel
    corner y-coordinates of the geometric mesh.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    lo = np.asarray(edge_min, dtype=np.float64)
    hi = np.asarray(edge_max, dtype=np.float64)
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    ys = mid[:, None] + half[:, None] * nodes[None, :]
    df = model.liftingline_circulation(
        ys.ravel(), FULL_SPAN, CHORD, ALPHA, U_inf=U_INF, a0=2.0 * np.pi, n_terms=LL_N_TERMS
    )
    gamma = np.asarray(df["Gamma"]).reshape(-1, n_points)
    return 0.5 * (gamma * weights[None, :]).sum(axis=1)


def _cl_model(model) -> float:
    "Integrated CL of the lifting line over the full span (rho=1)."
    y_grid = np.linspace(-HALF_SPAN, HALF_SPAN, 8001)
    df = model.liftingline_circulation(
        y_grid, FULL_SPAN, CHORD, ALPHA, U_inf=U_INF, a0=2.0 * np.pi, n_terms=LL_N_TERMS
    )
    return float(2.0 * np.trapezoid(df["Gamma"].to_numpy(), y_grid) / (U_INF * REF_AREA))


@pytest.mark.verification
def test_standalone_loading_matches_lifting_line_tutorial_resolution(_lifting_line):
    """Tutorial resolution (NS=14): whole-distribution L2, interior max and CL.

    The standalone flat-plate VLM must reproduce the Prandtl spanwise
    circulation within a few percent, on the actual geometric strips.
    """
    vlm = _solve_standalone(_flat_plate_aircraft(n_span=14))
    ys, totals, dy, edge_min, edge_max = _spanwise_loading(vlm)
    gamma_ll = np.abs(_lifting_line_cell_averaged(_lifting_line, edge_min, edge_max))

    root = float(np.max(totals))
    interior = np.abs(ys) < 0.8 * HALF_SPAN

    l2_int = float(np.sqrt(np.mean((totals - gamma_ll)[interior] ** 2)) / root)
    max_int = float(np.max(np.abs((totals - gamma_ll)[interior])) / root)
    l2_all = float(np.sqrt(np.mean((totals - gamma_ll) ** 2)) / root)

    cl_vlm = float(2.0 * np.trapezoid(totals, ys) / (U_INF * REF_AREA))
    cl_ll = _cl_model(_lifting_line)
    rel_cl = abs(cl_vlm - cl_ll) / cl_ll

    assert interior.sum() >= len(ys) // 2, "interior mask must cover at least half the span"
    assert l2_int <= 0.05, f"interior L2 {l2_int:.4f} > 5%"
    assert max_int <= 0.08, f"interior max {max_int:.4f} > 8%"
    assert l2_all <= 0.07, f"full-span L2 {l2_all:.4f} > 7%"
    assert rel_cl <= 0.06, f"integrated CL rel error {rel_cl:.4f} > 6%"


@pytest.mark.verification
def test_tip_taper_decreases_with_resolution(_lifting_line):
    """NS in {8, 14, 28, 56}: outer/root ratio decreases and LL peaks.

    The near-constant-loading regression would pin the outer-station gamma to
    its root plateau (~0.41 at NS=28).  A healthy finite wing must taper to the
    tips, track the lifting-line distribution at the interior, and approach the
    LL tip drop as the mesh bunches at the ends.
    """
    ratios: list[float] = []
    for ns in NS_SWEEP:
        vlm = _solve_standalone(_flat_plate_aircraft(n_span=ns))
        ys, totals, _, _, _ = _spanwise_loading(vlm)
        root = float(np.max(totals))
        idx_tip = int(np.argmax(np.abs(ys)))
        ratios.append(float(totals[idx_tip] / root))

    assert all(a > b for a, b in zip(ratios, ratios[1:], strict=False)), (
        f"outer/root ratio must monotonically decrease with NS, got {ratios}"
    )
    # NS=28: must already be below the pre-fix plateau (~0.41)
    assert ratios[2] < 0.41, f"outer/root at NS=28 {ratios[2]:.4f} !< 0.41"
    assert ratios[-1] < 0.25, f"outer/root at NS=56 {ratios[-1]:.4f} !< 0.25"


@pytest.mark.verification
def test_loading_symmetry():
    """The mirror half must produce the mirror-image loading."""
    vlm = _solve_standalone(_flat_plate_aircraft(n_span=14))
    n = vlm.lattice.num_panels
    gamma = np.abs(vlm.lattice.circulation.to_numpy()[:n])
    vortex = vlm.lattice.vortex_points.to_numpy()[:n]
    y_mid = 0.5 * (vortex[:, 1, 1] + vortex[:, 2, 1])

    pos, neg = {}, {}
    for k in range(n):
        (pos if y_mid[k] >= 0 else neg).setdefault(round(abs(y_mid[k]), 6), []).append(gamma[k])

    y_pos = np.array(sorted(pos))
    y_neg = np.array(sorted(neg))
    np.testing.assert_allclose(y_pos, y_neg, atol=1e-6)
    for yy in y_pos:
        g_pos = sum(pos[yy])
        g_neg = sum(neg[yy])
        assert g_pos == pytest.approx(g_neg, rel=1e-6), f"asymmetric loading at |y|={yy}"


def _coupled_outer_ratio(vlm, time_step_size) -> float:
    """Outer-station Γ / root Γ for one coupled solve at the given dt."""
    n_p = vlm.lattice.num_panels
    vlm._last_reference_velocity = (
        REF_VELOCITY  # wake_offset = reference_velocity * time_step_size needs the cached ref
    )
    vlm.solve(
        external_velocity=np.tile(REF_VELOCITY, (n_p, 1)),
        time_step_size=time_step_size,
        coupled=True,
    )
    n = vlm.lattice.num_panels
    gamma = np.abs(vlm.lattice.circulation.to_numpy()[:n])
    vortex = vlm.lattice.vortex_points.to_numpy()[:n]
    y_mid = 0.5 * (vortex[:, 1, 1] + vortex[:, 2, 1])
    grouped: dict[float, float] = {}
    for k in range(n):
        grouped[round(y_mid[k], 6)] = grouped.get(round(y_mid[k], 6), 0.0) + gamma[k]
    ys = np.array(sorted(grouped))
    totals = np.array([grouped[round(y, 6)] for y in ys])
    root = float(np.max(totals))
    return float(totals[int(np.argmax(np.abs(ys)))] / root)


@pytest.mark.verification
def test_coupled_tip_taper_depends_on_mesh_resolution_not_dt():
    """Disentangle Issue #3: the too-full coupled tip is a mesh-resolution
    artifact, NOT a dt=None / wake_offset defect.

    For a fixed mesh the coupled outer/root Γ ratio is essentially flat across
    dt (wake_offset = reference_velocity·time_step_size only shifts it a few percent), while changing the
    mesh resolution — uniform coarse vs the tutorial's tip-refined geometric
    mesh — moves it by tens of percent.  In particular the geometric tip mesh
    at the same NS must produce a strictly lower (more tapered) outer ratio than
    the uniform mesh, at dt=None AND at finite dt.  This guards against someone
    "fixing" the coupled tip by touching the wake-offset logic, which is not the
    lever.
    """
    vlm = _solve_standalone(_flat_plate_aircraft(n_span=28), mesh="uniform")
    r_uniform_dt0 = _coupled_outer_ratio(vlm, time_step_size=None)
    vlm = _solve_standalone(_flat_plate_aircraft(n_span=28), mesh="uniform")
    r_uniform_dt = _coupled_outer_ratio(vlm, time_step_size=0.0125)
    vlm = _solve_standalone(_flat_plate_aircraft(n_span=28), mesh="geom")
    r_geom_dt0 = _coupled_outer_ratio(vlm, time_step_size=None)
    vlm = _solve_standalone(_flat_plate_aircraft(n_span=28), mesh="geom")
    r_geom_dt = _coupled_outer_ratio(vlm, time_step_size=0.0125)

    # wake_offset is NOT the lever: dt barely moves the ratio on a fixed mesh
    assert abs(r_uniform_dt - r_uniform_dt0) < 0.05, (
        f"wake_offset must not dominate the coupled tip (uniform: {r_uniform_dt0:.4f} -> {r_uniform_dt:.4f})"
    )
    assert abs(r_geom_dt - r_geom_dt0) < 0.05, (
        f"wake_offset must not dominate the coupled tip (geom: {r_geom_dt0:.4f} -> {r_geom_dt:.4f})"
    )
    # mesh resolution IS the lever: geometric tip bunching lowers the ratio
    assert r_geom_dt0 < r_uniform_dt0 - 0.05, (
        f"geometric tip mesh must taper more than uniform at dt=None "
        f"({r_geom_dt0:.4f} vs {r_uniform_dt0:.4f})"
    )
    assert r_geom_dt < r_uniform_dt - 0.05, (
        f"geometric tip mesh must taper more than uniform at dt>0 "
        f"({r_geom_dt:.4f} vs {r_uniform_dt:.4f})"
    )
