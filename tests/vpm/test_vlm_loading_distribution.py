"""
VLM loading-distribution tests (chord-wise / span-wise force extraction).

VLMLoadingDistribution is a pure reduction of per-panel lattice data, so the
core invariant is exact bookkeeping: sectional forces summed over stations
must reproduce the lattice force totals, for the spanwise AND the chordwise
tables, on plain, rotated, and mirrored surfaces.

Also covered:
  - span-axis robustness: a plate rotated 90° about x (span along global z)
    must produce the same sane dy/cl as the y-aligned plate.  A previous
    version hard-coded the global y-axis (dy = |Δy| → 1e-10) which made
    L_prime/cl blow up by ~10 orders of magnitude for fins/rotor blades.
  - symmetry surfaces: mirror half present, stations symmetric about y=0.
  - per-surface sampling flag plumbing (sample_surface_forces).
"""

import numpy as np
import pytest

from source.solvers.VPM.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.VPM.boundary_elements.vlm.geometry.aircraft import (
    Aircraft,
    Wing,
    WingSegment,
)
from source.solvers.VPM.boundary_elements.vlm.solver import VLMLoadingDistribution
from source.solvers.VPM.boundary_elements.vlm.solver.vlm_solver import VLMSolver

DENSITY = 1.225
ALPHA_DEG = 5.0


@pytest.fixture(scope="module", autouse=True)
def _taichi_cpu():
    """Taichi must be initialised before any VLMLattice is created."""
    import taichi as ti

    ti.init(arch=ti.cpu)


def _flat_plate_aircraft(uid="plate", n_chord=4, n_span=12, span=4.0, chord=0.5, symmetry=0):
    """Rectangular flat plate in the z=0 plane, span along +y.

    WingSegment convention (see aircraft.py): a→b is the LEADING EDGE (span
    direction), a→d is the chord direction — matches the flatPlate tutorial's
    flat_plate_surface.json.
    """
    wing = Wing(uid=f"{uid}_wing", symmetry=symmetry)
    wing.add_segment(
        WingSegment(
            uid="seg1",
            vertices={
                "a": np.array([0.0, 0.0, 0.0]),
                "b": np.array([0.0, span, 0.0]),
                "c": np.array([chord, span, 0.0]),
                "d": np.array([chord, 0.0, 0.0]),
            },
            panels_chord=n_chord,
            panels_span=n_span,
        )
    )
    aircraft = Aircraft(uid=uid)
    aircraft.add_wing(wing)
    return aircraft


def _solve_static(vlm, u_ref):
    """One uncoupled steady solve + postprocess (computes panel forces)."""
    n_p = vlm.lattice.num_panels
    v_ext = np.tile(np.asarray(u_ref, dtype=float), (n_p, 1))
    vlm.solve(V_external=v_ext, dt=None, coupled=False)
    vlm.compute_postprocess(v_ext, np.asarray(u_ref, dtype=float), DENSITY, dt=None, coupled=False)


def _extract(vlm, name, u_ref):
    return VLMLoadingDistribution.extract_distributions(
        vlm, name, np.asarray(u_ref, dtype=float), DENSITY
    )


def _u_alpha():
    """Freestream at ALPHA_DEG incidence in the x-z plane."""
    a = np.deg2rad(ALPHA_DEG)
    return np.array([np.cos(a), 0.0, np.sin(a)])


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_spanwise_and_chordwise_sums_match_lattice_totals():
    """Σ sectional forces == Σ panel forces — exact reduction, no leakage."""
    n_chord, n_span = 4, 12
    aircraft = _flat_plate_aircraft(n_chord=n_chord, n_span=n_span)
    vlm = VLMSolver(
        VLMSetup(
            surfaces=(VLMSurfaceSetup(aircraft, sample_forces=True),),
            max_panels=256,
            linear_solver="SCIPY",
        )
    )
    vlm.generate_mesh()
    _solve_static(vlm, _u_alpha())

    dists = _extract(vlm, "plate", _u_alpha())
    sp, ch = dists["spanwise"], dists["chordwise"]

    assert len(sp) == n_span
    assert len(ch) == n_span * n_chord

    f_total = vlm.lattice.get_forces().sum(axis=0)

    f_span = sp[["Fx_sec", "Fy_sec", "Fz_sec"]].to_numpy().sum(axis=0)
    np.testing.assert_allclose(f_span, f_total, rtol=1e-5, atol=1e-8)

    f_chord = ch[["Fx", "Fy", "Fz"]].to_numpy().sum(axis=0)
    np.testing.assert_allclose(f_chord, f_total, rtol=1e-5, atol=1e-8)

    # L' integrates back to the projected total lift: Σ L'·dy == F_total·L̂
    u_hat = _u_alpha()
    l_hat = np.array([0.0, 0.0, 1.0]) - u_hat[2] * u_hat
    l_hat /= np.linalg.norm(l_hat)
    lift_total = float(f_total @ l_hat)
    lift_integrated = float((sp["L_prime"] * sp["dy"]).sum())
    assert lift_integrated == pytest.approx(lift_total, rel=1e-5)


def test_spanwise_stations_are_sane():
    """Stations use physical span edges and keep cell centres inside the tips."""
    aircraft = _flat_plate_aircraft(n_chord=4, n_span=12, span=4.0, chord=0.5)
    vlm = VLMSolver(
        VLMSetup(
            surfaces=(VLMSurfaceSetup(aircraft, sample_forces=True),),
            max_panels=256,
            linear_solver="SCIPY",
        )
    )
    vlm.generate_mesh()
    _solve_static(vlm, _u_alpha())

    sp = _extract(vlm, "plate", _u_alpha())["spanwise"]

    y = sp["y"].to_numpy()
    assert np.all(np.diff(y) > 0), "spanwise stations must be sorted and distinct"
    assert sp["y_over_b"].between(-1.0, 1.0, inclusive="neither").all()
    np.testing.assert_allclose(sp["dy"], 4.0 / 12, rtol=1e-4)
    np.testing.assert_allclose(sp["chord_local"], 0.5, rtol=1e-4)

    # Regression: the old exporter inferred b from the first and last cell
    # centres, incorrectly labelling them as the physical tips (±1).  The
    # one-sided 0..4 m plate has its mid-span at y=2 m and a physical span of
    # 4 m, so the cell-centred coordinates are 2*(y-2)/4.
    np.testing.assert_allclose(sp["y_over_b"], 2.0 * (y - 2.0) / 4.0, rtol=1e-5)
    assert sp["span_edge_min"].min() == pytest.approx(0.0, abs=1e-6)
    assert sp["span_edge_max"].max() == pytest.approx(4.0, abs=1e-6)

    cl = sp["cl"].to_numpy()
    assert np.all(cl > 0.0), "flat plate at +5° must lift everywhere"
    # finite wing: mid-span carries more load than the tip stations
    assert cl[len(cl) // 2] > cl[0]
    assert cl[len(cl) // 2] > cl[-1]


def test_rotated_surface_span_axis():
    """Plate rotated 90° about x (span along z): dy and cl must stay sane.

    Regression: the span axis used to be hard-coded to global y, so a
    vertical surface got dy = max(|Δy|, 1e-10) = 1e-10 and L_prime/cl
    exploded by ~10 orders of magnitude.
    """
    aircraft = _flat_plate_aircraft(uid="fin", n_chord=4, n_span=12, span=4.0, chord=0.5)
    # span +y → +z, plate normal +z → −y: sideslip onto the fin lifts in −y
    vlm = VLMSolver(
        VLMSetup(
            surfaces=(VLMSurfaceSetup(aircraft, rotation_deg=(90.0, 0.0, 0.0)),),
            max_panels=256,
            linear_solver="SCIPY",
        )
    )
    vlm.generate_mesh()
    a = np.deg2rad(ALPHA_DEG)
    u_ref = np.array([np.cos(a), -np.sin(a), 0.0])
    _solve_static(vlm, u_ref)

    sp = _extract(vlm, "fin", u_ref)["spanwise"]

    assert len(sp) == 12
    # station width must be the true strip width, not the y-shadow (~0)
    np.testing.assert_allclose(sp["dy"], 4.0 / 12, rtol=1e-4)
    # forces still sum to the lattice total
    f_total = vlm.lattice.get_forces().sum(axis=0)
    f_span = sp[["Fx_sec", "Fy_sec", "Fz_sec"]].to_numpy().sum(axis=0)
    np.testing.assert_allclose(f_span, f_total, rtol=1e-5, atol=1e-8)
    # the fin carries side force; per-station magnitudes must be physical
    assert np.all(np.abs(sp["Fy_sec"]) < 1e3)
    assert np.isfinite(sp["L_prime"]).all() and np.isfinite(sp["cl"]).all()
    assert np.abs(sp["L_prime"]).max() < 1e3, "L_prime exploded → span axis broken"


def test_symmetry_surface_has_both_halves():
    """symmetry=2 wing: mirror half present, stations symmetric, totals match."""
    n_span = 8
    aircraft = _flat_plate_aircraft(
        uid="sym_plate", n_chord=3, n_span=n_span, span=2.0, chord=0.5, symmetry=2
    )
    vlm = VLMSolver(
        VLMSetup(
            surfaces=(VLMSurfaceSetup(aircraft),),
            max_panels=256,
            linear_solver="SCIPY",
        )
    )
    vlm.generate_mesh()
    _solve_static(vlm, _u_alpha())

    sp = _extract(vlm, "sym_plate", _u_alpha())["spanwise"]

    assert set(sp["half"]) == {"orig", "mirror"}
    assert len(sp) == 2 * n_span

    # mirrored stations sit at −y of the originals
    y_orig = np.sort(sp[sp.half == "orig"]["y"].to_numpy())
    y_mirr = np.sort(-sp[sp.half == "mirror"]["y"].to_numpy())
    np.testing.assert_allclose(y_orig, y_mirr, atol=1e-5)

    # symmetric loading: cl(y) == cl(−y)
    sp_sorted = sp.sort_values("y")
    cl = sp_sorted["cl"].to_numpy()
    np.testing.assert_allclose(cl, cl[::-1], rtol=1e-3)

    f_total = vlm.lattice.get_forces().sum(axis=0)
    f_span = sp[["Fx_sec", "Fy_sec", "Fz_sec"]].to_numpy().sum(axis=0)
    np.testing.assert_allclose(f_span, f_total, rtol=1e-5, atol=1e-8)


def test_sampling_flag_plumbing():
    """Global default and per-surface override of sample_surface_forces."""
    a1 = _flat_plate_aircraft(uid="s1", n_chord=2, n_span=2)
    a2 = _flat_plate_aircraft(uid="s2", n_chord=2, n_span=2)
    a3 = _flat_plate_aircraft(uid="s3", n_chord=2, n_span=2)

    vlm = VLMSolver(
        VLMSetup(
            surfaces=(
                VLMSurfaceSetup(a1),
                VLMSurfaceSetup(a2, sample_forces=False),
                VLMSurfaceSetup(a3, sample_forces=True),
            ),
            max_panels=64,
            sample_surface_forces=True,
        )
    )

    assert vlm._surface_sampling == {"s1": True, "s2": False, "s3": True}
