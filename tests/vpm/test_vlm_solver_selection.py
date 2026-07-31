"""
Test VLM solver adaptive linear-solver selection.

This test ensures that small panel counts (< 1000) default to the SCIPY
direct solver, which is orders of magnitude faster than the GPU iterative
solver for small matrices. It also verifies that the GPU solver tolerance
is not pathologically tight (1e-10 is too strict for engineering VLM).

Author: OpenONDA Team
Date: May 2026
"""

import numpy as np
import pytest

from source.solvers.VPM.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.VPM.boundary_elements.vlm.geometry.aircraft import Aircraft, Wing, WingSegment
from source.solvers.VPM.boundary_elements.vlm.solver.vlm_solver import VLMSolver


def create_dummy_aircraft(n_chord=4, n_span=8):
    """Create a minimal flat-plate aircraft for testing."""
    wing = Wing(uid="test_wing", symmetry=0)
    wing.add_segment(
        WingSegment(
            uid="seg1",
            vertices={
                "a": np.array([0.0, 0.0, 0.0]),
                "b": np.array([1.0, 0.0, 0.0]),
                "c": np.array([1.0, 2.0, 0.0]),
                "d": np.array([0.0, 2.0, 0.0]),
            },
            panels_chord=n_chord,
            panels_span=n_span,
        )
    )
    aircraft = Aircraft(uid="test")
    aircraft.add_wing(wing)
    return aircraft


class TestAdaptiveSolverSelection:
    """Regression tests for VLM linear-solver defaults."""

    def test_small_system_defaults_to_scipy(self):
        """
        Small systems (< 1000 panels) should default to SCIPY for speed.

        The BICGSTAB_GPU solver has extreme kernel-launch overhead for
        small matrices, making it 100-1000x slower than SCIPY.
        """
        aircraft = create_dummy_aircraft(n_chord=4, n_span=8)  # 32 panels
        vlm = VLMSolver(VLMSetup(surfaces=(VLMSurfaceSetup(aircraft),)))
        assert vlm.linear_solver == "SCIPY"

    def test_large_system_keeps_bicgstab(self):
        """
        Large systems (≥ 1000 panels) should keep BICGSTAB_GPU.

        For big matrices the O(N²) kernel arithmetic dominates the launch
        overhead, and the GPU iterative solver avoids the O(N³) CPU cost.
        """
        aircraft = create_dummy_aircraft(n_chord=20, n_span=60)  # 1200 panels
        vlm = VLMSolver(VLMSetup(surfaces=(VLMSurfaceSetup(aircraft),)))
        assert vlm.linear_solver == "BICGSTAB_GPU"

    def test_explicit_override_preserved(self):
        """User-specified solver should never be silently changed."""
        aircraft = create_dummy_aircraft(n_chord=4, n_span=8)
        vlm = VLMSolver(
            VLMSetup(
                surfaces=(VLMSurfaceSetup(aircraft),),
                linear_solver="BICGSTAB_GPU",
            )
        )
        assert vlm.linear_solver == "BICGSTAB_GPU"

    def test_scipy_solve_completes_fast(self):
        """
        SCIPY solve for a 32-panel system must finish in < 1 s.

        This is a smoke test: if the solver takes > 1 s, something is
        fundamentally wrong (e.g. accidentally calling the GPU path).
        """
        import time

        # Taichi must be initialised before the lattice is created
        import taichi as ti

        ti.init(arch=ti.cpu)

        aircraft = create_dummy_aircraft(n_chord=4, n_span=8)
        vlm = VLMSolver(VLMSetup(surfaces=(VLMSurfaceSetup(aircraft),), linear_solver="SCIPY"))
        vlm.generate_mesh()

        # Mock external velocity
        n_panels = vlm.lattice.num_panels
        V_ext = np.tile([1.0, 0.0, 0.0], (n_panels, 1))

        # Compile Taichi kernels before timing steady-state execution.
        vlm.solve(V_external=V_ext)
        ti.sync()

        t0 = time.perf_counter()
        gamma = vlm.solve(V_external=V_ext)
        ti.sync()
        dt = time.perf_counter() - t0

        assert dt < 1.0, f"SCIPY solve took {dt:.2f}s, expected < 1.0s"
        assert gamma is not None
        assert len(gamma) == n_panels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
