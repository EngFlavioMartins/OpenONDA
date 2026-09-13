"""Expose the omitted old-circulation term in the experimental RK virtual row."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests/vpm")]

from _flat_plate_geometry import create_flat_plate
import numpy as np
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver


def main():
    """Compare completed and virtual rows on identical static geometry/offsets."""
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    try:
        wing = create_flat_plate(
            chord=1, span=2, angle_of_attack_degrees=6, n_chordwise_panels=1, n_spanwise_panels=2
        )
        vlm = VLMSolver(
            VLMSetup(surfaces=(VLMSurfaceSetup(wing),), dtype="f64", wake_core_overlap=2.5)
        )
        vlm.generate_mesh()
        vlm.solve(np.array([2.0, 0.0, 0.0]))
        lattice = vlm.lattice
        # One chordwise panel per strip, so local and cumulative circulation
        # coincide; solve() alone does not refresh the postprocess cumulative field.
        old = lattice.circulation.to_numpy()
        lattice.cumulative_circulation_old.from_numpy(old)
        elapsed = 0.01
        offsets = lattice.wake_offset.to_numpy()
        offsets[:] = [2 * elapsed, 0, 0]
        lattice.wake_offset.from_numpy(offsets)
        vlm._bound_transport_ready = False
        completed_matrix, old_velocity, strip_map = vlm._near_wake_particle_influence()

        class Uniform:
            def compute_target_velocity(self, particles, points, **kwargs):
                return np.tile([2.0, 0.0, 0.0], (len(points), 1))

        geometry = vlm._ensure_stage_geometry_fields(elapsed)
        partial = vlm._near_wake_stage_influence(geometry, None, Uniform(), elapsed)
        matrix_error = float(np.max(np.abs(partial - completed_matrix[:, strip_map])))
        missing_rhs = float(np.max(np.abs(old_velocity)))
        assert matrix_error < 1e-12
        assert missing_rhs > 1e-4
        result = {
            "status": "CONFIRMED_RESPONSIVE_FORMULATION_GAP",
            "matrix_max_difference": matrix_error,
            "omitted_old_circulation_normal_velocity": missing_rhs,
            "explanation": "Identical geometry and convection give identical new-row matrices, but only the accepted row includes the nonzero old-circulation RHS. The responsive solve must not be production-qualified from its small algebraic residual.",
        }
        Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        ti.reset()


if __name__ == "__main__":
    main()
