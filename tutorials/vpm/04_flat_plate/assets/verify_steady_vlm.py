#!/usr/bin/env python3
"""Isolate finite-chord VLM versus lifting-line differences without particle coupling.

The mesh study changes chordwise and spanwise resolution separately. The AR
study probes the high-aspect-ratio assumption of lifting-line theory. Neither
study qualifies the time step, regularization or particle resolution of VPM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import taichi as ti

from openonda.tutorial_runner import case_package
from openonda.tutorial_support.vpm_flat_plate.generate_surface import create_flat_plate
from source.solvers.vpm.boundary_elements.vlm.config import VLMMeshSetup, VLMSetup, VLMSurfaceSetup
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver

__package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"
from .theoretical_model import lifting_line_polar


def trefftz_drag(vortex_points, circulation, direction):
    """Independent drag from the straight, freestream-aligned prescribed wake.

    Project each horseshoe's two wake filaments onto the plane normal to the
    stream. An infinite filament induces Gamma*(e_stream cross r)/(2*pi*r^2).
    Half the resulting transverse-velocity KJ sum gives unit-density induced
    drag. This checks the force integral for the solved loading, not the accuracy
    of that loading or a relaxed finite-core particle wake.
    """
    ends = vortex_points[:, 1:3]
    centers = ends.mean(axis=1)
    velocity = np.zeros_like(centers)
    for side, sign in [(0, -1), (1, 1)]:
        displacement = centers[:, None, :] - ends[None, :, side, :]
        displacement -= np.einsum("ijk,k->ij", displacement, direction)[:, :, None] * direction
        radius_squared = np.sum(displacement**2, axis=2)
        if np.any(radius_squared <= 0):
            raise ValueError("Trefftz evaluation point lies on a wake filament")
        velocity += np.sum(
            sign
            * circulation[None, :, None]
            * np.cross(direction, displacement)
            / (2 * np.pi * radius_squared[:, :, None]),
            axis=1,
        )
    force = 0.5 * np.sum(circulation[:, None] * np.cross(velocity, ends[:, 1] - ends[:, 0]), axis=0)
    return float(force @ direction)


def solve(nc, ns, aspect_ratio, angle=5.0):
    alpha = np.radians(angle)
    solver = VLMSolver(
        VLMSetup(
            surfaces=(
                VLMSurfaceSetup(
                    create_flat_plate(
                        chord=1,
                        span=aspect_ratio,
                        angle_of_attack_degrees=0,
                        n_chordwise_panels=nc,
                        n_spanwise_panels=ns,
                    )
                ),
            ),
            mesh=VLMMeshSetup.geometric(ratio=4, region="end"),
            dtype="f64",
            freestream_velocity=(10 * np.cos(alpha), 0, 10 * np.sin(alpha)),
        )
    )
    solver.generate_mesh()
    solver.advance_time(0.0125, 0.0125)
    incoming = np.broadcast_to(solver.freestream_velocity, (solver.lattice.n_panels, 3)).copy()
    solver.solve(incoming)
    solver.compute_postprocess(incoming, np.array(solver.freestream_velocity), 1.0)
    force = solver.compute_forces(1.0, np.array(solver.freestream_velocity))
    cl, cd = float(force["lift_coefficient"]), float(force["drag_coefficient"])
    n = solver.lattice.n_panels
    cd_far = trefftz_drag(
        solver.lattice.vortex_point_position.to_numpy()[:n],
        solver.lattice.circulation.to_numpy()[:n],
        np.array(solver.freestream_velocity) / 10,
    ) / (50 * aspect_ratio)
    ref_cl, ref_cd = lifting_line_polar(angle, aspect_ratio)
    return dict(
        chordwise_panels=nc,
        half_span_panels=ns,
        aspect_ratio=aspect_ratio,
        angle_degrees=angle,
        CL=cl,
        CD=cd,
        CD_trefftz=cd_far,
        relative_near_far_drag_difference=cd / cd_far - 1,
        lifting_line_CL=float(ref_cl),
        lifting_line_CD=float(ref_cd),
        relative_CL_difference=cl / ref_cl - 1,
        relative_CD_difference=cd / ref_cd - 1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=4)
    result = {
        "scope": "Standalone steady VLM with its prescribed wake; no VPM, no LES, no fitting",
        "mesh_refinement": [
            solve(nc, ns, 10.0) for nc, ns in [(4, 7), (8, 14), (8, 28), (16, 28), (16, 56)]
        ],
        "aspect_ratio": [solve(8, 28, ar) for ar in [5.0, 10.0, 20.0, 40.0]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
