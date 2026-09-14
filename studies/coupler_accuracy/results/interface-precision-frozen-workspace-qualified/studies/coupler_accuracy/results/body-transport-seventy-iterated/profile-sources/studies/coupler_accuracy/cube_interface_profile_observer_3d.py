#!/usr/bin/env python3
"""Observe matched 3D cube profiles only after accepted coupling intervals.

The lines are observations of the full three-dimensional solvers. FVM profiles
use affine 12-cell interpolation. A second reference reconstruction uses the
identical small-FVM stencil, separating field error from stencil truncation at
the artificial boundary. No sampled reference value feeds either hybrid solver.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np

import openonda.fvm as fvm
from source.coupler.vorticity_transfer import _particle_state_snapshot
from studies.coupler_accuracy import cube_interface_cadence_3d as cadence
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_coupled_trial import MatchedComparison
from studies.coupler_accuracy.experimental_interface_iteration import FVM_FIELDS


def state_signature(observer, particle_solver):
    values = {}
    for prefix, solver in (("full", observer.full), ("small", observer.small)):
        for name in FVM_FIELDS:
            values[prefix + "_" + name] = np.asarray(getattr(solver, name))
        values[prefix + "_clock"] = np.array([solver.time, solver.step])
    values.update({"vpm_" + name: value for name, value in _particle_state_snapshot(particle_solver).items()})
    values["vpm_clock"] = np.array([particle_solver.time, particle_solver.step])
    values["panel_strength"] = particle_solver.panel_solver.lattice.source_strength.to_numpy()
    result = {}
    for name, value in values.items():
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
        result[name] = digest.hexdigest()
    return result


def apply_stencil(velocity, indices, weights):
    return np.sum(np.asarray(velocity)[indices] * weights[:, :, None], axis=1)


@contextmanager
def accepted_profiles(args, record):
    original = MatchedComparison.measure
    state = {}

    def measure(observer, particle_solver, **kwargs):
        original(observer, particle_solver, **kwargs)
        tick = observer.small.step
        final_tick = args.steps * args.substeps
        if tick % args.profile_every_fvm_steps and tick != final_tick:
            return
        before = state_signature(observer, particle_solver)
        if not state:
            x = np.linspace(-3., 10., 209)
            points = np.concatenate([np.column_stack((x, np.full_like(x, y), np.zeros_like(x))) for y in (0., .75)])
            fluid = np.max(np.abs(points), axis=1) > .5
            small = fluid & (np.max(np.abs(points), axis=1) <= 1.5)
            full_centres = observer.full.get_cell_centre_coordinates()
            small_centres = observer.small.get_cell_centre_coordinates()
            full_sampler = fvm.LineSampler(start=[-3., 0., 0.], end=[10., 0., 0.], n_points=len(points), k=12, reconstruction="affine")
            small_sampler = fvm.LineSampler(start=[-1.5, 0., 0.], end=[1.5, 0., 0.], n_points=int(small.sum()), k=12, reconstruction="affine")
            # Two parallel lines share one cached native interpolation operator.
            full_sampler.points = points[fluid]
            small_sampler.points = points[small]
            full_indices, full_weights = full_sampler._interpolation_stencil(full_centres)
            small_indices, small_weights = small_sampler._interpolation_stencil(small_centres)
            # The cropped mesh recomputes centroids with a different summation
            # order. Retain the original trial's geometric matching threshold;
            # both sampled fields below use exactly the same small-mesh weights.
            coordinate_difference = float(np.max(np.abs(small_centres - full_centres[observer.ids])))
            np.testing.assert_allclose(small_centres, full_centres[observer.ids], rtol=0, atol=1e-13)
            record["shared_cell_coordinate_maximum_difference"] = coordinate_difference
            geometry = args.output / "profile-geometry.npz"
            np.savez_compressed(geometry, position=points, fluid_mask=fluid, small_mask=small,
                                full_cell_centres=full_centres, small_cell_centres=small_centres,
                                full_indices=full_indices, full_weights=full_weights,
                                small_indices=small_indices, small_weights=small_weights, shared_cell_ids=observer.ids)
            record["geometry"] = hash_file(geometry)
            state.update(points=points, fluid=fluid, small=small, full_indices=full_indices,
                         full_weights=full_weights, small_indices=small_indices, small_weights=small_weights)
        full_velocity = observer.full.get_velocity_field().copy()
        small_velocity = observer.small.get_velocity_field().copy()
        full_profile = apply_stencil(full_velocity, state["full_indices"], state["full_weights"])
        small_profile = apply_stencil(small_velocity, state["small_indices"], state["small_weights"])
        matched_reference = apply_stencil(full_velocity[observer.ids], state["small_indices"], state["small_weights"])
        particle_profile = particle_solver.compute_velocity_at_points(
            state["points"][state["fluid"]], include_freestream=True, include_body=True,
        )
        after = state_signature(observer, particle_solver)
        assert before == after, "Profile observation changed numerical solution fields or clocks"
        assert all(np.all(np.isfinite(value)) for value in (full_profile, small_profile, matched_reference, particle_profile))
        path = args.output / f"profiles-fvm-step-{tick:06d}.npz"
        panel = particle_solver.panel_solver.lattice
        count = panel.n_panels
        np.savez_compressed(
            path, full_cell_velocity=full_velocity, small_cell_velocity=small_velocity,
            full_profile=full_profile, small_profile=small_profile,
            reference_on_small_stencil=matched_reference, vpm_profile=particle_profile,
            particle_position=particle_solver.particle_position,
            particle_vortex_strength=particle_solver.particle_vortex_strength,
            particle_core_radius=particle_solver.particle_core_radius,
            panel_vertices=panel.vertex_position.to_numpy()[:count], panel_normals=panel.normal.to_numpy()[:count],
            panel_source_strength=panel.source_strength.to_numpy()[:count],
        )
        row = {"fvm_step": tick, "coupling_step": particle_solver.step,
               "physical_time": observer.seed_time + particle_solver.time,
               "fields": hash_file(path), "observer_state_bitwise_unchanged": True,
               "state_fingerprints": before}
        record["frames"].append(row)
        (args.output / "profile-observation-3d.json").write_text(json.dumps(record, indent=2) + "\n")

    MatchedComparison.measure = measure
    try:
        yield
    finally:
        MatchedComparison.measure = original


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [Path(__file__).resolve(), Path(cadence.__file__).resolve(),
             ROOT / "source/solvers/fvm/sampling/fields.py", ROOT / "source/solvers/vpm/core/solver.py"]
    sources = [hash_file(path) for path in paths]
    record = {
        "schema": "openonda-interface-profile-observation-3d/1", "status": "running", "spatial_dimensions": 3,
        "profile_every_fvm_steps": args.profile_every_fvm_steps, "sources": sources, "frames": [],
        "sampling": {"line_x_bounds": [-3., 10.], "line_y": [0., .75], "line_z": 0., "spacing": .0625,
                     "fvm_reconstruction": "affine", "fvm_neighbours": 12,
                     "body_points_excluded": True, "small_profile_bounds": [-1.5, 1.5]},
        "limitations": [
            "Profiles observe full 3D fields along two lines; this does not replace any 3D solver operation with a 2D approximation.",
            "The same small-FVM stencil evaluates hybrid and reference cell fields. The full-reference line also uses its full mesh, and this separate stencil effect is retained.",
            "Primary fields and clocks must be bitwise unchanged by each observation. A separate advancing control must still qualify any cache effects.",
            "Only accepted coupling states are observed, including the common initial state. No reference field feeds the hybrid.",
        ],
    }
    try:
        with accepted_profiles(args, record):
            cadence.run(args)
        assert record["frames"][-1]["fvm_step"] == args.steps * args.substeps
        assert [hash_file(path) for path in paths] == sources
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if args.output.exists():
            for path in paths:
                target = args.output / "profile-sources" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
            child = args.output / "interface-cadence-3d.json"
            if child.exists():
                record["cadence_report"] = hash_file(child)
            (args.output / "profile-observation-3d.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--substeps", type=int, choices=(1, 5), required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--profile-every-fvm-steps", type=int, default=25)
    parser.add_argument("--particle-spacing", type=float, default=.0625)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--relaxation", type=float, default=1.)
    parser.add_argument("--normal-tolerance", type=float, default=1e-6)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if (args.steps < 1 or args.iterations < 0 or args.particle_spacing != .0625
            or args.profile_every_fvm_steps < 1 or args.profile_every_fvm_steps % args.substeps):
        parser.error("Require positive steps, matched medium spacing and a positive profile interval divisible by substeps")
    args.oracle, args.output = args.oracle.resolve(), args.output.resolve()
    run(args)
