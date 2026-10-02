"""Private CPU-only prototype of grouped radius-one wall moment solves.

This asset does not patch the solver. It uses the production scalar method as
the oracle and fallback, and does not change any geometry predicate or gate.
"""

import argparse
from contextlib import contextmanager
from functools import wraps
import json
import math
from pathlib import Path
import time

import numpy as np

if __package__:
    from .profile_diffusion_checkpoint import (
        array_digest, load_inputs, runtime_environment, source_identity,
    )
else:
    from profile_diffusion_checkpoint import (
        array_digest, load_inputs, runtime_environment, source_identity,
    )


def batched_radius_one(wall, calls):
    """Group equal visible support sizes; fall back to the exact scalar path.

    Each call is the original seven positional arguments of
    _wall_moment_correction. Near a decision boundary use the original scalar
    calculation as well; the original criteria are never loosened.
    """
    results = [None] * len(calls)
    groups = {}
    for row, (_, _, _, fluid, _, _, _) in enumerate(calls):
        groups.setdefault(int(np.count_nonzero(fluid)), []).append(row)
    diagnostics = {"batched_radius_one": 0, "scalar_fallback": 0, "groups": len(groups)}
    phase_seconds = {"pack": 0.0, "condition": 0.0, "solve_and_check": 0.0, "publish": 0.0}
    target = np.array([1.0, 0.0, 0.0, 0.0])
    for visible_count, members in groups.items():
        if visible_count < 4:
            continue
        started = time.perf_counter()
        nodes = [calls[row][1][calls[row][3]] for row in members]
        weights = np.stack([calls[row][2][calls[row][3]] for row in members])
        relative = np.stack(
            [
                (calls[row][4] + calls[row][5] * indices - calls[row][0]) / calls[row][5]
                for row, indices in zip(members, nodes, strict=True)
            ]
        )
        constraints = np.concatenate(
            (np.ones((len(members), 1, visible_count)), relative.swapaxes(1, 2)), axis=1
        )
        phase_seconds["pack"] += time.perf_counter() - started
        started = time.perf_counter()
        gram = constraints @ constraints.swapaxes(1, 2)
        try:
            condition = np.linalg.cond(gram)
        except np.linalg.LinAlgError:
            continue  # The unchanged scalar oracle decides the failure semantics.
        phase_seconds["condition"] += time.perf_counter() - started
        started = time.perf_counter()
        # Conservative grey bands select the scalar oracle, never an acceptance.
        selected = np.flatnonzero(np.isfinite(condition) & (condition < 1e10 * (1 - 1e-12)))
        if not len(selected):
            continue
        matrix = constraints[selected]
        defect = target - (matrix @ weights[selected, :, None])[..., 0]
        try:
            solution = np.linalg.solve(gram[selected], defect[..., None])[..., 0]
        except np.linalg.LinAlgError:
            continue
        correction = (matrix.swapaxes(1, 2) @ solution[..., None])[..., 0]
        corrected = weights[selected] + correction
        l1 = np.sum(np.abs(corrected), axis=1)
        residual = np.max(np.abs((matrix @ corrected[..., None])[..., 0] - target), axis=1)
        accepted = (
            np.all(np.isfinite(corrected), axis=1)
            & (l1 < 2.0 * (1 - 1e-12))
            & (residual < 1e-10 * (1 - 1e-6))
        )
        phase_seconds["solve_and_check"] += time.perf_counter() - started
        started = time.perf_counter()
        for selected_row in np.flatnonzero(accepted):
            local = selected[selected_row]
            row = members[local]
            results[row] = (nodes[local], correction[selected_row], 1)
            diagnostics["batched_radius_one"] += 1
        phase_seconds["publish"] += time.perf_counter() - started
    for row, result in enumerate(results):
        if result is None:
            results[row] = wall._wall_moment_correction(*calls[row])
            diagnostics["scalar_fallback"] += 1
    diagnostics["phase_seconds"] = phase_seconds
    return results, diagnostics


@contextmanager
def host_timers(owners_and_names, measured):
    """No per-call device synchronization: all wrapped operations are host-only."""
    saved = []
    for owner, name, label in owners_and_names:
        original = getattr(owner, name)
        local = name in vars(owner)

        def instrument(original, label):
            @wraps(original)
            def called(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    row = measured.setdefault(label, {"calls": 0, "seconds": 0.0})
                    row["calls"] += 1
                    row["seconds"] += time.perf_counter() - start

            return called

        setattr(owner, name, instrument(original, label))
        saved.append((owner, name, original, local))
    try:
        yield
    finally:
        for owner, name, original, local in reversed(saved):
            if local:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


def prepare_host_wall(checkpoint, boundary, anchor):
    """Bare pure-host mixin; never calls __init__ or allocates Taichi fields."""
    from source.solvers.vpm.physics.diffusion.grid import _GridDiffusionMixin

    wall = _GridDiffusionMixin.__new__(_GridDiffusionMixin)
    config = checkpoint["manifest"]["config"]["vpm"]
    viscous = config["viscous"]
    spacing, padding = viscous["gbd_grid_spacing"], viscous["gbd_domain_padding"]
    bounds = np.asarray(config["domain_bounds"], dtype=np.float64)
    slab = (config["induction"]["z_min"], config["induction"]["z_max"])
    margin = padding * spacing
    wall._max_grid_dims = tuple(
        max(5, math.ceil((bounds[2 * axis + 1] - bounds[2 * axis] + 2 * margin) / spacing) + 1)
        + int(axis == 2)
        for axis in range(3)
    )
    wall._grid_domain_bounds = bounds
    wall._fixed_grid_min = (bounds[::2] - margin).astype(np.float32)
    wall.configure_grid_lattice_anchor(anchor, spacing)
    wall._slip_slab_bounds = slab
    wall._body_mask_active = True
    wall._body_classifier = lambda points: boundary.contains(points, include_boundary=False)
    wall._body_segment_classifier = boundary.blocks_segments
    wall._body_query_bounds = boundary.bounds
    wall._body_geometry_revision = boundary.revision
    substeps = wall.gbd_diffusion_substep_count(
        viscous["kinematic_viscosity"], config["time_step_size"], spacing
    )
    halo_cells = max(3, substeps + 1)
    origin, shape = wall._lattice_aligned_bounds(
        checkpoint["particle"]["position"],
        spacing,
        padding,
        required_z_bounds=(slab[0] - halo_cells * spacing, slab[1] + halo_cells * spacing),
    )
    # Same float32 node quantization as the production mask upload, with no
    # device allocation/upload. Host preparation is excluded from solve timing.
    total = int(np.prod(shape))
    mask = np.empty(total, dtype=bool)
    for start in range(0, total, 262144):
        stop = min(total, start + 262144)
        ids = np.arange(start, stop)
        points = np.column_stack(
            (
                origin[0] + (ids // (shape[1] * shape[2])) * spacing,
                origin[1] + ((ids // shape[2]) % shape[1]) * spacing,
                origin[2] + (ids % shape[2]) * spacing,
            )
        )
        mask[start:stop] = wall._body_interior_at_particles(
            points.astype(np.float32).astype(np.float64)
        )
    wall._body_mask_host = mask.reshape(shape)
    wall._body_mask_cache_key = (
        0,
        boundary.revision,
        tuple(map(float, origin)),
        float(np.float32(spacing)),
        *shape,
        tuple(slab),
    )
    return wall, origin, shape, spacing


def compare_results(reference, candidate, strengths=None):
    if len(reference) != len(candidate):
        raise ValueError("Mismatched result counts")
    maximum = 0.0
    f32_changed = 0
    expanded = 0
    for row, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        if not np.array_equal(left[0], right[0]) or left[2] != right[2]:
            raise ValueError(f"Support/radius mismatch at stencil {row}")
        maximum = max(maximum, float(np.max(np.abs(left[1] - right[1]), initial=0)))
        scale = np.ones(3) if strengths is None else strengths[row]
        f32_changed += int(
            np.count_nonzero(
                (left[1][:, None] * scale).astype(np.float32)
                != (right[1][:, None] * scale).astype(np.float32)
            )
        )
        expanded += left[2] > 1
    return {
        "stencils": len(reference),
        "maximum_absolute_correction_difference": maximum,
        "f32_deposit_entries_changed": f32_changed,
        "expanded_support_stencils": expanded,
        "support_and_radius_identical": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--case-setup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--vtk-backend", choices=("Sequential", "STDThread"))
    parser.add_argument("--vtk-threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != (Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow") / "solution" or output.exists():
        raise ValueError("Require a new report in the ordinary solution directory")
    root = args.source_root.resolve()
    if args.vtk_backend:
        from vtkmodules.vtkCommonCore import vtkSMPTools

        if not vtkSMPTools.SetBackend(args.vtk_backend):
            raise ValueError("Requested private VTK backend is unavailable")
        vtkSMPTools.Initialize(args.vtk_threads)
    hashes = source_identity(root)
    checkpoint, boundary, anchor, admitted = load_inputs(
        root, args.checkpoint.resolve(), args.mesh.resolve(), args.case_setup.resolve()
    )
    start = time.perf_counter()
    wall, origin, shape, spacing = prepare_host_wall(checkpoint, boundary, anchor)
    prep_seconds = time.perf_counter() - start
    particle = checkpoint["particle"]
    calls, reference = [], []
    original = wall._wall_moment_correction

    def capture(*values):
        result = original(*values)
        calls.append(
            tuple(value.copy() if isinstance(value, np.ndarray) else value for value in values)
        )
        reference.append(result)
        return result

    wall._wall_moment_correction = capture
    timings = {}
    methods = [
        (wall, "_m4_wall_corrections", "whole_wall_corrections"),
        (wall, "_wall_moment_correction", "scalar_moment_with_capture"),
        (wall, "_body_blocked_segments", "folded_segment_visibility"),
        (boundary, "first_intersections", "all_body_intersections"),
        (boundary.bodies[0], "first_intersections", "triangulated_intersections"),
        (boundary.bodies[0], "signed_distance", "signed_distance"),
    ]
    with host_timers(methods, timings):
        wall._m4_wall_corrections(
            particle["position"], particle["vortex_strength"], origin, spacing, shape
        )
    del wall._wall_moment_correction
    point_strength = {
        tuple(point): value
        for point, value in zip(particle["position"], particle["vortex_strength"], strict=True)
    }
    strengths = [point_strength[tuple(call[0])] for call in calls]
    candidate, gates = batched_radius_one(wall, calls)
    comparison = compare_results(reference, candidate, strengths)
    measurements = []
    for repeat in range(args.repeats):
        row = {}
        for label in ("scalar", "batched") if repeat % 2 == 0 else ("batched", "scalar"):
            start = time.perf_counter()
            if label == "scalar":
                result = [wall._wall_moment_correction(*call) for call in calls]
            else:
                result, _ = batched_radius_one(wall, calls)
            row[label] = time.perf_counter() - start
            compare_results(reference, result, strengths)
        measurements.append(row)
    if source_identity(root) != hashes:
        raise RuntimeError("Source changed during host qualification")
    report = {
        "status": "complete_no_solver_edit",
        "device": "host_only_no_taichi_init",
        "scope": "Prototype radius-one solves on captured accepted-checkpoint wall stencils; not a continued solution",
        "source_root": str(root),
        "runtime_environment": runtime_environment(),
        "source_hashes": hashes,
        "input": admitted,
        "host_mask_prepare_seconds": prep_seconds,
        "active_grid_shape": shape,
        "active_grid_origin": origin.tolist(),
        "host_mask_sha256": array_digest(wall._body_mask_host),
        "host_subphase_timings_nested_not_additive": timings,
        "qualification": comparison,
        "gates": gates,
        "measurements": measurements,
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
