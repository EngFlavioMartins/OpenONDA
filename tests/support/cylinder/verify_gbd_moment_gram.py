"""CPU-only qualification of a closed-form GBD moment Gram, without solver imports.

The production implementation is not changed. Its pure NumPy functions are
loaded directly from the parsed grid.py source so concurrent FMM editing cannot
affect this diagnostic, and no Taichi runtime or numerical kernel is imported.
"""

import argparse
import ast
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import statistics
from time import perf_counter
import tracemalloc

import numpy as np


def load_reference(path):
    """Load the actual production pure recovery functions, not copied formulas."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id.startswith(("_GBD_", "_GRID_TRANSFER_CHUNK"))
            for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and (
            node.name.startswith("_gbd_") or node.name == "_nearest_visible_nodes"
        ):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "_GridDiffusionMixin":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in (
                    "_augment_moment_recovery_support",
                    "_redistribute_pruned_moments",
                ):
                    method.decorator_list = []
                    nodes.append(method)
    namespace = {"np": np, "Callable": Callable}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _cross_matrix(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def closed_moment_gram(position, vortex_strength, particle_spacing, *, chunk_size=65536):
    """Same nine constraints using 19 independent scalar weighted moments.

    A(r) = [I; C(r); Q(r)] with C(r)v=r×v and Q=(rrᵀ−|r|²I)/3.
    C Cᵀ=|r|²I−rrᵀ, C Q=−|r|²C/3, and
    Q Qᵀ=(|r|⁴I−|r|²rrᵀ)/9 eliminate the N×9×3 tensor.
    Keep the production weights/length scale and bounded chunk allocation.
    """
    magnitude = np.linalg.norm(vortex_strength, axis=1)
    total = float(magnitude.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("GBD moment support has no finite retained strength")
    weights = magnitude / total
    scale = max(
        float(np.sqrt(np.sum(weights * np.einsum("ij,ij->i", position, position)))),
        float(particle_spacing),
    )
    zeroth = 0.0
    first = np.zeros(3)
    second = np.zeros((3, 3))
    third = np.zeros(3)
    fourth = np.zeros((3, 3))
    for start in range(0, len(position), chunk_size):
        stop = min(start + chunk_size, len(position))
        r = position[start:stop] / scale
        w = weights[start:stop]
        radius_squared = np.einsum("ij,ij->i", r, r)
        weighted_radius_squared = w * radius_squared
        zeroth += w.sum(dtype=np.float64)
        first += np.einsum("i,ij->j", w, r)
        second += np.einsum("i,ij,ik->jk", w, r, r)
        third += np.einsum("i,ij->j", weighted_radius_squared, r)
        fourth += np.einsum("i,ij,ik->jk", weighted_radius_squared, r, r)
    identity = np.eye(3)
    gram = np.empty((9, 9))
    gram[:3, :3] = zeroth * identity
    gram[:3, 3:6] = -_cross_matrix(first)
    gram[:3, 6:] = (second - np.trace(second) * identity) / 3.0
    gram[3:6, 3:6] = np.trace(second) * identity - second
    gram[3:6, 6:] = -_cross_matrix(third) / 3.0
    gram[6:, 6:] = (np.trace(fourth) * identity - fourth) / 9.0
    gram[3:6, :3] = gram[:3, 3:6].T
    gram[6:, :3] = gram[:3, 6:].T
    gram[6:, 3:6] = gram[3:6, 6:].T
    return gram, scale, weights


def benchmark(function, position, strength, spacing, repeats):
    function(position, strength, spacing)
    seconds = []
    for _ in range(repeats):
        start = perf_counter()
        function(position, strength, spacing)
        seconds.append(perf_counter() - start)
    tracemalloc.start()
    result = function(position, strength, spacing)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "seconds": seconds,
        "median_seconds": statistics.median(seconds),
        "peak_traced_allocation_bytes": peak,
        "returned_live_allocation_bytes": current,
    }, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--particles", type=int, default=300000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic evidence: {args.output}")
    grid_path = args.source_root / "source/solvers/vpm/physics/diffusion/grid.py"
    reference = load_reference(grid_path)
    rng = np.random.default_rng(1407)
    clouds = [
        ("random", rng.normal(size=(args.particles, 3)), rng.normal(size=(args.particles, 3)))
    ]
    report = {
        "status": "running",
        "scope": "standalone Gram CPU algebra, not complete GBD or solver speedup",
        "grid_source_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        "spacing": 0.04,
        "measurements": [],
    }
    if args.checkpoint is not None:
        import h5py

        with h5py.File(args.checkpoint, "r") as saved:
            clouds.append(
                (
                    "native-checkpoint",
                    saved["particles/position"][:].astype(np.float64),
                    saved["particles/vortex_strength"][:].astype(np.float64),
                )
            )
        report["checkpoint"] = str(args.checkpoint.resolve())
        report["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    try:
        for name, position, strength in clouds:
            old_stats, old = benchmark(
                reference["_gbd_moment_gram"], position, strength, 0.04, args.repeats
            )
            new_stats, new = benchmark(closed_moment_gram, position, strength, 0.04, args.repeats)
            np.testing.assert_allclose(new[0], old[0], rtol=3e-13, atol=3e-14)
            np.testing.assert_array_equal(new[2], old[2])
            assert new[1] == old[1]
            report["measurements"].append(
                {
                    "cloud": name,
                    "particles": len(position),
                    "reference": old_stats,
                    "closed_form": new_stats,
                    "median_speedup": old_stats["median_seconds"] / new_stats["median_seconds"],
                    "relative_frobenius_error": float(
                        np.linalg.norm(new[0] - old[0]) / np.linalg.norm(old[0])
                    ),
                    "maximum_absolute_error": float(np.max(np.abs(new[0] - old[0]))),
                    "reference_rank_condition": reference["_gbd_moment_gram_quality"](old[0]),
                    "closed_rank_condition": reference["_gbd_moment_gram_quality"](new[0]),
                }
            )
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
