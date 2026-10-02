"""UNWIRED host census of finite-image mesh size and local-correction work.

This reads an immutable checkpoint; it never evaluates an induction field or
imports production numerical code. Pair totals are exhaustive cKDTree distance
counts, not extrapolations of a target sample. For every omitted pair r>=rc,
the Gaussian broadening reference bounds its u/J error by |Gamma|*B_tau(rc).
For a particular image, the source/target AABB separation d also lower-bounds
every r, so sum_j|Gamma_j|*B_tau(max(rc,d)) bounds that image at EVERY target.
Summing all supplied finite images gives the reported worst-target envelopes.

These analytic real-arithmetic bounds are evaluated in f64 with outward
geometric padding; this is NOT an interval or mesh/FFT/roundoff certificate.
The near-pair census retains the represented f64 cKDTree distance convention.
The cutoff screen does not select or modify a production numerical parameter.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np
from scipy.fft import next_fast_len
from scipy.spatial import cKDTree

from tests.vpm._gaussian_broadening_reference import singular_defect_bound


def finite_images(z_min, z_max, shells):
    if not (math.isfinite(z_min) and math.isfinite(z_max) and z_max > z_min):
        raise ValueError("finite ordered slab bounds required")
    if not isinstance(shells, int) or not 0 <= shells <= 128:
        raise ValueError("finite qualification supports 0..128 shells")
    period = 2 * (z_max - z_min)
    result = [(2 * z_min, True)]
    for shell in range(1, shells + 1):
        for k in (-shell, shell):
            result.extend(((k * period, False), (2 * z_min + k * period, True)))
    return result


def reflected_positions(position, shift, odd):
    result = np.asarray(position, dtype=np.float64).copy()
    if odd:
        result[:, 2] *= -1
    result[:, 2] += shift
    if not np.isfinite(result).all():
        raise FloatingPointError("transformed coordinate overflow")
    return result


def separated_boxes(a_min, a_max, b_min, b_max):
    """Outward-safe lower distance, with explicit ordinary-f64 caveat."""
    values = np.asarray([a_min, a_max, b_min, b_max], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("finite box bounds required")
    gap = np.maximum(np.maximum(values[0] - values[3], values[2] - values[1]), 0)
    padding = 64 * np.finfo(np.float64).eps * np.maximum(np.max(np.abs(values), axis=0), 1)
    return float(np.linalg.norm(np.maximum(gap - padding, 0)))


def image_bounds(a_min, a_max, shift, odd):
    lower, upper = np.array(a_min, copy=True), np.array(a_max, copy=True)
    if odd:
        lower[2], upper[2] = -a_max[2], -a_min[2]
    lower[2] += shift
    upper[2] += shift
    return lower, upper


def mesh_payload(position, spacing, order):
    """Actual compact two-family grid; payload excludes FFT plan workspace."""
    x = np.asarray(position, dtype=np.float64).copy()
    centre = 0.5 * float(x[:, 2].min()) + 0.5 * float(x[:, 2].max())
    x[:, 2] -= centre
    lower, upper = x.min(axis=0), x.max(axis=0)
    lower[2], upper[2] = min(lower[2], -upper[2]), max(upper[2], -lower[2])
    origin = np.floor(lower / spacing) * spacing - order * spacing
    raw = np.ceil((upper - origin) / spacing) + order + 1
    if not np.isfinite(raw).all() or np.any(raw > 2**30):
        raise ValueError("unrepresentable qualification grid")
    shape = tuple(int(value) for value in raw)
    padded = tuple(next_fast_len(2 * value - 1) for value in shape)
    real_nodes = math.prod(padded)
    complex_nodes = padded[0] * padded[1] * (padded[2] // 2 + 1)
    compact_nodes = math.prod(shape)
    # A deliberately explicit array inventory, NOT a guaranteed peak allocator
    # bound: potential+density(6S), kernel+inverse(2F), three complex arrays(6Fc).
    elements = 6 * compact_nodes + 2 * real_nodes + 6 * complex_nodes
    return {
        "spacing": spacing, "order": order, "compact_shape": shape,
        "fft_shape": padded, "compact_nodes": compact_nodes, "fft_real_nodes": real_nodes,
        "fft_complex_nodes": complex_nodes, "declared_float32_payload_bytes": 4 * elements,
        "declared_float64_payload_bytes": 8 * elements,
        "inventory": "potential+density 6S real; kernel+inverse 2F real; three Fc complex",
        "excludes": "FFT plan workspace, particle assignment buffers, backend/allocator overhead and existing solver fields",
    }


def count_corrections(position, images, cutoffs, *, progress=None):
    """Exhaustive directed target/source-image counts; no pair-list allocation."""
    position = np.asarray(position, dtype=np.float64)
    cutoffs = np.asarray(cutoffs, dtype=np.float64)
    if (position.ndim != 2 or position.shape[1:] != (3,) or not len(position)
            or not np.isfinite(position).all() or not np.isfinite(cutoffs).all()
            or np.any(cutoffs <= 0) or np.any(np.diff(cutoffs) <= 0)):
        raise ValueError("finite cloud and strictly increasing positive cutoffs required")
    lower, upper = position.min(axis=0), position.max(axis=0)
    tree = cKDTree(position, copy_data=True)
    totals = np.zeros(len(cutoffs), dtype=np.int64)
    records = []
    for index, (shift, odd) in enumerate(images):
        image_min, image_max = image_bounds(lower, upper, shift, odd)
        separation = separated_boxes(lower, upper, image_min, image_max)
        counts = np.zeros_like(totals)
        started = perf_counter()
        if separation <= cutoffs[-1]:
            image = reflected_positions(position, shift, odd)
            other = cKDTree(image, copy_data=True)
            counts = np.asarray(tree.count_neighbors(other, cutoffs, cumulative=True), dtype=np.int64)
            del image, other
        totals += counts
        record = {"index": index, "shift": shift, "odd": odd,
                  "aabb_distance_lower": separation, "near_pairs": counts.tolist(),
                  "seconds": perf_counter() - started}
        records.append(record)
        if progress is not None:
            progress(record)
    return totals, records


def correction_tail(absolute_strength, tau, cutoff, image_records):
    if not (math.isfinite(absolute_strength) and absolute_strength >= 0):
        raise ValueError("finite nonnegative absolute strength required")
    velocities, gradients = [], []
    for record in image_records:
        radius = max(cutoff, record["aabb_distance_lower"])
        bound = singular_defect_bound(radius, tau, absolute_strength)
        velocities.append(bound.velocity)
        gradients.append(bound.gradient)
    return {"velocity": math.fsum(velocities), "gradient_frobenius": math.fsum(gradients)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.08, 0.10, 0.12])
    parser.add_argument("--cutoff-ratios", type=float, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--spacing", type=float, default=0.04)
    parser.add_argument("--max-sources", type=int, default=400_000)
    args = parser.parse_args()
    if not all(math.isfinite(value) and value > 0 for value in (*args.taus, *args.cutoff_ratios, args.spacing)):
        parser.error("positive finite mesh/cutoff scales required")
    root = Path(__file__).resolve().parents[2]
    case = root / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
    output = args.output.resolve()
    if not output.is_relative_to(case / "solution"):
        raise ValueError("evidence belongs in normal cylinder solution directory")
    paths = [Path(__file__).resolve(), Path(__file__).with_name("_gaussian_broadening_reference.py")]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report = {"status": "running", "checkpoint": str(args.checkpoint.resolve()), "source_hashes": hashes,
              "runtime_qualified": False, "mesh_error_certified": False, "roundoff_certified": False,
              "fields_evaluated": False, "image_records": [], "cases": []}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)

    def publish():
        output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        import h5py
        started = perf_counter()
        with h5py.File(args.checkpoint, "r") as saved:
            if not 0 < len(saved["particles/position"]) <= args.max_sources:
                raise ValueError("source count exceeds explicit native census bound")
            x = saved["particles/position"][:].astype(np.float64)
            gamma = saved["particles/vortex_strength"][:].astype(np.float64)
            sigma = saved["particles/core_radius"][:].astype(np.float64)
            config = json.loads(saved["solver"].attrs["numerical_configuration"])
            report["time"] = float(saved["solver"].attrs["time"])
        if not all(np.isfinite(value).all() for value in (x, gamma, sigma)) or np.any(sigma <= 0):
            raise ValueError("nonfinite/invalid checkpoint source fields")
        if config["particle_kernel"] != "GAUSSIAN":
            raise ValueError("this correction census is Gaussian only")
        images = finite_images(config["induction"]["z_min"], config["induction"]["z_max"], 128)
        cutoffs = np.unique([tau * ratio for tau in args.taus for ratio in args.cutoff_ratios])
        strength = math.fsum(np.linalg.norm(gamma, axis=1).tolist())
        report.update(particles=len(x), bounds=[x.min(axis=0).tolist(), x.max(axis=0).tolist()],
                      core_min=float(sigma.min()), core_max=float(sigma.max()), absolute_strength=strength,
                      images=len(images), cutoffs=cutoffs.tolist(), pair_distance_convention="SciPy cKDTree f64 <= cutoff",
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      mesh_payloads=[mesh_payload(x, args.spacing, order) for order in (8, 10)])

        def progress(record):
            report["image_records"].append(record)
            if any(record["near_pairs"]):
                print(json.dumps({"event": "near_image", **record}), flush=True)
            publish()

        totals, records = count_corrections(x, images, cutoffs, progress=progress)
        for tau in args.taus:
            for ratio in args.cutoff_ratios:
                cutoff = tau * ratio
                index = int(np.flatnonzero(cutoffs == cutoff)[0])
                case = {"tau": tau, "cutoff_ratio": ratio, "cutoff": cutoff,
                        "source_core_admissible": bool(tau >= sigma.max()),
                        "exact_near_pairs": int(totals[index]),
                        "mean_near_pairs_per_target": float(totals[index] / len(x))}
                if case["source_core_admissible"]:
                    case["worst_target_absolute_omitted_correction_envelope"] = correction_tail(strength, tau, cutoff, records)
                report["cases"].append(case)
        report.update(status="complete", seconds=perf_counter() - started,
                      peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      envelope_scope="all finite513 images, omitted smooth-to-source-core correction only; f64-evaluated analytic bound")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["sources_changed"] = [path for path, digest in hashes.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
        unexpected = [name for name in sys.modules if name == "source" or name.startswith("source.") or name == "openonda" or name.startswith("openonda.")]
        if report["sources_changed"] or unexpected:
            report.update(status="failed", unexpected_numerical_imports=unexpected)
        publish()
    if report["status"] != "complete":
        raise RuntimeError("census failed evidence validation")


if __name__ == "__main__":
    main()
