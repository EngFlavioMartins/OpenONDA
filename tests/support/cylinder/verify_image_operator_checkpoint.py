"""Read-only, selected-target f64 direct-image audit of native induction profiles.

This is an independent all-pairs sum, not an FMM, a simulation continuation, or
an infinite-image reference. It evaluates exactly the finite shell family recorded
by the supplied profiles. Pair each total profile with its self-only profile to
isolate images without attributing a self-FMM change to the image operator.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def select_targets(position, strength, count=24, seed=20261001):
    """Strongest four, both extrema of each axis, then seeded random particles."""
    count = min(int(count), len(position))
    if count < 1:
        raise ValueError("At least one target and source particle are required")
    selected = []

    def add(index):
        if int(index) not in selected and len(selected) < count:
            selected.append(int(index))

    for index in np.argsort(-np.linalg.norm(strength, axis=1), kind="stable")[:4]:
        add(index)
    for axis in range(3):
        add(np.argmin(position[:, axis]))
        add(np.argmax(position[:, axis]))
    for index in np.random.default_rng(seed).permutation(len(position)):
        add(index)
        if len(selected) == count:
            break
    return np.asarray(selected, dtype=np.int64)


def image_family(z_min, z_max, last_shell):
    """Return physical source transforms, excluding the unreflected real source."""
    if not np.isfinite([z_min, z_max]).all() or z_max <= z_min or last_shell < 0:
        raise ValueError("Invalid slab bounds or final shell")
    images = []
    for shell in range(last_shell + 1):
        for k in (0,) if shell == 0 else (-shell, shell):
            for odd in (0, 1):
                if k == 0 and odd == 0:
                    continue
                images.append((2 * k * (z_max - z_min) + (2 * z_min if odd else 0), odd))
    return np.asarray(images, dtype=np.float64)


def direct_numpy(kernel, position, strength, radius, target_position, images):
    """Small independent host reference, using physical reflected axial vectors."""
    total = np.zeros((len(target_position), 12))
    absolute = np.zeros_like(total)
    for shift, odd in images:
        reflected_position = np.array(position, dtype=np.float64, copy=True)
        reflected_strength = np.array(strength, dtype=np.float64, copy=True)
        reflected_position[:, 2] = shift + (-1 if odd else 1) * position[:, 2]
        if odd:
            reflected_strength[:, :2] *= -1
        displacement = target_position[:, None, :] - reflected_position[None, :, :]
        # Passing the source radius twice gives the source-only field radius.
        velocity = kernel.velocity_pair(displacement, reflected_strength, radius, radius)
        gradient = kernel.gradient_pair(displacement, reflected_strength, radius, radius)
        pairs = np.concatenate((velocity, gradient.reshape(*velocity.shape[:-1], 9)), axis=-1)
        total += pairs.sum(axis=1)
        absolute += np.abs(pairs).sum(axis=1)
    return total, absolute


def make_direct_evaluator(radial_factors, tile_size=256):
    """Build a bounded-memory all-pairs evaluator; no induction backend imports."""
    import taichi as ti

    if tile_size < 1:
        raise ValueError("Source tile size must be positive")

    @ti.data_oriented
    class DirectImages:
        def __init__(self, position, strength, radius, targets, images):
            self.source_count = len(position)
            self.target_count = len(targets)
            self.source_tiles = (self.source_count + tile_size - 1) // tile_size
            self.image_count = len(images)
            self.position = ti.Vector.field(3, ti.f64, shape=self.source_count)
            self.strength = ti.Vector.field(3, ti.f64, shape=self.source_count)
            self.radius = ti.field(ti.f64, shape=self.source_count)
            self.targets = ti.Vector.field(3, ti.f64, shape=self.target_count)
            self.shifts = ti.field(ti.f64, shape=self.image_count)
            self.odd = ti.field(ti.i32, shape=self.image_count)
            self.total = ti.field(ti.f64, shape=(self.target_count, 12))
            self.absolute = ti.field(ti.f64, shape=(self.target_count, 12))
            self.position.from_numpy(np.asarray(position, dtype=np.float64))
            self.strength.from_numpy(np.asarray(strength, dtype=np.float64))
            self.radius.from_numpy(np.asarray(radius, dtype=np.float64))
            self.targets.from_numpy(np.asarray(targets, dtype=np.float64))
            self.shifts.from_numpy(np.asarray(images[:, 0], dtype=np.float64))
            self.odd.from_numpy(np.asarray(images[:, 1], dtype=np.int32))

        @ti.kernel
        def accumulate(self, image_start: ti.i32, image_stop: ti.i32):
            # Each work item reduces only a source tile. In particular, no
            # thread serializes all N sources or all 513 images for one target.
            for target, image, tile in ti.ndrange(
                self.target_count, (image_start, image_stop), self.source_tiles
            ):
                summed = ti.Vector.zero(ti.f64, 12)
                absolute = ti.Vector.zero(ti.f64, 12)
                for local in range(tile_size):
                    source = tile * tile_size + local
                    if source < self.source_count:
                        source_position = self.position[source]
                        gamma = self.strength[source]
                        if self.odd[image]:
                            source_position[2] = self.shifts[image] - source_position[2]
                            gamma[0] = -gamma[0]
                            gamma[1] = -gamma[1]
                        else:
                            source_position[2] += self.shifts[image]
                        displacement = self.targets[target] - source_position
                        sigma = self.radius[source]
                        factors = radial_factors(displacement.norm() / sigma, sigma, True)
                        cross = gamma.cross(displacement)
                        skew = ti.Matrix(
                            [
                                [0.0, -gamma[2], gamma[1]],
                                [gamma[2], 0.0, -gamma[0]],
                                [-gamma[1], gamma[0], 0.0],
                            ],
                            dt=ti.f64,
                        )
                        velocity = factors[0] * cross
                        gradient = factors[0] * skew - factors[1] * cross.outer_product(
                            displacement
                        )
                        for axis in ti.static(range(3)):
                            summed[axis] += velocity[axis]
                            absolute[axis] += ti.abs(velocity[axis])
                            for other in ti.static(range(3)):
                                component = 3 + 3 * axis + other
                                summed[component] += gradient[axis, other]
                                absolute[component] += ti.abs(gradient[axis, other])
                for component in ti.static(range(12)):
                    ti.atomic_add(self.total[target, component], summed[component])
                    ti.atomic_add(self.absolute[target, component], absolute[component])

        def evaluate(self, image_batch=16, progress=None):
            if image_batch < 1:
                raise ValueError("Image batch must be positive")
            self.total.fill(0.0)
            self.absolute.fill(0.0)
            for first in range(0, self.image_count, image_batch):
                last = min(first + image_batch, self.image_count)
                self.accumulate(first, last)
                ti.sync()
                if progress is not None:
                    progress(last, self.image_count)
            return self.total.to_numpy(), self.absolute.to_numpy()

    return DirectImages


def load_profile_pair(total_prefix, self_prefix, sample_configuration):
    """Refuse mismatched times/configurations/checkpoints or incomplete evidence."""
    reports = []
    for prefix in (total_prefix, self_prefix):
        report = json.loads(prefix.with_suffix(".json").read_text())
        if report.get("status") != "complete":
            raise ValueError(f"Incomplete profile: {prefix}")
        for key in ("checkpoint_sha256", "particles", "time", "configuration"):
            if report.get(key) != sample_configuration[key]:
                raise ValueError(f"Profile {prefix} has a different {key}")
        reports.append(report)
    tail = reports[0]["measurements"][-1]["tail"]
    if tail is None or reports[1]["measurements"][-1]["tail"] is not None:
        raise ValueError("Each --profile-prefixes pair must be TOTAL followed by SELF_ONLY")
    shell = int(tail["shell"])
    expected_evaluations = sample_configuration["particles"] * (1 + 4 * shell)
    if tail["target_evaluations"] != expected_evaluations:
        raise ValueError("Profile image count differs from the complete finite shell family")
    if reports[0]["source_root"] != reports[1]["source_root"]:
        raise ValueError("Total and self profiles must use the same source root")
    arrays = []
    for prefix in (total_prefix, self_prefix):
        with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as saved:
            fields = []
            for key, shape in (("velocity", (3,)), ("gradient", (3, 3)), ("rate", (3,))):
                value = saved[key]
                if (
                    value.shape != (sample_configuration["particles"], *shape)
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid {key} array in {prefix}")
                fields.append(
                    value.astype(np.float64).reshape(sample_configuration["particles"], -1)
                )
            arrays.append(np.concatenate(fields, axis=1))
    return reports, arrays, shell


def rate_from_gradient(gradient, strength, scheme):
    scheme = scheme.upper()
    if scheme == "TRANSPOSED":
        gradient = np.swapaxes(gradient, -1, -2)
    elif scheme == "MIXED":
        gradient = 0.5 * (gradient + np.swapaxes(gradient, -1, -2))
    elif scheme != "DIRECT":
        raise ValueError(f"Unknown stretching scheme: {scheme}")
    return np.einsum("...ij,...j->...i", gradient, strength)


def error_summary(actual, exact, sum_absolute, extraction_allowance):
    error = actual - exact
    scale = float(np.linalg.norm(exact))
    # This is a conditioning diagnostic, NOT a tolerance or a worst-case bound
    # for the f32 FMM arithmetic (whose operation graph is different).
    eps_sum_absolute = np.finfo(np.float32).eps * sum_absolute
    denominator = eps_sum_absolute + extraction_allowance
    unscaled_errors = int(np.count_nonzero((denominator == 0) & (error != 0)))
    return {
        "max_absolute_error": float(np.max(np.abs(error))),
        "rms_absolute_error": float(np.sqrt(np.mean(error**2))),
        "relative_l2_error": float(np.linalg.norm(error) / scale) if scale else None,
        "max_sum_absolute": float(np.max(sum_absolute)),
        "max_error_in_eps32_sum_absolute_units": None
        if unscaled_errors
        else float(
            np.max(
                np.divide(
                    np.abs(error), denominator, out=np.zeros_like(error), where=denominator > 0
                )
            )
        ),
        "nonzero_errors_without_conditioning_scale": unscaled_errors,
        "max_extraction_rounding_allowance": float(np.max(extraction_allowance)),
    }


def load_saved_reference(prefix, sample_configuration, position, strength, images):
    """Validate hash-verified immutable direct evidence without device work."""
    report_path = prefix.with_suffix(".json")
    archive_path = prefix.with_suffix(".npz")
    report = json.loads(report_path.read_text())
    for key in ("checkpoint_sha256", "particles", "time", "configuration"):
        if report.get(key) != sample_configuration[key]:
            raise ValueError(f"Saved reference has a different {key}")
    if report.get("status") != "complete" or report.get("precision") != "f64":
        raise ValueError("Saved reference must be a completed f64 direct audit")
    recorded_hash = report.get("archive_sha256")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise ValueError("Saved reference report must embed its archive SHA256")
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if archive_hash != recorded_hash:
        raise ValueError("Saved reference archive SHA256 mismatch")
    if report.get("images") != len(images) or 1 + 4 * int(report.get("last_shell", -1)) != len(
        images
    ):
        raise ValueError("Saved reference uses a different finite image family")
    with np.load(archive_path, allow_pickle=False) as saved:
        archive = {
            key: saved[key].copy()
            for key in ("indices", "position", "strength", "images", "direct", "sum_absolute")
        }
    indices = archive["indices"]
    target_count = report.get("targets")
    if not isinstance(target_count, int) or not 1 <= target_count <= len(position):
        raise ValueError("Invalid saved reference target count")
    if report.get("pair_evaluations") != len(position) * target_count * len(images):
        raise ValueError("Saved reference pair count disagrees with native input")
    if (
        indices.shape != (target_count,)
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= len(position))
        or len(np.unique(indices)) != target_count
    ):
        raise ValueError("Invalid saved reference target indices")
    if indices.tolist() != report.get("target_indices"):
        raise ValueError("Saved reference indices disagree with its report")
    expected_indices = select_targets(position, strength, target_count, report["seed"])
    if not np.array_equal(indices, expected_indices):
        raise ValueError("Saved reference target selection does not match the native checkpoint")
    for key, expected in (
        ("position", position[indices]),
        ("strength", strength[indices]),
        ("images", images),
    ):
        if not np.array_equal(archive[key], expected):
            raise ValueError(f"Saved reference {key} disagrees with native input")
    for key in ("direct", "sum_absolute"):
        value = archive[key]
        if (
            value.shape != (target_count, 15)
            or value.dtype != np.float64
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Invalid saved reference {key} array")
    if np.any(archive["sum_absolute"] < 0):
        raise ValueError("Saved reference absolute sums must be nonnegative")
    scheme = sample_configuration["configuration"]["induction"]["stretching_scheme"]
    for key in ("direct", "sum_absolute"):
        vector = strength[indices] if key == "direct" else np.abs(strength[indices])
        rate = rate_from_gradient(archive[key][:, 3:12].reshape(-1, 3, 3), vector, scheme)
        if not np.array_equal(rate, archive[key][:, 12:]):
            raise ValueError(f"Saved reference {key} rate is inconsistent with its gradient")
    source_information = {
        "prefix": str(prefix.resolve()),
        "archive_sha256": archive_hash,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "hash_verification": "embedded report",
    }
    return report, archive, source_information


def global_profile_differences(loaded, prefixes):
    """All-particle old/new differences; deliberately not a direct accuracy proof."""
    reference = loaded[0][1]
    comparisons = []
    for index, (_, candidate, _) in enumerate(loaded[1:], start=1):
        record = {
            "reference_total_prefix": str(prefixes[0][0].resolve()),
            "candidate_total_prefix": str(prefixes[index][0].resolve()),
            "particles": len(reference[0]),
            "scope": "all-particle profile differences only, NOT direct or absolute accuracy validation",
        }
        for kind, previous, current in (
            ("total", reference[0], candidate[0]),
            ("self", reference[1], candidate[1]),
            ("images", reference[0] - reference[1], candidate[0] - candidate[1]),
        ):
            record[kind] = {}
            for name, columns in (
                ("velocity", slice(0, 3)),
                ("gradient", slice(3, 12)),
                ("rate", slice(12, 15)),
            ):
                difference = current[:, columns] - previous[:, columns]
                scale = float(np.linalg.norm(previous[:, columns]))
                record[kind][name] = {
                    "max_absolute_difference": float(np.max(np.abs(difference))),
                    "rms_absolute_difference": float(np.sqrt(np.mean(difference**2))),
                    "relative_l2_difference": float(np.linalg.norm(difference) / scale)
                    if scale
                    else None,
                    "maximum_particle_vector_difference": float(
                        np.max(np.linalg.norm(difference, axis=1))
                    ),
                }
        comparisons.append(record)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--source-root", type=Path, help="Run the direct CUDA reference using these radial kernels"
    )
    mode.add_argument(
        "--saved-reference",
        type=Path,
        help="Reuse an existing direct JSON/NPZ prefix without GPU work",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--profile-prefixes",
        type=Path,
        nargs=2,
        action="append",
        required=True,
        metavar=("TOTAL", "SELF_ONLY"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument("--source-tile", type=int, default=256)
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        output.parent
        != (
            Path(__file__).resolve().parents[3]
            / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
        )
        / "solution"
    ):
        raise ValueError("Audit reports must stay in this tutorial's ordinary solution directory")
    if output.with_suffix(".json").exists() or output.with_suffix(".npz").exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {output}")
    import h5py

    with h5py.File(args.checkpoint, "r") as saved:
        config = json.loads(saved["solver"].attrs["numerical_configuration"])
        position = saved["particles/position"][:].astype(np.float64)
        strength = saved["particles/vortex_strength"][:].astype(np.float64)
        radius = saved["particles/core_radius"][:].astype(np.float64)
        clock = float(saved["solver"].attrs["time"])
    if not all(np.isfinite(value).all() for value in (position, strength, radius)):
        raise ValueError("Nonfinite checkpoint particle data")
    if np.any(radius <= 0):
        raise ValueError("Nonpositive source core radius")
    sample_configuration = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "particles": len(position),
        "time": clock,
        "configuration": config,
    }
    loaded = [
        load_profile_pair(*prefixes, sample_configuration) for prefixes in args.profile_prefixes
    ]
    shells = {item[2] for item in loaded}
    if len(shells) != 1:
        raise ValueError(
            "Profiles terminated at different shells; separate finite-family audits required"
        )
    last_shell = shells.pop()
    induction = config["induction"]
    images = image_family(induction["z_min"], induction["z_max"], last_shell)
    if args.saved_reference is not None:
        original, archive, saved_source_information = load_saved_reference(
            args.saved_reference,
            sample_configuration,
            position,
            strength,
            images,
        )
        indices = archive["indices"]
        exact, absolute = archive["direct"], archive["sum_absolute"]
        reference_metadata = {
            key: original[key]
            for key in (
                "reference_source_root",
                "arch",
                "precision",
                "seed",
                "source_tile",
                "image_batch",
                "pair_evaluations",
                "seconds_including_first_JIT",
                "reduction_roundoff_only_bound",
            )
        }
    else:
        root = args.source_root.resolve()
        sys.path.insert(0, str(root))
        import taichi as ti

        from source.solvers.vpm.kernels.base import make_device_vortex_kernels

        for name, module in tuple(sys.modules.items()):
            if (
                name.startswith(("source.", "openonda."))
                and getattr(module, "__file__", None)
                and not Path(module.__file__).resolve().is_relative_to(root)
            ):
                raise RuntimeError(f"Wrong reference source root: {module.__file__}")
        indices = select_targets(position, strength, args.target_count, args.seed)
        ti.init(arch=ti.cuda, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
        if ti.lang.impl.current_cfg().arch != ti.cuda:
            raise RuntimeError("CUDA is required; refusing a silent CPU fallback")
        device_kernels = make_device_vortex_kernels(config["particle_kernel"], ti.f64)
        evaluator = make_direct_evaluator(device_kernels["radial_factors_"], args.source_tile)(
            position, strength, radius, position[indices], images
        )
        started = time.perf_counter()
        exact, absolute = evaluator.evaluate(
            args.image_batch,
            progress=lambda done, count: print(
                json.dumps(
                    {"images_done": done, "images": count, "seconds": time.perf_counter() - started}
                ),
                flush=True,
            ),
        )
        elapsed = time.perf_counter() - started
        if not np.isfinite(exact).all() or not np.isfinite(absolute).all():
            raise ArithmeticError("Direct-image output is nonfinite")
        scheme = induction["stretching_scheme"]
        rate = rate_from_gradient(exact[:, 3:].reshape(-1, 3, 3), strength[indices], scheme)
        rate_absolute_bound = rate_from_gradient(
            absolute[:, 3:].reshape(-1, 3, 3), np.abs(strength[indices]), scheme
        )
        exact = np.concatenate((exact, rate), axis=1)
        absolute = np.concatenate((absolute, rate_absolute_bound), axis=1)
        archive = {
            "indices": indices,
            "position": position[indices],
            "strength": strength[indices],
            "images": images,
            "direct": exact,
            "sum_absolute": absolute,
        }
        unit_roundoff = np.finfo(np.float64).eps / 2
        reduction_terms = args.source_tile + evaluator.source_tiles * len(images)
        gamma = reduction_terms * unit_roundoff / (1 - reduction_terms * unit_roundoff)
        reference_metadata = {
            "reference_source_root": str(root),
            "arch": "cuda",
            "precision": "f64",
            "seed": args.seed,
            "source_tile": args.source_tile,
            "image_batch": args.image_batch,
            "pair_evaluations": len(position) * len(indices) * len(images),
            "seconds_including_first_JIT": elapsed,
            "reduction_roundoff_only_bound": {
                "gamma": gamma,
                "max_absolute": float(gamma * np.max(absolute)),
                "excludes": "pair arithmetic, radial-function evaluation, and final rate contraction",
            },
        }
        saved_source_information = None
    comparisons = []
    for index, ((reports, fields, _), prefixes) in enumerate(
        zip(loaded, args.profile_prefixes, strict=True)
    ):
        selected_total, selected_self = (field[indices] for field in fields)
        actual = selected_total - selected_self
        # Profiles store f32 total and self separately. Their subtraction is
        # f64 here, but cannot recover rounding incurred while publishing them.
        rounding = np.finfo(np.float32).eps * (np.abs(selected_total) + np.abs(selected_self))
        archive[f"profile_{index}_images"] = actual
        archive[f"profile_{index}_error"] = actual - exact
        archive[f"profile_{index}_extraction_allowance"] = rounding
        comparisons.append(
            {
                "total_prefix": str(prefixes[0].resolve()),
                "self_prefix": str(prefixes[1].resolve()),
                "source_root": reports[0]["source_root"],
                "profile_hashes": {
                    role: {
                        suffix: hashlib.sha256(prefix.with_suffix(suffix).read_bytes()).hexdigest()
                        for suffix in (".json", ".npz")
                    }
                    for role, prefix in zip(("total", "self"), prefixes, strict=True)
                },
                **{
                    name: error_summary(
                        actual[:, columns],
                        exact[:, columns],
                        absolute[:, columns],
                        rounding[:, columns],
                    )
                    for name, columns in (
                        ("velocity", slice(0, 3)),
                        ("gradient", slice(3, 12)),
                        ("rate", slice(12, 15)),
                    )
                },
            }
        )
    report = {
        **sample_configuration,
        **reference_metadata,
        "checkpoint": str(args.checkpoint.resolve()),
        "status": "complete",
        "mode": "saved-direct-reference" if saved_source_information else "fresh-direct-reference",
        "saved_reference": saved_source_information,
        "direct_pair_evaluations_this_run": 0
        if saved_source_information
        else reference_metadata["pair_evaluations"],
        "last_shell": last_shell,
        "images": len(images),
        "targets": len(indices),
        "target_indices": indices.tolist(),
        "selection": "strongest four, six axis extrema, seeded random without replacement",
        "scope": "finite image contribution only; no FMM, no infinite-tail or all-target validation",
        "gradient_layout": "row-major du_i/dx_j; arrays columns velocity(3), gradient(9), rate(3)",
        "sum_absolute_note": "per-pair component absolute sum; rate uses triangle bound from gradient",
        "conditioning_note": "eps32*sum_absolute is a scale, not an acceptance tolerance or FMM error bound",
        "self_subtraction_exclusion": "Images are total minus a separately profiled self field. Output writing rounding is quantified; repeat-to-repeat self-FMM accumulation variation is not independently bounded or removed.",
        "comparisons": comparisons,
        "global_profile_differences": global_profile_differences(loaded, args.profile_prefixes),
    }
    # Exclusive creation prevents accidentally replacing another qualification.
    with output.with_suffix(".npz").open("xb") as destination:
        np.savez(destination, **archive)
    report["archive_sha256"] = hashlib.sha256(output.with_suffix(".npz").read_bytes()).hexdigest()
    with output.with_suffix(".json").open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
