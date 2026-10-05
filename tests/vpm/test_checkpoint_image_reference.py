"""Qualify the independent native-checkpoint audit without FMM or CUDA."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.kernels.base import make_device_vortex_kernels, make_vortex_kernel

_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests/support/cylinder/verify_image_operator_checkpoint.py"
)
_SPEC = importlib.util.spec_from_file_location("checkpoint_image_reference", _PATH)
reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reference)


def test_complete_shell_family_and_deterministic_target_selection():
    images = reference.image_family(-0.48, 0.48, 128)
    assert images.shape == (513, 2)
    np.testing.assert_array_equal(images[0], [-0.96, 1])
    np.testing.assert_allclose(images[1:5], [[-1.92, 0], [-2.88, 1], [1.92, 0], [0.96, 1]])
    assert not np.any((images[:, 0] == 0) & (images[:, 1] == 0))
    rng = np.random.default_rng(9)
    position, strength = rng.normal(size=(2, 40, 3))
    chosen = reference.select_targets(position, strength)
    np.testing.assert_array_equal(chosen, reference.select_targets(position, strength))
    assert len(set(chosen)) == 24
    assert np.argmax(np.linalg.norm(strength, axis=1)) == chosen[0]
    for axis in range(3):
        assert np.argmin(position[:, axis]) in chosen
        assert np.argmax(position[:, axis]) in chosen


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "WINCKELMANS", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN"]
)
def test_tiled_f64_reduction_matches_numpy_physical_image_reference(kernel_name):
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    try:
        rng = np.random.default_rng(305)
        position = rng.uniform(-0.25, 0.25, size=(259, 3))
        strength = rng.normal(size=(259, 3)) * 0.002
        radius = rng.uniform(0.04, 0.13, size=259)
        images = reference.image_family(-0.3, 0.3, 2)
        targets = position[[0, 53, 258]].copy()
        # A target exactly on an odd-image source exercises the finite core
        # Jacobian limit; the middle target keeps off-axis/unequal-core pairs.
        targets[0, 2] = -0.6 - position[0, 2]
        expected, expected_abs = reference.direct_numpy(
            make_vortex_kernel(kernel_name), position, strength, radius, targets, images
        )
        radial = make_device_vortex_kernels(kernel_name, ti.f64)["radial_factors_"]
        evaluator = reference.make_direct_evaluator(radial)(
            position, strength, radius, targets, images
        )
        actual, actual_abs = evaluator.evaluate(image_batch=4)
        np.testing.assert_allclose(actual, expected, rtol=5e-11, atol=2e-12)
        np.testing.assert_allclose(actual_abs, expected_abs, rtol=5e-11, atol=2e-12)
        # Changing image launch batching must neither duplicate nor drop work.
        repeated, repeated_abs = evaluator.evaluate(image_batch=1)
        np.testing.assert_allclose(repeated, actual, rtol=2e-14, atol=2e-13)
        np.testing.assert_allclose(repeated_abs, actual_abs, rtol=2e-14, atol=2e-13)
    finally:
        ti.reset()


def test_physical_axial_reflection_equals_target_polar_parity():
    rng = np.random.default_rng(303)
    position = rng.uniform(-0.2, 0.2, size=(17, 3))
    strength = rng.normal(size=(17, 3))
    radius = rng.uniform(0.1, 0.4, size=17)
    targets = rng.normal(size=(3, 3))
    kernel = make_vortex_kernel("GAUSSIAN")
    images = reference.image_family(-0.3, 0.3, 2)
    physical, _ = reference.direct_numpy(kernel, position, strength, radius, targets, images)
    mapped = np.zeros_like(physical)
    for shift, odd in images:
        query = targets.copy()
        query[:, 2] = shift - targets[:, 2] if odd else targets[:, 2] - shift
        displacement = query[:, None, :] - position[None, :, :]
        velocity = kernel.velocity_pair(displacement, strength, radius, radius).sum(axis=1)
        gradient = kernel.gradient_pair(displacement, strength, radius, radius).sum(axis=1)
        if odd:
            parity = np.array([1, 1, -1])
            velocity *= parity
            gradient *= parity[None, :, None] * parity[None, None, :]
        mapped += np.concatenate((velocity, gradient.reshape(-1, 9)), axis=1)
    np.testing.assert_allclose(mapped, physical, rtol=3e-14, atol=2e-13)


def test_profile_pair_validation_checks_identity_and_shell_work(tmp_path):
    sample_configuration = {
        "checkpoint_sha256": "native",
        "particles": 2,
        "time": 11.0,
        "configuration": {"particle_kernel": "GAUSSIAN"},
    }
    total, self_only = tmp_path / "total", tmp_path / "self"
    base = {**sample_configuration, "source_root": "immutable", "status": "complete"}
    total_report = {**base, "measurements": [{"tail": {"shell": 2, "target_evaluations": 18}}]}
    self_report = {**base, "measurements": [{"tail": None}]}
    total.with_suffix(".json").write_text(json.dumps(total_report))
    self_only.with_suffix(".json").write_text(json.dumps(self_report))
    for prefix in (total, self_only):
        np.savez(
            prefix.with_suffix(".npz"),
            velocity=np.ones((2, 3), dtype=np.float32),
            gradient=np.ones((2, 3, 3), dtype=np.float32),
            rate=np.ones((2, 3), dtype=np.float32),
        )
    _, fields, shell = reference.load_profile_pair(total, self_only, sample_configuration)
    assert shell == 2
    assert fields[0].shape == (2, 15)
    wrong = {**sample_configuration, "time": 11.04}
    with pytest.raises(ValueError, match="different time"):
        reference.load_profile_pair(total, self_only, wrong)
    total_report["measurements"][0]["tail"]["target_evaluations"] -= 1
    total.with_suffix(".json").write_text(json.dumps(total_report))
    with pytest.raises(ValueError, match="image count"):
        reference.load_profile_pair(total, self_only, sample_configuration)


@pytest.mark.parametrize("scheme", ["DIRECT", "TRANSPOSED", "MIXED"])
def test_rate_absolute_triangle_bound(scheme):
    rng = np.random.default_rng(82)
    pair_gradients = rng.normal(size=(8, 3, 3))
    strength = rng.normal(size=3)
    individual = reference.rate_from_gradient(pair_gradients, strength, scheme)
    contracted = reference.rate_from_gradient(pair_gradients.sum(axis=0), strength, scheme)
    bound = reference.rate_from_gradient(
        np.abs(pair_gradients).sum(axis=0), np.abs(strength), scheme
    )
    np.testing.assert_allclose(contracted, individual.sum(axis=0))
    assert np.all(np.abs(individual).sum(axis=0) <= bound + 1e-14)


def _saved_evidence(tmp_path):
    rng = np.random.default_rng(306)
    position, strength = rng.normal(size=(2, 12, 3))
    sample_configuration = {
        "checkpoint_sha256": "native",
        "particles": 12,
        "time": 11.0,
        "configuration": {
            "induction": {"stretching_scheme": "TRANSPOSED", "z_min": -0.5, "z_max": 0.5}
        },
    }
    images = reference.image_family(-0.5, 0.5, 2)
    indices = reference.select_targets(position, strength, 10, 305)
    direct = rng.normal(size=(10, 12))
    absolute = np.abs(direct)
    direct = np.column_stack(
        (
            direct,
            reference.rate_from_gradient(
                direct[:, 3:].reshape(-1, 3, 3), strength[indices], "TRANSPOSED"
            ),
        )
    )
    absolute = np.column_stack(
        (
            absolute,
            reference.rate_from_gradient(
                absolute[:, 3:].reshape(-1, 3, 3), np.abs(strength[indices]), "TRANSPOSED"
            ),
        )
    )
    archive = {
        "indices": indices,
        "position": position[indices],
        "strength": strength[indices],
        "images": images.copy(),
        "direct": direct,
        "sum_absolute": absolute,
    }
    prefix = tmp_path / "direct"
    np.savez(prefix.with_suffix(".npz"), **archive)
    digest = hashlib.sha256(prefix.with_suffix(".npz").read_bytes()).hexdigest()
    report = {
        **sample_configuration,
        "status": "complete",
        "precision": "f64",
        "arch": "cuda",
        "last_shell": 2,
        "images": len(images),
        "targets": len(indices),
        "target_indices": indices.tolist(),
        "seed": 305,
        "source_tile": 256,
        "image_batch": 16,
        "pair_evaluations": len(position) * len(indices) * len(images),
        "seconds_including_first_JIT": 0.5,
        "reference_source_root": "immutable",
        "reduction_roundoff_only_bound": {
            "gamma": 1e-13,
            "max_absolute": 1e-12,
            "excludes": "pair arithmetic",
        },
    }
    report["archive_sha256"] = digest
    prefix.with_suffix(".json").write_text(json.dumps(report))
    return prefix, report, archive, sample_configuration, position, strength, images, digest


def test_saved_reference_hash_verification_preserves_values_and_rejects_changed_archive(tmp_path):
    prefix, _, original, sample_configuration, position, strength, images, digest = _saved_evidence(
        tmp_path
    )
    _, loaded, source_information = reference.load_saved_reference(
        prefix, sample_configuration, position, strength, images
    )
    assert source_information["archive_sha256"] == digest
    assert source_information["hash_verification"] == "embedded report"
    for key in original:
        np.testing.assert_array_equal(loaded[key], original[key])
    prefix.with_suffix(".npz").write_bytes(prefix.with_suffix(".npz").read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        reference.load_saved_reference(prefix, sample_configuration, position, strength, images)


@pytest.mark.parametrize(
    "corruption",
    [
        "sample_configuration",
        "indices",
        "position",
        "strength",
        "images",
        "nan",
        "rate",
        "shape",
        "shell",
    ],
)
def test_saved_reference_rejects_inconsistent_evidence_even_with_fresh_archive_hash(
    tmp_path, corruption
):
    prefix, report, archive, sample_configuration, position, strength, images, _ = _saved_evidence(
        tmp_path
    )
    if corruption == "sample_configuration":
        report["time"] += 0.04
    elif corruption == "indices":
        archive["indices"] = archive["indices"][::-1].copy()
        report["target_indices"] = archive["indices"].tolist()
    elif corruption in {"position", "strength", "images"}:
        archive[corruption].flat[0] += 0.1
    elif corruption == "nan":
        archive["direct"][0, 0] = np.nan
    elif corruption == "rate":
        archive["direct"][0, 12] += 0.1
    elif corruption == "shape":
        archive["direct"] = archive["direct"][:, :-1]
    else:
        report["last_shell"] += 1
    np.savez(prefix.with_suffix(".npz"), **archive)
    report["archive_sha256"] = hashlib.sha256(prefix.with_suffix(".npz").read_bytes()).hexdigest()
    prefix.with_suffix(".json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        reference.load_saved_reference(prefix, sample_configuration, position, strength, images)


def test_saved_reference_main_performs_no_device_work_and_publishes_global_differences(
    tmp_path, monkeypatch
):
    import h5py

    prefix, report, _, sample_configuration, position, strength, _, _ = _saved_evidence(tmp_path)
    checkpoint = tmp_path / "checkpoint.h5"
    with h5py.File(checkpoint, "w") as saved:
        solver = saved.create_group("solver")
        solver.attrs["numerical_configuration"] = json.dumps(sample_configuration["configuration"])
        solver.attrs["time"] = sample_configuration["time"]
        saved["particles/position"] = position
        saved["particles/vortex_strength"] = strength
        saved["particles/core_radius"] = np.ones(len(position))
    sample_configuration["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report["checkpoint_sha256"] = sample_configuration["checkpoint_sha256"]
    prefix.with_suffix(".json").write_text(json.dumps(report))
    pairs = []
    for index in range(2):
        pair = [tmp_path / f"total{index}", tmp_path / f"self{index}"]
        for role, target in enumerate(pair):
            tail = {"shell": 2, "target_evaluations": 108} if role == 0 else None
            target.with_suffix(".json").write_text(
                json.dumps(
                    {
                        **sample_configuration,
                        "status": "complete",
                        "source_root": f"source{index}",
                        "measurements": [{"tail": tail}],
                    }
                )
            )
            value = index + role
            np.savez(
                target.with_suffix(".npz"),
                velocity=np.full((12, 3), value, dtype=np.float32),
                gradient=np.full((12, 3, 3), value, dtype=np.float32),
                rate=np.full((12, 3), value, dtype=np.float32),
            )
        pairs.append(pair)
    solution = tmp_path / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution"
    solution.mkdir(parents=True)
    output = solution / "comparison"
    monkeypatch.setattr(
        reference,
        "__file__",
        str(tmp_path / "tests/support/cylinder/verify_image_operator_checkpoint.py"),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved-reference mode must never initialize a GPU or evaluate pairs")

    monkeypatch.setattr(ti, "init", forbidden)
    monkeypatch.setattr(reference, "make_direct_evaluator", forbidden)
    arguments = [
        "verify",
        "--saved-reference",
        str(prefix),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
    ]
    for pair in pairs:
        arguments.extend(["--profile-prefixes", *map(str, pair)])
    monkeypatch.setattr(sys, "argv", arguments)
    reference.main()
    actual = json.loads(output.with_suffix(".json").read_text())
    assert actual["mode"] == "saved-direct-reference"
    assert actual["direct_pair_evaluations_this_run"] == 0
    assert len(actual["comparisons"]) == 2
    difference = actual["global_profile_differences"][0]
    assert difference["particles"] == 12
    assert "NOT direct" in difference["scope"]
    assert difference["total"]["velocity"]["max_absolute_difference"] == 1
    assert difference["images"]["velocity"]["max_absolute_difference"] == 0
    assert "not independently bounded" in actual["self_subtraction_exclusion"]
    assert (
        hashlib.sha256(output.with_suffix(".npz").read_bytes()).hexdigest()
        == actual["archive_sha256"]
    )
    with pytest.raises(FileExistsError):
        reference.main()
