"""Read-only inputs and independent comparisons for unwired GPU qualification.

No production solver imports, field publication, restart, or backend admission.
The saved f64 direct oracle is authenticated by its existing admission helper;
it covers selected targets only, not an all-target accuracy certificate.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

from tests.vpm._slip_periodic_gaussian_oracle import gaussian_pairs


def finite_blocks(last_shell=128):
    """Same complete doubling blocks as the unchanged finite image controller."""
    if isinstance(last_shell, bool) or not isinstance(last_shell, int) or last_shell < 1:
        raise ValueError("positive integer last shell required")
    start = 0
    blocks = []
    while start <= last_shell:
        expected_end = 0 if start == 0 else (1 if start == 1 else 2 * start - 2)
        end = min(last_shell, expected_end)
        images = [(k, odd) for shell in range(start, end + 1)
                  for k in ((0,) if shell == 0 else (-shell, shell))
                  for odd in (False, True) if k != 0 or odd]
        blocks.append({"start": start, "end": end, "complete": end == expected_end,
                       "images": images})
        start = end + 1
    return blocks


def authenticate_inputs(checkpoint, oracle, *, max_sources=400_000):
    checkpoint, oracle = Path(checkpoint).resolve(), Path(oracle).resolve()
    with h5py.File(checkpoint, "r") as saved:
        count = len(saved["particles/position"])
        if not 0 < count <= max_sources:
            raise ValueError("native source count exceeds qualification cap")
        x = saved["particles/position"][:]
        gamma = saved["particles/vortex_strength"][:]
        sigma = saved["particles/core_radius"][:]
        config = json.loads(saved["solver"].attrs["numerical_configuration"])
        clock = float(saved["solver"].attrs["time"])
    if (x.shape != (count, 3) or gamma.shape != x.shape or sigma.shape != (count,)
            or not all(np.isfinite(v).all() for v in (x, gamma, sigma)) or np.any(sigma <= 0)):
        raise ValueError("invalid native source arrays")
    slab = config["induction"]
    if config["particle_kernel"] != "GAUSSIAN" or slab["method"] != "SLIP_SLAB":
        raise ValueError("this qualification is for Gaussian slab images only")
    identity = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "particles": count, "time": clock, "configuration": config}
    root = Path(__file__).resolve().parents[2]
    module_path = root / "tests/support/cylinder/verify_image_operator_checkpoint.py"
    spec = importlib.util.spec_from_file_location("_finite_mesh_saved_oracle_admission", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    images = module.image_family(slab["z_min"], slab["z_max"], slab["max_shells"] - 1)
    _, archive, provenance = module.load_saved_oracle(oracle, identity, x, gamma, images)
    for array in (x, gamma, sigma):
        array.setflags(write=False)
    return {"position": x, "strength": gamma, "core": sigma, "identity": identity,
            "oracle": archive, "oracle_provenance": provenance,
            "blocks": finite_blocks(slab["max_shells"] - 1), "checkpoint": str(checkpoint),
            "admission_helper_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest()}


def compare_fields(velocity, gradient, expected_velocity, expected_gradient):
    """Report both absolute and norm-relative errors; never hide zero truth."""
    result = {}
    for label, current, reference in (("velocity", velocity, expected_velocity),
                                      ("gradient", gradient, expected_gradient)):
        current, reference = np.asarray(current, dtype=np.float64), np.asarray(reference, dtype=np.float64)
        if current.shape != reference.shape or not all(np.isfinite(v).all() for v in (current, reference)):
            raise ValueError("field shapes/finiteness disagree")
        difference = current - reference
        denominator = float(np.linalg.norm(reference))
        error = float(np.linalg.norm(difference))
        per_point = np.linalg.norm(difference.reshape(len(current), -1), axis=1)
        true_point = np.linalg.norm(reference.reshape(len(current), -1), axis=1)
        relative = np.divide(per_point, true_point, out=np.zeros_like(per_point), where=true_point > 0)
        result[label] = {"absolute_l2": error, "relative_l2": error / denominator if denominator else None,
                         "max_absolute_component": float(np.max(np.abs(difference), initial=0)),
                         "point_absolute_norms": per_point.tolist(),
                         "point_reference_norms": true_point.tolist(),
                         "point_relative_norms": [float(r) if t > 0 else None
                                                  for r, t in zip(relative, true_point, strict=True)]}
    return result


def native_tail_maxima(velocity, gradient):
    """Same component order and f32 norm arithmetic as native shell maxima."""
    u, j = np.asarray(velocity, dtype=np.float32), np.asarray(gradient, dtype=np.float32)
    if u.shape != (len(u), 3) or j.shape != (len(u), 3, 3):
        raise ValueError("tail fields require matching vector/Jacobian shapes")
    vsq, jsq = np.zeros(len(u), np.float32), np.zeros(len(u), np.float32)
    for a in range(3):
        vsq += u[:, a] * u[:, a]
        for b in range(3):
            jsq += j[:, a, b] * j[:, a, b]
    if not np.isfinite(vsq).all() or not np.isfinite(jsq).all():
        raise FloatingPointError("nonfinite native tail norms")
    return float(np.sqrt(vsq).max(initial=0)), float(np.sqrt(jsq).max(initial=0))


def direct_small_block(source_x, source_gamma, source_sigma, targets, descriptors, *, zmin, zmax,
                       max_pairs=20_000_000, chunk=16_384):
    """Independent f64 host finite-block sum, explicitly bounded work/memory.

    Intended for the nearest block and a sparse target selection, not all
    targets or the complete 513-image operator. No local cutoff is applied.
    """
    images = list(descriptors)
    x, gamma, sigma, q = [np.asarray(v, dtype=np.float64)
                           for v in (source_x, source_gamma, source_sigma, targets)]
    if len(x) * len(q) * len(images) > max_pairs or chunk < 1:
        raise ValueError("independent direct pair budget exceeded")
    out_u, out_j = np.zeros((len(q), 3)), np.zeros((len(q), 3, 3))
    for k, odd in images:
        if type(odd) is not bool or not isinstance(k, int):
            raise ValueError("explicit integer/boolean image descriptors required")
        shift = 2 * k * (zmax - zmin) + (2 * zmin if odd else 0)
        for first in range(0, len(x), chunk):
            xx, gg = x[first:first + chunk].copy(), gamma[first:first + chunk].copy()
            xx[:, 2] = shift + (-1 if odd else 1) * xx[:, 2]
            if odd:
                gg[:, :2] *= -1
            u, j = gaussian_pairs(q[:, None] - xx[None], gg[None], sigma[None, first:first + chunk])
            out_u += u.sum(axis=1)
            out_j += j.sum(axis=1)
    if not np.isfinite(out_u).all() or not np.isfinite(out_j).all():
        raise FloatingPointError("independent native direct field is nonfinite")
    return out_u, out_j
