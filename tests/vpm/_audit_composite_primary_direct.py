"""UNWIRED host audit: direct pair-mean primary plus authenticated image truth.

No Taichi or GPU imports, source changes, physical advancement, or backend
admission. The primary includes its finite Gaussian own-particle Jacobian;
the saved image oracle instead uses source-only cores and excludes primary.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tests.vpm._finite_slab_native_qualification import authenticate_inputs, compare_fields
from tests.vpm._slip_periodic_gaussian_oracle import gaussian_pairs


def primary_direct(position, strength, core, indices, *, chunk=8192, max_pairs=10_000_000):
    """Independent Gaussian pair evaluation; identical source index is included."""
    x, gamma, sigma = (np.asarray(value, dtype=np.float64) for value in (position, strength, core))
    indices = np.asarray(indices)
    if (x.ndim != 2 or x.shape[1:] != (3,) or gamma.shape != x.shape or sigma.shape != (len(x),)
            or indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(x))
            or not all(np.isfinite(value).all() for value in (x, gamma, sigma)) or np.any(sigma <= 0)):
        raise ValueError("finite source vectors, positive cores and valid particle indices required")
    if (isinstance(chunk, bool) or not isinstance(chunk, int) or not 1 <= chunk <= 65536
            or isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 0
            or len(x)*len(indices) > max_pairs):
        raise ValueError("bounded direct pair work required")
    u, j = np.zeros((len(indices), 3)), np.zeros((len(indices), 3, 3))
    absolute_u, absolute_j = np.zeros_like(u), np.zeros_like(j)
    for first in range(0, len(x), chunk):
        last = min(first+chunk, len(x))
        displacement = x[indices, None]-x[None, first:last]
        pair_core = .5*(sigma[indices, None]+sigma[None, first:last])
        du, dj = gaussian_pairs(displacement, gamma[None, first:last], pair_core)
        u += du.sum(axis=1)
        j += dj.sum(axis=1)
        absolute_u += np.abs(du).sum(axis=1)
        absolute_j += np.abs(dj).sum(axis=1)
    if not all(np.isfinite(value).all() for value in (u, j, absolute_u, absolute_j)):
        raise FloatingPointError("nonfinite independent primary field")
    return u, j, absolute_u, absolute_j


def transposed_rate_f32(gradient, strength, *, fused=False):
    """Ordered three-term f32 J^T Gamma, with explicit separate/FMA rounding.

    The FMA variant evaluates each f32 product-plus-accumulator exactly enough
    in f64 before one f32 rounding. Neither is called bitwise-native, because
    the CUDA compiler may choose another reassociation. Both protect the same
    transposed contraction, not a direct or symmetric-gradient substitute.
    """
    gradient, strength = np.asarray(gradient, np.float32), np.asarray(strength, np.float32)
    if gradient.shape != (len(strength), 3, 3) or strength.shape != (len(strength), 3):
        raise ValueError("matching full-J and strength arrays required")
    rate = np.zeros_like(strength)
    for component in range(3):
        for axis in range(3):
            if fused:
                rate[:, component] = (gradient[:, axis, component].astype(np.float64)*strength[:, axis]
                                      +rate[:, component].astype(np.float64)).astype(np.float32)
            else:
                rate[:, component] += gradient[:, axis, component]*strength[:, axis]
    return rate


def _rate_error(actual, reference):
    # Reuse the same transparent vector norm/pointwise report schema.
    return compare_fields(actual, np.zeros((len(actual), 3, 3)), reference,
                          np.zeros((len(actual), 3, 3)))["velocity"]


def _load_fields(prefix, identity, *, composite):
    prefix = Path(prefix).resolve().with_suffix("")
    report = json.loads(prefix.with_suffix(".json").read_text())
    if report.get("status") != "complete":
        raise ValueError("only completed native field profiles can be audited")
    if composite:
        if (report.get("identity") != identity or not report.get("checkpoint_unchanged")
                or not report.get("source_fields_unchanged") or report.get("sources_changed")):
            raise ValueError("composite report does not authenticate the unchanged source checkpoint")
    elif (report.get("checkpoint_sha256") != identity["checkpoint_sha256"]
          or report.get("configuration") != identity["configuration"]
          or report.get("particles") != identity["particles"]
          or report.get("source_files_changed_during_run")):
        raise ValueError("baseline report does not match the source identity")
    path = prefix.with_suffix(".npz")
    with np.load(path, allow_pickle=False) as saved:
        fields = {key: saved[key].copy() for key in ("velocity", "gradient", "rate")}
    count = identity["particles"]
    if any(fields[name].shape != shape or not np.isfinite(fields[name]).all()
           for name, shape in (("velocity", (count, 3)), ("gradient", (count, 3, 3)), ("rate", (count, 3)))):
        raise ValueError("invalid profiled field shapes/finiteness")
    return fields, {"report": str(prefix.with_suffix(".json")), "archive": str(path),
                    "report_sha256": hashlib.sha256(prefix.with_suffix(".json").read_bytes()).hexdigest(),
                    "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "oracle", "baseline", "candidate", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    solution = root/"tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution"
    output = args.output.resolve()
    if output.parent != solution or output.suffix != ".json":
        raise ValueError("audit evidence must remain in the ordinary cylinder solution directory")
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError("refusing to overwrite independent accuracy evidence")
    numerical = [Path(__file__).resolve(), Path(__file__).with_name("_slip_periodic_gaussian_oracle.py"),
                 Path(__file__).with_name("_finite_slab_native_qualification.py")]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in numerical}
    data = authenticate_inputs(args.checkpoint, args.oracle)
    if data["identity"]["configuration"]["induction"]["stretching_scheme"] != "TRANSPOSED":
        raise ValueError("this audit explicitly qualifies the native transposed convention")
    baseline, old_provenance = _load_fields(args.baseline, data["identity"], composite=False)
    candidate, new_provenance = _load_fields(args.candidate, data["identity"], composite=True)
    indices = data["oracle"]["indices"]
    x, gamma, sigma = data["position"], data["strength"], data["core"]
    report = {"status": "running", "production_admissible": False, "identity": data["identity"],
              "source_hashes": hashes, "saved_image_oracle": data["oracle_provenance"],
              "baseline": old_provenance, "candidate": new_provenance,
              "target_indices": indices.tolist(), "direct_primary_pairs": len(x)*len(indices),
              "finite_image_count": sum(len(block["images"]) for block in data["blocks"]),
              "primary_core": "arithmetic mean of source and target particle radii",
              "image_core": "source radius only; saved independent f64 finite-image oracle",
              "own_particle": "included: u=0 and J=[Gamma]cross/(3*pi^(3/2)*sigma^3)",
              "background_velocity": "none in either audited induction profile",
              "scope": "24 selected targets, complete primary+same finite513 image family; not all-target or infinite-tail accuracy"}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    try:
        started = time.perf_counter()
        pu, pj, au, aj = primary_direct(x, gamma, sigma, indices)
        report["primary_direct_seconds"] = time.perf_counter()-started
        images = data["oracle"]["direct"]
        iu, ij = images[:, :3], images[:, 3:12].reshape(-1, 3, 3)
        exact_u, exact_j = pu+iu, pj+ij
        selected_gamma = gamma[indices].astype(np.float64)
        exact_rate = np.einsum("nji,nj->ni", exact_j, selected_gamma)
        rate32 = transposed_rate_f32(exact_j, selected_gamma)
        rate32_fma = transposed_rate_f32(exact_j, selected_gamma, fused=True)
        split32 = transposed_rate_f32(pj, selected_gamma)+transposed_rate_f32(ij, selected_gamma)
        own_u, own_j = gaussian_pairs(np.zeros_like(selected_gamma), selected_gamma, sigma[indices])
        report["primary_core_unique_values"] = np.unique(sigma).tolist()
        report["own_gradient_max_frobenius"] = float(np.linalg.norm(own_j, axis=(1, 2)).max(initial=0))
        report["own_velocity_max"] = float(np.abs(own_u).max(initial=0))
        report["comparisons"] = {}
        for name, fields in (("baseline", baseline), ("candidate", candidate)):
            selected = {key: value[indices].astype(np.float64) for key, value in fields.items()}
            summary = compare_fields(selected["velocity"], selected["gradient"], exact_u, exact_j)
            summary["rate_float64_physical"] = _rate_error(selected["rate"], exact_rate)
            summary["rate_transposed_f32_reference"] = _rate_error(selected["rate"], rate32)
            summary["rate_transposed_f32_fma_reference"] = _rate_error(selected["rate"], rate32_fma)
            summary["rate_primary_plus_images_f32_reference"] = _rate_error(selected["rate"], split32)
            summary["stored_rate_vs_own_stored_J_f32"] = _rate_error(
                selected["rate"], transposed_rate_f32(selected["gradient"], selected_gamma))
            report["comparisons"][name] = summary
        report["float32_contraction_note"] = (
            "Three explicit transposed terms; separate and fused-rounding references are both reported. "
            "Stored native rate accumulates primary plus image blocks separately, so it need not be bitwise "
            "equal to a contraction of the separately rounded stored whole Jacobian.")
        archive = output.with_suffix(".npz")
        with archive.open("xb") as stream:
            np.savez(stream, indices=indices, position=x[indices], strength=gamma[indices], core=sigma[indices],
                     primary_velocity=pu, primary_gradient=pj, primary_absolute_velocity=au,
                     primary_absolute_gradient=aj, image_velocity=iu, image_gradient=ij,
                     exact_velocity=exact_u, exact_gradient=exact_j, exact_rate=exact_rate,
                     exact_rate_f32=rate32, exact_rate_fma32=rate32_fma, exact_rate_split32=split32,
                     own_gradient=own_j)
        report.update(status="complete", archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["sources_changed"] = [str(path) for path in numerical if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)]]
        report["checkpoint_unchanged"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == data["identity"]["checkpoint_sha256"]
        if report["sources_changed"] or not report["checkpoint_unchanged"]:
            report["status"] = "failed"
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if report["status"] != "complete":
        raise RuntimeError("accuracy evidence inputs changed")


if __name__ == "__main__":
    main()
