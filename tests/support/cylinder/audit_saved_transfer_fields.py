"""Read-only planar curl/divergence comparison from a frozen native state.

The accepted checkpoint follows VPM evolution and renewal. Spatial differences
localize transfer representation errors; they do not prove force causality.
All direct Gaussian sums use bounded CPU blocks and native saved coefficients.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from source.solvers.fvm.fields.diagnostics import compute_continuity_error
from tests.support.cylinder.audit_saved_wall_circulation import digest, load_donor_geometry


def represented_curl(points, sources, strength, radius, span):
    """Return exact sampled planar Gaussian curl in 1/s, with bounded memory."""
    result = np.zeros(len(points))
    for probe in range(0, len(points), 128):
        destination = points[probe : probe + 128, :2]
        for donor in range(0, len(sources), 2048):
            delta = destination[:, None, :] - sources[None, donor : donor + 2048, :2]
            squared = np.einsum("fpi,fpi->fp", delta, delta)
            sigma2 = radius[None, donor : donor + 2048] ** 2
            coefficient = strength[None, donor : donor + 2048] / (span * math.pi * sigma2)
            result[probe : probe + len(destination)] += np.sum(
                coefficient * np.exp(-squared / sigma2), axis=1
            )
    return result


def statistics(values, weight):
    return {
        "weighted_mean": float(np.average(values, weights=weight)),
        "weighted_rms": float(np.sqrt(np.average(values**2, weights=weight))),
        "maximum_absolute": float(np.max(np.abs(values), initial=0)),
    }


def comparison(a, b, weight):
    denominator = np.sum(weight * a**2)
    return {
        "fvm_curl": statistics(a, weight),
        "represented_vpm_curl": statistics(b, weight),
        "curl_error": statistics(b - a, weight),
        "relative_curl_l2_error": float(np.sqrt(np.sum(weight * (b - a) ** 2) / denominator)),
        "best_amplitude_ratio": float(np.sum(weight * a * b) / denominator),
        "curl_rms_ratio": float(np.sqrt(np.sum(weight * b**2) / denominator)),
    }


def regions(points, fluid):
    radius = np.linalg.norm(points[:, :2], axis=1)
    x, y = points[:, 0], points[:, 1]
    return {
        "fluid_transfer_region": fluid,
        "wall_distance_below_0.10": fluid & (radius < 0.60),
        "wall_distance_0.10_to_0.50": fluid & (radius >= 0.60) & (radius < 1.0),
        "body_wake": fluid & (x > 0.5) & (x < 1.6) & (np.abs(y) < 0.7),
        "wake_release": fluid & (x >= 1.6) & (np.abs(y) < 0.7),
        "transfer_blending_belt": fluid & ((x < -1.01) | (x > 1.81) | (np.abs(y) > 1.01)),
    }


def audit(directory):
    metadata_path = directory / "checkpoint/checkpoint_info.json"
    metadata = json.loads(metadata_path.read_text())
    state_path = metadata_path.parent / metadata["checkpoint_files"]["fvm"]
    vpm_path = metadata_path.parent / metadata["checkpoint_files"]["vpm"]
    paths = (metadata_path, state_path, vpm_path, directory / "coupled_mesh.npz")
    hashes = {str(path): digest(path) for path in paths}
    mesh, geometry, state, gradient, boundary, wall, trace = load_donor_geometry(
        directory, state_path
    )
    if wall.revision not in metadata["config"]["solid_geometry"]["wall_revisions"]:
        raise ValueError("Reconstructed wall does not match checkpoint")
    with h5py.File(vpm_path) as saved:
        source = np.asarray(saved["particles/position"], dtype=float)
        strength = np.asarray(saved["particles/vortex_strength"], dtype=float)[:, 2]
        radius = np.asarray(saved["particles/core_radius"], dtype=float)
    h = metadata["config"]["vpm"]["viscous"]["particle_spacing"]
    span = metadata["config"]["vpm"]["induction"]["planar_span"]
    anchor = metadata["config"]["transfer_lattice"]["anchor"]
    box = metadata["config"]["coupler"]["transfer_region_bounds"]
    axes = []
    for axis, name in enumerate("xy"):
        axes.append(
            anchor[axis]
            + h
            * np.arange(
                math.ceil((box[f"{name}min"] - anchor[axis]) / h),
                math.floor((box[f"{name}max"] - anchor[axis]) / h) + 1,
            )
        )
    x, y = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    fluid = ~boundary.contains(points, include_boundary=False)
    cell_velocity = state["velocity"][: mesh["n_cells"]]
    curls = {name: np.zeros(len(points)) for name in ("untapered", "tapered")}
    divergences = {name: np.zeros(len(points)) for name in ("untapered", "tapered")}
    for axis, component, sign in ((0, 1, 1), (1, 0, -1)):
        for side in (-1, 1):
            faces = points.copy()
            faces[:, axis] += side * h / 2
            visible = ~boundary.contains(faces, include_boundary=False)
            velocity = np.zeros_like(faces)
            velocity[visible] = trace.prepare(faces[visible]).sample(cell_velocity, gradient)
            phase = np.clip(boundary.signed_distance(faces) / h, 0, 1)
            tapered = velocity * (phase**2 * (3 - 2 * phase))[:, None]
            for name, values in (("untapered", velocity), ("tapered", tapered)):
                curls[name] += sign * side * values[:, component] / h
                divergences[name] += side * values[:, axis] / h
    vpm_curl = represented_curl(points, source, strength, radius, span)
    lattice_rows = {}
    for name, selected in regions(points, fluid).items():
        row = {"probe_count": int(selected.sum())}
        for method in curls:
            row[method] = comparison(
                curls[method][selected], vpm_curl[selected], np.ones(selected.sum())
            )
            row[method]["compatible_divergence"] = statistics(
                divergences[method][selected], np.ones(selected.sum())
            )
        lattice_rows[name] = row

    centres = geometry["cell_centre"]
    volume = geometry["cell_volume"]
    lsq_curl = gradient[:, 0, 1] - gradient[:, 1, 0]
    lsq_divergence = np.trace(gradient, axis1=1, axis2=2)
    flux_divergence = compute_continuity_error(state["volumetric_face_flux"], mesh) / volume
    cell_rows = {}
    inside = (
        (centres[:, 0] >= box["xmin"])
        & (centres[:, 0] <= box["xmax"])
        & (centres[:, 1] >= box["ymin"])
        & (centres[:, 1] <= box["ymax"])
    )
    # All 7741 native centres are evaluated in bounded blocks; no particle-pair array.
    cell_vpm_curl = represented_curl(centres, source, strength, radius, span)
    for name, selected in regions(centres, inside).items():
        cell_rows[name] = {
            "cell_count": int(selected.sum()),
            **comparison(lsq_curl[selected], cell_vpm_curl[selected], volume[selected]),
            "lsq_divergence": statistics(lsq_divergence[selected], volume[selected]),
            "accepted_flux_divergence": statistics(flux_divergence[selected], volume[selected]),
        }
    if any(digest(path) != hashes[str(path)] for path in paths):
        raise RuntimeError("Frozen inputs changed during read-only audit")
    report = {
        "time": metadata["time"],
        "step": metadata["coupling_step"],
        "particles": len(source),
        "particle_spacing": h,
        "input_sha256": hashes,
        "interpretation": (
            "Accepted post-evolution, post-renewal state; spatial curl and divergence errors "
            "do not isolate their evolved force effect. Compatible divergence uses the same "
            "half-cell collocated-velocity samples as transferred compatible curl. Native "
            "accepted face-flux divergence is computed independently."
        ),
        "uniform_lattice": lattice_rows,
        "native_cell_centres": cell_rows,
    }
    np.savez_compressed(
        directory / "transfer_curl_divergence_fields.npz",
        lattice_position=points,
        lattice_fluid=fluid,
        lattice_vpm_curl=vpm_curl,
        lattice_untapered_curl=curls["untapered"],
        lattice_tapered_curl=curls["tapered"],
        lattice_untapered_divergence=divergences["untapered"],
        lattice_tapered_divergence=divergences["tapered"],
        cell_position=centres,
        cell_volume=volume,
        cell_lsq_curl=lsq_curl,
        cell_vpm_curl=cell_vpm_curl,
        cell_lsq_divergence=lsq_divergence,
        cell_flux_divergence=flux_divergence,
    )
    destination = directory / "transfer_curl_divergence.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(audit(args.directory))
