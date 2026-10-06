"""Measure one wall-free remeshing operator on a frozen native cylinder wake.

Translate the same saved cloud rigidly, then compare its exact Gaussian field
with complete-support M4-prime and six-point Lagrange remeshing. There is no
time integration, molecular diffusion, pruning, wall correction or renewal.
The six-point scatter is an isolated test operator, not a native planar model.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from source.coupler.stable_renewal import vortex_invariants
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.vpm.physics.diffusion.grid import _lagrange6_weights
from source.solvers.vpm.physics.diffusion.planar import scatter_planar
from tests.support.cylinder.audit_saved_wall_circulation import gaussian_velocity_and_gradient

REPOSITORY = Path(__file__).resolve().parents[3]
SOURCE_PATHS = (
    "source/solvers/vpm/physics/diffusion/planar.py",
    "source/solvers/vpm/physics/diffusion/grid.py",
    "source/coupler/stable_renewal.py",
    "source/coupler/solver.py",
    "tests/support/cylinder/audit_saved_wall_circulation.py",
    "tests/support/cylinder/measure_frozen_wake_remeshing.py",
)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def array_digest(values):
    values = np.ascontiguousarray(values)
    hasher = hashlib.sha256()
    hasher.update(str(values.dtype).encode())
    hasher.update(str(values.shape).encode())
    hasher.update(values.view(np.uint8))
    return hasher.hexdigest()


def complete_lattice(position, anchor, spacing):
    """Use one common, five-node halo for both complete scatter operators."""
    cells = (np.asarray(position, dtype=float)[:, :2] - anchor[:2]) / spacing
    lower = np.floor(cells.min(axis=0)).astype(np.int64) - 5
    upper = np.ceil(cells.max(axis=0)).astype(np.int64) + 5
    shape = upper - lower + 1
    origin = np.asarray(anchor, dtype=float).copy()
    origin[:2] += lower * spacing
    return origin, tuple(int(value) for value in shape)


def scatter_lagrange6(position, strength, origin, spacing, shape):
    """Host six-point tensor scatter using the existing native 3D weights."""
    scaled = (np.asarray(position, dtype=float)[:, :2] - origin[:2]) / spacing
    base = np.floor(scaled).astype(np.int64)
    fraction = scaled - base
    wx = _lagrange6_weights(fraction[:, 0])
    wy = _lagrange6_weights(fraction[:, 1])
    grid = np.zeros((*shape, 3), dtype=float)
    for first in range(6):
        for second in range(6):
            ix, iy = base[:, 0] + first - 2, base[:, 1] + second - 2
            weight = wx[:, first] * wy[:, second]
            active = weight != 0
            if np.any((ix[active] < 0) | (ix[active] >= shape[0])
                      | (iy[active] < 0) | (iy[active] >= shape[1])):
                raise ValueError("Six-point scatter exceeds complete padded support")
            np.add.at(grid[:, :, 2], (ix[active], iy[active]),
                      weight[active] * strength[active, 2])
    return grid


def grid_particles(grid, origin, spacing, dtype):
    """Commit every exactly nonzero node; no threshold or moment correction."""
    ix, iy = np.nonzero(grid[:, :, 2] != 0)
    position = np.tile(origin, (len(ix), 1))
    position[:, 0] += ix * spacing
    position[:, 1] += iy * spacing
    strength = grid[ix, iy].copy()
    return position.astype(dtype), strength.astype(dtype), position, strength


def moment_report(before_position, before_strength, after_position, after_strength,
                  spacing, capacity, correction_limit):
    """Apply the unchanged native moment guard to the isolated complete scatter."""
    from source.coupler.solver import _validate_gbd_moment_recovery

    if len(after_position) == 0 or len(after_position) > capacity:
        raise RuntimeError("Complete operator violates the recorded particle capacity")
    if not all(np.isfinite(field).all() for field in
               (before_position, before_strength, after_position, after_strength)):
        raise RuntimeError("The isolated operator produced a non-finite field")
    before = vortex_invariants(before_position, before_strength)
    after = vortex_invariants(after_position, after_strength)
    magnitude = max(float(np.abs(before_strength[:, 2]).sum()), np.finfo(float).tiny)
    length = max(spacing, float(np.linalg.norm(before_position, axis=1).max()))
    recovery = {
        "applied": False,
        "nonzero_node_count": len(after_position),
        "retained_node_count": len(after_position),
        "pruned_node_count": 0,
        "support_augmented_node_count": 0,
        "correction_fraction": 0.0,
        "normalized_vortex_strength_residual": float(np.linalg.norm(
            after.total_vortex_strength - before.total_vortex_strength) / magnitude),
        "normalized_linear_impulse_residual": float(np.linalg.norm(
            after.linear_impulse - before.linear_impulse) / (magnitude * length)),
        "normalized_angular_impulse_residual": float(np.linalg.norm(
            after.angular_impulse - before.angular_impulse) / (magnitude * length**2)),
    }
    _validate_gbd_moment_recovery(recovery, correction_limit)
    centre = np.mean(before_position[:, :2], axis=0)
    scale = max(spacing, float(np.ptp(before_position[:, :2], axis=0).max()))

    def moments(position, strength):
        q = (np.asarray(position, dtype=float)[:, :2] - centre) / scale
        basis = np.array([np.ones(len(q)), q[:, 0], q[:, 1], q[:, 0]**2,
                          q[:, 0] * q[:, 1], q[:, 1]**2])
        return basis @ np.asarray(strength, dtype=float)[:, 2]

    original_moments = moments(before_position, before_strength)
    new_moments = moments(after_position, after_strength)
    return {
        "native_moment_guard": recovery,
        "unchanged_native_residual_limit": 1e-5,
        "recorded_correction_fraction_limit": correction_limit,
        "particle_capacity": capacity,
        "particle_count": len(after_position),
        "original_strength_l1": magnitude,
        "strength_l1": float(np.abs(after_strength[:, 2]).sum()),
        "circulation_before": before.total_vortex_strength.tolist(),
        "circulation_after": after.total_vortex_strength.tolist(),
        "linear_impulse_change": (after.linear_impulse - before.linear_impulse).tolist(),
        "angular_impulse_change": (after.angular_impulse - before.angular_impulse).tolist(),
        "scaled_xy_moment_names": ["1", "x", "y", "x2", "xy", "y2"],
        "scaled_xy_moment_residuals_over_strength_l1":
            ((new_moments - original_moments) / magnitude).tolist(),
    }


def remesh(position, strength, anchor, spacing, kernel, capacity, correction_limit):
    origin, shape = complete_lattice(position, anchor, spacing)
    if kernel == "M4_PRIME":
        grid = scatter_planar(position, strength, origin, spacing, *shape)
    elif kernel == "LAGRANGE6_TEST_ONLY":
        grid = scatter_lagrange6(position, strength, origin, spacing, shape)
    else:
        raise ValueError("Unknown isolated remeshing operator")
    output, circulation, uncast_position, uncast_strength = grid_particles(
        grid, origin, spacing, position.dtype)
    committed = moment_report(position, strength, output, circulation,
                              spacing, capacity, correction_limit)
    uncast = moment_report(position, strength, uncast_position, uncast_strength,
                           spacing, capacity, correction_limit)
    committed.update({
        "float64_before_commit_moments": uncast,
        "grid_shape": list(shape), "grid_origin": origin.tolist(),
        "complete_halo_nodes_each_side": 5,
        "position_sha256": array_digest(output),
        "vortex_strength_sha256": array_digest(circulation),
        "committed_position_dtype": str(output.dtype),
        "committed_strength_dtype": str(circulation.dtype),
        "physical_diffusion_interval_seconds": 0.0,
        "pruning_threshold": 0.0,
        "exact_zero_nodes_omitted": int(np.prod(shape)) - len(output),
        "wall_mask_or_wall_correction": False,
    })
    return output, circulation, committed


def induced_fields(points, position, strength, sigma, span, target_block_size):
    """Bound the independent Gaussian workspace in both source and target axes."""
    # Preserve stored FP32 values exactly, then evaluate coefficients in FP64.
    # Otherwise NumPy keeps strength/(2*pi) in FP32 before multiplying by the
    # double-precision kernel, compromising the independent Gaussian check.
    position = np.asarray(position, dtype=np.float64)
    gamma = np.asarray(strength, dtype=np.float64)[:, 2] / span
    points = np.asarray(points, dtype=np.float64)
    velocity = np.empty_like(points, dtype=float)
    jacobian = np.empty((len(points), 3, 3), dtype=float)
    for start in range(0, len(points), target_block_size):
        stop = start + target_block_size
        velocity[start:stop], jacobian[start:stop] = gaussian_velocity_and_gradient(
            points[start:stop], position, gamma, sigma)
    curl = jacobian[:, 1, 0] - jacobian[:, 0, 1]
    if not (np.isfinite(velocity).all() and np.isfinite(jacobian).all()):
        raise RuntimeError("The Gaussian operator produced a non-finite field")
    return velocity, jacobian, curl


def paired_statistics(exact, remeshed, weights=None):
    exact = np.asarray(exact, dtype=float)
    remeshed = np.asarray(remeshed, dtype=float)
    weights = np.ones(len(exact)) if weights is None else np.asarray(weights)
    extension = (slice(None),) + (None,) * (exact.ndim - 1)
    weight = weights[extension]
    norm = float(np.sum(weight * exact**2))
    output_norm = float(np.sum(weight * remeshed**2))
    error = remeshed - exact
    error_norm = float(np.sum(weight * error**2))
    maximum = float(np.max(np.abs(error)))
    return {
        "exact_rms": float(np.sqrt(norm / weights.sum())),
        "remeshed_rms": float(np.sqrt(output_norm / weights.sum())),
        "error_rms": float(np.sqrt(error_norm / weights.sum())),
        "maximum_absolute_error": maximum,
        "relative_l2_error": None if norm == 0 else float(np.sqrt(error_norm / norm)),
        "rms_ratio": None if norm == 0 else float(np.sqrt(output_norm / norm)),
        "least_squares_amplitude_gain": None if norm == 0 else float(
            np.sum(weight * remeshed * exact) / norm),
    }


def wake_probes(cut, spacing):
    """Resolve h-scale content in a 25-by-40 interior wake strip, ten h past crop."""
    x = cut + 10 * spacing + np.arange(25) * spacing
    y = (np.arange(40) - 19.5) * spacing
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size))), (25, 40)


def curl_spectral_bands(exact, remeshed, shape, spacing):
    """Report finite-window spectral content and gain; this is not a force test."""
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    basis = np.column_stack((np.ones(i.size), i.ravel(), j.ravel()))
    window = np.outer(np.hanning(shape[0]), np.hanning(shape[1]))

    def spectrum(values):
        detrended = values - basis @ np.linalg.lstsq(basis, values, rcond=None)[0]
        return np.fft.fft2(detrended.reshape(shape) * window)

    original, output = spectrum(exact), spectrum(remeshed)
    power = np.abs(original)**2
    total = float(power.sum())
    fx = np.fft.fftfreq(shape[0], d=spacing)
    fy = np.fft.fftfreq(shape[1], d=spacing)
    frequency = np.hypot(fx[:, None], fy[None, :])
    wavelength = np.divide(1., frequency, out=np.full(shape, np.inf), where=frequency > 0)
    rows = {}
    for name, lower, upper in (
        ("wavelength_at_least_0.5m", 0.5, np.inf),
        ("wavelength_0.2_to_0.5m", 0.2, 0.5),
        ("wavelength_0.1_to_0.2m", 0.1, 0.2),
        ("wavelength_0.08_to_0.1m", 0.08, 0.1),
        ("wavelength_below_0.08m", 0., 0.08),
    ):
        keep = (frequency > 0) & (wavelength >= lower) & (wavelength < upper)
        p = float(power[keep].sum())
        rows[name] = {
            "exact_curl_windowed_power_fraction": None if total == 0 else p / total,
            "rms_gain": None if p == 0 else float(np.sqrt(
                np.sum(np.abs(output[keep])**2) / p)),
            "least_squares_gain": None if p == 0 else float(
                np.sum(np.real(output[keep] * np.conj(original[keep]))) / p),
            "mode_count": int(keep.sum()),
        }
    return {
        "method": "Subtract a least-squares affine plane; apply separable Hann window; "
                  "use radial FFT wavelengths. Finite window, crop and leakage affect bands.",
        "bands": rows,
    }


def boundary_components(fields, normals):
    velocity, jacobian, _ = fields
    normal_velocity = np.einsum("fi,fi->f", velocity, normals)
    derivative = np.einsum("fij,fj->fi", jacobian, normals)
    tangential_normal_derivative = derivative - np.einsum(
        "fi,fi->f", derivative, normals)[:, None] * normals
    return normal_velocity, tangential_normal_derivative


def load_frozen(directory):
    metadata_path = directory / "checkpoint/checkpoint_info.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["format_version"] != 13 or metadata["kind"] != "openonda.coupled_backup":
        raise ValueError("A current native coupled checkpoint is required")
    if metadata["coupling_step"] != 1150 or metadata["time"] != 46.0:
        raise ValueError("This diagnostic requires the frozen 46 s checkpoint")
    configuration = metadata["config"]
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(encoded.encode()).hexdigest() != metadata["config_sha256"]:
        raise ValueError("Native coupled configuration hash is invalid")
    paths = {"metadata": metadata_path,
             "mesh": directory / "coupled_mesh.npz",
             "trace": directory / "wall_trace_fields.npz"}
    for name, relative in metadata["checkpoint_files"].items():
        path = (metadata_path.parent / relative).resolve()
        if not path.is_relative_to(metadata_path.parent.resolve()):
            raise ValueError("Checkpoint artifact escapes its frozen directory")
        if digest(path) != metadata["file_sha256"][name]:
            raise ValueError(f"Frozen native checkpoint hash mismatch: {name}")
        paths[name] = path
    with h5py.File(paths["vpm"], "r") as saved:
        attributes = dict(saved["solver"].attrs)
        if str(attributes["backup_format_version"]) != "10.3":
            raise ValueError("A current native VPM checkpoint is required")
        if float(attributes["time"]) != 46.0 or int(attributes["step"]) != 1150:
            raise ValueError("Native VPM clock differs from the committed manifest")
        vpm_configuration = json.loads(attributes["numerical_configuration"])
        if vpm_configuration != configuration["vpm"]:
            raise ValueError("Native VPM configuration differs from the coupled manifest")
        encoded = json.dumps(vpm_configuration, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(encoded.encode()).hexdigest() != attributes[
                "numerical_configuration_sha256"]:
            raise ValueError("Native VPM configuration hash is invalid")
        particles = {name: value[:] for name, value in saved["particles"].items()}
    if not all(np.isfinite(values).all() for values in particles.values()):
        raise ValueError("Native particle state is non-finite")
    position, strength = particles["position"], particles["vortex_strength"]
    if len(position) != int(attributes["n_particles_total"]):
        raise ValueError("Native particle coverage is incomplete")
    induction = vpm_configuration["induction"]
    viscous = vpm_configuration["viscous"]
    spacing, span = viscous["particle_spacing"], induction["planar_span"]
    if (induction["method"] != "PLANAR" or spacing != 0.04 or span != 1.0
            or viscous["gbd_remeshing_kernel"] != "M4_PRIME"
            or viscous["kinematic_viscosity"] != 1 / 150
            or vpm_configuration["time_step_size"] != 0.04):
        raise ValueError("Frozen numerical/physical settings differ from the admitted cylinder")
    if (not np.all(position[:, 2] == 0) or not np.all(strength[:, :2] == 0)
            or position.dtype != np.dtype("float32") or strength.dtype != position.dtype
            or not np.all(particles["core_radius"] == np.float32(spacing))
            or not np.all(particles["particle_volume"] == np.float32(spacing**2 * span))):
        raise ValueError("Saved fields do not represent the required one-plane Gaussian wake")
    with np.load(paths["trace"], allow_pickle=False) as saved:
        traces = {name: saved[name].copy() for name in
                  ("numericalBoundary_centre", "numericalBoundary_normal", "numericalBoundary_area")}
    if (traces["numericalBoundary_centre"].shape != (356, 3)
            or not all(np.isfinite(field).all() for field in traces.values())
            or not np.all(traces["numericalBoundary_area"] > 0)
            or not np.allclose(np.linalg.norm(traces["numericalBoundary_normal"], axis=1), 1,
                               rtol=0, atol=1e-12)):
        raise ValueError("Frozen outer-face traces are incomplete or invalid")
    mesh = load_native_mesh(paths["mesh"])
    wall_nodes = []
    for patch in mesh["boundary"]:
        if patch["type"] == "wall":
            for face in range(patch["start_face"], patch["start_face"] + patch["n_faces"]):
                nodes = np.asarray(mesh["faces"][face])
                wall_nodes.extend(nodes[nodes >= 0])
    maximum_wall_x = float(mesh["vertex_position"][wall_nodes, 0].max())
    return metadata, particles, traces, maximum_wall_x, paths


def measure(frozen_directory, output_directory, target_block_size=48):
    if target_block_size < 1 or target_block_size > 64:
        raise ValueError("Use a target block size between 1 and 64 for bounded memory")
    output_directory.mkdir(parents=True, exist_ok=False)
    metadata, particles, traces, maximum_wall_x, paths = load_frozen(frozen_directory)
    inputs = {str(path.resolve()): digest(path) for path in paths.values()}
    sources = {relative: digest(REPOSITORY / relative) for relative in SOURCE_PATHS}
    position, strength = particles["position"], particles["vortex_strength"]
    config = metadata["config"]
    spacing = config["vpm"]["viscous"]["particle_spacing"]
    span = config["vpm"]["induction"]["planar_span"]
    sigma = float(particles["core_radius"][0])
    anchor = np.asarray(config["transfer_lattice"]["anchor"], dtype=float)
    capacity = config["vpm"]["max_n_particles"]
    correction_limit = config["coupler"]["transfer_discretization_error_limit"]
    normals = traces["numericalBoundary_normal"]
    areas = traces["numericalBoundary_area"]
    outer = traces["numericalBoundary_centre"]
    start = time.perf_counter()
    rows = {}
    with h5py.File(output_directory / "operator_fields.h5", "w") as saved:
        saved.attrs["schema"] = "openonda-frozen-wake-remeshing-fields/1"
        saved.attrs["jacobian_convention"] = "J[velocity_component,coordinate_derivative]"
        saved.attrs["source_physical_time_seconds"] = metadata["time"]
        saved.attrs["physical_time_advanced_seconds"] = 0.0
        saved.create_dataset("outer_face_points", data=outer)
        saved.create_dataset("outer_face_normal", data=normals)
        saved.create_dataset("outer_face_area", data=areas)
        for cut in (1., 3.):
            keep = position[:, 0] > cut
            original_position = position[keep].copy()
            original_strength = strength[keep].copy()
            if len(original_position) == 0:
                raise ValueError("The admitted wake subset is empty")
            if float(original_position[:, 0].min()) - maximum_wall_x <= 3 * spacing:
                raise ValueError("Six-point remeshing support does not clear the stationary wall")
            probes, shape = wake_probes(cut, spacing)
            points = np.vstack((outer, probes))
            group = saved.create_group(f"x_above_{cut:g}")
            group.create_dataset("wake_points", data=probes)
            group.attrs["wake_probe_shape"] = shape
            subset_rows = {}
            for fraction in (0., .25, .5, .75):
                phase_start = time.perf_counter()
                shifted = original_position.astype(float)
                shifted[:, :2] += fraction * spacing
                shifted = shifted.astype(original_position.dtype)
                phase = group.create_group(f"shift_{fraction:g}")
                exact_start = time.perf_counter()
                exact = induced_fields(points, shifted, original_strength, sigma, span,
                                       target_block_size)
                exact_cost = time.perf_counter() - exact_start
                exact_un, exact_gt = boundary_components(
                    (exact[0][:len(outer)], exact[1][:len(outer)], exact[2][:len(outer)]), normals)
                for name, field in zip(("velocity", "jacobian", "curl_z"), exact, strict=True):
                    phase.create_dataset("exact_" + name, data=field)
                cell_fraction = ((shifted[:, :2].astype(float) - anchor[:2]) / spacing) % 1
                phase_row = {
                    "rigid_shift_cells": [fraction, fraction],
                    "rigid_shift_metres": [fraction * spacing, fraction * spacing, 0.],
                    "exact_comparator": "Same Gaussian sigma, span, shifted FP32 positions and "
                                        "original FP32 circulation; no freestream added.",
                    "shifted_position_sha256": array_digest(shifted),
                    "original_vortex_strength_sha256": array_digest(original_strength),
                    "fractional_lattice_coordinate_strength_weighted_histogram": [
                        np.histogram(cell_fraction[:, axis], bins=np.linspace(0, 1, 17),
                                     weights=np.abs(original_strength[:, 2]))[0].tolist()
                        for axis in (0, 1)],
                    "fractional_lattice_histogram_edges": np.linspace(0, 1, 17).tolist(),
                    "exact_induction_wall_seconds": exact_cost,
                    "operators": {},
                }
                for kernel in ("M4_PRIME", "LAGRANGE6_TEST_ONLY"):
                    remesh_start = time.perf_counter()
                    new_position, new_strength, moments = remesh(
                        shifted, original_strength, anchor, spacing, kernel,
                        capacity, correction_limit)
                    scatter_cost = time.perf_counter() - remesh_start
                    field_start = time.perf_counter()
                    result = induced_fields(points, new_position, new_strength, sigma, span,
                                            target_block_size)
                    field_cost = time.perf_counter() - field_start
                    un, gt = boundary_components(
                        (result[0][:len(outer)], result[1][:len(outer)], result[2][:len(outer)]),
                        normals)
                    row = {
                        "moment_and_capacity_guards": moments,
                        "scatter_wall_seconds": scatter_cost,
                        "induction_wall_seconds": field_cost,
                        "outer_induced_normal_velocity": paired_statistics(exact_un, un, areas),
                        "outer_induced_tangential_normal_gradient": paired_statistics(
                            exact_gt, gt, areas),
                        "interior_wake_induced_velocity": paired_statistics(
                            exact[0][len(outer):], result[0][len(outer):]),
                        "interior_wake_jacobian": paired_statistics(
                            exact[1][len(outer):], result[1][len(outer):]),
                        "interior_wake_gaussian_curl": paired_statistics(
                            exact[2][len(outer):], result[2][len(outer):]),
                        "interior_wake_gaussian_curl_spectral_bands": curl_spectral_bands(
                            exact[2][len(outer):], result[2][len(outer):], shape, spacing),
                    }
                    phase_row["operators"][kernel] = row
                    subgroup = phase.create_group(kernel)
                    for name, field in zip(("velocity", "jacobian", "curl_z"), result, strict=True):
                        subgroup.create_dataset(name, data=field)
                    print(f"cut={cut:g} shift={fraction:g} kernel={kernel} "
                          f"nodes={len(new_position)} scatter={scatter_cost:.3f}s "
                          f"induction={field_cost:.3f}s", flush=True)
                phase_row["total_wall_seconds"] = time.perf_counter() - phase_start
                subset_rows[str(fraction)] = phase_row
                saved.flush()
            rows[f"x_above_{cut:g}"] = {
                "source_particle_count": int(keep.sum()),
                "original_position_sha256": array_digest(original_position),
                "original_strength_sha256": array_digest(original_strength),
                "minimum_source_x": float(original_position[:, 0].min()),
                "maximum_wall_x": maximum_wall_x,
                "minimum_six_point_wall_clearance_metres": float(
                    original_position[:, 0].min()) - maximum_wall_x - 3 * spacing,
                "wake_probe_count": len(probes), "wake_probe_shape": list(shape),
                "wake_probe_bounds": np.column_stack((probes.min(0), probes.max(0))).tolist(),
                "wake_probe_spacing_metres": spacing,
                "wake_probe_distance_from_crop_metres": 10 * spacing,
                "phases": subset_rows,
            }
    if any(digest(path) != value for path, value in inputs.items()):
        raise RuntimeError("Frozen checkpoint inputs changed during the operator measurement")
    if any(digest(REPOSITORY / path) != value for path, value in sources.items()):
        raise RuntimeError("Numerical or measurement source changed during the operator measurement")
    report = {
        "schema": "openonda-frozen-wake-remeshing/1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_physical_time_seconds": metadata["time"],
        "source_coupling_step": metadata["coupling_step"],
        "physical_time_advanced_seconds": 0.,
        "native_checkpoint_format": metadata["format_version"],
        "native_vpm_checkpoint_format": "10.3",
        "source_configuration_sha256": metadata["config_sha256"],
        "frozen_input_sha256": inputs, "numerical_and_measurement_sources_sha256": sources,
        "target_block_size": target_block_size, "source_block_size": 2048,
        "spacing_metres": spacing, "gaussian_sigma_metres": sigma,
        "represented_span_metres": span, "particle_capacity": capacity,
        "native_physical_viscosity_not_applied": config["vpm"]["viscous"]["kinematic_viscosity"],
        "wall_seconds": time.perf_counter() - start,
        "operator_fields_sha256": digest(output_directory / "operator_fields.h5"),
        "subsets": rows,
        "limitations": [
            "Rigid translation with uniform imposed shift isolates one scatter. Real RK2 "
            "advection has spatially varying displacements and physical GBD diffusion.",
            "Both exact and remeshed clouds use the same fixed particle crop. Far-field "
            "effects of the discarded cloud are absent from both; results condition on the crop.",
            "The local 1000-point strip is ten h past the crop edge and resolves h-scale curl. "
            "Its finite-window spectrum is descriptive; it does not establish a force cause.",
            "All field metrics exclude the unchanged freestream. The Gaussian kernel, sigma, "
            "span and actual FP32 storage are common; no additional transfer filtering is applied.",
            "Only exact-zero nodes are omitted. Native threshold pruning, moment recovery, "
            "solid support and FVM renewal are absent. Capacity and native residual guards apply.",
            "LAGRANGE6_TEST_ONLY is not an admitted native planar case or restart setting.",
            "One frozen 46 s cloud and diagonal shifts do not measure mature force amplitudes "
            "or demonstrate stability for a changing displacement field or near a wall.",
        ],
    }
    (output_directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_directory": str(output_directory), "wall_seconds": report["wall_seconds"],
                      "physical_time_advanced_seconds": 0.}, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--target-block-size", type=int, default=48)
    arguments = parser.parse_args()
    measure(arguments.frozen_directory.resolve(), arguments.directory.resolve(),
            arguments.target_block_size)


if __name__ == "__main__":
    main()
