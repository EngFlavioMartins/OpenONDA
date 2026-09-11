#!/usr/bin/env python3
"""Capture evidence around one fresh GBD step, after compute-slot release.

The accepted-step change includes RK advection/stretching and diffusion.
Current public diagnostics do not isolate every remapping contribution.
Supply six CPU threads externally; this script is a real one-step CFD run.
"""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zlib

import h5py
import numpy as np

import openonda.vpm as vpm
from openonda.tutorial_runner import load_case_module
from source.solvers.vpm.kernels.base import make_vortex_kernel

launcher = load_case_module(Path(__file__).resolve().parents[1], "assets.run_seeded_gbd_fmm")
CASE_NAME = "gbd_breakdown_fmm_cpu_root_200000_first_event"


def moments(position, strength, core):
    """Gaussian vector strength and unbounded impulse moments, in float64."""
    return {
        "vortex_strength": strength.sum(axis=0).tolist(),
        "linear_impulse": (0.5 * np.cross(position, strength).sum(axis=0)).tolist(),
        "angular_impulse": (
            (np.cross(position, np.cross(position, strength)) - core[:, None] ** 2 * strength).sum(
                axis=0
            )
            / 3
        ).tolist(),
        "strength_l1": float(np.linalg.norm(strength, axis=1).sum()),
    }


def host_field(position, strength, core, targets):
    """Evaluate the regularized source field at a small common target set."""
    kernel = make_vortex_kernel("GAUSSIAN")
    velocity, curl = [], []
    for target in targets:
        displacement = target - position
        velocity.append(kernel.velocity_pair(displacement, strength, core, core).sum(axis=0))
        gradient = kernel.gradient_pair(displacement, strength, core, core).sum(axis=0)
        curl.append(
            [
                gradient[2, 1] - gradient[1, 2],
                gradient[0, 2] - gradient[2, 0],
                gradient[1, 0] - gradient[0, 1],
            ]
        )
    return {"velocity": np.asarray(velocity), "curl_vorticity": np.asarray(curl)}


def field_change(actual, reference):
    return {
        name: {
            "relative_l2": float(
                np.linalg.norm(actual[name] - reference[name]) / np.linalg.norm(reference[name])
            ),
            "maximum_absolute_component": float(np.abs(actual[name] - reference[name]).max()),
        }
        for name in reference
    }


def observe_recovery(original, particles, core_ratio, records):
    """Wrap one existing recovery call in this separate diagnostic process.

    Inputs are hashed around the call. The production operator is called
    exactly once, and its unmodified return object is returned to production.
    """

    def wrapped(*args, **kwargs):
        grid, magnitude, ix, iy, iz, origin, spacing = args[:7]
        arrays = tuple(value for value in args if isinstance(value, np.ndarray)) + tuple(
            value for value in kwargs.values() if isinstance(value, np.ndarray)
        )
        hashes_before = [
            hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() for array in arrays
        ]
        pre_gbd = {
            name: getattr(particles, name + "_cpu")().astype(float)
            for name in ("position", "vortex_strength", "core_radius")
        }
        retained_position = np.column_stack(
            [origin[0] + ix * spacing, origin[1] + iy * spacing, origin[2] + iz * spacing]
        ).astype(float)
        raw_retained = grid[ix, iy, iz].astype(float)
        corrected = original(*args, **kwargs)
        hashes_after = [
            hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() for array in arrays
        ]
        assert hashes_before == hashes_after
        nonzero = np.nonzero(np.any(grid != 0, axis=-1))
        full_position = np.column_stack(
            [origin[axis] + nonzero[axis] * spacing for axis in range(3)]
        ).astype(float)
        full_strength = grid[nonzero].astype(float)
        full_norm = np.linalg.norm(full_strength, axis=1)
        full_l1 = float(full_norm.sum())
        retained_l1 = float(np.linalg.norm(raw_retained, axis=1).sum())
        full_core = np.full(len(full_position), core_ratio * spacing)
        retained_core = np.full(len(retained_position), core_ratio * spacing)
        selected = np.unique(
            np.r_[np.linspace(0, len(full_position) - 1, 16, dtype=int), np.argsort(full_norm)[-8:]]
        )
        targets = full_position[selected] + spacing * np.array([0.17, 0.23, 0.31])
        full_field = host_field(full_position, full_strength, full_core, targets)
        raw_field = host_field(retained_position, raw_retained, retained_core, targets)
        corrected_field = host_field(
            retained_position, corrected.astype(float), retained_core, targets
        )
        pre_field = host_field(
            pre_gbd["position"], pre_gbd["vortex_strength"], pre_gbd["core_radius"], targets
        )
        full_moments = moments(full_position, full_strength, full_core)
        corrected_moments = moments(retained_position, corrected.astype(float), retained_core)
        records.append(
            {
                "production_calls": 1,
                "input_arrays_unchanged": True,
                "full_grid_nonzero_nodes": len(full_position),
                "final_retained_nodes_after_support_and_cap": len(retained_position),
                "complete_diffused_grid_l1": full_l1,
                "final_raw_retained_l1": retained_l1,
                "actual_discarded_l1": full_l1 - retained_l1,
                "actual_discarded_l1_fraction": (full_l1 - retained_l1) / full_l1,
                "correction_l1_fraction": float(
                    np.linalg.norm(corrected.astype(float) - raw_retained, axis=1).sum() / full_l1
                ),
                "pre_gbd_particle_moments": moments(
                    pre_gbd["position"], pre_gbd["vortex_strength"], pre_gbd["core_radius"]
                ),
                "complete_diffused_grid_moments": full_moments,
                "raw_retained_moments": moments(retained_position, raw_retained, retained_core),
                "corrected_retained_moments": corrected_moments,
                "closure_residuals_scaled_by_l1_and_R0": {
                    name: float(
                        np.linalg.norm(np.asarray(corrected_moments[name]) - full_moments[name])
                        / full_l1
                    )
                    for name in ("vortex_strength", "linear_impulse", "angular_impulse")
                },
                "R0_metres": 1.0,
                "target_positions": targets.tolist(),
                "pruning_field_change": field_change(raw_field, full_field),
                "pruning_and_recovery_field_change": field_change(corrected_field, full_field),
                "total_gbd_field_change_including_scatter_and_diffusion": field_change(
                    corrected_field, pre_field
                ),
                "production_closure_diagnostics": dict(kwargs.get("diagnostics") or {}),
                "scope": "The pre-GBD source is after RK and before diffusion. Full-grid versus retained comparisons isolate pruning/recovery. Pre-GBD versus final includes scatter, physical diffusion and regeneration, not advection. No energy-conservation claim is made from point subsets.",
            }
        )
        return corrected

    return wrapped


def plane_arrays(path):
    """Read the current writer's raw-appended compressed vector fields."""
    data = path.read_bytes()
    marker = data.index(b"<AppendedData")
    root = ET.fromstring(data[:marker] + b"</VTKFile>")
    assert root.attrib["header_type"] == "UInt32"
    assert root.attrib["byte_order"] == "LittleEndian"
    assert root.attrib["compressor"] == "vtkZLibDataCompressor"
    start = data.index(b"_", data.index(b">", marker)) + 1
    fields = {}
    for entry in root.iter("DataArray"):
        assert entry.attrib["type"] == "Float32"
        offset = start + int(entry.attrib["offset"])
        count, block_size, last_size = np.frombuffer(data, "<u4", 3, offset)
        sizes = np.frombuffer(data, "<u4", int(count), offset + 12)
        cursor = offset + 12 + 4 * int(count)
        chunks = []
        for index, size in enumerate(sizes):
            chunk = zlib.decompress(data[cursor : cursor + int(size)])
            assert len(chunk) == (last_size if index == count - 1 else block_size)
            chunks.append(chunk)
            cursor += int(size)
        fields[entry.attrib["Name"]] = (
            np.frombuffer(b"".join(chunks), "<f4").reshape(-1, 3).astype(float)
        )
    return fields


def retained_fraction_evidence(log):
    """Bound the reported pre-support loss, including a reported population cap."""
    events = {}
    current = None
    for line in log.splitlines():
        if line.startswith("Event") and "gaussian blob diffusion" in line:
            current = line.split("|", 1)[1].strip()
            events.setdefault(current, {})
        elif line.startswith("  ") and "|" in line and current is not None:
            key, value = line.strip().split("|", 1)
            events[current][key.strip()] = value.strip()
        elif line.strip():
            current = None
    regeneration = events.get("gaussian blob diffusion regeneration", {})
    cap = events.get("gaussian blob diffusion population cap", {})
    retained = cap.get("strength fraction, net", regeneration.get("strength fraction retained"))
    return {
        "events": events,
        "reported_retained_fraction": None if retained is None else float(retained),
        "pre_support_discarded_l1_fraction_interval": None
        if retained is None
        else [max(0.0, 1 - float(retained) - 0.5e-6), min(1.0, 1 - float(retained) + 0.5e-6)],
        "population_cap_logged": bool(cap),
        "final_raw_survivor_discarded_l1_fraction": None,
        "gap": "Six-decimal log fractions precede support augmentation/exchanges and moment correction. Exact final raw-survivor L1, total discarded norm, and pre/post isolated grid fields are not publicly exposed.",
    }


def observation_qualification(records):
    """Require one fully observed production recovery call for this one-step probe."""
    passed = len(records) == 1 and records[0]["production_calls"] == 1
    return {
        "status": "pass" if passed else "fail",
        "recovery_records": len(records),
        "production_calls": sum(record["production_calls"] for record in records),
        "reason": "Exactly one recovery call observed"
        if passed
        else "Expected exactly one recovery record and one production call; numerical qualification fails until the missing/unexpected event is understood",
    }


def run():
    case = launcher.build_case(steps=1, wall_minutes=30, name=CASE_NAME)
    initial = [ring.build() for ring in case.initial_conditions]
    # Match the solver's initial f32 upload before forming float64 moments.
    before = {
        name: np.concatenate([getattr(ring, name) for ring in initial])
        .astype(np.float32)
        .astype(float)
        for name in ("position", "vortex_strength", "core_radius")
    }
    solver = vpm.VPMSolver(case)
    recovery_records = []
    original = solver.physics._redistribute_pruned_moments
    solver.physics._redistribute_pruned_moments = observe_recovery(
        original, solver.particles, case.numerics.viscous.core_radius_ratio, recovery_records
    )
    try:
        solver.run()
    finally:
        del solver.physics._redistribute_pruned_moments
    directory = Path(case.directory) / case.backup.directory
    report = {
        "case": CASE_NAME,
        "run_status": solver.run_status,
        "accepted_step": int(solver.step),
        "time": float(solver.time),
        "gbd_diffusion_substeps": solver.physics.last_gbd_diffusion_substeps,
        "moment_recovery": solver.physics.last_gbd_moment_recovery,
        "instrumented_recovery": recovery_records,
        "observation_qualification": observation_qualification(recovery_records),
        "instrumentation": "Process-local observation of the existing private recovery boundary. No source edits; production called once per record and returned unchanged. Not installed in the qualification/continuation launcher.",
        "tail_evidence": retained_fraction_evidence((directory / "vpm.log").read_text()),
        "initial_moments": moments(
            before["position"], before["vortex_strength"], before["core_radius"]
        ),
        "scope": "Initial-to-first-accepted-step change, including RK evolution and GBD; not isolated remap error. Public closure diagnostics compare the final recovered nodes with the unpruned diffused grid.",
    }
    if solver.step == 1:
        with h5py.File(directory / "vpm_000001.h5", "r") as checkpoint:
            report["final_moments"] = moments(
                *(
                    checkpoint["particles"][name][:].astype(float)
                    for name in ("position", "vortex_strength", "core_radius")
                )
            )
        samples = Path(case.directory) / "samples" / case.samplers.directory
        plane_before = plane_arrays(samples / "core_section_000000.vts")
        plane_after = plane_arrays(samples / "core_section_000001.vts")
        np.testing.assert_array_equal(plane_before["Points"], plane_after["Points"])
        report["core_plane_change"] = {
            name: {
                "relative_l2": float(
                    np.linalg.norm(plane_after[name] - plane_before[name])
                    / np.linalg.norm(plane_before[name])
                ),
                "maximum_absolute_component": float(
                    np.abs(plane_after[name] - plane_before[name]).max()
                ),
            }
            for name in ("velocity", "vorticity")
        }
    (directory / "first_event_evidence.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if report["observation_qualification"]["status"] != "pass":
        raise RuntimeError(report["observation_qualification"]["reason"])


if __name__ == "__main__":
    run()
