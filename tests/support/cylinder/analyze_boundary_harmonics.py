"""Measure actual planar VPM boundary harmonics against native reference traces.

This read-only diagnostic freezes published one-second particle snapshots and
reconstructs the native Gaussian planar velocity and its normal derivative with
streamed NumPy sums. It creates no solver, device, or particle evolution. The
reference traces come from the independently advanced mature reference case.
The different flow-time windows and the one-second sampling limit are explicit.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

from .analyze_force_cycles import freeze_csv

REPOSITORY = Path(__file__).resolve().parents[3]
CASE = REPOSITORY / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
STUDY = Path(__file__).resolve().parent / "force_boundary_study/20261005T214514Z"
REFERENCE = Path("/tmp/openonda-cylinder-boundary-study-20261005")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def freeze_file(source, destination):
    before = digest(source)
    shutil.copyfile(source, destination)
    if digest(source) != before or digest(destination) != before:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Source publication changed: {source}")
    return {"source": str(source), "snapshot": str(destination), "sha256": before}


def gaussian_velocity_normal_derivative(target, normal, position, strength, radius, span):
    """Return native filament velocity and J n with at most 1024 source columns."""
    velocity = np.zeros((len(target), 3))
    derivative = np.zeros_like(velocity)
    for start in range(0, len(position), 1024):
        selected = slice(start, start + 1024)
        dx = target[:, None, 0] - position[None, selected, 0]
        dy = target[:, None, 1] - position[None, selected, 1]
        squared_distance = dx * dx + dy * dy
        inverse_core_squared = 1 / radius[None, selected] ** 2
        q = squared_distance * inverse_core_squared
        small = q <= 0.05
        f = np.empty_like(q)
        df = np.empty_like(q)
        large = ~small
        decay = np.exp(-q[large])
        f[large] = -np.expm1(-q[large]) / squared_distance[large]
        df[large] = ((q[large] + 1) * decay - 1) / squared_distance[large] ** 2
        qs = q[small]
        inv = np.broadcast_to(inverse_core_squared, q.shape)[small]
        f[small] = inv * (1 - qs / 2 + qs**2 / 6 - qs**3 / 24 + qs**4 / 120 - qs**5 / 720)
        df[small] = inv**2 * (-0.5 + qs / 3 - qs**2 / 8 + qs**3 / 30 - qs**4 / 144 + qs**5 / 840)
        circulation_weight = strength[None, selected, 2] / (2 * np.pi * span)
        velocity[:, 0] -= np.sum(circulation_weight * dy * f, axis=1)
        velocity[:, 1] += np.sum(circulation_weight * dx * f, axis=1)
        projected_distance = dx * normal[:, None, 0] + dy * normal[:, None, 1]
        derivative[:, 0] -= np.sum(
            circulation_weight * (f * normal[:, None, 1] + 2 * dy * df * projected_distance),
            axis=1,
        )
        derivative[:, 1] += np.sum(
            circulation_weight * (f * normal[:, None, 0] + 2 * dx * df * projected_distance),
            axis=1,
        )
    return velocity, derivative


def check_gaussian_derivative():
    points = np.array([[0.0, 0.0, 0], [0.003, -0.004, 0], [0.2, 0.1, 0]])
    normal = np.array([[0.6, 0.8, 0]] * len(points))
    arguments = (np.zeros((1, 3)), np.array([[0.0, 0.0, 0.7]]), np.array([0.04]), 1.0)
    velocity, derivative = gaussian_velocity_normal_derivative(points, normal, *arguments)
    epsilon = 1e-7
    plus = gaussian_velocity_normal_derivative(points + epsilon * normal, normal, *arguments)[0]
    minus = gaussian_velocity_normal_derivative(points - epsilon * normal, normal, *arguments)[0]
    error = float(np.max(abs((plus - minus) / (2 * epsilon) - derivative)))
    if error > 1e-6 or np.any(velocity[0] != 0):
        raise ValueError("Gaussian derivative or source-centre check failed")
    return {"normal_derivative_finite_difference_maximum_error": error, "epsilon": epsilon}


def design_matrix(time, frequency):
    centred = time - (time[0] + time[-1]) / 2
    phase = 2 * np.pi * frequency * centred
    return np.column_stack(
        (
            np.ones(len(time)),
            centred,
            np.cos(phase),
            np.sin(phase),
            np.cos(2 * phase),
            np.sin(2 * phase),
        )
    )


def fit_harmonics(time, values, frequency):
    shape = values.shape[1:]
    design = design_matrix(time, frequency)
    flattened = values.reshape(len(time), -1)
    coefficient = np.linalg.lstsq(design, flattened, rcond=None)[0]
    residual = flattened - design @ coefficient
    return {
        "mean": coefficient[0].reshape(shape),
        "linear_slope": coefficient[1].reshape(shape),
        "fundamental_amplitude": np.hypot(coefficient[2], coefficient[3]).reshape(shape),
        "second_harmonic_amplitude": np.hypot(coefficient[4], coefficient[5]).reshape(shape),
        "residual_rms": np.sqrt(np.mean(residual**2, axis=0)).reshape(shape),
    }


def force_frequency(values, start, end):
    selected = (values["time"] >= start - 1e-8) & (values["time"] <= end + 1e-8)
    time = values["time"][selected]
    lift = values["lift_coefficient"][selected]

    def residual(frequency):
        design = design_matrix(time, frequency)
        coefficient = np.linalg.lstsq(design, lift, rcond=None)[0]
        return float(np.mean((lift - design @ coefficient) ** 2))

    grid = np.linspace(0.15, 0.22, 701)
    index = int(np.argmin([residual(value) for value in grid]))
    bounded = minimize_scalar(
        residual,
        bounds=(grid[max(index - 2, 0)], grid[min(index + 2, len(grid) - 1)]),
        method="bounded",
        options={"xatol": 1e-12},
    )
    frequency = float(bounded.x)
    fits = {
        name: fit_harmonics(time, values[name][selected], frequency)
        for name in ("lift_coefficient", "drag_coefficient")
    }
    return frequency, {
        "window": [float(time[0]), float(time[-1])],
        "sample_count": len(time),
        "fundamental_frequency": frequency,
        "period": 1 / frequency,
        "force_harmonics": {
            name: {key: float(value) for key, value in fitted.items()}
            for name, fitted in fits.items()
        },
    }


def weighted_magnitude(value, weights):
    squared = value**2 if value.ndim == 1 else np.sum(value**2, axis=-1)
    return float(np.sqrt(np.average(squared, weights=weights)))


def summarize(fits, area, centre):
    xmin, xmax = np.min(centre[:, 0]), np.max(centre[:, 0])
    ymin, ymax = np.min(centre[:, 1]), np.max(centre[:, 1])
    sides = {
        "all": np.ones(len(centre), dtype=bool),
        "upstream": np.isclose(centre[:, 0], xmin, atol=1e-10),
        "downstream": np.isclose(centre[:, 0], xmax, atol=1e-10),
        "lower": np.isclose(centre[:, 1], ymin, atol=1e-10),
        "upper": np.isclose(centre[:, 1], ymax, atol=1e-10),
    }
    return {
        side: {
            "face_count": int(np.count_nonzero(selected)),
            "area": float(np.sum(area[selected])),
            **{
                field: {
                    quantity: weighted_magnitude(value[selected], area[selected])
                    for quantity, value in fitted.items()
                }
                for field, fitted in fits.items()
            },
        }
        for side, selected in sides.items()
    }


def ratio_comparison(coupled, reference):
    return {
        side: {
            field: {
                quantity: coupled[side][field][quantity] / value if value > 1e-12 else None
                for quantity, value in reference[side][field].items()
            }
            for field in ("normal_velocity", "tangential_gradient")
        }
        for side in reference
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--start", type=int, default=30)
    parser.add_argument("--end", type=int, default=50)
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.study.resolve() / f"boundary_harmonics_{stamp}"
    output.mkdir()
    frozen = output / "snapshots"
    frozen.mkdir()
    sources = {}
    geometry_source = args.study / "wall_trace_20261005T215034Z/coupled_mesh.npz"
    sources["mesh"] = freeze_file(geometry_source, frozen / "coupled_mesh.npz")
    sources["reference_traces"] = freeze_file(
        args.reference / "reference_traces.h5", frozen / "reference_traces.h5"
    )
    coupled_forces, sources["coupled_forces"] = freeze_csv(
        args.case / "samples/forces_history.csv", frozen / "coupled_forces.csv"
    )
    reference_forces, sources["reference_forces"] = freeze_csv(
        args.reference / "reference/samples/forces_history.csv", frozen / "reference_forces.csv"
    )
    mesh = load_native_mesh(frozen / "coupled_mesh.npz")
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    patch = next(item for item in mesh["boundary"] if item["name"] == "numericalBoundary")
    faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
    centre = geometry["face_centre"][faces]
    area = geometry["face_area"][faces]
    normal = geometry["face_area_vector"][faces] / area[:, None]
    with h5py.File(frozen / "reference_traces.h5") as h:
        if not h.attrs["complete"]:
            raise ValueError("Reference trace publication is incomplete")
        distance, indices = cKDTree(h["face_centre"][:]).query(centre)
        if np.max(distance) > 1e-10 or len(np.unique(indices)) != len(faces):
            raise ValueError("Reference and coupled boundary face centres differ")
        if (
            np.max(abs(h["face_normal"][:][indices] - normal)) > 1e-10
            or np.max(abs(h["face_area"][:][indices] - area)) > 1e-10
        ):
            raise ValueError("Reference and coupled boundary geometry differ")
        reference_time = h["time"][:]
        reference_fields = {
            "normal_velocity": h["normal_velocity"][:][:, indices],
            "tangential_gradient": h["tangential_gradient"][:][:, indices],
        }
        reference_attributes = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in h.attrs.items()
        }
    coupled_time = []
    coupled_normal = []
    coupled_gradient = []
    particle_records = []
    configuration = None
    for flow_time in range(args.start, args.end + 1):
        step = round(flow_time / 0.04)
        name = f"vpm_{step:06d}.h5"
        sources[name] = freeze_file(args.case / "solution/vpm" / name, frozen / name)
        with h5py.File(frozen / name) as h:
            attributes = dict(h["solver"].attrs)
            values = {
                key: h["particles"][key][:].astype(np.float64)
                for key in ("position", "vortex_strength", "core_radius", "particle_volume")
            }
        current_config = json.loads(attributes["numerical_configuration"])
        if configuration is None:
            configuration = current_config
        if (
            current_config != configuration
            or current_config["induction"]["method"] != "PLANAR"
            or current_config["induction"]["kernel"] != "GAUSSIAN"
        ):
            raise ValueError("Snapshot configuration changed or kernel differs")
        if abs(attributes["time"] - flow_time) > 1e-8 or attributes["step"] != step:
            raise ValueError("Published particle clock differs from snapshot time")
        if any(not np.isfinite(value).all() for value in values.values()) or np.any(
            values["core_radius"] <= 0
        ):
            raise ValueError("Invalid particle snapshot")
        span = current_config["induction"]["planar_span"]
        velocity, derivative = gaussian_velocity_normal_derivative(
            centre,
            normal,
            values["position"],
            values["vortex_strength"],
            values["core_radius"],
            span,
        )
        velocity += attributes["freestream_velocity"]
        normal_velocity = np.einsum("ij,ij->i", velocity, normal)
        correction = float(np.dot(normal_velocity, area) / np.sum(area))
        normal_velocity -= correction
        tangent = derivative - np.einsum("ij,ij->i", derivative, normal)[:, None] * normal
        coupled_time.append(flow_time)
        coupled_normal.append(normal_velocity)
        coupled_gradient.append(tangent)
        particle_records.append(
            {
                "time": flow_time,
                "step": step,
                "particle_count": len(values["position"]),
                "span": span,
                "background": np.asarray(attributes["freestream_velocity"]).tolist(),
                "core_radius_range": [
                    float(values["core_radius"].min()),
                    float(values["core_radius"].max()),
                ],
                "particle_volume_range": [
                    float(values["particle_volume"].min()),
                    float(values["particle_volume"].max()),
                ],
                "mean_normal_flux_correction": correction,
            }
        )
        print(f"Reconstructed {flow_time:g} s, {len(values['position'])} particles", flush=True)
    coupled_time = np.asarray(coupled_time, dtype=float)
    coupled_fields = {
        "normal_velocity": np.asarray(coupled_normal),
        "tangential_gradient": np.asarray(coupled_gradient),
    }
    coupled_frequency, coupled_force_fit = force_frequency(coupled_forces, args.start, args.end)
    reference_frequency, reference_force_fit = force_frequency(
        reference_forces, reference_time[0], reference_time[-1]
    )
    fits = {
        "coupled": {
            field: fit_harmonics(coupled_time, values, coupled_frequency)
            for field, values in coupled_fields.items()
        },
        "reference": {
            field: fit_harmonics(reference_time, values, reference_frequency)
            for field, values in reference_fields.items()
        },
    }
    integer_seconds = np.arange(round(reference_time[0]), round(reference_time[-1]) + 1)
    sampled_indices = np.argmin(abs(reference_time[:, None] - integer_seconds), axis=0)
    fits["reference_one_second_samples"] = {
        field: fit_harmonics(
            reference_time[sampled_indices], values[sampled_indices], reference_frequency
        )
        for field, values in reference_fields.items()
    }
    summaries = {label: summarize(values, area, centre) for label, values in fits.items()}
    arrays = {
        "face_centre": centre,
        "face_area": area,
        "face_normal": normal,
        "coupled_time": coupled_time,
        "reference_time": reference_time,
    }
    arrays.update({f"coupled_{field}": values for field, values in coupled_fields.items()})
    arrays.update({f"reference_{field}": values for field, values in reference_fields.items()})
    for label, values in fits.items():
        arrays.update(
            {
                f"{label}_{field}_{quantity}": value
                for field, fitted in values.items()
                for quantity, value in fitted.items()
            }
        )
    np.savez_compressed(output / "boundary_harmonic_fields.npz", **arrays)
    source_paths = [
        Path(__file__),
        REPOSITORY / "source/solvers/vpm/physics/induction/planar.py",
        REPOSITORY / "source/coupler/boundary.py",
    ]
    report = {
        "schema": "openonda-planar-boundary-harmonics/1",
        "captured_at": datetime.now(UTC).isoformat(),
        "output": str(output),
        "source_files": sources,
        "numerical_source_sha256": {str(path): digest(path) for path in source_paths},
        "analytic_checks": check_gaussian_derivative(),
        "face_count": len(faces),
        "reference_attributes": reference_attributes,
        "coupled_configuration": configuration,
        "coupled_particle_snapshots": particle_records,
        "force_frequency_fits": {"coupled": coupled_force_fit, "reference": reference_force_fit},
        "boundary_amplitudes": summaries,
        "coupled_to_reference_ratios": ratio_comparison(
            summaries["coupled"], summaries["reference"]
        ),
        "reference_one_second_to_native_ratios": ratio_comparison(
            summaries["reference_one_second_samples"], summaries["reference"]
        ),
        "method": {
            "induction": "Exact native Gaussian planar formula with per-particle core radius; float64 streamed accumulation of stored float32 sources; circulation=stored_strength_z/span.",
            "volume": "Native particle_volume=h_xy^2*span; strengths carry circulation*span. Volume is not multiplied into induction a second time.",
            "gradient": "J[i,j]=d(u_i)/d(x_j); gt=J*n-((J*n) dot n)*n.",
            "flux_balance": "Area-weighted constant removed from coupled un, matching reference closed-surface trace balancing; gt unchanged.",
            "harmonic_fit": "Independent force-derived fundamental in each run; simultaneous constant, linear drift, fundamental and second harmonic least squares at each face. Report area-weighted RMS harmonic amplitudes, independent of phase offsets.",
            "units": {"normal_velocity": "m/s", "tangential_gradient": "1/s"},
        },
        "limitations": [
            "Coupled 30–50 s and mature reference 100–112 s are different accepted-time windows; this measures input differences, not their causal effect on forces.",
            "One-second particle output resolves the fitted first and second harmonics only; higher harmonics may alias and are not interpreted.",
            "A weak coupled shedding mode can itself yield weaker VPM traces, so attenuation alone cannot establish whether boundary inputs caused or inherited that weakness.",
            "The native reference traces use reconstructed reference fields on the small-domain faces, not a particle representation or a simultaneously coupled trajectory.",
            "Stored endpoint fields are measured; intra-exchange interpolation and Picard history are not reconstructed by this diagnostic.",
        ],
    }
    (output / "boundary_harmonic_statistics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "force_frequency_fits": report["force_frequency_fits"],
                "ratios": report["coupled_to_reference_ratios"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
