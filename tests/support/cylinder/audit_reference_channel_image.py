"""Test slip-wall images against simultaneous mature reference fields."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .audit_reference_particle_image import ReferenceParticleImage
from .audit_saved_wall_circulation import (
    digest,
    gaussian_velocity_and_gradient,
    velocity_statistics,
)
from .audit_slip_channel_induction import channel_image_velocity_gradient
from .project_reference_volume_images import auxiliary_image_admission
from .run_boundary_condition_study import balanced_trace


def audit(options):
    configuration = json.loads(options.configuration.read_text())["config"]
    model = ReferenceParticleImage(options.inputs, configuration, support="production")
    paths = (
        options.configuration,
        options.volumes,
        options.images,
        options.inputs / "reference_mesh.npz",
        options.inputs / "reference_backup.npz",
    )
    hashes = {str(path): digest(path) for path in paths}
    rows = []
    with h5py.File(options.volumes) as volumes, h5py.File(options.images) as images:
        points, normal, area = (
            volumes[key][:] for key in ("face_centre", "face_normal", "face_area")
        )
        for index in options.frames:
            variants, report, _ = model.sample(
                volumes["velocity"][index],
                volumes["velocity_gradient"][index],
                repetitions=1,
                outer_repetitions=10,
                include_cut_cells=False,
            )
            admission = auxiliary_image_admission(report)
            position, strength, _ = variants["full_reference_native_masked"]
            correction, gradient = channel_image_velocity_gradient(
                points, position.astype(float), strength[:, 2].astype(float) / model.span, 10.0
            )
            original = images["reference_support/velocity"][index]
            exact = images["exact_reference/velocity"][index]
            corrected, _, _, _ = balanced_trace(
                original + correction,
                images["reference_support/jacobian"][index] + gradient,
                normal,
                area,
            )
            mirrored = position.astype(float).copy()
            mirrored[:, 0] = -16.0 - mirrored[:, 0]
            mirrored_circulation = -strength[:, 2].astype(float) / model.span
            inlet_velocity, inlet_gradient = channel_image_velocity_gradient(
                points, mirrored, mirrored_circulation, 10.0
            )
            point_velocity, point_gradient = gaussian_velocity_and_gradient(
                points, mirrored, mirrored_circulation, model.h
            )
            inlet_velocity += point_velocity
            inlet_gradient += point_gradient
            matched, _, _, _ = balanced_trace(
                original + correction + inlet_velocity,
                images["reference_support/jacobian"][index] + gradient + inlet_gradient,
                normal,
                area,
            )
            rows.append(
                {
                    "time": float(volumes["time"][index]),
                    "cold_image_admission": admission,
                    "image_correction": velocity_statistics(correction, normal, area),
                    "free_space_error": velocity_statistics(original - exact, normal, area),
                    "slip_channel_error": velocity_statistics(corrected - exact, normal, area),
                    "slip_channel_and_inlet_normal_error": velocity_statistics(
                        matched - exact, normal, area
                    ),
                    "inlet_normal_correction": velocity_statistics(inlet_velocity, normal, area),
                    "correction_mean_streamwise_velocity": float(
                        np.average(correction[:, 0], weights=area)
                    ),
                    "correction_mean_transverse_velocity": float(
                        np.average(correction[:, 1], weights=area)
                    ),
                    "maximum_image_gradient": float(np.max(np.abs(gradient))),
                }
            )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(
            {
                "scope": "Independent cold reference images; exact y=+/-10 m slip-wall harmonic correction and reflected vortices enforcing induced Un=0 at x=-8 m. No evolution or force proof.",
                "input_sha256": hashes,
                "measurements": rows,
            },
            indent=2,
        )
        + "\n"
    )
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise RuntimeError("Scientific source changed during the detached comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--volumes", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 67, 134])
    audit(parser.parse_args())
