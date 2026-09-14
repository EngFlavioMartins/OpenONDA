"""Isolate the effect of CS radii on ring self/mutual induction at saved states.

This is an instantaneous operator contrast, not a continued counterfactual
solution. Positions and strengths stay fixed. No checkpoint is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from openonda.tutorial_runner import case_package
from source.solvers.vpm.kernels import make_vortex_kernel

if not __package__:
    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

from . import legacy_les as setup_les


def representative_particles(cloud, group, count=32):
    """Choose near-centre particles in uniform azimuth bins of one ring group."""
    selected = np.flatnonzero(cloud["group_id"] == group)
    position = cloud["position"][selected]
    radial = np.hypot(position[:, 1], position[:, 2])
    weights = np.linalg.norm(cloud["vortex_strength"][selected], axis=1) / radial
    centre_x = np.average(position[:, 0], weights=weights)
    centre_r = np.average(radial, weights=weights)
    distance = (position[:, 0] - centre_x) ** 2 + (radial - centre_r) ** 2
    angle = np.mod(np.arctan2(position[:, 2], position[:, 1]), 2 * np.pi)
    bins = np.minimum((angle * count / (2 * np.pi)).astype(int), count - 1)
    targets = [selected[local[np.argmin(distance[local])]]
               for sector in range(count) if len(local := np.flatnonzero(bins == sector))]
    centre_variance = np.average(distance, weights=weights)
    blob_variance = np.average(cloud["core_radius"][selected] ** 2, weights=weights)
    return np.asarray(targets), dict(
        centroid_x=float(centre_x), major_radius=float(centre_r),
        material_cross_section_variance=float(centre_variance),
        actual_mean_sigma_squared=float(blob_variance),
        actual_blob_fraction_of_width=float(blob_variance / (centre_variance + blob_variance)),
    )


def induction(cloud, targets, source_group, radii):
    """Direct Gaussian particle transport operator, including pair-mean sigma."""
    kernel = make_vortex_kernel("GAUSSIAN")
    sources = np.flatnonzero(cloud["group_id"] == source_group)
    velocity = np.zeros((len(targets), 3))
    for start in range(0, len(sources), 2048):
        indices = sources[start:start + 2048]
        delta = cloud["position"][targets, None, :] - cloud["position"][indices][None, :, :]
        velocity += kernel.velocity_pair(
            delta, cloud["vortex_strength"][indices][None, :, :],
            radii[targets, None], radii[indices][None, :],
        ).sum(axis=1)
    return velocity


def audit(cloud, time, sigma0, viscosity):
    radii = {
        "actual": cloud["core_radius"],
        "molecular_only": np.full(len(cloud["position"]), np.sqrt(sigma0**2 + 4 * viscosity * time)),
        "initial_radius": np.full(len(cloud["position"]), sigma0),
    }
    rows = []
    for group in (0, 1):
        targets, geometry = representative_particles(cloud, group)
        for model, sigma in radii.items():
            own = induction(cloud, targets, group, sigma)
            mutual = induction(cloud, targets, 1 - group, sigma)
            rows.append(dict(
                group=group, time=time, radius_model=model, **geometry,
                n_probes=len(targets), mean_sigma=float(sigma.mean()),
                self_axial_velocity=float(own[:, 0].mean()),
                mutual_axial_velocity=float(mutual[:, 0].mean()),
                total_axial_velocity=float((own[:, 0] + mutual[:, 0]).mean()),
            ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / "solution" / args.run
    metadata = json.loads((folder / "vpm_metadata.json").read_text())
    numerics = metadata["configuration"]["numerics"]
    first = metadata["configuration"]["initial_conditions"][0]
    spacing = first["distribution"]["spacing"]
    sigma0 = spacing * first["distribution"]["core_radius_ratio"]
    viscosity = first["kinematic_viscosity"]
    initial = setup_les.build_case(
        "baseline", particle_spacing=spacing, particle_core_radius=sigma0,
    )
    if first["disturbance"]["amplitude"] != 0 or not np.isclose(viscosity, np.pi / 3000):
        raise ValueError("This diagnostic is scoped to the unperturbed Re=3000 control")
    particles = [ring.build() for ring in initial.initial_conditions]
    fields = ("position", "vortex_strength", "core_radius", "group_id")
    cloud = {key: np.concatenate([getattr(p, key) for p in particles]).astype(float) for key in fields}
    results = audit(cloud, 0.0, sigma0, viscosity)
    hashes = {}
    for step in args.steps:
        snapshot = folder / f"vpm_{step:06d}.h5"
        hashes[snapshot.name] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        with h5py.File(snapshot) as handle:
            cloud = {key: handle[f"particles/{key}"][:].astype(float) for key in fields}
        results.extend(audit(cloud, step * numerics["time_step_size"], sigma0, viscosity))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        run=args.run, diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_sha256=hashes, molecular_viscosity=viscosity, initial_particle_radius=sigma0,
        limitation="Instantaneous sigma-only contrasts on fixed positions and strengths; not time-evolved solutions.",
        rows=results,
    ), indent=2) + "\n")
    for row in results:
        print(f"t={row['time']:.3f} group={row['group']} {row['radius_model']:15s} "
              f"self={row['self_axial_velocity']:.6f} mutual={row['mutual_axial_velocity']:.6f}")


if __name__ == "__main__":
    main()
