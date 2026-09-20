#!/usr/bin/env python3
"""Read saved solutions and compare panel-free velocity reconstructions.

Run from the repository root with:
    python -m studies.panel_removal.diagnose --output /tmp/panel-removal.json

No simulation state, case settings, samples or checkpoints are modified.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pyvista as pv

from .operators import (
    affine_sample,
    boundary_targets,
    error_metrics,
    polygon_induction,
    polygon_sources,
    volume_induction,
)

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tutorials/coupled_fvm_vpm"
FREESTREAM = np.array([1.0, 0.0, 0.0])


def frame_at(collection, time):
    for item in ET.parse(collection).findall(".//DataSet"):
        if abs(float(item.get("timestep")) - time) < 1e-8:
            return collection.parent / item.get("file")
    raise ValueError(f"No saved frame at t={time} in {collection}")


def fingerprint(path):
    """Hash source data, including every MPI piece of a parallel VTK frame."""
    paths = [path]
    if path.suffix == ".pvtu":
        paths.extend(path.parent / p.get("Source") for p in ET.parse(path).findall(".//Piece"))
    digest = hashlib.sha256()
    for source in paths:
        digest.update(source.name.encode())
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {"path": str(path.relative_to(ROOT)), "sha256_with_pieces": digest.hexdigest()}


def panel_for(stl, precision="f64"):
    import taichi as ti

    from source.solvers.vpm.boundary_elements.panels.geometry.stl_io import load_stl
    from source.solvers.vpm.boundary_elements.panels.solver.panel_solver import PanelSolver

    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=4, offline_cache=False)
    triangles, _ = load_stl(str(stl))
    panel = PanelSolver(
        max_n_panels=len(triangles),
        float_dtype=precision,
        coupling_scope="fvm_vpm",
        freestream_velocity=FREESTREAM,
        collect_timing=True,
    )
    panel.add_surface("body", str(stl))
    panel.initialize()
    return panel, triangles


def solve_incident(panel, incident, time):
    import taichi as ti

    dtype = ti.f64 if panel.float_dtype == "f64" else ti.f32
    field = ti.Vector.field(3, dtype=dtype, shape=len(incident))
    field.from_numpy(incident.astype(np.float64 if dtype == ti.f64 else np.float32))
    panel.solve(FREESTREAM, field, time)


def cylinder_study(provenance):
    case = CASES / "01_cylinder_shedding_flow"
    collection = case / "reference_flow/solution/fine/fvm.pvd"
    targets = boundary_targets(2)
    rows, quadrature = [], []
    for time in (80.0, 90.0, 100.0):
        path = frame_at(collection, time)
        provenance.append(fingerprint(path))
        grid = pv.read(path)
        cut = grid.slice(normal="z", origin=[0, 0, 0])
        centres = cut.cell_centers().points[:, :2]
        reference = affine_sample(centres, cut["velocity"][:, :2], targets)
        for crop, bounds in (
            ("full", None),
            ("vpm_domain", (-5, 15, -5, 5)),
            ("transfer_box", (-1.25, 1.25, -1.25, 1.25)),
        ):
            omega = np.array(cut["vorticity"][:, 2])
            if bounds is not None:
                keep = (
                    (centres[:, 0] >= bounds[0])
                    & (centres[:, 0] <= bounds[1])
                    & (centres[:, 1] >= bounds[2])
                    & (centres[:, 1] <= bounds[3])
                )
                omega[~keep] = 0
            sources, weights = polygon_sources(cut.points, cut.faces, omega)
            for span in (1.0, 12.0, -1.0) if crop == "full" else (-1.0,):
                prediction = polygon_induction(targets, sources, weights, span) + FREESTREAM[:2]
                rows.append(
                    dict(time=time, crop=crop, span=span, **error_metrics(prediction, reference))
                )
        if time == 100.0:
            for order in (4, 8, 16):
                sources, weights = polygon_sources(
                    cut.points, cut.faces, cut["vorticity"][:, 2], order
                )
                prediction = polygon_induction(targets, sources, weights) + FREESTREAM[:2]
                quadrature.append(dict(order=order, **error_metrics(prediction, reference)))

    panel, triangles = panel_for(case / "assets/cylinder_long.stl")
    provenance.append(fingerprint(case / "assets/cylinder_long.stl"))
    panel.solve(FREESTREAM, None, 0.0)
    theta = np.arange(64) * (2 * np.pi / 64)
    analytic = []
    for radius in (0.55, 0.75, 1.0, 1.5, 2.0, 4.0):
        points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros(64)))
        exact = np.column_stack(
            (
                1 - 0.25 / radius**2 * np.cos(2 * theta),
                -0.25 / radius**2 * np.sin(2 * theta),
                np.zeros(64),
            )
        )
        analytic.append(
            dict(
                radius=radius,
                **error_metrics(panel.compute_induced_velocity(points) + FREESTREAM, exact),
            )
        )
    panel.solve(FREESTREAM, None, 0.0)
    uniform_diagnostic = panel.results["diagnostic_history"][-1].copy()

    # grid is the final t=100 reference. Replay the actual 1D slab, not an
    # infinitely replicated wake, against the 12D body used by this tutorial.
    position = grid.cell_centers().points
    strength = grid["vorticity"] * grid["cell_volume"][:, None]
    sigma = 0.5 * np.cbrt(grid["cell_volume"])
    points = np.column_stack((targets, np.zeros(len(targets))))
    base = volume_induction(points, position, strength, sigma) + FREESTREAM
    centres = panel.lattice.panel_centre.to_numpy()[: len(triangles)]
    solve_incident(panel, volume_induction(centres, position, strength, sigma), 100.0)
    correction = panel.compute_induced_velocity(points)
    reference3 = np.column_stack((reference, np.zeros(len(reference))))
    edges = np.stack(
        [np.linalg.norm(triangles[:, (i + 1) % 3] - triangles[:, i], axis=1) for i in range(3)],
        axis=1,
    )
    # The analytical baseline above uses f64 to isolate the formulation; also
    # check the tutorial's configured field precision on the CPU backend.
    panel32, _ = panel_for(case / "assets/cylinder_long.stl", precision="f32")
    panel32.solve(FREESTREAM, None, 0.0)
    radius = 1.5
    ring = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros(64)))
    exact32 = np.column_stack(
        (
            1 - 0.25 / radius**2 * np.cos(2 * theta),
            -0.25 / radius**2 * np.sin(2 * theta),
            np.zeros(64),
        )
    )
    return {
        "reconstruction": rows,
        "quadrature": quadrature,
        "potential_rings": analytic,
        "uniform_panel_diagnostic": uniform_diagnostic,
        "f32_cpu_potential_ring": {
            "radius": radius,
            **error_metrics(panel32.compute_induced_velocity(ring) + FREESTREAM, exact32),
            "panel_diagnostic": panel32.results["diagnostic_history"][-1],
        },
        "slab_replay": {
            "without_panel": error_metrics(base, reference3),
            "with_panel": error_metrics(base + correction, reference3),
            "panel_diagnostic": panel.results["diagnostic_history"][-1],
        },
        "geometry": {
            "panels": len(triangles),
            "max_edge_ratio": float(np.max(edges.max(axis=1) / edges.min(axis=1))),
            "collocation_points_in_resolved_span": int(np.count_nonzero(abs(centres[:, 2]) < 0.5)),
        },
    }


def cube_study(provenance):
    case = CASES / "02_cube_flow"
    collection = case / "reference_flow/solution/fine/fine.pvd"
    targets = boundary_targets(3)
    panel, triangles = panel_for(case / "assets/cube.stl")
    provenance.append(fingerprint(case / "assets/cube.stl"))
    centres = panel.lattice.panel_centre.to_numpy()[: len(triangles)]
    rows, particle_rows = [], []
    for time in (5.0, 10.0, 20.0):
        path = frame_at(collection, time)
        provenance.append(fingerprint(path))
        grid = pv.read(path)
        position = grid.cell_centers().points
        reference = affine_sample(position, grid["velocity"], targets)
        strength = grid["vorticity"] * grid["cell_volume"][:, None]
        for scale in (0.25, 0.5, 1.0):
            prediction = (
                volume_induction(targets, position, strength, scale * np.cbrt(grid["cell_volume"]))
                + FREESTREAM
            )
            rows.append(dict(time=time, sigma_over_h=scale, **error_metrics(prediction, reference)))
        particle_frame = frame_at(case / "solution/vpm.pvd", time).with_suffix(".h5")
        provenance.append(fingerprint(particle_frame))
        with h5py.File(particle_frame) as handle:
            particles = handle["particles"]
            position = particles["position"][:].astype(float)
            strength = particles["vortex_strength"][:].astype(float)
            # Production Gaussian uses exp(-(r/core_radius)**2).
            sigma = particles["core_radius"][:].astype(float) / np.sqrt(2)
        base = volume_induction(targets, position, strength, sigma) + FREESTREAM
        solve_incident(panel, volume_induction(centres, position, strength, sigma), time)
        correction = panel.compute_induced_velocity(targets)
        particle_rows.append(
            {
                "time": time,
                "particles": len(position),
                "without_panel": error_metrics(base, reference),
                "with_panel": error_metrics(base + correction, reference),
                "panel_correction": error_metrics(correction, np.zeros_like(correction)),
                "panel_diagnostic": panel.results["diagnostic_history"][-1],
            }
        )
    return {"reference_reconstruction": rows, "saved_particle_replay": particle_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("cylinder", "cube", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = []
    result = {
        "scope": "Offline reconstruction and frozen-state replay; no trajectory/force validation",
        "velocity_error_normalization": "U_inf = 1 m/s",
        "reference_sampling": "12-neighbour affine fit; 320 cylinder or 216 cube targets",
        "sources": provenance,
        "study_code": [
            fingerprint(Path(__file__)),
            fingerprint(Path(__file__).with_name("operators.py")),
        ],
    }
    for name, study in (("cylinder", cylinder_study), ("cube", cube_study)):
        if args.case in (name, "all"):
            case = CASES / ("01_cylinder_shedding_flow" if name == "cylinder" else "02_cube_flow")
            for relative in (
                "setup.py",
                "reference_flow/setup.py",
                "reference_flow/solution/fine/fvm_metadata.json",
            ):
                provenance.append(fingerprint(case / relative))
            print(f"Running {name} reconstruction study", flush=True)
            result[name] = study(provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
