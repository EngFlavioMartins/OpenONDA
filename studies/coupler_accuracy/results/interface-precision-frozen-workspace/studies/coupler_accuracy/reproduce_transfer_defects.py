#!/usr/bin/env python3
"""Reproduce the removed transfer formulas against independent field identities.

The legacy quantities below explicitly evaluate the pre-fix formulas. They
are not a second production implementation or a claim of a full baseline run.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from source.coupler import CouplerSetup
from source.coupler.stable_renewal import (
    gaussian_represented_vortex_strength,
    vortex_strength_from_velocity_trace,
)
from source.coupler.vorticity_transfer import VorticityTransfer

ROOT = Path(__file__).resolve().parents[2]
box = np.array([-1.0, 1.0] * 3)
h = 0.0625
cell_spacing = (0.25, 0.5, 0.5)
axes = [np.arange(-1 + d / 2, 1, d) for d in cell_spacing]
position = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
cfg = CouplerSetup(
    transfer_method="buffered_m4_renewal",
    transfer_region_bounds=tuple(box),
    transfer_vorticity_cutoff=0,
)
coupler = SimpleNamespace(
    setup=cfg,
    kinematic_viscosity=0.01,
    fvm_box=box,
    vpm_core_radius_ratio=1.0,
    vpm_particle_spacing=h,
    vpm_time_step_size=0.01,
    vpm_solver=SimpleNamespace(viscous_scheme="GBD"),
)
donors = SimpleNamespace(
    setup=SimpleNamespace(boundaries=[]),
    get_cell_centre_coordinates=lambda: position,
    get_cell_volume=lambda: np.full(len(position), np.prod(cell_spacing)),
)
transfer = VorticityTransfer(coupler)
transfer.setup(donors)
lattice = transfer._stable_renewal_lattice
inside = np.all(np.abs(lattice.positions) < 0.75, axis=1)
distance, _ = cKDTree(position).query(lattice.positions)
phase = np.clip((distance - h) / h, 0, 1)
legacy_weight = 1 - phase * phase * (3 - 2 * phase)
gradient = np.zeros((len(position), 3, 3))
gradient[:, 1, 0], gradient[:, 0, 1] = -0.5, 0.5
velocity = position @ gradient[0]
strength = vortex_strength_from_velocity_trace(
    lattice.positions, h, lambda q: transfer._velocity_trace.sample(q, velocity, gradient)
)
current_target = strength[inside] * lattice.mesh_weight[inside, None] / h**3
shape = (17, 17, 17)
gamma = np.zeros((*shape, 3))
gamma[8, 8, 8, 2] = 1
legacy_peak = gaussian_filter(gamma[..., 2], 1 / np.sqrt(2), mode="constant", truncate=5)[8, 8, 8]
peak = gaussian_represented_vortex_strength(gamma.reshape(-1, 3), shape, 1, core_radius=1)
peak = peak[np.ravel_multi_index((8, 8, 8), shape), 2]
report = {
    "schema": "openonda-coupler-operator-audit/1",
    "setup": {
        "fvm_cell_spacing": cell_spacing,
        "particle_spacing": h,
        "test_nodes": int(inside.sum()),
    },
    "legacy_formula": {
        "zero_confidence_fraction": float(np.mean(legacy_weight[inside] == 0)),
        "mean_target_vorticity_z": float(legacy_weight[inside].mean()),
        "gaussian_peak_relative_error": float(legacy_peak * np.pi**1.5 - 1),
    },
    "current_implementation": {
        "constant_vorticity_max_error": float(np.max(np.abs(current_target - [0, 0, 1]))),
        "mean_target_vorticity_z": float(current_target[:, 2].mean()),
        "gaussian_peak_relative_error": float(peak * np.pi**1.5 - 1),
    },
    "finite_filament_span_3D_velocity_ratio": {
        str(r): float(1.5 / np.sqrt(1.5**2 + r * r)) for r in (1, 2, 4)
    },
    "source_sha256": {
        str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in map(
            Path,
            (
                "source/coupler/stable_renewal.py",
                "source/coupler/vorticity_transfer.py",
                "source/coupler/geometry.py",
            ),
        )
    },
}
output = Path(__file__).resolve().parent / "results/operator-audit.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
