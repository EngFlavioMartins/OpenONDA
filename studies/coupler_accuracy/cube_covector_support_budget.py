"""Separate occupied-support quadrature from a resolved periodic source integral.

This is a frozen self-induced-flow control, not a replacement for the native
body-bounded source budget. The full periodic source integral should vanish.
The fluid-only integral can be nonzero because of the excluded cube. Comparing
that integral with existing-particle quadrature isolates the effect of occupied
support. No particles are inserted, masked, projected, or advanced.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from cube_represented_evolution_budget import PeriodicCloud
import numpy as np
from threadpoolctl import threadpool_limits


def measure(fields, radius, spacing, lengths):
    """Integrate the same resolved source over full, fluid and occupied domains."""
    started = time.perf_counter()
    position, strength = fields["particle_position"], fields["particle_strength"]
    grid = PeriodicCloud(position, fields["points"], spacing, lengths, radius)
    w_hat = grid.vector_spectrum()
    for i in range(3):
        w_hat[..., i] = grid.gaussian * grid.deposit(strength[:, i])
    u_hat = grid.induce(w_hat)
    ell = grid.vector_field(w_hat - grid.project(w_hat.copy()))
    del w_hat
    # Exact cube volume fractions remove changes in the excluded volume as
    # quadrature is refined; the source still has midpoint quadrature error.
    fraction = []
    for i in range(3):
        axis = grid.origin[i] + spacing * np.arange(grid.shape[i])
        overlap = np.minimum(axis + spacing / 2, 0.5) - np.maximum(axis - spacing / 2, -0.5)
        fraction.append(np.maximum(overlap, 0) / spacing)
    solid_weight = (
        fraction[0][:, None, None] * fraction[1][None, :, None] * fraction[2][None, None, :]
    )
    full, solid, occupied = [], [], []
    for i in range(3):
        source = np.zeros(grid.shape)
        for j in range(3):
            gradient = grid.invert(1j * grid.wave[i] * u_hat[..., j])
            source -= 2 * gradient * ell[..., j]
        full.append(float(source.sum() * spacing**3))
        solid.append(float(np.sum(source * solid_weight) * spacing**3))
        occupied.append(float(source[grid.source_index].sum() * 0.06**3))
    fluid = np.asarray(full) - solid
    return {
        "quadrature_spacing": spacing,
        "periodic_lengths": lengths,
        "excluded_cube_volume": float(solid_weight.sum() * spacing**3),
        "whole_periodic_source_integral": full,
        "solid_source_integral": solid,
        "fluid_source_integral": fluid.tolist(),
        "occupied_particle_source_integral": occupied,
        "occupied_minus_full_fluid": (np.asarray(occupied) - fluid).tolist(),
        "seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.audit / "fields_t6.npz"
    with np.load(source) as data:
        fields = {key: data[key].astype(float) for key in data.files}
    with np.load(args.audit / "particles_accepted_t6.npz") as data:
        radius = data["core_radius"]
    np.testing.assert_array_equal(radius, np.full(len(radius), radius[0]))
    report = {
        "scope": __doc__,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    for spacing, lengths in (
        (0.03, (7.2, 3.6, 3.6)),
        (0.02, (7.2, 3.6, 3.6)),
        (0.03, (9.6, 4.8, 4.8)),
    ):
        with threadpool_limits(limits=2):
            row = measure(fields, float(radius[0]), spacing, lengths)
        report["cases"].append(row)
        (args.output / "support.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
