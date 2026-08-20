"""Behaviour of the built-in ring diagnostics sampler."""

import csv
from types import SimpleNamespace

import numpy as np

from openonda.vpm import RingDiagnosticsSampler


def test_builtin_ring_sampler_writes_each_particle_group(tmp_path):
    theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    positions = []
    circulations = []
    groups = []
    for group_id, center in enumerate((-0.5, 0.5)):
        position = np.column_stack(
            (
                np.full_like(theta, center),
                np.cos(theta),
                np.sin(theta),
            )
        )
        circulation = np.column_stack(
            (
                np.zeros_like(theta),
                -np.sin(theta),
                np.cos(theta),
            )
        )
        positions.append(position)
        circulations.append(circulation)
        groups.append(np.full(len(theta), group_id))

    solver = SimpleNamespace(
        particles_positions=np.vstack(positions),
        particle_vortex_strength=np.vstack(circulations),
        particles_group_ids=np.concatenate(groups),
    )
    output = tmp_path / "ring_diagnostics.csv"
    RingDiagnosticsSampler().save_csv(solver, output, time=0.25, step=5)

    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["group_id"]) for row in rows] == [0, 1]
    assert [int(row["step"]) for row in rows] == [5, 5]
    np.testing.assert_allclose([float(row["major_radius"]) for row in rows], 1.0)
