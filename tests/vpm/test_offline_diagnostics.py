"""Offline allocation follows saved populations, including shrinking wakes."""

import h5py
import pytest

from source.solvers.vpm.diagnostics.offline import OfflineFlowDiagnostics


@pytest.mark.parametrize("counts", [(100, 3), (3, 100), (0, 0)])
def test_offline_capacity_covers_every_frame(tmp_path, counts):
    for step, count in enumerate(counts):
        with h5py.File(tmp_path / f"vpm_{step:06d}.h5", "w") as archive:
            archive.create_group("solver").attrs["n_particles_total"] = count
    diagnostics = OfflineFlowDiagnostics(tmp_path)
    assert diagnostics._estimate_max_particles() == max(1, *counts)


def test_offline_does_not_hide_missing_population_metadata(tmp_path):
    with h5py.File(tmp_path / "vpm_000001.h5", "w") as archive:
        archive.create_group("solver")
    diagnostics = OfflineFlowDiagnostics(tmp_path)
    with pytest.raises(KeyError):
        diagnostics._estimate_max_particles()
