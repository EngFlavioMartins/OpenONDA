"""One complete FVM extrusion layer couples to infinite planar filaments."""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler import CouplerSetup
from source.coupler import vorticity_transfer as module
from source.solvers.vpm.physics.induction.planar import PlanarInduction


def _donors(z_levels=(0.5,)):
    return (
        np.array(
            np.meshgrid(
                np.linspace(-0.875, 0.875, 8),
                np.linspace(-0.875, 0.875, 8),
                z_levels,
                indexing="ij",
            )
        )
        .reshape(3, -1)
        .T
    )


@pytest.mark.parametrize(
    "z_levels, heights", [((0.5,), (1.0,)), ((0.1, 0.35, 0.75), (0.2, 0.3, 0.5))]
)
def test_complete_planar_stacks_accept_single_or_unequal_layers(z_levels, heights):
    donors = _donors(z_levels)
    area = 0.25**2 * (1.0 + donors[:, 0] ** 2)
    volume = area * np.tile(heights, 64)
    order = np.random.default_rng(817).permutation(len(donors))
    groups, counts = module._planar_cell_stack_groups(donors[order], volume[order])
    np.testing.assert_array_equal(counts, len(z_levels))
    for group in range(len(counts)):
        np.testing.assert_array_equal(np.sort(donors[order][groups == group, 2]), z_levels)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "mixed_levels", "volume_profile"])
def test_planar_stacks_reject_partial_or_nonextruded_donors(defect):
    donors = _donors((0.25, 0.75))
    volume = np.full(len(donors), 0.25**2 * 0.5)
    if defect == "missing":
        donors, volume = donors[1:], volume[1:]
    elif defect == "duplicate":
        donors[0] = donors[1]
    elif defect == "mixed_levels":
        donors[:2, 2] += 0.01
    else:
        volume[0] *= 1.1
    with pytest.raises(ValueError, match="extruded"):
        module._planar_cell_stack_groups(donors, volume)


def _transfer(z_levels=(0.5,), span=1.0):
    donors = _donors(z_levels)
    driver = SimpleNamespace(
        setup=CouplerSetup(transfer_region_bounds=(-0.75, 0.75, -0.75, 0.75, 0, 1)),
        kinematic_viscosity=0.01,
        fvm_box=np.array([-1, 1, -1, 1, 0, 1]),
        vpm_core_radius_ratio=1.0,
        vpm_particle_spacing=0.125,
        vpm_time_step_size=0.01,
        vpm_solver=SimpleNamespace(
            induction=PlanarInduction(span=span, plane_z=0.5),
            viscous_scheme="GBD",
            setup=SimpleNamespace(
                viscous=SimpleNamespace(gbd_threshold_mode="absolute", gbd_threshold=0.0)
            ),
        ),
    )
    fvm = SimpleNamespace(
        setup=SimpleNamespace(boundaries=[]),
        get_cell_centre_coordinates=lambda: donors,
        get_cell_volume=lambda: np.full(len(donors), 0.25**2 / len(z_levels)),
    )
    transfer = module.VorticityTransfer(driver)
    transfer.setup(fvm)
    return donors, transfer


@pytest.mark.parametrize("span", [1.0, 2.5])
def test_single_layer_cached_trace_has_exact_circulation_and_span_units(monkeypatch, span):
    donors, transfer = _transfer(span=span)
    captured = []

    def capture(vpm, *, lattice, fvm_vortex_strength_at_node, **kwargs):
        captured.append(fvm_vortex_strength_at_node(lattice.positions))
        return captured[-1]

    monkeypatch.setattr(module, "replace_particles_from_buffered_m4_renewal", capture)
    velocity = np.tile([100.0, 0, 0], (len(donors), 1))
    gradient = np.zeros((len(donors), 3, 3))
    transfer._transfer_buffered_m4_renewal(
        None, fvm_velocity=velocity, fvm_velocity_gradient=gradient
    )
    np.testing.assert_array_equal(captured[-1], 0)
    stencils = transfer._buffered_trace_stencils
    assert len(stencils) == 4

    omega = 1e-6
    velocity[:, 1] = omega * donors[:, 0]
    gradient[:, 0, 1] = omega
    transfer._transfer_buffered_m4_renewal(
        None, fvm_velocity=velocity, fvm_velocity_gradient=gradient
    )
    assert transfer._buffered_trace_stencils is stencils
    lattice = transfer._stable_renewal_lattice
    interior = np.all(np.abs(lattice.positions[:, :2]) < 0.6, axis=1)
    expected = np.zeros((np.count_nonzero(interior), 3))
    expected[:, 2] = omega * 0.125**2 * span
    np.testing.assert_allclose(captured[-1][interior], expected, rtol=1e-7, atol=1e-15)
    assert lattice.shape[2] == 1
    assert lattice.particle_volume == pytest.approx(0.125**2 * span)
    np.testing.assert_array_equal(lattice.positions[:, 2], 0.5)
    assert transfer.last_spanwise_metrics["span_velocity_max"] == 0
    assert transfer.last_spanwise_metrics["span_variation_max"] == 0


@pytest.mark.parametrize("defect", ["span_velocity", "span_variation"])
def test_single_layer_support_retains_planar_velocity_guards(monkeypatch, defect):
    donors, transfer = _transfer((0.25, 0.75) if defect == "span_variation" else (0.5,))
    velocity = np.tile([1.0, 0, 0], (len(donors), 1))
    if defect == "span_velocity":
        velocity[:, 2] = 0.01
    else:
        velocity[0, 0] += 0.01
    monkeypatch.setattr(
        module,
        "replace_particles_from_buffered_m4_renewal",
        lambda *args, **kwargs: pytest.fail("invalid donor state reached renewal"),
    )
    with pytest.raises(RuntimeError, match="lost spanwise invariance"):
        transfer._transfer_buffered_m4_renewal(
            None, fvm_velocity=velocity, fvm_velocity_gradient=np.zeros((len(donors), 3, 3))
        )


@pytest.mark.integration
@pytest.mark.parametrize("ranks", [1, 2])
def test_one_layer_periodic_native_planar_coupling_and_shutdown(tmp_path, ranks):
    if ranks > 1 and (find_spec("mpi4py") is None or find_spec("petsc4py") is None):
        pytest.skip("MPI and PETSc are required")
    if ranks > 1 and not (
        Path(sys.executable).with_name("mpiexec").is_file() or shutil.which("mpiexec")
    ):
        pytest.skip("mpiexec is required")
    output = tmp_path / f"planar-{ranks}-rank"
    environment = os.environ.copy()
    environment.pop("_OPENONDA_MPI_CHILD", None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    environment["TI_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path / "taichi-cache")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_planar_single_layer_smoke.py")),
            str(output),
            str(ranks),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-8000:]
    report = json.loads((output / "qualification.json").read_text())
    assert report["ranks"] == ranks
    assert report["global_cells"] == 16
    assert report["layers_per_stack"] == 1
    assert report["particle_volume"] == 0.125**2
    assert report["exterior_absolute_strength"] > 1e-4
    assert report["span_velocity_max"] < 1e-3
    assert report["span_variation_max"] == 0
    for rank in range(ranks):
        assert json.loads((output / f"rank-{rank}.json").read_text()) == {
            "rank": rank,
            "steps": 2,
            "time": 0.02,
            "closed": True,
        }
