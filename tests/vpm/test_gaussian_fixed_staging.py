"""Growing Gaussian prefixes keep fixed transfer shapes and publication indices."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.base import _HOST_TRANSFER_CHUNK_SIZE, PhysicsBase
from source.solvers.vpm.physics.induction import slip_slab


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    owned = ti.lang.impl.get_runtime().prog is None
    if owned:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=1)
    yield
    if owned:
        ti.reset()


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
def test_snapshot_growth_reuses_fixed_shapes_and_one_download_per_field(monkeypatch, dtype):
    scalar_type = np.float32 if dtype == ti.f32 else np.float64
    chunk = _HOST_TRANSFER_CHUNK_SIZE
    capacity = chunk + 19
    physics = PhysicsBase("GAUSSIAN", 1, dtype, max_evaluation_points=1)
    position = ti.Vector.field(3, dtype, shape=capacity)
    core = ti.field(dtype, shape=capacity)
    values = (np.arange(capacity * 3).reshape(capacity, 3) / 13).astype(scalar_type)
    radii = (np.arange(capacity) / 17).astype(scalar_type)
    values[-7:] = np.nan
    radii[-7:] = np.nan
    position.from_numpy(values)
    core.from_numpy(radii)
    transfers, downloads = [], []
    for name in ("_extract_vec3_field_prefix", "_extract_scalar_field_prefix"):
        original = getattr(physics, name)

        def extract(source, buffer, start, count, original=original):
            transfers.append((id(buffer), buffer.shape, buffer.dtype, start, count))
            return original(source, buffer, start, count)

        monkeypatch.setattr(physics, name, extract)
    for name in ("_download_vector_field", "_download_scalar_field"):
        original = getattr(physics, name)

        def download(source, count, original=original):
            downloads.append((source, count))
            return original(source, count)

        monkeypatch.setattr(physics, name, download)
    monkeypatch.setattr(position, "to_numpy", lambda: pytest.fail("capacity download"))
    monkeypatch.setattr(core, "to_numpy", lambda: pytest.fail("capacity download"))

    for count in (0, 7, chunk + 1, chunk + 12):
        first = max(0, count - 3)
        sources, cores, targets = slip_slab._active_snapshots(
            physics, ((position, first), (core, count), (position, count))
        )
        assert downloads[-2:] == [(position, count), (core, count)]
        assert sources.dtype == cores.dtype == targets.dtype == scalar_type
        np.testing.assert_array_equal(sources, values[:first])
        np.testing.assert_array_equal(targets, values[:count])
        np.testing.assert_array_equal(cores, radii[:count])
        if first:
            assert np.shares_memory(sources, targets)
    assert {shape for _, shape, _, _, _ in transfers} == {(chunk, 3), (chunk,)}
    assert len({identity for identity, _, _, _, _ in transfers}) == 2
    assert all(dt == scalar_type for _, _, dt, _, _ in transfers)
    assert [(start, count) for _, shape, _, start, count in transfers if shape == (chunk, 3)] == [
        (0, 7),
        (0, chunk),
        (chunk, 1),
        (0, chunk),
        (chunk, 12),
    ]


def test_f64_sources_remain_exact_with_f32_physics_and_empty_prefixes():
    physics = PhysicsBase("GAUSSIAN", 1, ti.f32, max_evaluation_points=1)
    position = ti.Vector.field(3, ti.f64, shape=5)
    radius = ti.field(ti.f64, shape=5)
    values = np.arange(15).reshape(5, 3) + 2**-35
    radii = np.arange(5) + 2**-36
    position.from_numpy(values)
    radius.from_numpy(radii)
    for count in (0, 3, 5):
        x, sigma = slip_slab._active_snapshots(physics, ((position, count), (radius, count)))
        assert x.dtype == sigma.dtype == np.float64
        np.testing.assert_array_equal(x, values[:count])
        np.testing.assert_array_equal(sigma, radii[:count])
    assert physics._host_transfer_buffer("vector", position, "upload").dtype == np.float32
    assert physics._host_transfer_buffer("vector", position, "download").dtype == np.float64


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_stage_publication_preserves_chunk_offsets_rates_and_fixed_shapes(monkeypatch, dtype, mode):
    chunk = _HOST_TRANSFER_CHUNK_SIZE
    count, capacity = chunk + 5, chunk + 11
    scalar_type = np.float32 if dtype == ti.f32 else np.float64
    physics = PhysicsBase("GAUSSIAN", 1, dtype, max_evaluation_points=1)
    velocity = ti.Vector.field(3, dtype, shape=capacity)
    gradient = ti.Matrix.field(3, 3, dtype, shape=capacity)
    strength = ti.Vector.field(3, dtype, shape=capacity)
    rate = ti.Vector.field(3, dtype, shape=capacity)
    rng = np.random.default_rng(18)
    host_u = rng.normal(size=(count, 3)).astype(scalar_type)
    host_j = rng.normal(size=(count, 3, 3)).astype(scalar_type)
    strengths = rng.normal(size=(capacity, 3)).astype(scalar_type)
    strength.from_numpy(strengths)
    velocity.fill(3)
    gradient.fill(5)
    rate.fill(7)
    original = slip_slab._publish_mesh_chunk
    transfers = []

    def publish(vector, matrix, *args):
        transfers.append((id(vector), id(matrix), vector.shape, matrix.shape, args[5], args[6]))
        return original(vector, matrix, *args)

    monkeypatch.setattr(slip_slab, "_publish_mesh_chunk", publish)
    for active in (3, count):
        slip_slab._publish_mesh_results(
            physics,
            host_u[:active],
            host_j[:active],
            velocity,
            gradient,
            strength,
            rate,
            physics._zero_velocity,
            active,
            mode,
            1,
            True,
            True,
            True,
            False,
        )
    expected_u, expected_j = (
        np.full((capacity, 3), 3.0, dtype=scalar_type),
        np.full((capacity, 3, 3), 5.0, dtype=scalar_type),
    )
    expected_u[:3] += host_u[:3]
    expected_j[:3] += host_j[:3]
    expected_u[:count] += host_u
    expected_j[:count] += host_j
    contraction = host_j if mode == 0 else host_j.transpose(0, 2, 1)
    if mode == 2:
        contraction = scalar_type(0.5) * (host_j + host_j.transpose(0, 2, 1))
    contribution = np.einsum("nij,nj->ni", contraction, strengths[:count])
    expected_rate = np.full((capacity, 3), 7.0, dtype=scalar_type)
    expected_rate[:3] += contribution[:3]
    expected_rate[:count] += contribution
    np.testing.assert_array_equal(velocity.to_numpy(), expected_u)
    np.testing.assert_array_equal(gradient.to_numpy(), expected_j)
    np.testing.assert_allclose(
        rate.to_numpy(),
        expected_rate,
        rtol=8 * np.finfo(scalar_type).eps,
        atol=8 * np.finfo(scalar_type).eps,
    )
    assert len({(vector, matrix) for vector, matrix, *_ in transfers}) == 1
    assert [(v, j, start, active) for _, _, v, j, start, active in transfers] == [
        ((chunk, 3), (chunk, 3, 3), 0, 3),
        ((chunk, 3), (chunk, 3, 3), 0, chunk),
        ((chunk, 3), (chunk, 3, 3), chunk, 5),
    ]


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
def test_partial_query_publication_keeps_disabled_outputs_and_adds_freestream(dtype):
    chunk = _HOST_TRANSFER_CHUNK_SIZE
    count, capacity = chunk + 3, chunk + 7
    scalar_type = np.float32 if dtype == ti.f32 else np.float64
    physics = PhysicsBase("GAUSSIAN", 1, dtype, max_evaluation_points=1)
    velocity = ti.Vector.field(3, dtype, shape=capacity)
    gradient = ti.Matrix.field(3, 3, dtype, shape=capacity)
    rate = ti.Vector.field(3, dtype, shape=capacity)
    background = ti.Vector.field(3, dtype, shape=())
    background[None] = [0.5, -0.25, 1.0]
    velocity.fill(19)
    gradient.fill(23)
    rate.fill(29)
    host_u = np.full((count, 3), 0.125, dtype=scalar_type)
    host_j = np.full((count, 3, 3), 0.0625, dtype=scalar_type)
    slip_slab._publish_mesh_results(
        physics,
        host_u,
        host_j,
        velocity,
        gradient,
        rate,
        rate,
        background,
        count,
        0,
        0,
        False,
        True,
        False,
        True,
    )
    expected = np.full((capacity, 3), 19.0, dtype=scalar_type)
    expected[:count] = host_u + [0.5, -0.25, 1.0]
    np.testing.assert_array_equal(velocity.to_numpy(), expected)
    np.testing.assert_array_equal(gradient.to_numpy(), 23)
    np.testing.assert_array_equal(rate.to_numpy(), 29)
    slip_slab._publish_mesh_results(
        physics,
        host_u,
        host_j,
        velocity,
        gradient,
        rate,
        rate,
        background,
        count,
        0,
        0,
        True,
        False,
        False,
        False,
    )
    np.testing.assert_array_equal(velocity.to_numpy(), expected)
    np.testing.assert_array_equal(gradient.to_numpy(), 23)
    expected_rate = np.full((capacity, 3), 29.0, dtype=scalar_type)
    expected_rate[:count] = 0
    np.testing.assert_array_equal(rate.to_numpy(), expected_rate)
