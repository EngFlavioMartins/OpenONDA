"""
GBD/DVH active-box workspace: numerical neutrality and bounded dispatch.

The grid-diffusion path used to set the active grid to the whole declared
``vpm_domain_bounds`` whenever the domain was pre-configured (the Vulkan path
always is), and then transferred the whole allocation with ``to_numpy`` /
``from_numpy``.  On a device with ``maxComputeWorkGroupCount = 65535`` that
overflows the dispatch for any large domain and surfaces only as
``Dispatch error : RhiResult(not_supported)``.

The fix replaces both with an active box that tracks the particle cloud while
staying on the pre-configured lattice, plus chunked host transfers.  That is a
change in *work performed*, never in the diffusion equation, and these tests
pin exactly that.
"""

import numpy as np
import pytest
import taichi as ti

from source.solvers.VPM.runtime.backend import reset_taichi_backend

H = 0.05
DOMAIN = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]


@pytest.fixture
def diffusion():
    reset_taichi_backend()
    ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32, random_seed=0)
    from source.solvers.VPM.physics.diffusion import DiffusionPhysics

    d = DiffusionPhysics(max_particles=4096)
    d.configure_max_grid_extent(DOMAIN, H)
    yield d
    reset_taichi_backend()


def _cloud(n=400, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=0.08, size=(n, 3)).astype(np.float32)
    circ = (rng.normal(size=(n, 3)) * 1e-3).astype(np.float32)
    return pos, circ


# ── lattice phase ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_active_box_origin_stays_on_the_configured_lattice(diffusion):
    """Origin must differ from the anchor by a whole number of cells.

    That is what preserves the fixed grid phase the full-domain path provided,
    and with it the asymmetric flat-end artefact it was introduced to avoid.
    """
    anchor = np.asarray(diffusion._fixed_grid_min, dtype=np.float64)
    for seed in range(4):
        pos, _ = _cloud(seed=seed)
        grid_min, _ = diffusion._lattice_aligned_bounds(pos, H, 3.0)
        steps = (np.asarray(grid_min, dtype=np.float64) - anchor) / H
        assert np.allclose(steps, np.rint(steps), atol=1e-4), (
            f"origin off-lattice by {np.abs(steps - np.rint(steps)).max():.2e} cells"
        )


@pytest.mark.unit
def test_active_box_covers_the_cloud_and_fits_the_allocation(diffusion):
    cap = np.asarray(diffusion._max_grid_dims)
    anchor = np.asarray(diffusion._fixed_grid_min, dtype=np.float64)
    for seed in range(4):
        pos, _ = _cloud(seed=seed)
        grid_min, dims = diffusion._lattice_aligned_bounds(pos, H, 3.0)
        dims = np.asarray(dims)
        offset = np.rint((np.asarray(grid_min, dtype=np.float64) - anchor) / H).astype(int)
        assert (offset >= 0).all()
        assert (offset + dims <= cap).all(), "active box escapes the allocation"
        hi = np.asarray(grid_min) + (dims - 1) * H
        assert (pos.min(axis=0) >= np.asarray(grid_min) - 1e-6).all()
        assert (pos.max(axis=0) <= hi + 1e-6).all(), "cloud not covered"


@pytest.mark.unit
@pytest.mark.parametrize("axis,side", [(axis, side) for axis in range(3) for side in (-1, 1)])
def test_active_box_clamps_safely_at_every_domain_face(diffusion, axis, side):
    rng = np.random.default_rng(12 + 2 * axis + (side > 0))
    pos = rng.uniform(-0.08, 0.08, size=(200, 3)).astype(np.float32)
    pos[:, axis] += side * 0.415

    grid_min, dims = diffusion._lattice_aligned_bounds(pos, H, 3.0)
    anchor = np.asarray(diffusion._fixed_grid_min, dtype=np.float64)
    cap = np.asarray(diffusion._max_grid_dims)
    offset = np.rint((np.asarray(grid_min, dtype=np.float64) - anchor) / H).astype(int)
    grid_max = np.asarray(grid_min) + (np.asarray(dims) - 1) * H

    assert (offset >= 0).all()
    assert (offset + np.asarray(dims) <= cap).all()
    assert (pos.min(axis=0) >= np.asarray(grid_min) - 1e-6).all()
    assert (pos.max(axis=0) <= grid_max + 1e-6).all()


@pytest.mark.unit
def test_active_box_ignores_particles_already_outside_retention_domain(diffusion):
    pos, _ = _cloud()
    pos = np.vstack((pos, np.array([[100.0, 100.0, 100.0]], dtype=np.float32)))

    _, dims = diffusion._lattice_aligned_bounds(pos, H, 3.0)

    assert int(np.prod(dims)) < int(np.prod(diffusion._max_grid_dims)) // 2


@pytest.mark.unit
def test_active_box_is_much_smaller_than_the_full_domain(diffusion):
    pos, _ = _cloud()
    _, dims = diffusion._lattice_aligned_bounds(pos, H, 3.0)
    active = int(np.prod(dims))
    full = int(np.prod(diffusion._max_grid_dims))
    assert active < full // 2, f"active {active} vs full {full} — no reduction"


# ── bounded host transfers ──────────────────────────────────────────────────


@pytest.mark.unit
def test_bounded_vec_grid_transfer_roundtrip(diffusion):
    """Upload then download must reproduce the payload bit-for-bit in f32."""
    nx, ny, nz = 7, 5, 6
    diffusion._ensure_grid_capacity(nx, ny, nz)
    rng = np.random.default_rng(1)
    payload = rng.normal(size=(nx, ny, nz, 3)).astype(np.float32)

    diffusion._upload_active_vec_grid(diffusion._current_grid, payload, nx, ny, nz)
    out = diffusion._download_active_vec_grid(diffusion._current_grid, nx, ny, nz)

    np.testing.assert_array_equal(out, payload)


@pytest.mark.unit
def test_bounded_transfer_spans_multiple_chunks():
    """Exercise the chunk loop, not just a single sub-chunk transfer."""
    reset_taichi_backend()
    ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32, random_seed=0)
    try:
        from source.solvers.VPM.physics import diffusion as diffusion_module
        from source.solvers.VPM.physics.diffusion import DiffusionPhysics

        nx = ny = nz = 50
        assert nx * ny * nz > diffusion_module._GRID_TRANSFER_CHUNK

        d = DiffusionPhysics(max_particles=256)
        d.configure_max_grid_extent([0.0, (nx - 1) * H] * 3, H)
        d._ensure_grid_capacity(nx, ny, nz)

        rng = np.random.default_rng(2)
        payload = rng.normal(size=(nx, ny, nz, 3)).astype(np.float32)
        d._upload_active_vec_grid(d._current_grid, payload, nx, ny, nz)
        out = d._download_active_vec_grid(d._current_grid, nx, ny, nz)
        np.testing.assert_array_equal(out, payload)
    finally:
        reset_taichi_backend()


@pytest.mark.unit
def test_bounded_scalar_grid_upload(diffusion):
    nx, ny, nz = 6, 4, 5
    diffusion._ensure_grid_capacity(nx, ny, nz)
    rng = np.random.default_rng(3)
    payload = rng.random((nx, ny, nz)).astype(np.float32)
    diffusion._upload_active_scalar_grid(diffusion._nu_eff_grid, payload, nx, ny, nz)
    got = diffusion._nu_eff_grid.to_numpy()[:nx, :ny, :nz]
    np.testing.assert_array_equal(got, payload)


@pytest.mark.unit
def test_batched_m4_scatter_matches_single_dispatch(diffusion):
    """Splitting particle deposits must leave the accumulated grid unchanged."""
    from source.solvers.VPM.particles.container import Particles

    position, circulation = _cloud(n=37, seed=44)
    particles = Particles(max_particles=64)
    particles.add_vortex_particles(
        position=position,
        velocity=np.zeros_like(position),
        vortex_strength=circulation,
        core_radius=np.full(len(position), H, dtype=np.float32),
        volume=np.full(len(position), H**3, dtype=np.float32),
        kinematic_viscosity=np.full(len(position), 1.0e-3, dtype=np.float32),
    )
    grid_min, (nx, ny, nz) = diffusion._lattice_aligned_bounds(position, H, 3.0)
    nx, ny, nz = diffusion._ensure_grid_capacity(nx, ny, nz)
    gmin = np.asarray(grid_min, dtype=float)

    diffusion._zero_grid_kernel(diffusion._current_grid, nx, ny, nz)
    diffusion._m4_scatter_gpu_kernel(
        particles.position,
        particles.vortex_strength,
        diffusion._current_grid,
        *gmin,
        H,
        nx,
        ny,
        nz,
        0,
        len(position),
    )
    ti.sync()
    single = diffusion._download_active_vec_grid(diffusion._current_grid, nx, ny, nz)

    diffusion._zero_grid_kernel(diffusion._current_grid, nx, ny, nz)
    for start in range(0, len(position), 4):
        count = min(4, len(position) - start)
        diffusion._m4_scatter_gpu_kernel(
            particles.position,
            particles.vortex_strength,
            diffusion._current_grid,
            *gmin,
            H,
            nx,
            ny,
            nz,
            start,
            count,
        )
        ti.sync()
    batched = diffusion._download_active_vec_grid(diffusion._current_grid, nx, ny, nz)

    np.testing.assert_array_equal(batched, single)


# ── numerical neutrality: active box vs the former full-domain grid ─────────


def _run_gbd(d, pos, circ, force_full_domain: bool):
    """Run one GBD firing, optionally pinning the old full-domain active box."""
    from source.solvers.VPM.particles.container import Particles

    n = len(pos)
    particles = Particles(max_particles=4096)
    particles.add_vortex_particles(
        position=pos,
        velocity=np.zeros((n, 3), np.float32),
        vortex_strength=circ,
        core_radius=np.full(n, 1.5 * H, np.float32),
        volume=np.full(n, H**3, np.float32),
        kinematic_viscosity=np.full(n, 1e-3, np.float32),
    )
    if force_full_domain:
        original = d._lattice_aligned_bounds
        d._lattice_aligned_bounds = lambda p, h, pad: (
            np.asarray(d._fixed_grid_min, dtype=np.float32).copy(),
            tuple(d._max_grid_dims),
        )
    try:
        d.gbd_diffusion(
            particles, time_step_size=1.0e-3, particle_spacing=H, nu=1.0e-3, domain_padding=3.0
        )
    finally:
        if force_full_domain:
            d._lattice_aligned_bounds = original
    n_out = particles.n_particles
    assert d._ping is True
    return {
        "n": n_out,
        "position": particles.position_cpu()[:n_out].copy(),
        "circulation": particles.vortex_strength_cpu()[:n_out].copy(),
        "radius": particles.core_radius_cpu()[:n_out].copy(),
        "group_id": particles.group_id_cpu()[:n_out].copy(),
        "total_circulation": particles.vortex_strength_cpu()[:n_out].sum(axis=0),
    }


@pytest.mark.verification
def test_active_box_gbd_matches_full_domain_gbd(diffusion):
    """The active-box workspace must not change the diffusion result.

    Both runs use the same lattice phase, so every node that carries vorticity
    exists in both grids at the same physical coordinate; only the count of
    empty nodes differs.
    """
    pos, circ = _cloud()
    new = _run_gbd(diffusion, pos, circ, force_full_domain=False)
    old = _run_gbd(diffusion, pos, circ, force_full_domain=True)

    assert new["n"] == old["n"], f"particle count changed: {new['n']} vs {old['n']}"

    order_new = np.lexsort(new["position"].T)
    order_old = np.lexsort(old["position"].T)
    for key, tol in (
        ("position", 1e-6),
        ("circulation", 1e-9),
        ("radius", 1e-6),
    ):
        np.testing.assert_allclose(
            new[key][order_new], old[key][order_old], rtol=0, atol=tol, err_msg=f"{key} changed"
        )
    np.testing.assert_array_equal(new["group_id"][order_new], old["group_id"][order_old])
    np.testing.assert_allclose(
        new["total_circulation"], old["total_circulation"], rtol=0, atol=1e-9
    )


# ── Vulkan dispatch regression ─────────────────────────────────────────────


@pytest.mark.gpu
def test_vulkan_large_allocation_small_active_box():
    """A large allocation must stay usable as long as dispatches are bounded.

    This is the shape of the cubeFlow failure: an allocation whose whole-field
    dispatch exceeds maxComputeWorkGroupCount, driven only over a small box.
    """
    reset_taichi_backend()
    try:
        ti.init(arch=ti.vulkan, default_fp=ti.f32, default_ip=ti.i32)
    except Exception:
        pytest.skip("Vulkan backend unavailable")

    try:
        from source.solvers.VPM.physics.diffusion import DiffusionPhysics

        nx_full, ny_full, nz_full = 260, 200, 200
        assert nx_full * ny_full * nz_full > 65535 * 128, "allocation too small to test the limit"

        d = DiffusionPhysics(max_particles=1024)
        h = 0.05
        d.configure_max_grid_extent(
            [0.0, (nx_full - 1) * h, 0.0, (ny_full - 1) * h, 0.0, (nz_full - 1) * h], h
        )
        nx, ny, nz = 12, 10, 11
        d._ensure_grid_capacity(nx, ny, nz)

        d._zero_grid_kernel(d._current_grid, nx, ny, nz)
        ti.sync()

        rng = np.random.default_rng(0)
        payload = rng.normal(size=(nx, ny, nz, 3)).astype(np.float32)
        d._upload_active_vec_grid(d._current_grid, payload, nx, ny, nz)
        got = d._download_active_vec_grid(d._current_grid, nx, ny, nz)
        ti.sync()
        np.testing.assert_array_equal(got, payload)
    finally:
        reset_taichi_backend()
