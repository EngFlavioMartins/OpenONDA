"""Gaussian slab dispatch: real Taichi publication with a bounded fake session.

These are ordering/lifecycle tests, NOT Gaussian numerical qualification. CPU
runtime rejection is separately checked before patching CUDA admission for the
fake session; no CuPy allocation or physical solver advance occurs here.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.config.fingerprint import _canonical_value
from source.solvers.vpm.core import solver as solver_module
from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction import slip_slab as module
from source.solvers.vpm.physics.induction.direct import DirectInduction
from source.solvers.vpm.physics.induction.fmm import FMMInduction
from source.solvers.vpm.physics.induction.gaussian_mesh.session import GaussianSlabPolicy
from source.solvers.vpm.physics.induction.reuse_backends import StandardFMMReuseContract


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    owned = ti.lang.impl.get_runtime().prog is None
    if owned:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    yield
    if owned:
        ti.reset()


class FakeSession:
    def __init__(self, **kwargs):
        self.kwargs, self.calls, self.closes = kwargs, [], 0
        self.failure = None
        self.cleanup_uncertain = False

    def evaluate(self, x, gamma, sigma, targets, *, source_only):
        self.calls.append((x.copy(), gamma.copy(), sigma.copy(), targets.copy(), source_only))
        if self.failure:
            raise self.failure
        dtype = np.dtype(self.kwargs["dtype"])
        u = np.broadcast_to([0.5, -0.25, 0.75], (len(targets), 3)).astype(dtype).copy()
        j = (
            np.broadcast_to(np.arange(9).reshape(3, 3) / 100, (len(targets), 3, 3))
            .astype(dtype)
            .copy()
        )
        if not len(x):
            u.fill(0)
            j.fill(0)
        return (
            u,
            j,
            {"shell": 2, "velocity_tail_bound": 1e-7, "gradient_tail_bound": 2e-7, "seconds": 0.0},
        )

    def close(self):
        self.closes += 1
        if self.cleanup_uncertain:
            raise RuntimeError("injected uncertain cleanup")


class Harness:
    def __init__(self, monkeypatch, scheme="TRANSPOSED"):
        monkeypatch.setattr(module, "_mesh_runtime_is_cuda", lambda: True)
        # This fixture qualifies publication/order with a fake session, not
        # optional CUDA-library availability (covered by separate tests).
        monkeypatch.setattr(module, "_admit_mesh_installation", lambda: None)
        monkeypatch.setattr(module, "_new_mesh_session", FakeSession)
        self.physics = PhysicsBase("GAUSSIAN", 2, ti.f64, max_evaluation_points=3)
        self.slab = module.SlipSlabInduction(
            DirectInduction(stretching_scheme=scheme),
            z_min=-0.5,
            z_max=0.5,
        ).bind(self.physics)
        self.x = ti.Vector.field(3, ti.f64, shape=2)
        self.g = ti.Vector.field(3, ti.f64, shape=2)
        self.r = ti.field(ti.f64, shape=2)
        self.q = ti.Vector.field(3, ti.f64, shape=3)
        self.u = ti.Vector.field(3, ti.f64, shape=3)
        self.j = ti.Matrix.field(3, 3, ti.f64, shape=3)
        self.rate = ti.Vector.field(3, ti.f64, shape=2)
        self.background = ti.Vector.field(3, ti.f64, shape=())
        self.x.from_numpy(np.array([[0.1, 0.2, 0.15], [-0.2, 0.05, -0.27]]))
        self.g.from_numpy(np.array([[0.2, -0.3, 0.7], [-0.4, 0.15, -0.2]]))
        self.r.from_numpy(np.array([0.12, 0.14]))
        self.q.from_numpy(np.array([[0.3, 0.1, 0.5], [0.2, 0.4, -0.5], [0, 0, 0]]))
        self.background[None] = [1, 2, 3]
        self.u.fill(19)
        self.j.fill(23)
        self.rate.fill(29)

    def stage(self, *, gradient=True, rate=True, base=False):
        (self.slab.base if base else self.slab).evaluate_stage(
            position=self.x,
            vortex_strength=self.g,
            core_radius=self.r,
            count=2,
            velocity_out=self.u,
            vortex_strength_rate_out=self.rate,
            velocity_gradient_out=self.j if gradient else None,
            strength_rate_enabled=rate,
        )

    def targets(self, *, sources=2, velocity=True, gradient=True, free=True, count=3):
        self.slab.evaluate_targets(
            target_position=self.q,
            source_position=self.x,
            source_vortex_strength=self.g,
            source_core_radius=self.r,
            target_velocity=self.u if velocity else None,
            target_velocity_gradient=self.j if gradient else None,
            target_count=count,
            source_count=sources,
            include_freestream=free,
            background_velocity=self.background,
        )


@pytest.mark.parametrize("scheme", ["DIRECT", "TRANSPOSED", "MIXED"])
@pytest.mark.parametrize(
    "gradient,rate", [(True, True), (False, True), (True, False), (False, False)]
)
def test_stage_preserves_primary_and_adds_correct_stretching(monkeypatch, scheme, gradient, rate):
    h = Harness(monkeypatch, scheme)
    h.stage(gradient=gradient, rate=rate, base=True)
    primary = (h.u.to_numpy(), h.j.to_numpy(), h.rate.to_numpy())
    h.stage(gradient=gradient, rate=rate)
    u, j, _ = h.slab._mesh_session.evaluate(
        h.x.to_numpy(), h.g.to_numpy(), h.r.to_numpy(), h.x.to_numpy(), source_only=False
    )
    np.testing.assert_allclose(h.u.to_numpy()[:2], primary[0][:2] + u, rtol=1e-13, atol=1e-13)
    if gradient:
        np.testing.assert_allclose(h.j.to_numpy()[:2], primary[1][:2] + j, rtol=1e-13, atol=1e-13)
    else:
        np.testing.assert_array_equal(h.j.to_numpy(), primary[1])
    contraction = j if scheme == "DIRECT" else j.transpose(0, 2, 1)
    if scheme == "MIXED":
        contraction = (j + j.transpose(0, 2, 1)) / 2
    expected = primary[2] + np.einsum("nij,nj->ni", contraction, h.g.to_numpy()) if rate else 0
    np.testing.assert_allclose(h.rate.to_numpy(), expected, rtol=1e-13, atol=1e-13)
    assert not h.slab._mesh_session.calls[0][-1]
    assert h.slab.last_tail["contract"] == "gaussian_interval_remainder_v1"
    assert h.slab._block_velocity.shape == (1,)


@pytest.mark.parametrize(
    "velocity,gradient,free", [(True, True, True), (True, False, False), (False, True, True)]
)
def test_target_role_is_whole_source_only_and_freestream_once(
    monkeypatch, velocity, gradient, free
):
    h = Harness(monkeypatch)
    monkeypatch.setattr(
        h.slab.base, "evaluate_targets", lambda **_: pytest.fail("primary double count")
    )
    h.targets(velocity=velocity, gradient=gradient, free=free)
    if velocity:
        expected = np.broadcast_to([0.5, -0.25, 0.75], (3, 3)) + ([1, 2, 3] if free else 0)
        np.testing.assert_array_equal(h.u.to_numpy(), expected)
    else:
        np.testing.assert_array_equal(h.u.to_numpy(), 19)
    if gradient:
        np.testing.assert_array_equal(
            h.j.to_numpy(), np.broadcast_to(np.arange(9).reshape(3, 3) / 100, (3, 3, 3))
        )
    else:
        np.testing.assert_array_equal(h.j.to_numpy(), 23)
    assert h.slab._mesh_session.calls[0][-1]


def test_tail_failure_precedes_primary_and_preserves_outputs(monkeypatch):
    h = Harness(monkeypatch)
    h.slab._mesh_session = FakeSession(dtype="float64")
    h.slab._mesh_session.failure = RuntimeError("tail admission rejected")
    monkeypatch.setattr(
        h.slab.base, "evaluate_stage", lambda **_: pytest.fail("early primary publication")
    )
    with pytest.raises(RuntimeError, match="tail admission"):
        h.stage()
    np.testing.assert_array_equal(h.u.to_numpy(), 19)
    np.testing.assert_array_equal(h.j.to_numpy(), 23)
    np.testing.assert_array_equal(h.rate.to_numpy(), 29)


def test_empty_alias_mutation_and_rebind_contract(monkeypatch):
    h = Harness(monkeypatch)
    h.targets(sources=0)
    np.testing.assert_array_equal(h.u.to_numpy(), np.broadcast_to([1, 2, 3], (3, 3)))
    np.testing.assert_array_equal(h.j.to_numpy(), 0)
    session = h.slab._mesh_session
    h.x[0] = [0.11, 0.2, 0.15]
    h.targets()
    assert session.calls[-1][0][0, 0] == 0.11  # no field-identity shortcut
    h.slab.bind(h.physics)
    assert session.closes == 1 and h.slab._mesh_session is None
    h.u = h.q
    with pytest.raises(ValueError, match="must not alias"):
        h.targets()
    h.slab.gaussian_mesh_policy = None
    with pytest.raises(RuntimeError, match="controls changed"):
        h.targets()


def test_prebind_runtime_and_kernel_admission_has_no_base_side_effect(monkeypatch):
    physics = PhysicsBase("GAUSSIAN", 1, ti.f64, max_evaluation_points=1)
    slab = module.SlipSlabInduction(
        DirectInduction(),
        z_min=-1,
        z_max=1,
        gaussian_mesh_policy=GaussianSlabPolicy(backend="cupy_cuda"),
    )
    monkeypatch.setattr(slab.base, "bind", lambda *_, **__: pytest.fail("unsupported base bind"))
    monkeypatch.setattr(module, "_mesh_runtime_is_cuda", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        slab.bind(physics)
    monkeypatch.setattr(module, "_mesh_runtime_is_cuda", lambda: True)
    with pytest.raises(ValueError, match="GAUSSIAN"):
        slab.bind(physics, kernel=make_vortex_kernel("WINCKELMANS"))
    assert slab.physics is None and slab._mesh_session is None


def test_default_policy_build_fingerprint_and_fmm_reuse_decline():
    default = module.SlipSlabInduction(DirectInduction(), z_min=-1, z_max=1)
    assert default.gaussian_mesh_policy == GaussianSlabPolicy()
    assert _canonical_value(default) == _canonical_value(default.build())
    policy = GaussianSlabPolicy()
    slab = module.SlipSlabInduction(
        DirectInduction(), z_min=-1, z_max=1, gaussian_mesh_policy=policy
    )
    copied = slab.build()
    assert copied.base is not slab.base and copied.gaussian_mesh_policy == policy
    identity = _canonical_value(slab)
    assert identity["gaussian_mesh_policy"]["tail_contract"] == policy.tail_contract
    copied.gaussian_mesh_policy = replace(policy, max_sources=999)
    assert identity != _canonical_value(copied)
    assert StandardFMMReuseContract(slab)() is None


def test_mesh_shell_ceiling_rejected_before_binding():
    physics = PhysicsBase("GAUSSIAN", 1, ti.f64, max_evaluation_points=1)
    with pytest.raises(ValueError, match="1024-shell capacity"):
        module.SlipSlabInduction(
            DirectInduction(),
            z_min=-1,
            z_max=1,
            max_shells=1025,
            gaussian_mesh_policy=GaussianSlabPolicy(),
        ).bind(physics)
    # Other radial kernels retain their existing shell controls.
    assert (
        module.SlipSlabInduction(DirectInduction(), z_min=-1, z_max=1, max_shells=1025).max_shells
        == 1025
    )


def test_gaussian_operator_cannot_be_disabled_or_called_as_reflected_kernel(monkeypatch):
    with pytest.raises(TypeError, match="cannot disable"):
        module.SlipSlabInduction(DirectInduction(), z_min=-1, z_max=1, gaussian_mesh_policy=None)
    h = Harness(monkeypatch)
    with pytest.raises(RuntimeError, match="field mesh operator"):
        h.slab._images(h.x, h.g, h.r, h.q, 2, 3, h.u, h.j)


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
def test_active_snapshots_skip_padding_and_download_each_distinct_field_once(monkeypatch, dtype):
    scalar_type = np.float32 if dtype == ti.f32 else np.float64
    physics = PhysicsBase("GAUSSIAN", 1, dtype, max_evaluation_points=1)
    position = ti.Vector.field(3, dtype, shape=17)
    radius = ti.field(dtype, shape=17)
    position.fill(float("nan"))
    radius.fill(float("nan"))
    position[0], position[1], position[2] = [1, 2, 3], [4, 5, 6], [7, 8, 9]
    radius[0], radius[1] = 0.1, 0.2
    original = physics._download_vector_field
    calls = []

    def download(source, count):
        calls.append((source, count))
        return original(source, count)

    monkeypatch.setattr(physics, "_download_vector_field", download)
    monkeypatch.setattr(position, "to_numpy", lambda: pytest.fail("capacity download"))
    monkeypatch.setattr(radius, "to_numpy", lambda: pytest.fail("capacity download"))
    sources, cores, targets = module._active_snapshots(
        physics, ((position, 2), (radius, 2), (position, 3))
    )
    assert calls == [(position, 3)]
    assert sources.dtype == cores.dtype == targets.dtype == scalar_type
    np.testing.assert_array_equal(sources, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(targets, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_array_equal(cores, np.array([0.1, 0.2], scalar_type))
    assert np.shares_memory(sources, targets)


@pytest.mark.parametrize("failure", [False, True])
def test_solver_closes_only_owned_mesh_and_never_resets_after_uncertainty(monkeypatch, failure):
    slab = module.SlipSlabInduction(DirectInduction(), z_min=-1, z_max=1)
    session = slab._mesh_session = FakeSession()
    session.cleanup_uncertain = failure
    events = []
    monkeypatch.setattr(
        slab.base, "close", lambda: pytest.fail("shared base closed"), raising=False
    )
    owner = SimpleNamespace(
        _closed=False,
        induction=slab,
        _run_started=True,
        _initial_conditions_built=True,
        _backend_claimed=True,
        stage_rhs=SimpleNamespace(close=lambda: events.append("cache")),
        _particle_snapshot_buffers={},
    )
    monkeypatch.setattr(solver_module, "reset_taichi_backend", lambda **_: events.append("reset"))
    if failure:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="uncertain cleanup"):
                solver_module.VPMSolver.close(owner)
        assert owner._backend_claimed and "reset" not in events
        assert session.closes == 1
    else:
        solver_module.VPMSolver.close(owner)
        assert events == ["cache", "reset"] and session.closes == 1


def test_unknown_induction_close_is_not_called(monkeypatch):
    owner = SimpleNamespace(
        _closed=False,
        induction=SimpleNamespace(
            close_mesh_session=lambda: pytest.fail("custom shared backend closed")
        ),
        _run_started=True,
        _initial_conditions_built=True,
        _backend_claimed=True,
        _particle_snapshot_buffers={},
    )
    monkeypatch.setattr(solver_module, "reset_taichi_backend", lambda **_: None)
    solver_module.VPMSolver.close(owner)
    assert owner._closed


def _stage_reservation(monkeypatch, harness, ensure):
    state = SimpleNamespace(_reclaim_stage_cache=None, _ensure_workspace=ensure)
    monkeypatch.setattr(
        harness.slab.base,
        "stage_workspace",
        lambda count, *, reclaim: FMMInduction.stage_workspace(state, count, reclaim=reclaim),
        raising=False,
    )
    return state


def test_source_scratch_precedes_mesh_admission_and_warm_cache_is_retained(monkeypatch):
    h = Harness(monkeypatch)
    old = h.slab._mesh_session = FakeSession(dtype="float64")
    capacity, events = 1, []

    def ensure(count):
        nonlocal capacity
        events.append("reserve")
        if count > capacity:
            state._reclaim_stage_cache()
            capacity = count

    state = _stage_reservation(monkeypatch, h, ensure)

    def new_session(**kwargs):
        assert capacity >= 2 and old.closes == 1
        events.append("mesh")
        return FakeSession(**kwargs)

    monkeypatch.setattr(module, "_new_mesh_session", new_session)
    h.stage()
    session = h.slab._mesh_session
    assert events == ["reserve", "mesh"]
    assert state._reclaim_stage_cache is None and session.closes == 0
    h.stage()
    assert events == ["reserve", "mesh", "reserve"]
    assert h.slab._mesh_session is session and len(session.calls) == 2
    assert session.closes == 0 and old.closes == 1
    assert state._reclaim_stage_cache is None


def test_interaction_growth_reclaims_mesh_but_preserves_completed_host_results(monkeypatch):
    h = Harness(monkeypatch)
    h.stage(base=True)
    primary = h.u.to_numpy(), h.j.to_numpy(), h.rate.to_numpy()
    h.u.fill(19)
    h.j.fill(23)
    h.rate.fill(29)
    state = _stage_reservation(monkeypatch, h, lambda count: None)
    original = h.slab.base.evaluate_stage
    reclaimed = []

    def grow_then_compute(**kwargs):
        session = h.slab._mesh_session
        assert len(session.calls) == 1  # Image admission has completed.
        state._reclaim_stage_cache()
        reclaimed.append(session)
        assert session.closes == 1 and h.slab._mesh_session is None
        np.testing.assert_array_equal(h.u.to_numpy(), 19)
        np.testing.assert_array_equal(h.j.to_numpy(), 23)
        np.testing.assert_array_equal(h.rate.to_numpy(), 29)
        original(**kwargs)

    monkeypatch.setattr(h.slab.base, "evaluate_stage", grow_then_compute)
    h.stage()
    image_u = np.broadcast_to([.5, -.25, .75], (2, 3))
    image_j = np.broadcast_to(np.arange(9).reshape(3, 3) / 100, (2, 3, 3))
    np.testing.assert_allclose(h.u.to_numpy()[:2], primary[0][:2] + image_u, rtol=1e-13)
    np.testing.assert_allclose(h.j.to_numpy()[:2], primary[1][:2] + image_j, rtol=1e-13)
    expected_rate = primary[2] + np.einsum("nji,nj->ni", image_j, h.g.to_numpy())
    np.testing.assert_allclose(h.rate.to_numpy(), expected_rate, rtol=1e-13)
    np.testing.assert_array_equal(h.u.to_numpy()[2:], 19)
    np.testing.assert_array_equal(h.j.to_numpy()[2:], 23)
    assert len(reclaimed) == 1 and state._reclaim_stage_cache is None
    assert h.slab.last_tail["contract"] == "gaussian_interval_remainder_v1"


def test_interaction_growth_cleanup_failure_precedes_all_output_publication(monkeypatch):
    h = Harness(monkeypatch)
    state = _stage_reservation(monkeypatch, h, lambda count: None)
    failed = []

    def fail_growth(**kwargs):
        session = h.slab._mesh_session
        assert len(session.calls) == 1
        session.cleanup_uncertain = True
        failed.append(session)
        state._reclaim_stage_cache()
        pytest.fail("primary computation survived uncertain cache cleanup")

    monkeypatch.setattr(h.slab.base, "evaluate_stage", fail_growth)
    with pytest.raises(RuntimeError, match="injected uncertain cleanup"):
        h.stage()
    with pytest.raises(RuntimeError, match="cleanup remains uncertain"):
        h.stage()
    assert len(failed) == 1 and failed[0].closes == 1
    assert h.slab._mesh_session is failed[0]
    assert state._reclaim_stage_cache is None
    np.testing.assert_array_equal(h.u.to_numpy(), 19)
    np.testing.assert_array_equal(h.j.to_numpy(), 23)
    np.testing.assert_array_equal(h.rate.to_numpy(), 29)
