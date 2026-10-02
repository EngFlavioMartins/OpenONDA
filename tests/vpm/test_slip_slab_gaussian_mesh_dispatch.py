"""Optional mesh dispatch: real Taichi publication with a bounded fake session.

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
        j = np.broadcast_to(np.arange(9).reshape(3, 3)/100, (len(targets), 3, 3)).astype(dtype).copy()
        if not len(x):
            u.fill(0)
            j.fill(0)
        return u, j, {"shell": 2, "velocity_tail_bound": 1e-7,
                      "gradient_tail_bound": 2e-7, "seconds": 0.0}

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
            DirectInduction(stretching_scheme=scheme), z_min=-0.5, z_max=0.5,
            gaussian_mesh_policy=GaussianSlabPolicy(),
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
            position=self.x, vortex_strength=self.g, core_radius=self.r, count=2,
            velocity_out=self.u, vortex_strength_rate_out=self.rate,
            velocity_gradient_out=self.j if gradient else None, strength_rate_enabled=rate,
        )

    def targets(self, *, sources=2, velocity=True, gradient=True, free=True, count=3):
        self.slab.evaluate_targets(
            target_position=self.q, source_position=self.x, source_vortex_strength=self.g,
            source_core_radius=self.r, target_velocity=self.u if velocity else None,
            target_velocity_gradient=self.j if gradient else None, target_count=count,
            source_count=sources, include_freestream=free, background_velocity=self.background,
        )


@pytest.mark.parametrize("scheme", ["DIRECT", "TRANSPOSED", "MIXED"])
@pytest.mark.parametrize("gradient,rate", [(True, True), (False, True), (True, False), (False, False)])
def test_stage_preserves_primary_and_adds_correct_stretching(monkeypatch, scheme, gradient, rate):
    h = Harness(monkeypatch, scheme)
    h.stage(gradient=gradient, rate=rate, base=True)
    primary = (h.u.to_numpy(), h.j.to_numpy(), h.rate.to_numpy())
    h.stage(gradient=gradient, rate=rate)
    u, j, _ = h.slab._mesh_session.evaluate(
        h.x.to_numpy(), h.g.to_numpy(), h.r.to_numpy(), h.x.to_numpy(), source_only=False)
    np.testing.assert_allclose(h.u.to_numpy()[:2], primary[0][:2]+u, rtol=1e-13, atol=1e-13)
    if gradient:
        np.testing.assert_allclose(h.j.to_numpy()[:2], primary[1][:2]+j, rtol=1e-13, atol=1e-13)
    else:
        np.testing.assert_array_equal(h.j.to_numpy(), primary[1])
    contraction = j if scheme == "DIRECT" else j.transpose(0, 2, 1)
    if scheme == "MIXED":
        contraction = (j+j.transpose(0, 2, 1))/2
    expected = primary[2] + np.einsum("nij,nj->ni", contraction, h.g.to_numpy()) if rate else 0
    np.testing.assert_allclose(h.rate.to_numpy(), expected, rtol=1e-13, atol=1e-13)
    assert not h.slab._mesh_session.calls[0][-1]
    assert h.slab.last_tail["contract"] == "gaussian_interval_remainder_v1"
    assert h.slab._block_velocity.shape == (1,)


@pytest.mark.parametrize("velocity,gradient,free", [(True, True, True), (True, False, False), (False, True, True)])
def test_target_role_is_whole_source_only_and_freestream_once(monkeypatch, velocity, gradient, free):
    h = Harness(monkeypatch)
    monkeypatch.setattr(h.slab.base, "evaluate_targets", lambda **_: pytest.fail("primary double count"))
    h.targets(velocity=velocity, gradient=gradient, free=free)
    if velocity:
        expected = np.broadcast_to([0.5, -0.25, 0.75], (3, 3)) + ([1, 2, 3] if free else 0)
        np.testing.assert_array_equal(h.u.to_numpy(), expected)
    else:
        np.testing.assert_array_equal(h.u.to_numpy(), 19)
    if gradient:
        np.testing.assert_array_equal(h.j.to_numpy(), np.broadcast_to(np.arange(9).reshape(3, 3)/100, (3, 3, 3)))
    else:
        np.testing.assert_array_equal(h.j.to_numpy(), 23)
    assert h.slab._mesh_session.calls[0][-1]


def test_tail_failure_precedes_primary_and_preserves_outputs(monkeypatch):
    h = Harness(monkeypatch)
    h.slab._mesh_session = FakeSession(dtype="float64")
    h.slab._mesh_session.failure = RuntimeError("tail admission rejected")
    monkeypatch.setattr(h.slab.base, "evaluate_stage", lambda **_: pytest.fail("early primary publication"))
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
    slab = module.SlipSlabInduction(DirectInduction(), z_min=-1, z_max=1,
                                   gaussian_mesh_policy=GaussianSlabPolicy(backend="cupy_cuda"))
    monkeypatch.setattr(slab.base, "bind", lambda *_, **__: pytest.fail("unsupported base bind"))
    monkeypatch.setattr(module, "_mesh_runtime_is_cuda", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        slab.bind(physics)
    monkeypatch.setattr(module, "_mesh_runtime_is_cuda", lambda: True)
    with pytest.raises(ValueError, match="GAUSSIAN"):
        slab.bind(physics, kernel=make_vortex_kernel("WINCKELMANS"))
    assert slab.physics is None and slab._mesh_session is None


def test_policy_build_fingerprint_and_legacy_reuse_decline():
    legacy = module.SlipSlabInduction(DirectInduction(), z_min=-1, z_max=1)
    assert "gaussian_mesh_policy" not in _canonical_value(legacy)
    assert _canonical_value(legacy) == _canonical_value(legacy.build())
    policy = GaussianSlabPolicy()
    slab = module.SlipSlabInduction(DirectInduction(), z_min=-1, z_max=1, gaussian_mesh_policy=policy)
    copied = slab.build()
    assert copied.base is not slab.base and copied.gaussian_mesh_policy == policy
    identity = _canonical_value(slab)
    assert identity["gaussian_mesh_policy"]["tail_contract"] == policy.tail_contract
    copied.gaussian_mesh_policy = replace(policy, max_sources=999)
    assert identity != _canonical_value(copied)
    assert StandardFMMReuseContract(slab)() is None


def test_mesh_shell_ceiling_rejected_before_binding():
    with pytest.raises(ValueError, match="1024-shell capacity"):
        module.SlipSlabInduction(DirectInduction(), z_min=-1, z_max=1, max_shells=1025,
                                gaussian_mesh_policy=GaussianSlabPolicy())
    # No new restriction is silently imposed on the unchanged legacy operator.
    assert module.SlipSlabInduction(DirectInduction(), z_min=-1, z_max=1, max_shells=1025).max_shells == 1025


@pytest.mark.parametrize("failure", [False, True])
def test_solver_closes_only_owned_mesh_and_never_resets_after_uncertainty(monkeypatch, failure):
    slab = module.SlipSlabInduction(DirectInduction(), z_min=-1, z_max=1)
    session = slab._mesh_session = FakeSession()
    session.cleanup_uncertain = failure
    events = []
    monkeypatch.setattr(slab.base, "close", lambda: pytest.fail("shared base closed"), raising=False)
    owner = SimpleNamespace(_closed=False, induction=slab, _run_started=True,
                            _initial_conditions_built=True, _backend_claimed=True,
                            stage_rhs=SimpleNamespace(close=lambda: events.append("cache")),
                            _particle_snapshot_buffers={})
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
    owner = SimpleNamespace(_closed=False, induction=SimpleNamespace(
        close_mesh_session=lambda: pytest.fail("custom shared backend closed")),
        _run_started=True, _initial_conditions_built=True, _backend_claimed=True,
        _particle_snapshot_buffers={})
    monkeypatch.setattr(solver_module, "reset_taichi_backend", lambda **_: None)
    solver_module.VPMSolver.close(owner)
    assert owner._closed
