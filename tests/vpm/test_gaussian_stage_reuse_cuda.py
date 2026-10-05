"""Opt-in real CUDA integration; run only in the serialized GPU window.

OPENONDA_FENV_EXTENSION names the already built optional guard. No extension
is compiled, dependency installed, or simulator advanced by this test.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.base import StageRates, StageState
from source.solvers.vpm.physics.induction.fmm.device import FMMInduction
from source.solvers.vpm.physics.induction.gaussian_mesh.session import GaussianSlabSettings
from source.solvers.vpm.physics.induction.reuse_backends import FMMReuseConditions
from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction
from source.solvers.vpm.physics.stage_rhs import StageRHS

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.qualification,
    pytest.mark.skipif(
        not os.environ.get("OPENONDA_FENV_EXTENSION"),
        reason="explicit serialized CUDA/FENV qualification required",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def cuda_runtime():
    path = os.environ.get("OPENONDA_FENV_EXTENSION")
    if not path:
        pytest.skip("optional compiled FENV guard required")
    name = "source.solvers.vpm.numerics._fenv"
    spec = importlib.util.spec_from_file_location(name, path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    sys.modules[name] = bridge
    owned = ti.lang.impl.get_runtime().prog is None
    if owned:
        ti.init(arch=ti.cuda, default_fp=ti.f32, offline_cache=True, cpu_max_num_threads=2)
    assert ti.lang.impl.current_cfg().arch == ti.cuda
    yield
    if owned:
        ti.reset()


class Fields:
    def __init__(self, scheme):
        self.physics = PhysicsBase("GAUSSIAN", 3, ti.f32, max_evaluation_points=8)
        self.slab = SlipSlabInduction(
            FMMInduction(stretching_scheme=scheme),
            z_min=-0.5,
            z_max=0.5,
            max_shells=3,
            gaussian_mesh_settings=GaussianSlabSettings(max_sources=3, max_query_points=8),
        ).bind(self.physics)
        self.x, self.g, self.u, self.rate = [ti.Vector.field(3, ti.f32, shape=3) for _ in range(4)]
        self.r = ti.field(ti.f32, shape=3)
        self.j = ti.Matrix.field(3, 3, ti.f32, shape=3)
        self.x.from_numpy(
            np.array([[0.0, 0.0, 0.0], [0.05, -0.03, 0.02], [-0.02, 0.04, -0.05]], np.float32)
        )
        self.g.from_numpy(np.array([[1, 2, 3], [-1, 2, 1], [2, -1, 2]], np.float32) * 1e-6)
        self.r.fill(0.04)

    def run(self, rhs, time=0.0, *, gradient=True, rate=True):
        rhs.evaluate(
            StageState(self.x, self.g, self.r, 3, time=time),
            time,
            StageRates(self.u, self.rate, self.j if gradient else None, strength_rate_enabled=rate),
        )
        return tuple(field.to_numpy() for field in (self.u, self.j, self.rate))

    def close(self):
        self.slab.close_mesh_session()
        self.slab.base.close()


@pytest.mark.parametrize("scheme", ["DIRECT", "TRANSPOSED", "MIXED"])
def test_real_complete_hit_subsets_role_change_guards_and_providers(scheme):
    h = Fields(scheme)
    events = []

    class Provider:
        def add_stage_rates(self, state, time, rates):
            events.append(("provider", time))
            rates.velocity.from_numpy(rates.velocity.to_numpy() + np.float32(time))
            rates.vortex_strength_rate.fill(41)

    rhs = StageRHS(h.slab, (Provider(),), strength_enabled=False)

    def guard(state):
        events.append(("guard", state.time))
        if state.time == 4:
            h.g[0] = h.g[0] * 2

    rhs.position_guard = guard
    try:
        assert FMMReuseConditions(h.slab)() is None
        first = h.run(rhs)
        # FMM scratch can grow during a first miss. Warm without assuming
        # that optional cold capacity changes are semantic cache hits.
        h.run(rhs)
        before = rhs.induction_reuse_statistics
        h.j.fill(97)
        second = h.run(rhs, time=1.0, gradient=False, rate=False)
        np.testing.assert_array_equal(second[0], first[0] + np.float32(1))
        np.testing.assert_array_equal(second[1], 97)
        np.testing.assert_array_equal(second[2], 0)
        assert rhs.induction_reuse_statistics.hits == before.hits + 1
        assert h.slab.last_tail["exact_stage_reuse"]
        assert "finite_evaluation" not in h.slab.last_tail["mesh"]
        # Real coherent-query role change, not a fake identity edit.
        other_gamma = ti.Vector.field(3, ti.f32, shape=3)
        other_gamma.from_numpy(h.g.to_numpy() * np.float32(1.25))
        h.slab.evaluate_targets(
            target_position=h.x,
            source_position=h.x,
            source_vortex_strength=other_gamma,
            source_core_radius=h.r,
            target_velocity=h.u,
            target_velocity_gradient=h.j,
            target_count=3,
            source_count=3,
            include_freestream=False,
            background_velocity=h.physics._zero_velocity,
        )
        query_field = h.slab._mesh_session._field
        before = rhs.induction_reuse_statistics
        third = h.run(rhs, time=2.0)
        np.testing.assert_array_equal(third[0], first[0] + np.float32(2))
        assert rhs.induction_reuse_statistics.hits == before.hits + 1
        assert h.slab._mesh_session._field is query_field
        assert h.slab._mesh_session._role is True
        assert h.slab.last_actual_mesh_work["mesh"]["source_only_primary"]
        # A closed private field must not be masked even though its fields
        # are not dependencies of the cached particle answer.
        h.slab._mesh_session.closed = True
        try:
            with pytest.raises(RuntimeError, match="closed"):
                h.run(rhs, time=3.0)
            assert not rhs.induction_reuse._valid
        finally:
            h.slab._mesh_session.closed = False
        # Source mutation in the external guard must precede the exact check.
        before = rhs.induction_reuse_statistics
        h.run(rhs, time=4.0)
        assert rhs.induction_reuse_statistics.misses == before.misses + 1
        assert events == (
            [(who, time) for time in (0.0, 0.0, 1.0, 2.0) for who in ("guard", "provider")]
            + [("guard", 3.0), ("guard", 4.0), ("provider", 4.0)]
        )
        # Every-request count/alias guard precedes an otherwise exact hit.
        h.u.fill(59)
        with pytest.raises(ValueError, match="alias"):
            rhs.induction_reuse.evaluate_stage(
                position=h.x,
                vortex_strength=h.g,
                core_radius=h.r,
                count=3,
                velocity_out=h.x,
                vortex_strength_rate_out=h.rate,
                velocity_gradient_out=h.j,
            )
        np.testing.assert_array_equal(h.u.to_numpy(), 59)
        assert not rhs.induction_reuse._valid
        # Public int(flag)==1 behavior is preserved by request-level bypass,
        # not replaced by bool(flag) publication from the cache.
        rhs.induction_reuse.evaluate_stage(
            position=h.x,
            vortex_strength=h.g,
            core_radius=h.r,
            count=3,
            velocity_out=h.u,
            vortex_strength_rate_out=h.rate,
            velocity_gradient_out=h.j,
            strength_rate_enabled=2,
        )
        np.testing.assert_array_equal(h.rate.to_numpy(), 0)
        assert rhs.induction_reuse_statistics.bypasses >= 1
    finally:
        rhs.close()
        h.close()
