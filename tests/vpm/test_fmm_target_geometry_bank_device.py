"""Small real-backend qualification; launch only in a coordinated JIT window."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.fmm.device import FMMInduction
from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction
from tests.vpm._fmm_target_geometry_bank_prototype import target_geometry_bank_factory


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    ti.init(arch=ti.cpu, cpu_max_num_threads=2, offline_cache=False)
    yield
    ti.reset()


class Harness:
    def __init__(self, kernel):
        self.n = 11
        rng = np.random.default_rng(8061)
        self.x = ti.Vector.field(3, ti.f32, shape=self.n)
        self.gamma = ti.Vector.field(3, ti.f32, shape=self.n)
        self.radius = ti.field(ti.f32, shape=self.n)
        self.velocity = ti.Vector.field(3, ti.f32, shape=self.n)
        self.gradient = ti.Matrix.field(3, 3, ti.f32, shape=self.n)
        self.rate = ti.Vector.field(3, ti.f32, shape=self.n)
        self.positions = rng.uniform(-0.35, 0.35, (self.n, 3)).astype(np.float32)
        self.x.from_numpy(self.positions)
        self.gamma.from_numpy(rng.normal(0, 1e-7, (self.n, 3)).astype(np.float32))
        self.radius.from_numpy(rng.uniform(0.04, 0.13, self.n).astype(np.float32))
        physics = PhysicsBase(kernel, self.n, ti.f32, max_evaluation_points=4)
        self.slab = SlipSlabInduction(
            FMMInduction(), z_min=-0.48, z_max=0.48, tail_tolerance=1e-4, max_shells=3
        ).bind(physics)

    def run(self):
        self.slab.evaluate_stage(
            position=self.x, vortex_strength=self.gamma, core_radius=self.radius,
            count=self.n, velocity_out=self.velocity, velocity_gradient_out=self.gradient,
            vortex_strength_rate_out=self.rate,
        )
        observations = {key: self.slab.last_tail[key] for key in (
            "shell", "block_start", "relative", "velocity", "gradient", "target_batches"
        )}
        return [self.velocity.to_numpy(), self.gradient.to_numpy(), self.rate.to_numpy()], observations

    def close(self):
        self.slab.base._release_target_workspace()
        self.slab.base.workspace.destroy()


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_bank_preserves_complete_slab_fields_tail_and_mutation_between_scopes(kernel):
    h = Harness(kernel)
    try:
        for version in range(2):
            if version:
                h.positions[:, 0] += 0.17
                h.x.from_numpy(h.positions)
            baseline, baseline_tail = h.run()
            with target_geometry_bank_factory() as records:
                result, tail = h.run()
            for actual, expected in zip(result, baseline, strict=True):
                np.testing.assert_array_equal(actual, expected)
            assert tail == baseline_tail
            assert len(records) == 1
            record = records[0]
            assert record["builds"] == 3 and record["hits"] == 6
            assert record["peak_payload_bytes"] == 96 * 16
            assert record["status"] == "complete"
        with target_geometry_bank_factory(max_bytes=0) as records:
            fallback, tail = h.run()
        for actual, expected in zip(fallback, baseline, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert tail == baseline_tail and records[0]["hits"] == 0
        assert records[0]["fallbacks"] == 9
    finally:
        h.close()


def test_actual_snode_alias_and_scalar_component_alias_decline_scope():
    h = Harness("GAUSSIAN")
    try:
        # Zero sources returns before numerical work; scope admission still
        # checks actual storage identity, including a distinct scalar wrapper.
        for output in (h.x, h.x.get_scalar_field(2)):
            with target_geometry_bank_factory() as records:
                h.slab._images(h.x, h.gamma, h.radius, h.x, 0, h.n, output, h.gradient)
            assert records[0]["disabled_reason"] == "unsupported-or-aliased-scope"
            assert records[0]["peak_payload_bytes"] == 0
    finally:
        h.close()


def test_retained_real_allocation_is_fresh_for_each_logical_scope():
    h = Harness("GAUSSIAN")
    try:
        first_positions = h.positions.copy()
        first, first_tail = h.run()
        h.positions[:, 1] += 0.23
        second_positions = h.positions.copy()
        h.x.from_numpy(h.positions)
        second, second_tail = h.run()
        with target_geometry_bank_factory() as records:
            h.x.from_numpy(first_positions)
            restored, restored_tail = h.run()
            h.x.from_numpy(second_positions)
            changed, changed_tail = h.run()
        for actual, expected in zip(restored, first, strict=True):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(changed, second, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert changed_tail == second_tail and restored_tail == first_tail
        assert len(records) == 2
        assert records[0]["allocation_pool_after_scope"]["allocations"] == 1
        assert records[1]["allocation_pool_after_scope"]["allocations"] == 1
        assert records[1]["allocation_pool_after_scope"]["reuses"] == 1
        assert all(record["builds"] == 3 and record["hits"] == 6 for record in records)
        assert records[-1]["allocation_pool_at_factory_exit"]["current_payload_bytes"] == 0
        assert records[-1]["allocation_pool_at_factory_exit"]["releases"] == 1
    finally:
        h.close()
