"""Tiny real CPU-backend qualification for the unwired reuse capability."""

from dataclasses import replace

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.base import StageRates, StageState
from source.solvers.vpm.physics.induction.fmm.device import FMMInduction
from source.solvers.vpm.physics.induction.reuse import ExactContentInductionReuse
from source.solvers.vpm.physics.induction.reuse_backends import StandardFMMReuseContract
from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction
from source.solvers.vpm.physics.stage_rhs import StageRHS


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    ti.init(arch=ti.cpu, cpu_max_num_threads=2, offline_cache=False)
    yield
    ti.reset()


class Harness:
    def __init__(self, kernel="GAUSSIAN", scheme="TRANSPOSED", slab=False):
        self.count = 3
        self.physics = PhysicsBase(
            particle_kernel=kernel,
            max_n_particles=3,
            accumulator_dtype=ti.f32,
            max_evaluation_points=16,
        )
        base = FMMInduction(stretching_scheme=scheme)
        self.backend = (
            SlipSlabInduction(base, z_min=-1.0, z_max=1.0, tail_tolerance=1e-4, max_shells=3)
            if slab
            else base
        ).bind(self.physics)
        self.base = base
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=3)
        self.strength = ti.Vector.field(3, dtype=ti.f32, shape=3)
        self.radius = ti.field(dtype=ti.f32, shape=3)
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=3)
        self.gradient = ti.Matrix.field(3, 3, dtype=ti.f32, shape=3)
        self.rate = ti.Vector.field(3, dtype=ti.f32, shape=3)
        self.xyz = np.array([[0.1, -0.2, -0.3], [-0.2, 0.2, 0.1], [0.3, 0.1, 0.2]], np.float32)
        self.gamma = np.array([[0.2, -0.5, 0.1], [-0.1, 0.2, -0.2], [0.2, 0.4, 0.3]], np.float32)
        if slab:
            # A tiny but nonzero source keeps this operator qualification to
            # three shells. Production tail settings are not modified.
            self.gamma *= 1e-7
        self.position.from_numpy(self.xyz)
        self.strength.from_numpy(self.gamma)
        self.radius.from_numpy(np.array([0.08, 0.15, 0.11], np.float32))
        self.run(self.backend)  # Finish normal lazy source/target allocation.
        self.provider = StandardFMMReuseContract(self.backend)
        assert self.provider() is not None
        self.cache = ExactContentInductionReuse(
            self.backend, max_particles=3, contract_provider=self.provider
        )

    def run(self, evaluator=None, *, gradient=True, rate=True, time=0.0):
        if evaluator is None:
            evaluator = self.cache
        self.gradient.fill(37)
        evaluator.evaluate_stage(
            position=self.position,
            vortex_strength=self.strength,
            core_radius=self.radius,
            count=self.count,
            velocity_out=self.velocity,
            vortex_strength_rate_out=self.rate,
            velocity_gradient_out=self.gradient if gradient else None,
            strength_rate_enabled=rate,
            stage_time=time,
        )
        return self.velocity.to_numpy(), self.gradient.to_numpy(), self.rate.to_numpy()

    def close(self):
        self.cache.close()
        self.base._release_target_workspace()
        self.base.workspace.destroy()


@pytest.mark.parametrize(
    "kernel,scheme",
    [
        ("GAUSSIAN", "DIRECT"),
        ("WINCKELMANS", "TRANSPOSED"),
        ("HIGH_ORDER_GAUSSIAN", "MIXED"),
        ("SUPER_GAUSSIAN", "DIRECT"),
    ],
)
def test_real_fmm_output_subsets_are_cache_equivalent(kernel, scheme):
    h = Harness(kernel, scheme)
    try:
        for gradient, rate in [(False, False), (False, True), (True, False), (True, True)]:
            expected = h.run(h.backend, gradient=gradient, rate=rate)
            actual = h.run(gradient=gradient, rate=rate, time=31.0)
            for left, right in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(left, right)
        assert h.cache.statistics.misses == 1
        assert h.cache.statistics.hits == 3
    finally:
        h.close()


@pytest.mark.parametrize("kernel", ["WINCKELMANS"])
def test_real_slab_subsets_tail_and_internal_span_guard(kernel, monkeypatch):
    h = Harness(kernel, slab=True)
    try:
        for gradient, rate in [(False, False), (False, True), (True, False), (True, True)]:
            expected = h.run(h.backend, gradient=gradient, rate=rate)
            actual = h.run(gradient=gradient, rate=rate, time=52.0)
            for left, right in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(left, right)
        assert h.cache.statistics.hits == 3
        physical_tail = {
            key: h.backend.last_tail[key]
            for key in ("shell", "block_start", "relative", "velocity", "gradient")
        }
        h.backend.last_tail.update(relative=99, shell=99, target_batches=12345, seconds=12.5)
        stage_calls = h.base.diagnostics.stage_evaluations
        h.run()
        assert {key: h.backend.last_tail[key] for key in physical_tail} == physical_tail
        assert h.backend.last_tail["target_batches"] == 12345
        assert h.backend.last_tail["seconds"] == 12.5
        assert h.base.diagnostics.stage_evaluations == stage_calls
        original_key = h.provider().operator_key
        for name, value in (
            ("z_min", -2.0),
            ("z_max", 2.0),
            ("tail_tolerance", 1e-5),
            ("max_shells", 5),
            ("velocity_scale", 2.0),
            ("gradient_scale", 2.0),
            ("stretching_scheme", "MIXED"),
        ):
            with monkeypatch.context() as patch:
                patch.setattr(h.backend, name, value)
                assert h.provider().operator_key != original_key
        h.physics._zero_velocity[None] = [0.1, 0, 0]
        assert h.provider().operator_key != original_key
        h.physics._zero_velocity[None] = [0, 0, 0]
        h.backend._z_min_field[None] = 0.5
        h.velocity.fill(19)
        with pytest.raises(RuntimeError, match="escaped"):
            h.run()
        np.testing.assert_array_equal(h.velocity.to_numpy(), 19)
        assert not h.cache._valid
    finally:
        h.close()


def test_exact_operator_dependencies_and_unknown_bindings_decline(monkeypatch):
    h = Harness()
    try:
        original = h.provider().operator_key
        owners = [
            (h.base, "_stretching_mode", 0),
            (h.base, "stretching_scheme", "MIXED"),
            (h.base.workspace, "gradient_tail_cutoff", 123.0),
            (h.base.workspace.tree, "theta_sq", 0.005),
            (h.physics, "max_evaluation_points", 7),
        ]
        for owner, name, value in owners:
            with monkeypatch.context() as patch:
                patch.setattr(owner, name, value)
                assert h.provider().operator_key != original
        field = h.base.workspace.tree.regularization_tail_cutoff
        old = field[None]
        field[None] = old + 1
        assert h.provider().operator_key != original
        field[None] = old
        table = h.base.workspace._derivative_coefficient
        old = table[0, 0]
        table[0, 0] = old + 1
        assert h.provider().operator_key != original
        table[0, 0] = old
        with monkeypatch.context() as patch:
            patch.setattr(h.physics, "_copy_vec3", lambda *args: None)
            assert h.provider() is None
        with monkeypatch.context() as patch:
            patch.setattr(h.base, "_radial_factors", lambda *args: 0)
            assert h.provider() is None
        for name in (
            "_gaussian_q",
            "_gaussian_zeta",
            "_gaussian_radial_factors",
            "_winckelmans_radial_factors",
        ):
            with monkeypatch.context() as patch:
                patch.setattr(h.base.workspace.tree, name, lambda *args: 0)
                assert h.provider() is None
        with monkeypatch.context() as patch:
            patch.setattr(h.base, "stretching_scheme", [])
            assert h.provider() is None
        with monkeypatch.context() as patch:
            patch.setattr(h.provider, "_runtime_program", object())
            assert h.provider() is None
        with monkeypatch.context() as patch:
            patch.setattr(
                h.base, "kernel", replace(make_vortex_kernel("GAUSSIAN"), q_function=lambda x: x)
            )
            assert h.provider() is None
        with monkeypatch.context() as patch:
            patch.setattr(h.base, "evaluate_stage", lambda **kwargs: None)
            assert h.provider() is None
        h.run()
        with h.base.fixed_source_targets(h.position, h.strength, h.radius, h.count):
            assert h.provider() is None
            with pytest.raises(RuntimeError, match="immutable-source"):
                h.run()
        h.base.workspace.destroy()
        assert h.provider() is None
    finally:
        h.close()


def test_slab_target_ancestry_capacity_and_layout_are_part_of_contract(monkeypatch):
    h = Harness("WINCKELMANS", slab=True)
    try:
        target = h.base._target_workspace
        assert target is not None
        original = h.provider().operator_key
        with monkeypatch.context() as patch:
            patch.setattr(target, "target_path_capacity", target.target_path_capacity + 1)
            assert h.provider().operator_key != original
        # Contract checks only; never evaluate a deliberately invalid field
        # binding.  A same-owner alias still must not preserve the cache key.
        for name, substitute in (
            ("target_path", target.target_path_length),
            ("target_path_length", target.leaf_nodes),
            ("target_path_error", target.leaf_count),
        ):
            with monkeypatch.context() as patch:
                patch.setattr(target, name, substitute)
                assert h.provider().operator_key != original
        assert h.provider().operator_key == original
    finally:
        h.close()


def test_unsupported_classes_and_rebind_identity():
    class CustomFMM(FMMInduction):
        pass

    assert StandardFMMReuseContract(object())() is None
    assert StandardFMMReuseContract(CustomFMM())() is None
    assert StandardFMMReuseContract(FMMInduction())() is None
    h = Harness()
    try:
        key = h.provider().operator_key
        h.base.bind(h.physics)
        assert h.provider().operator_key != key
    finally:
        h.close()


@pytest.mark.parametrize("kernel", ["HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN"])
def test_uncertified_slab_direct_target_bindings_decline(kernel):
    physics = PhysicsBase(particle_kernel=kernel, max_n_particles=1, max_evaluation_points=1)
    slab = SlipSlabInduction(FMMInduction(), z_min=-1, z_max=1).bind(physics)
    try:
        assert StandardFMMReuseContract(slab)() is None
    finally:
        slab.base.workspace.destroy()


def test_hit_invalidates_stale_target_tree_and_restores_only_rate_observations():
    h = Harness()
    try:
        expected = h.run()
        rate_defect = h.base.diagnostics.last_uncorrected_rate_defect
        h.strength.from_numpy(h.gamma * 2)
        with h.base.fixed_source_targets(h.position, h.strength, h.radius, h.count):
            pass
        h.strength.from_numpy(h.gamma)
        h.base.diagnostics.stage_evaluations += 100
        h.base.diagnostics.last_tree_build_seconds = 876.0
        h.base.diagnostics.last_uncorrected_rate_defect = 123.0
        actual_calls = h.base.diagnostics.stage_evaluations
        actual = h.run()
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(left, right)
        assert h.cache.statistics.hits == 1
        assert h.base.diagnostics.last_uncorrected_rate_defect == rate_defect
        assert h.base.diagnostics.stage_evaluations == actual_calls
        assert h.base.diagnostics.last_tree_build_seconds == 876.0
        assert h.base._last_tree_key is None
        assert h.base._source_moments_ready is False
        with h.base.fixed_source_targets(
            h.position, h.strength, h.radius, h.count, reuse_current_tree=True
        ):
            np.testing.assert_array_equal(
                h.base.workspace.tree.vortex_strength.to_numpy()[:3], h.gamma
            )
    finally:
        h.close()


def test_actual_stage_rhs_guard_precedes_check_and_providers_run_on_hits():
    h = Harness()
    events = []

    class Provider:
        def add_stage_rates(self, state, stage_time, rates):
            events.append(("provider", stage_time, h.cache.statistics.hits))
            rates.velocity.from_numpy(rates.velocity.to_numpy() + stage_time)
            rates.vortex_strength_rate.fill(17)

    def guard(state):
        events.append(("guard", state.time, h.cache.statistics.hits))
        if state.time == 3:
            h.position[0] = h.position[0] + ti.Vector([0.01, 0, 0])

    rhs = StageRHS(h.cache, (Provider(),), strength_enabled=False)
    rhs.position_guard = guard
    rates = StageRates(h.velocity, h.rate, h.gradient, strength_rate_enabled=False)
    try:
        baseline = h.run(h.backend)[0]
        for time in (1, 2, 3):
            state = StageState(h.position, h.strength, h.radius, h.count, time=float(time))
            rhs.evaluate(state, float(time), rates)
            if time < 3:
                np.testing.assert_array_equal(h.velocity.to_numpy(), baseline + time)
            np.testing.assert_array_equal(h.rate.to_numpy(), 0)
        assert h.cache.statistics.hits == 1
        assert h.cache.statistics.misses == 2
        assert events == [
            ("guard", 1.0, 0),
            ("provider", 1.0, 0),
            ("guard", 2.0, 0),
            ("provider", 2.0, 1),
            ("guard", 3.0, 1),
            ("provider", 3.0, 1),
        ]
    finally:
        h.close()
