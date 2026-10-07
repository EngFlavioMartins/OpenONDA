"""Host allocation lifetime/validation tests; finite GPU accuracy is qualified separately."""

from concurrent.futures import ThreadPoolExecutor
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh import session


@pytest.fixture
def harness(monkeypatch):
    events, fields = [], []
    # Real error_bounds require the restoring IEEE guard even after another
    # test initializes Taichi with flush-to-zero host arithmetic.
    pytest.importorskip("source.solvers.vpm.numerics._fenv")
    monkeypatch.setattr(session, "require_round_to_nearest", lambda: None)

    # The classifier itself has independent tests. This seam isolates serial
    # session allocation lifetime; it must not be mistaken for cutoff proof evidence.
    def classify(snapshot, *args, cutoff, images):
        zmin, zmax = struct.unpack("!dd", snapshot.source_data.slab_bytes)
        return SimpleNamespace(
            omitted_distance_lower=cutoff * 0.999,
            world_images=tuple(
                (2 * k * (zmax - zmin) + (2 * zmin if odd else 0.0), odd) for k, odd in images
            ),
        )

    monkeypatch.setattr(session, "_classification", classify)

    class Field:
        cp = SimpleNamespace(asnumpy=lambda x: np.array(x, copy=True))

        def __init__(self, *arrays, **kwargs):
            assert all(item.closed for item in fields)
            self.closed, self.fail = False, False
            self.sources, self.kwargs = arrays[:3], kwargs
            points = np.concatenate((arrays[0], arrays[3]))
            self.logical_lower, self.logical_upper = points.min(axis=0), points.max(axis=0)
            self.images = None
            fields.append(self)
            events.append("create")

        def prepare(self, images):
            assert not self.closed
            self.images = images
            zmin, zmax = self.kwargs["zmin"], self.kwargs["zmax"]
            self._prepared_world_images = tuple(
                (2 * k * (zmax - zmin) + (2 * zmin if odd else 0.0), odd) for k, odd in images
            )
            events.append("prepare")

        def can_evaluate_targets(self, targets):
            assert not self.closed
            return bool(
                np.all(targets >= self.logical_lower) and np.all(targets <= self.logical_upper)
            )

        def evaluate_prepared(self, targets):
            assert not self.closed
            if self.fail:
                raise RuntimeError("injected finite-field failure")
            events.append("query")
            return np.ones((len(targets), 3)), np.ones((len(targets), 3, 3)), {}

        def close(self):
            assert not self.closed
            self.closed = True
            events.append("close")

    monkeypatch.setattr(session, "_new_field", Field)
    x = np.array([[0.0, 0.0, -0.2], [0.1, 0.3, 0.2]], np.float32)
    strength = np.array([[1e-7, -1e-7, 2e-7], [-1e-7, 2e-7, -1e-7]], np.float32)
    sigma = np.full(2, 0.04, np.float32)
    engine = session.GaussianSlabFieldSession(
        z_min=-0.5, z_max=0.5, tail_tolerance=1e-4, max_shells=129
    )
    yield engine, x, strength, sigma, fields, events
    if engine.cleanup_uncertain:
        with pytest.raises(RuntimeError, match="cleanup remains uncertain"):
            engine.close()
    else:
        engine.close()


def test_exact_snapshot_and_role_storage(harness):
    engine, x, g, sigma, fields, events = harness
    _, _, first = engine.evaluate(x, g, sigma, x)
    _, _, second = engine.evaluate(x.copy(), g.copy(), sigma.copy(), x.copy())
    assert not first["source_snapshot_hit"] and not first["field_reused"]
    assert second["source_snapshot_hit"] and second["field_reused"]
    json.dumps(second, allow_nan=False)
    assert len(fields) == 1 and len(fields[0].images) == 4 * first["shell"] + 1
    assert first["shell"] < engine.max_shells - 1
    _, _, third = engine.evaluate(x, g, sigma, x, source_only=True)
    assert third["source_snapshot_hit"] and not third["field_reused"]
    assert third["finite_images"] == 4 * third["shell"] + 2
    assert fields[1].images.count((0, False)) == 1
    assert events[-4:] == ["close", "create", "prepare", "query"]


@pytest.mark.parametrize("which", [0, 1, 2])
def test_source_mutation_replaces_before_allocation(harness, which):
    engine, x, g, sigma, fields, _ = harness
    source = [x, g, sigma]
    engine.evaluate(*source, x)
    saved = tuple(value.copy() for value in fields[0].sources)
    source[which].flat[0] += 0.000001
    _, _, record = engine.evaluate(*source, x)
    assert not record["source_snapshot_hit"] and len(fields) == 2
    assert fields[0].closed
    for expected, captured in zip(saved, fields[0].sources, strict=True):
        np.testing.assert_array_equal(expected, captured)
        assert not captured.flags.writeable


def test_image_selection_requires_both_velocity_and_gradient_bounds(harness, monkeypatch):
    engine, x, g, sigma, fields, _ = harness
    monkeypatch.setattr(
        session, "query_tail_bound",
        lambda *args, shells, **kwargs: SimpleNamespace(
            velocity_upper=2e-5, gradient_upper=1e-3 * (7 / shells)**4,
        ),
    )
    _, _, result = engine.evaluate(x, g, sigma, x)
    assert result["shell"] == 15
    assert result["velocity_tail_bound"] <= engine.tolerance
    assert result["gradient_tail_bound"] <= engine.tolerance
    assert len(fields[0].images) == 61


def test_wider_query_tail_replaces_images_and_retains_larger_valid_sum(harness, monkeypatch):
    engine, x, g, sigma, fields, _ = harness

    def bound(snapshot, lower, upper, *, shells):
        required = 15 if upper[0] > 0.05 else 3
        return SimpleNamespace(
            velocity_upper=2e-4 if shells < required else 2e-5, gradient_upper=0.0,
        )

    monkeypatch.setattr(session, "query_tail_bound", bound)
    _, _, near = engine.evaluate(x, g, sigma, x[:1])
    _, _, wider = engine.evaluate(x, g, sigma, x[1:])
    assert near["shell"] == 3 and wider["shell"] == 15
    assert wider["field_rebuild_reason"] == "image_shells_changed"
    assert fields[0].closed
    _, _, reused = engine.evaluate(x, g, sigma, x[:1])
    assert reused["shell"] == 15 and reused["field_reused"]
    assert len(fields) == 2


def test_particle_growth_and_query_enlargement(harness):
    engine, x, g, sigma, fields, _ = harness
    engine.evaluate(x, g, sigma, x)
    x2, g2, sigma2 = np.repeat(x, 2, axis=0), np.repeat(g, 2, axis=0), np.repeat(sigma, 2)
    _, _, grew = engine.evaluate(x2, g2, sigma2, x)
    assert not grew["source_snapshot_hit"]
    queries = x.copy()
    queries[:, 0] += 1.0
    _, _, widened = engine.evaluate(x2, g2, sigma2, queries)
    assert widened["source_snapshot_hit"] and not widened["field_reused"]
    assert len(fields) == 3 and all(field.closed for field in fields[:-1])


def test_disjoint_initial_query_boxes_reuse_source_wide_field(harness):
    engine, x, g, sigma, fields, events = harness
    _, _, initial = engine.evaluate(x, g, sigma, x[:1], source_only=True)
    # Outside the first query's zero-volume box, but inside the Gaussian field's
    # logical source-covered domain. No sampler order or coordinates are special.
    _, _, disjoint = engine.evaluate(x, g, sigma, x[1:], source_only=True)
    assert not initial["field_reused"] and disjoint["field_reused"]
    assert disjoint["field_rebuild_reason"] is None
    assert disjoint["query_lower"] != initial["query_lower"]
    assert len(fields) == 1 and events.count("prepare") == 1
    assert disjoint["field_build_seconds"] == disjoint["field_prepare_seconds"] == 0


def test_domain_hit_still_checks_new_query_error_bounds(harness, monkeypatch):
    engine, x, g, sigma, fields, events = harness
    engine.evaluate(x, g, sigma, x[:1], source_only=True)
    prior_queries = events.count("query")
    monkeypatch.setattr(
        session,
        "query_tail_bound",
        lambda *a, **k: SimpleNamespace(velocity_upper=2.0, gradient_upper=0.0),
    )
    with pytest.raises(RuntimeError, match="truncation exceeds"):
        engine.evaluate(x, g, sigma, x[1:], source_only=True)
    assert fields[0].closed and events.count("query") == prior_queries


def test_failed_domain_validation_revokes_field(harness):
    engine, x, g, sigma, fields, events = harness
    engine.evaluate(x, g, sigma, x[:1], source_only=True)

    def failed(_):
        raise RuntimeError("injected logical metadata failure")

    fields[0].can_evaluate_targets = failed
    prior_queries = events.count("query")
    with pytest.raises(RuntimeError, match="logical metadata"):
        engine.evaluate(x, g, sigma, x[1:], source_only=True)
    assert fields[0].closed and engine._field is None
    assert events.count("query") == prior_queries


def test_failure_revokes_field_and_publishes_nothing(harness):
    engine, x, g, sigma, fields, _ = harness
    u, j, _ = engine.evaluate(x, g, sigma, x)
    fields[0].fail = True
    with pytest.raises(RuntimeError, match="injected"):
        engine.evaluate(x, g, sigma, x)
    assert fields[0].closed and engine._field is None
    _, _, retry = engine.evaluate(x, g, sigma, x)
    assert retry["source_snapshot_hit"] and not retry["field_reused"]
    np.testing.assert_array_equal(u, np.ones_like(u))
    np.testing.assert_array_equal(j, np.ones_like(j))


@pytest.mark.parametrize("failure_type", [MemoryError, RuntimeError])
def test_evaluation_failure_survives_failed_cleanup_and_context_exit(harness, failure_type):
    engine, x, g, sigma, fields, _ = harness
    engine.evaluate(x, g, sigma, x)
    field = fields[0]
    failure = failure_type("original finite-field failure")
    cleanup_failure = RuntimeError("injected GPU drain failure")

    def fail_query(_):
        raise failure

    def fail_close():
        raise cleanup_failure

    field.evaluate_prepared, field.close = fail_query, fail_close
    with pytest.raises(failure_type, match="original finite-field failure") as captured, engine:
        engine.evaluate(x, g, sigma, x)
    assert captured.value is failure
    assert engine.closed and engine.cleanup_uncertain
    assert engine._field is None and engine._failed_field is field
    assert any("injected GPU drain failure" in note for note in failure.__notes__)
    assert any("session cleanup also failed" in note for note in failure.__notes__)
    with pytest.raises(RuntimeError, match="closed"):
        engine.evaluate(x, g, sigma, x)
    assert len(fields) == 1


def test_context_exit_without_primary_failure_propagates_cleanup_failure(harness):
    engine, x, g, sigma, fields, _ = harness
    cleanup_failure = RuntimeError("injected GPU drain failure")

    def fail_close():
        raise cleanup_failure

    with pytest.raises(RuntimeError, match="injected GPU drain failure") as captured, engine:
        engine.evaluate(x, g, sigma, x)
        fields[0].close = fail_close
    assert captured.value is cleanup_failure
    assert engine.closed and engine.cleanup_uncertain
    assert engine._failed_field is fields[0]


def test_tail_check_precedes_device_creation(harness, monkeypatch):
    engine, x, g, sigma, fields, _ = harness
    monkeypatch.setattr(
        session,
        "query_tail_bound",
        lambda *a, **k: SimpleNamespace(velocity_upper=2.0, gradient_upper=0.0),
    )
    with pytest.raises(RuntimeError, match="truncation exceeds"):
        engine.evaluate(x, g, sigma, x)
    assert not fields


def test_bad_query_does_not_publish_or_reallocate(harness):
    engine, x, g, sigma, fields, _ = harness
    engine.evaluate(x, g, sigma, x)
    with pytest.raises(ValueError, match="finite query"):
        engine.evaluate(x, g, sigma, x * np.nan)
    assert len(fields) == 1 and not fields[0].closed


def test_empty_query_and_close_release(harness):
    engine, x, g, sigma, fields, _ = harness
    engine.evaluate(x, g, sigma, x)
    u, j, evidence = engine.evaluate(x, g, sigma, np.empty((0, 3), np.float32))
    assert u.shape == (0, 3) and j.shape == (0, 3, 3) and evidence["empty_field"]
    assert fields[0].closed
    engine.close()
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.evaluate(x, g, sigma, x)


def test_changed_controls_cannot_reuse_old_fields(harness):
    engine, x, g, sigma, fields, _ = harness
    engine.evaluate(x, g, sigma, x)
    engine.settings = session.GaussianSlabSettings(mesh=session.GaussianMeshParameters(order=8))
    with pytest.raises(RuntimeError, match="controls changed"):
        engine.evaluate(x, g, sigma, x)
    assert fields[0].closed and len(fields) == 1


def test_cross_thread_call_cannot_revoke_field(harness):
    engine, x, g, sigma, fields, _ = harness
    engine.evaluate(x, g, sigma, x)
    with ThreadPoolExecutor(1) as pool:
        for future in (pool.submit(engine.evaluate, x, g, sigma, x), pool.submit(engine.close)):
            with pytest.raises(RuntimeError, match="another thread"):
                future.result()
    assert not fields[0].closed and engine._field is fields[0]


def test_constructor_failure_poison_is_not_retryable(harness, monkeypatch):
    engine, x, g, sigma, fields, _ = harness

    def failed(*args, **kwargs):
        raise MemoryError("injected uncertain constructor cleanup")

    monkeypatch.setattr(session, "_new_field", failed)
    with pytest.raises(MemoryError, match="uncertain"):
        engine.evaluate(x, g, sigma, x)
    assert engine.closed and not fields
    with pytest.raises(RuntimeError, match="closed"):
        engine.evaluate(x, g, sigma, x)


def test_confirmed_constructor_cleanup_preserves_primary_failure_and_retry(harness, monkeypatch):
    engine, x, g, sigma, fields, _ = harness
    normal_factory = session._new_field
    failure = MemoryError("injected clean memory validation")

    def failed(*args, **kwargs):
        raise session._FieldConstructionError(failure, SimpleNamespace(closed=True))

    monkeypatch.setattr(session, "_new_field", failed)
    with pytest.raises(MemoryError, match="clean memory validation") as captured:
        engine.evaluate(x, g, sigma, x)
    assert captured.value is failure
    assert not engine.closed and not engine.cleanup_uncertain and not fields
    monkeypatch.setattr(session, "_new_field", normal_factory)
    engine.evaluate(x, g, sigma, x)
    assert len(fields) == 1


def test_confirmed_constructor_cleanup_failure_retains_field_and_blocks_reset(harness, monkeypatch):
    engine, x, g, sigma, _, _ = harness
    field = SimpleNamespace(closed=True)
    failure = MemoryError("original construction allocation")

    def failed(*args, **kwargs):
        raise session._FieldConstructionError(failure, field, RuntimeError("GPU drain failed"))

    monkeypatch.setattr(session, "_new_field", failed)
    with pytest.raises(MemoryError, match="original construction allocation"):
        engine.evaluate(x, g, sigma, x)
    assert engine.closed and engine.cleanup_uncertain and engine._failed_field is field
    assert "GPU drain failed" in failure.__notes__[0]


@pytest.mark.parametrize("cleanup_failed", [False, True])
def test_real_factory_retains_failed_constructor_handle(monkeypatch, cleanup_failed):
    from source.solvers.vpm.physics.induction.gaussian_mesh import fields

    events = []

    class FailedOwner:
        def __init__(self):
            events.append(self)
            raise MemoryError("constructor failed before publication")

        def close(self):
            self.closed = not cleanup_failed
            if cleanup_failed:
                raise RuntimeError("failed drain")

    monkeypatch.setattr(fields, "GaussianImageFields", FailedOwner)
    with pytest.raises(session._FieldConstructionError) as captured:
        session._new_field()
    outcome = captured.value
    assert outcome.field is events[0] and isinstance(outcome.failure, MemoryError)
    assert (outcome.cleanup_failure is not None) == cleanup_failed


def test_close_failure_poison_is_not_retryable(harness):
    engine, x, g, sigma, fields, _ = harness
    engine.evaluate(x, g, sigma, x)

    def failed_close():
        raise RuntimeError("injected failed GPU drain")

    fields[0].close = failed_close
    with pytest.raises(RuntimeError, match="drain"):
        engine.evaluate(x, g, sigma, x, source_only=True)
    assert engine.closed and engine._failed_field is fields[0]
    with pytest.raises(RuntimeError, match="closed"):
        engine.evaluate(x, g, sigma, x)


def test_non_nearest_rejected_before_touching_field(harness, monkeypatch):
    engine, x, g, sigma, fields, _ = harness
    engine.evaluate(x, g, sigma, x)

    def reject():
        raise RuntimeError("round-to-nearest required")

    monkeypatch.setattr(session, "require_round_to_nearest", reject)
    with pytest.raises(RuntimeError, match="round-to-nearest"):
        engine.evaluate(x, g, sigma, x)
    assert engine._field is fields[0] and not fields[0].closed


def test_runtime_image_list_must_equal_error_bound_even_on_cache_hit(harness):
    engine, x, g, sigma, fields, _ = harness
    engine.evaluate(x, g, sigma, x)
    fields[0]._prepared_world_images = ((0.0, False),)
    with pytest.raises(RuntimeError, match="descriptors differ"):
        engine.evaluate(x, g, sigma, x)
    assert fields[0].closed and engine._field is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "unknown"},
        {"tail_error_method": "unchecked"},
        {"max_sources": 1_000_001},
        {"max_sources": True},
        {"max_total_bytes": 1},
        {"max_plan_bytes": 3 * 1024**3},
        {"mesh": {}},
    ],
)
def test_explicit_settings_rejects_invalid_conditions(kwargs):
    with pytest.raises((ValueError, TypeError)):
        session.GaussianSlabSettings(**kwargs)
