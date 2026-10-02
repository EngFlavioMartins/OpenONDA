"""No-runtime tests of the paired sampler replay, not numerical qualification."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.vpm._profile_gaussian_sampler_reuse import (
    InitialQueryBoxOwner,
    difference,
    owner_domain_policy,
    prepare_requests,
    run_sequence,
)


def test_old_predicate_is_initial_box_not_allocated_domain_and_close_delegates():
    closed = []
    real = SimpleNamespace(close=lambda: closed.append(True), _prepared_world_images=((0., True),))
    initial = np.array([[0., 0., 0.], [1., 1., 1.]], np.float32)
    owner = InitialQueryBoxOwner(real, initial)
    initial[:] = 10
    assert owner.can_evaluate_targets(np.array([[.2, .3, .5]]))
    assert not owner.can_evaluate_targets(np.array([[.2, -1e-9, .5]]))
    assert owner._prepared_world_images == real._prepared_world_images
    owner.close()
    assert closed == [True]


@pytest.mark.parametrize("mode,wrapped", [("old_query_box", True), ("logical_domain", False)])
def test_factory_instrumentation_is_scoped_and_counts_actual_success(mode, wrapped):
    real = SimpleNamespace()
    def factory(*args, **kwargs):
        return real
    module = SimpleNamespace(_new_field_owner=factory)
    ledger = {"successful_owner_builds": 0}
    with pytest.raises(RuntimeError, match="deliberate"), owner_domain_policy(module, mode, ledger):
        owner = module._new_field_owner(None, None, None, np.zeros((1, 3)))
        assert isinstance(owner, InitialQueryBoxOwner) == wrapped
        assert ledger["successful_owner_builds"] == 1
        raise RuntimeError("deliberate")
    assert module._new_field_owner is factory


def test_failed_construction_is_not_counted_and_factory_is_restored():
    def fail(*_args, **_kwargs):
        raise RuntimeError("construction failure")
    module = SimpleNamespace(_new_field_owner=fail)
    ledger = {"successful_owner_builds": 0}
    with pytest.raises(RuntimeError, match="construction"), owner_domain_policy(module, "old_query_box", ledger):
        module._new_field_owner(None, None, None, np.zeros((1, 3)))
    assert ledger["successful_owner_builds"] == 0
    assert module._new_field_owner is fail


def test_sequence_uses_existing_source_role_gates_and_closes_final_owner():
    events = []
    module = SimpleNamespace()

    def make_owner(*args, **kwargs):
        events.append("build")
        return SimpleNamespace(can_evaluate_targets=lambda _: True,
                               close=lambda: events.append("close"))

    module._new_field_owner = make_owner

    class FakeSession:
        def __init__(self, **kwargs):
            events.append("session")
            self.owner = None

        def evaluate(self, *args, source_only):
            assert source_only is True
            events.extend(("source_check", "role_check", "query_certificate"))
            query = args[3]
            hit = self.owner is not None and self.owner.can_evaluate_targets(query)
            if not hit:
                if self.owner is not None:
                    self.owner.close()
                self.owner = module._new_field_owner(*args)
            return query.copy(), np.zeros((len(query), 3, 3)), {"field_owner_hit": hit}

        def close(self):
            if self.owner:
                self.owner.close()
            events.append("session_close")

    module.GaussianSlabFieldSession = FakeSession
    requests = [{"name": str(i), "points": np.full((2, 3), i)} for i in range(2)]
    for mode, expected_builds in (("old_query_box", 2), ("logical_domain", 1)):
        events.clear()
        record, fields = run_sequence(module, {}, None, (None, None, None), requests,
                                      mode=mode, synchronize=lambda: None)
        assert record["successful_owner_builds"] == expected_builds
        assert events.count("source_check") == events.count("role_check") == events.count("query_certificate") == 2
        assert events.count("close") == expected_builds
        assert events[0] == "session" and events[-1] == "session_close"
        assert record["complete_sequence_seconds"] >= sum(item["seconds"] for item in record["requests"])
        assert set(fields) == {"0_velocity", "0_gradient", "1_velocity", "1_gradient"}


def test_due_request_order_and_backend_coordinate_rounding_are_explicit():
    def schedule(interval):
        return SimpleNamespace(is_final_only=False,
                               is_due=lambda step, time, dt: step % interval == 0)
    first = SimpleNamespace(file_name="phase", line_points=np.array([[1/3, 0., 0.]]), schedule=schedule(1))
    second = SimpleNamespace(file_name="slice", grid_points=np.ones((2, 3), np.float32), schedule=schedule(10))
    actual = prepare_requests([first, second], step=281, time=11.24, dt=.04)
    other = prepare_requests([first, second], step=280, time=11.2, dt=.04)
    assert [item["name"] for item in actual] == ["phase"]
    assert [item["name"] for item in other] == ["phase", "slice"]
    assert actual[0]["points"].dtype == np.float32 and not actual[0]["points"].flags.writeable
    assert actual[0]["constructed_coordinates"] != actual[0]["backend_coordinates"]


def test_all_points_comparison_rejects_broadcasting_and_nonfinite_inputs():
    with pytest.raises(ValueError):
        difference(np.zeros((1, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        difference(np.full((1, 3), np.nan), np.zeros((1, 3)))
    reference = np.zeros((9, 3), np.float32)
    current = reference.copy()
    current[-1, -1] = .125
    assert difference(current, reference)["max_component"] == .125
