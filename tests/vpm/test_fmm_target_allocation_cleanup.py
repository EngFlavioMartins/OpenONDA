"""Target scratch storage under injected allocation/upload failures.

Only the production class's plain Python run phases methods are compiled from
its AST. Fake field_groups and fields replace every Taichi object, so these tests
neither import Taichi nor initialize/compile a numerical runtime.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_TARGETS = (
    Path(__file__).resolve().parents[2] / "source/solvers/vpm/physics/induction/fmm/targets.py"
)


def _resource_methods_class(namespace):
    parsed = ast.parse(_TARGETS.read_text(encoding="utf-8"), filename=str(_TARGETS))
    target = next(
        item
        for item in parsed.body
        if isinstance(item, ast.ClassDef) and item.name == "FMMTargetEvaluator"
    )
    names = {"__init__", "_initialize", "_allocate_pairs", "destroy"}
    target.body = [
        item for item in target.body if isinstance(item, ast.FunctionDef) and item.name in names
    ]
    target.decorator_list = []
    for method in target.body:
        assert not method.decorator_list, "Run events extraction must not include Taichi kernels"
    module = ast.Module(body=[target], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_TARGETS), "exec"), namespace)
    return namespace["FMMTargetEvaluator"]


@pytest.fixture
def rig():
    state = SimpleNamespace(
        events=[],
        field_groups=[],
        failure_at=None,
        cleanup_failure_at=None,
        failure=RuntimeError("injected allocation/upload failure"),
        syncs=0,
    )

    def check(event):
        state.events.append(event)
        if event == state.failure_at:
            state.failure_at = None
            raise state.failure
        if event == state.cleanup_failure_at:
            state.cleanup_failure_at = None
            raise RuntimeError("injected cleanup failure")

    class Field:
        def __init__(self, fields):
            self.fields = fields

        def from_numpy(self, _array):
            self.fields.uploads += 1
            check(f"{self.fields.kind}:upload:{self.fields.uploads}")

    class DeviceFields:
        def __init__(self, kind):
            self.kind = kind
            self.placements = self.uploads = self.destroys = 0
            self.finalized = False
            state.field_groups.append(self)
            check(f"{kind}:created")

        def _place(self, *_args, **_kwargs):
            self.placements += 1
            check(f"{self.kind}:place:{self.placements}")
            return Field(self)

        scalar = vector = matrix = _place

        def finalize(self):
            self.finalized = True
            check(f"{self.kind}:finalize")

        def destroy(self):
            self.destroys += 1
            assert self.destroys == 1, f"{self.kind} was destroyed twice"
            check(f"{self.kind}:destroy")

    def tree_factory(**_kwargs):
        check("tree:allocation")
        fields = DeviceFields("tree")
        fields.max_tree_depth_guard = 96
        return fields

    def fields_factory():
        field_groups = sum(fields.kind != "tree" for fields in state.field_groups)
        check(f"fields:allocation:{field_groups}")
        return DeviceFields("main" if field_groups == 0 else "pairs")

    def sync():
        state.syncs += 1
        check(f"sync:{state.syncs}")

    namespace = {
        "math": math,
        "np": np,
        "ti": SimpleNamespace(f32="f32", i32="i32", i64="i64", sync=sync),
        "TaichiTreecode": tree_factory,
        "_DeviceFields": fields_factory,
        "_MAX_PAIR_CAPACITY": 1_000_000,
        "_LOCAL_COUNT": 3,
        "_DERIVATIVE_ORDER": 1,
        "_DERIVATIVE_COUNT": 3,
        "_DERIVATIVE_TERMS": 2,
        "_MULTI_INDICES": ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        "_INITIAL_PAIRS_PER_TARGET": 32,
        "_M2L_BATCH_SIZE": 100,
        "_NEAR_LANES": 64,
        "_target_derivative_tables": lambda: tuple(np.zeros(1) for _ in range(5)),
    }
    state.target_class = _resource_methods_class(namespace)
    state.source = SimpleNamespace(
        kernel_name="GAUSSIAN",
        velocity_tail_cutoff=5.0,
        gradient_tail_cutoff=5.0,
        tree=SimpleNamespace(theta_sq=0.01, regularization_tail_cutoff={None: 5.0}),
    )
    return state


@pytest.mark.parametrize(
    "failure_at",
    [
        "tree:allocation",
        "fields:allocation:0",
        "main:place:1",
        "main:place:20",
        "main:finalize",
        "main:upload:1",
        "main:upload:7",
        "sync:1",
        "fields:allocation:1",
        "pairs:place:1",
        "pairs:place:17",
        "pairs:finalize",
    ],
)
def test_partial_constructor_releases_every_acquired_field(rig, failure_at):
    rig.failure_at = failure_at
    target = rig.target_class.__new__(rig.target_class)
    with pytest.raises(RuntimeError) as caught:
        target.__init__(rig.source, 3)
    assert caught.value is rig.failure
    assert all(fields.destroys == 1 for fields in rig.field_groups), rig.events
    assert target.tree is None and target._fields is None and target._pair_fields is None
    # Explicit retry of cleanup is harmless, including very early failures.
    target.destroy()
    assert all(fields.destroys == 1 for fields in rig.field_groups)


def test_successful_destroy_is_idempotent_and_does_not_own_source(rig):
    source_tree = rig.source.tree
    target = rig.target_class(rig.source, 3)
    assert [fields.kind for fields in rig.field_groups] == ["tree", "main", "pairs"]
    assert all(fields.destroys == 0 for fields in rig.field_groups)
    target.destroy()
    target.destroy()
    assert all(fields.destroys == 1 for fields in rig.field_groups)
    assert target.tree is None and target._fields is None and target._pair_fields is None
    assert rig.source.tree is source_tree


def test_constructor_cleanup_error_does_not_mask_original_upload_error(rig):
    rig.failure_at = "main:upload:1"
    rig.cleanup_failure_at = "main:destroy"
    target = rig.target_class.__new__(rig.target_class)
    with pytest.raises(RuntimeError) as caught:
        target.__init__(rig.source, 3)
    assert caught.value is rig.failure
    assert all(fields.destroys == 1 for fields in rig.field_groups), rig.events
    assert any("cleanup" in note for note in getattr(caught.value, "__notes__", ()))


def test_destroy_attempts_other_owners_if_one_release_fails(rig):
    target = rig.target_class(rig.source, 3)
    rig.cleanup_failure_at = "pairs:destroy"
    with pytest.raises(RuntimeError, match="cleanup failure"):
        target.destroy()
    assert all(fields.destroys == 1 for fields in rig.field_groups), rig.events
    target.destroy()


@pytest.mark.parametrize("kwargs", [{"max_targets": 0}, {"max_images": 0}, {"max_pairs": 0}])
def test_invalid_configuration_allocates_nothing(rig, kwargs):
    arguments = {"max_targets": 3} | kwargs
    target = rig.target_class.__new__(rig.target_class)
    with pytest.raises(ValueError):
        target.__init__(rig.source, **arguments)
    assert not rig.field_groups
    target.destroy()
