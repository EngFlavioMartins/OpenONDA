"""Host conditions qualification; no CuPy, GPU, or physical solver is run."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.induction.gaussian_mesh import reuse_conditions as module
from source.solvers.vpm.physics.induction.gaussian_mesh.session import GaussianSlabSettings
from source.solvers.vpm.physics.induction.reuse import InductionReuseConditions, _typed_value_key


class Scalar:
    dtype = ti.f32
    shape = ()

    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return self.value

    def __setitem__(self, index, value):
        self.value = value


@pytest.fixture
def host_conditions(monkeypatch):
    events = []
    primary = InductionReuseConditions(
        ("qualified fake primary",),
        True,
        True,
        True,
        True,
        lambda: ("primary proof",),
        lambda state: events.append(("restore primary", state)),
    )
    slab = SimpleNamespace(
        physics=SimpleNamespace(accumulator_dtype=ti.f32, _zero_velocity=Scalar((0.0, 0.0, 0.0))),
        kernel=SimpleNamespace(name="GAUSSIAN"),
        gaussian_mesh_settings=GaussianSlabSettings(),
        z_min=-0.5,
        z_max=0.5,
        tail_tolerance=1e-4,
        max_shells=129,
        velocity_scale=1.0,
        gradient_scale=2.0,
        stretching_scheme="TRANSPOSED",
        _z_min_field=Scalar(-0.5),
        _z_max_field=Scalar(0.5),
        _span_excess=Scalar(0.0),
        _mesh_session=None,
        _uses_mesh=lambda: events.append("binding") or True,
        _mesh_fields=lambda *args: events.append(("fields", args)),
        _mesh_count=lambda count, cap: events.append(("capacity", count, cap)),
        last_tail={
            "shell": 128,
            "velocity": 1e-6,
            "gradient": 2e-6,
            "relative": 1e-6,
            "tail_error_method": "gaussian_interval_remainder_v1",
            "seconds": 11.0,
            "mesh": {
                "source_sha256": "source",
                "source_only_primary": False,
                "velocity_tail_bound": 1e-6,
                "gradient_tail_bound": 2e-6,
                "finite_evaluation": {"passes": 45},
                "seconds": 11.0,
            },
        },
    )
    provider = module.GaussianSlabReuseConditions.__new__(module.GaussianSlabReuseConditions)
    provider.backend = slab
    provider._supported = lambda: True
    provider._base_conditions = lambda: primary
    monkeypatch.setattr(module, "_check_rounding", lambda: events.append("rounding"))
    return provider, slab, events


def _request(count=3):
    return {
        "position": object(),
        "vortex_strength": object(),
        "core_radius": object(),
        "count": count,
        "velocity_out": object(),
        "vortex_strength_rate_out": object(),
        "velocity_gradient_out": None,
        "strength_rate_enabled": False,
        "stage_time": 31.0,
    }


def test_request_validation_runs_public_layout_caps_and_rounding(host_conditions):
    provider, _, events = host_conditions
    provider().request_check(**_request())
    assert events[0] == "binding"
    assert events[1][0] == "fields"
    assert events[2] == ("capacity", 3, 1_000_000)
    assert events[3] == "rounding"
    events.clear()
    provider().request_check(**_request(0))
    assert "rounding" not in events


@pytest.mark.parametrize("name", ["velocity_out", "vortex_strength_rate_out"])
def test_missing_mandatory_output_fails_before_any_publication(host_conditions, name):
    provider, _, events = host_conditions
    args = _request()
    args[name] = None
    with pytest.raises(ValueError, match="require velocity"):
        provider().request_check(**args)
    assert events == ["binding"]


def test_controls_and_actual_device_planes_change_key_but_role_does_not(host_conditions):
    provider, slab, _ = host_conditions
    original = _typed_value_key(provider().operator_key)
    # Even an unrelated coherent-query field cannot change mathematical
    # dependence of the independently copied particle result.
    slab._mesh_session = SimpleNamespace(_role=True, _field=object())
    assert _typed_value_key(provider().operator_key) == original
    slab._z_min_field[None] = -0.4
    assert _typed_value_key(provider().operator_key) != original
    slab._z_min_field[None] = -0.5
    slab.gaussian_mesh_settings = replace(slab.gaussian_mesh_settings, max_sources=17)
    assert _typed_value_key(provider().operator_key) != original


def test_only_mathematical_observations_are_restored(host_conditions):
    provider, slab, events = host_conditions
    conditions = provider()
    saved = conditions.capture_diagnostics()
    slab.last_tail = {"shell": 99, "seconds": 23.0, "mesh": {"finite_evaluation": {"passes": 79}}}
    current = slab.last_tail.copy()
    field = object()
    slab._mesh_session = SimpleNamespace(_role=True, _field=field, calls=17)
    conditions.restore_diagnostics(saved)
    assert slab.last_tail["shell"] == 128
    assert slab.last_tail["exact_stage_reuse"]
    assert slab.last_tail["seconds"] == 0
    assert "finite_evaluation" not in slab.last_tail["mesh"]
    assert slab.last_actual_mesh_work == current
    assert slab._mesh_session._field is field and slab._mesh_session.calls == 17
    assert events == [("restore primary", ("primary proof",))]
    conditions.restore_diagnostics(saved)
    assert slab.last_actual_mesh_work == current


@pytest.mark.parametrize(
    "fault", ["closed", "uncertain", "thread", "controls", "plans", "correction", "stream"]
)
def test_existing_session_run_events_never_hidden(host_conditions, fault):
    provider, slab, events = host_conditions

    def fail():
        raise RuntimeError("injected field thread or stream")

    device = SimpleNamespace(
        check_context=fail if fault == "stream" else lambda: events.append("device")
    )
    field = SimpleNamespace(
        _check_context=lambda: device.check_context(),
        _plans=SimpleNamespace(closed=fault == "plans", memory_pool=device),
        _correction=SimpleNamespace(closed=fault == "correction", _memory_pool=device),
        _prepared_images=((0, True),),
        _compact_fields=object(),
    )
    config = (
        slab.z_min,
        slab.z_max,
        slab.tail_tolerance,
        slab.max_shells,
        slab.velocity_scale,
        slab.gradient_scale,
        "float32",
        slab.gaussian_mesh_settings,
        "cupy_cuda",
    )
    slab._mesh_session = SimpleNamespace(
        _check_thread=fail if fault == "thread" else lambda: None,
        closed=fault == "closed",
        cleanup_uncertain=fault == "uncertain",
        execution_backend="cupy_cuda",
        _configuration=config,
        _configuration_key=lambda: () if fault == "controls" else config,
        _field=field,
    )
    with pytest.raises(RuntimeError):
        provider().request_check(**_request())


def test_standard_binding_changes_decline_without_validating_custom_code(monkeypatch):
    assert module._standard_modules()
    monkeypatch.setattr(module.fields.GaussianImageFields, "evaluate", lambda *args: None)
    assert not module._standard_modules()


def test_unsupported_backends_decline():
    assert module.GaussianSlabReuseConditions(SimpleNamespace(base=None))() is None


@pytest.mark.parametrize("flag", [0, 1, 2, 0.5, "1", None])
def test_non_boolean_rate_flag_is_not_silently_reinterpreted(host_conditions, flag):
    provider, _, _ = host_conditions
    args = _request()
    args["strength_rate_enabled"] = flag
    assert provider().request_check(**args) is False


@pytest.mark.parametrize(
    "field,name",
    [
        (module.fields, "GaussianImageFields"),
        (module.runtime, "CUDAMemoryPool"),
        (module.tail_error_bounds, "prepare_tail_source"),
    ],
)
def test_module_rebinding_before_any_field_declines(monkeypatch, field, name):
    assert module._standard_modules()
    monkeypatch.setattr(field, name, lambda *args: None)
    assert not module._standard_modules()


def test_python_replacement_cannot_validate_host_rounding(monkeypatch):
    bridge = SimpleNamespace(
        __name__="source.solvers.vpm.numerics._fenv",
        capture=lambda: None,
        enter_default=lambda token: None,
        restore=lambda token: None,
        round_to_nearest=lambda: True,
    )
    monkeypatch.setattr(module.ieee, "_bridge", lambda: bridge)
    with pytest.raises(RuntimeError, match="standard compiled"):
        module._check_rounding()


@pytest.mark.parametrize("name", tuple(module._TRANSFER_METHODS))
@pytest.mark.parametrize("scope", ["instance", "class"])
def test_custom_transfer_methods_decline_exact_stage_reuse(monkeypatch, name, scope):
    cls = module.physics_base.PhysicsBase
    physics = cls.__new__(cls)
    assert module._standard_transfers(physics)
    monkeypatch.setattr(physics if scope == "instance" else cls, name, lambda *args: None)
    assert not module._standard_transfers(physics)


def test_changed_transfer_chunk_size_declines_exact_stage_reuse(monkeypatch):
    cls = module.physics_base.PhysicsBase
    physics = cls.__new__(cls)
    assert module._standard_transfers(physics)
    monkeypatch.setattr(module.physics_base, "_HOST_TRANSFER_CHUNK_SIZE", 32768)
    assert not module._standard_transfers(physics)


def _blocked_session(slab):
    current = module.session.GaussianSlabFieldSession(
        z_min=slab.z_min,
        z_max=slab.z_max,
        tail_tolerance=slab.tail_tolerance,
        max_shells=slab.max_shells,
        velocity_scale=slab.velocity_scale,
        gradient_scale=slab.gradient_scale,
        dtype="float32",
        settings=slab.gaussian_mesh_settings,
        execution_backend="cupy_cuda",
    )
    field = module.blocked_fields.GaussianBlockedCUDAFields(
        np.array([[0.1, -0.2, 0.17]]),
        np.array([[0.2, -0.3, 0.7]]),
        np.array([0.04]),
        np.array([[0.15, -0.1, 0.19]]),
        zmin=slab.z_min,
        zmax=slab.z_max,
        tau=0.12,
        spacing=0.025,
        cutoff=0.6,
    )
    field.prepare(((0, True),))
    wrapper = module.execution.PortableGaussianImageFields.__new__(
        module.execution.PortableGaussianImageFields
    )
    wrapper.closed, wrapper.execution_backend, wrapper._implementation = False, "cupy_cuda", field
    current._field = wrapper
    slab._mesh_session = current
    return field


def test_standard_blocked_field_preserves_exact_stage_reuse_validation(
    host_conditions, monkeypatch
):
    provider, slab, _ = host_conditions
    slab.base = provider._base_backend = object()
    slab.physics = module.physics_base.PhysicsBase.__new__(module.physics_base.PhysicsBase)
    field = _blocked_session(slab)
    monkeypatch.setattr(module, "_standard_methods", lambda *args: True)
    assert module.GaussianSlabReuseConditions._supported(provider)
    module._check_session(slab)
    monkeypatch.setattr(field, "_split", lambda *args: None)
    assert not module.GaussianSlabReuseConditions._supported(provider)


@pytest.mark.parametrize("fault", ["closed", "failed", "leaf", "thread", "images"])
def test_blocked_field_run_events_fault_is_not_hidden_by_stage_hit(host_conditions, fault):
    _, slab, _ = host_conditions
    field = _blocked_session(slab)
    if fault == "closed":
        field.closed = True
    elif fault == "failed":
        field._failed_field = object()
    elif fault == "leaf":
        field._leaf = object()
    elif fault == "thread":
        field._thread = -1
    else:
        field._prepared_images = None
    with pytest.raises(RuntimeError):
        module._check_session(slab)


@pytest.mark.parametrize("name", ["_split", "_evaluate_leaf", "evaluate_prepared"])
def test_blocked_kernel_delegation_overrides_decline_reuse(monkeypatch, name):
    assert module._standard_modules()
    monkeypatch.setattr(module.blocked_fields.GaussianBlockedCUDAFields, name, lambda *args: None)
    assert not module._standard_modules()
