"""Host contract qualification; no CuPy, GPU, or physical solver is run."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import taichi as ti

from source.solvers.vpm.physics.induction.gaussian_mesh import reuse_contract as module
from source.solvers.vpm.physics.induction.gaussian_mesh.session import GaussianSlabPolicy
from source.solvers.vpm.physics.induction.reuse import InductionReuseContract, _canonical_key


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
def host_contract(monkeypatch):
    events = []
    primary = InductionReuseContract(
        ("qualified fake primary",), True, True, True, True,
        lambda: ("primary proof",), lambda state: events.append(("restore primary", state)))
    slab = SimpleNamespace(
        physics=SimpleNamespace(accumulator_dtype=ti.f32, _zero_velocity=Scalar((0., 0., 0.))),
        kernel=SimpleNamespace(name="GAUSSIAN"), gaussian_mesh_policy=GaussianSlabPolicy(),
        z_min=-.5, z_max=.5, tail_tolerance=1e-4, max_shells=129,
        velocity_scale=1., gradient_scale=2., stretching_scheme="TRANSPOSED",
        _z_min_field=Scalar(-.5), _z_max_field=Scalar(.5), _span_excess=Scalar(0.),
        _mesh_session=None,
        _uses_mesh=lambda: events.append("binding") or True,
        _mesh_fields=lambda *args: events.append(("fields", args)),
        _mesh_count=lambda count, cap: events.append(("capacity", count, cap)),
        last_tail={"shell": 128, "velocity": 1e-6, "gradient": 2e-6, "relative": 1e-6,
                   "contract": "gaussian_interval_remainder_v1", "seconds": 11.,
                   "mesh": {"source_sha256": "source", "source_only_primary": False,
                            "velocity_tail_bound": 1e-6, "gradient_tail_bound": 2e-6,
                            "finite_evaluation": {"passes": 45}, "seconds": 11.}})
    provider = module.StandardGaussianSlabReuseContract.__new__(module.StandardGaussianSlabReuseContract)
    provider.backend = slab
    provider._supported = lambda: True
    provider._base_contract = lambda: primary
    monkeypatch.setattr(module, "_admit_rounding", lambda: events.append("rounding"))
    return provider, slab, events


def _request(count=3):
    return {"position": object(), "vortex_strength": object(), "core_radius": object(), "count": count,
            "velocity_out": object(), "vortex_strength_rate_out": object(),
            "velocity_gradient_out": None, "strength_rate_enabled": False, "stage_time": 31.}


def test_request_admission_runs_public_layout_caps_and_rounding(host_contract):
    provider, _, events = host_contract
    provider().request_admission(**_request())
    assert events[0] == "binding"
    assert events[1][0] == "fields"
    assert events[2] == ("capacity", 3, 1_000_000)
    assert events[3] == "rounding"
    events.clear()
    provider().request_admission(**_request(0))
    assert "rounding" not in events


@pytest.mark.parametrize("name", ["velocity_out", "vortex_strength_rate_out"])
def test_missing_mandatory_output_fails_before_any_publication(host_contract, name):
    provider, _, events = host_contract
    args = _request()
    args[name] = None
    with pytest.raises(ValueError, match="require velocity"):
        provider().request_admission(**args)
    assert events == ["binding"]


def test_controls_and_actual_device_planes_change_key_but_role_does_not(host_contract):
    provider, slab, _ = host_contract
    original = _canonical_key(provider().operator_key)
    # Even an unrelated coherent-query owner cannot change mathematical
    # dependence of the independently copied particle result.
    slab._mesh_session = SimpleNamespace(_role=True, _owner=object())
    assert _canonical_key(provider().operator_key) == original
    slab._z_min_field[None] = -.4
    assert _canonical_key(provider().operator_key) != original
    slab._z_min_field[None] = -.5
    slab.gaussian_mesh_policy = replace(slab.gaussian_mesh_policy, max_sources=17)
    assert _canonical_key(provider().operator_key) != original


def test_only_mathematical_observations_are_restored(host_contract):
    provider, slab, events = host_contract
    contract = provider()
    saved = contract.capture_diagnostics()
    slab.last_tail = {"shell": 99, "seconds": 23., "mesh": {"finite_evaluation": {"passes": 79}}}
    current = slab.last_tail.copy()
    owner = object()
    slab._mesh_session = SimpleNamespace(_role=True, _owner=owner, calls=17)
    contract.restore_diagnostics(saved)
    assert slab.last_tail["shell"] == 128
    assert slab.last_tail["exact_stage_reuse"]
    assert slab.last_tail["seconds"] == 0
    assert "finite_evaluation" not in slab.last_tail["mesh"]
    assert slab.last_actual_mesh_work == current
    assert slab._mesh_session._owner is owner and slab._mesh_session.calls == 17
    assert events == [("restore primary", ("primary proof",))]
    contract.restore_diagnostics(saved)
    assert slab.last_actual_mesh_work == current


@pytest.mark.parametrize("fault", ["closed", "uncertain", "thread", "controls", "plans", "correction", "stream"])
def test_existing_session_lifecycle_never_hidden(host_contract, fault):
    provider, slab, events = host_contract
    def fail():
        raise RuntimeError("injected owner thread or stream")
    device = SimpleNamespace(admit=fail if fault == "stream" else lambda: events.append("device"))
    owner = SimpleNamespace(_admit=lambda: device.admit(),
                            _plans=SimpleNamespace(closed=fault == "plans", owner=device),
                            _correction=SimpleNamespace(closed=fault == "correction", _owner=device),
                            _prepared_images=((0, True),), _compact_fields=object())
    config = (slab.z_min, slab.z_max, slab.tail_tolerance, slab.max_shells,
              slab.velocity_scale, slab.gradient_scale, "float32", slab.gaussian_mesh_policy, "cupy_cuda")
    slab._mesh_session = SimpleNamespace(
        _admit_thread=fail if fault == "thread" else lambda: None,
        closed=fault == "closed", cleanup_uncertain=fault == "uncertain", execution_backend="cupy_cuda",
        _configuration=config, _configuration_key=lambda: () if fault == "controls" else config,
        _owner=owner)
    with pytest.raises(RuntimeError):
        provider().request_admission(**_request())


def test_standard_binding_changes_decline_without_certifying_custom_code(monkeypatch):
    assert module._standard_modules()
    monkeypatch.setattr(module.fields.GaussianImageFields, "evaluate", lambda *args: None)
    assert not module._standard_modules()


def test_unsupported_backends_decline():
    assert module.StandardGaussianSlabReuseContract(SimpleNamespace(base=None))() is None


@pytest.mark.parametrize("flag", [0, 1, 2, .5, "1", None])
def test_non_boolean_rate_flag_is_not_silently_reinterpreted(host_contract, flag):
    provider, _, _ = host_contract
    args = _request()
    args["strength_rate_enabled"] = flag
    assert provider().request_admission(**args) is False


@pytest.mark.parametrize("owner,name", [(module.fields, "GaussianImageFields"),
                                       (module.runtime, "DeviceOwner"),
                                       (module.certificate, "prepare_tail_source")])
def test_module_rebinding_before_any_owner_declines(monkeypatch, owner, name):
    assert module._standard_modules()
    monkeypatch.setattr(owner, name, lambda *args: None)
    assert not module._standard_modules()


def test_python_replacement_cannot_certify_host_rounding(monkeypatch):
    bridge = SimpleNamespace(__name__="source.solvers.vpm.numerics._fenv",
                             capture=lambda: None, enter_default=lambda token: None,
                             restore=lambda token: None, round_to_nearest=lambda: True)
    monkeypatch.setattr(module.ieee, "_bridge", lambda: bridge)
    with pytest.raises(RuntimeError, match="standard compiled"):
        module._admit_rounding()
