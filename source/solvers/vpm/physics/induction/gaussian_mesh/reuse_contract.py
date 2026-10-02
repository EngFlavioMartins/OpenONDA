"""Exact particle-stage reuse for the standard Gaussian slab.

Other radial kernels use the reflected-source FMM/slab contract.
Only the *complete pure particle operator* is cached: pair-mean-core primary
FMM plus source-core images. Coherent source-only queries are never cached by
this adapter. They may replace the session's private grid without invalidating
an independently owned, exact-content particle result.

A previous successful result carries its mathematical tail/correction proof
for exactly the same ordered x/Gamma/core values and controls. No interval
proof is inferred for a new source or query. Every request still undergoes
the public layout/capacity/runtime/FENV guards before the device exact check;
the StageRHS position guard and all external providers remain outside reuse.
"""

from copy import deepcopy
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
import inspect

import taichi as ti

from ....numerics import ieee
from .. import gaussian_tail, slip_slab
from ..gaussian_tail import _interval, certificate
from ..reuse import InductionReuseContract, _primitive_key
from ..reuse_backends import StandardFMMReuseContract, _field_layout, _standard_methods
from . import (
    coordinates,
    correction,
    correction_admission,
    error_bounds,
    execution,
    fields,
    planning,
    policy,
    runtime,
    session,
    stencil,
)

_MODULES = (
    slip_slab,
    session,
    policy,
    fields,
    runtime,
    correction,
    stencil,
    coordinates,
    planning,
    correction_admission,
    error_bounds,
    execution,
    gaussian_tail,
    certificate,
    _interval,
    ieee,
)
_CLASSES = (
    session.GaussianSlabFieldSession,
    session.GaussianSlabPolicy,
    policy.GaussianMeshParameters,
    fields.GaussianImageFields,
    runtime.DeviceOwner,
    runtime.FFTPlanPair,
    correction.GaussianCoreCorrectionGPU,
    execution.PortableGaussianImageFields,
)
# Snapshot actual standard implementations, not names or arbitrary closures.
# Imports are CuPy-lazy. Untracked custom numerical code always falls through
# to the public operator; this is not a purity declaration for custom code.
_CLASS_BINDINGS = {
    cls: {
        name: value
        for name, value in vars(cls).items()
        if callable(value) or isinstance(value, (property, staticmethod, classmethod))
    }
    for cls in _CLASSES
}
_MODULE_BINDINGS = {
    module: {
        name: value
        for name, value in vars(module).items()
        if inspect.isfunction(value)
        or inspect.isclass(value)
        or (name.isupper() and _primitive_key(value))
    }
    for module in _MODULES
}
_INSTANCE_METHODS = {
    cls: frozenset(name for name in methods if not name.startswith("__"))
    for cls, methods in _CLASS_BINDINGS.items()
}
_MATHEMATICAL_TAIL = ("shell", "velocity", "gradient", "relative", "contract")
_MATHEMATICAL_MESH = (
    "empty_field",
    "source_sha256",
    "source_only_primary",
    "finite_images",
    "shell",
    "velocity_tail_bound",
    "gradient_tail_bound",
    "velocity_correction_bound",
    "gradient_correction_bound",
    "omitted_distance_lower",
    "certificate_scope",
)


def _standard_classes():
    return all(
        vars(cls).get(name) is method
        for cls, methods in _CLASS_BINDINGS.items()
        for name, method in methods.items()
    )


def _standard_instance(instance, cls):
    return type(instance) is cls and _INSTANCE_METHODS[cls].isdisjoint(vars(instance))


def _standard_modules():
    # Identity also detects reassignment of equal-looking numeric constants.
    return _standard_classes() and all(
        vars(module).get(name) is value
        for module, bindings in _MODULE_BINDINGS.items()
        for name, value in bindings.items()
    )


def _policy_key(value):
    if is_dataclass(value):
        return (
            type(value).__name__,
            tuple(
                (item.name, _policy_key(getattr(value, item.name)))
                for item in dataclass_fields(value)
            ),
        )
    return value


def _admit_rounding():
    # A replaced extension function must not turn the runtime guard into a
    # purported purity declaration. Accept only this extension's real C API.
    bridge = ieee._bridge()
    for name in ("capture", "enter_default", "restore", "round_to_nearest"):
        function = getattr(bridge, name, None)
        if (
            not inspect.isbuiltin(function)
            or function.__name__ != name
            or function.__module__ != bridge.__name__
            or getattr(function, "__self__", None) is not bridge
        ):
            raise RuntimeError("Gaussian stage reuse requires the standard compiled FENV guard")
    ieee.require_round_to_nearest()


def _admit_session(slab):
    """Do not hide a dead/wrong-stream owner behind a pure-result hit.

    Healthy absence or a changed role is allowed: no cached field depends on
    retaining that disposable GPU grid. No owner is created or changed here.
    """
    current = slab._mesh_session
    if current is None:
        return
    current._admit_thread()
    if current.closed or current.cleanup_uncertain:
        raise RuntimeError("Gaussian slab field session is closed or cleanup is uncertain")
    if current._configuration_key() != current._configuration:
        raise RuntimeError("Gaussian slab controls changed; construct a new session")
    expected = (
        slab.z_min,
        slab.z_max,
        slab.tail_tolerance,
        slab.max_shells,
        slab.velocity_scale,
        slab.gradient_scale,
        "float32",
        slab.gaussian_mesh_policy,
        current.execution_backend,
    )
    if current._configuration != expected:
        raise RuntimeError("Gaussian slab session does not match its bound operator")
    owner = current._owner
    if owner is None:
        return
    if type(owner) is execution.PortableGaussianImageFields:
        if owner.closed or owner.execution_backend != "cupy_cuda":
            raise RuntimeError("Gaussian slab reusable CUDA owner is unavailable")
        owner = owner._owner
    owner._admit()
    plans, local = owner._plans, owner._correction
    if plans is None or plans.closed or local is None or local.closed:
        raise RuntimeError("Gaussian slab private field resources are closed")
    plans.owner.admit()
    local._owner.admit()
    if owner._prepared_images is None or owner._compact_fields is None:
        raise RuntimeError("Gaussian slab private field resources are incomplete")


class StandardGaussianSlabReuseContract:
    """Whitelist for f32 CUDA Gaussian Slab + standard primary FMM only."""

    def __init__(self, backend):
        self.backend = backend
        self._base_backend = getattr(backend, "base", None)
        self._base_contract = StandardFMMReuseContract(self._base_backend)

    def _supported(self):
        slab = self.backend
        if (
            not _standard_modules()
            or not _standard_methods(slab, slip_slab.SlipSlabInduction)
            or slab.base is not self._base_backend
            or type(slab.gaussian_mesh_policy) is not session.GaussianSlabPolicy
            or type(slab.gaussian_mesh_policy.mesh) is not policy.GaussianMeshParameters
        ):
            return False
        current = slab._mesh_session
        if current is not None:
            if not _standard_instance(current, session.GaussianSlabFieldSession):
                return False
            owner = current._owner
            if owner is not None:
                if type(owner) is execution.PortableGaussianImageFields:
                    if (
                        not _standard_instance(owner, execution.PortableGaussianImageFields)
                        or owner.closed
                        or owner.execution_backend != "cupy_cuda"
                    ):
                        return False
                    owner = owner._owner
                if not _standard_instance(owner, fields.GaussianImageFields):
                    return False
                for item, cls in (
                    (owner._owner, runtime.DeviceOwner),
                    (owner._plans, runtime.FFTPlanPair),
                    (owner._correction, correction.GaussianCoreCorrectionGPU),
                ):
                    if item is not None and not _standard_instance(item, cls):
                        return False
                if owner._correction is not None and not _standard_instance(
                    owner._correction._owner, runtime.DeviceOwner
                ):
                    return False
        return True

    def _admit_request(self, **args):
        slab = self.backend
        if not slab._uses_mesh():
            raise RuntimeError("Gaussian slab stage reuse requires its bound mesh operator")
        if args["velocity_out"] is None or args["vortex_strength_rate_out"] is None:
            raise ValueError("particle mesh stages require velocity and strength-rate outputs")
        if type(args["strength_rate_enabled"]) is not bool:
            # The public mesh uses int(flag)==1, whereas private publication
            # uses a Boolean. Only the actual StageRHS Boolean contract is
            # admitted; never silently reinterpret e.g. flag=2 on a hit.
            # Preserve the public operator's accepted int/np.bool_ semantics.
            return False
        slab._mesh_fields(
            args["position"],
            args["vortex_strength"],
            args["core_radius"],
            args["position"],
            args["count"],
            args["count"],
            args["velocity_out"],
            args["velocity_gradient_out"],
            args["vortex_strength_rate_out"],
        )
        slab._mesh_count(args["count"], slab.gaussian_mesh_policy.max_sources)
        # Empty public stages return without certificate construction. Avoid
        # imposing a dependency on their otherwise dependency-free no-op.
        if args["count"]:
            _admit_rounding()
            _admit_session(slab)

    def __call__(self):
        if not self._supported():
            return None
        slab = self.backend
        primary = self._base_contract()
        if primary is None:
            return None
        # f64/custom kernels deliberately bypass until separately qualified.
        if slab.physics.accumulator_dtype != ti.f32 or slab.kernel.name != "GAUSSIAN":
            return None
        key = (
            "gaussian-slab-exact-stage-v1",
            primary.operator_key,
            id(slab),
            id(slab.physics),
            _policy_key(slab.gaussian_mesh_policy),
            slab.z_min,
            slab.z_max,
            slab.tail_tolerance,
            slab.max_shells,
            slab.velocity_scale,
            slab.gradient_scale,
            slab.stretching_scheme,
            float(slab._z_min_field[None]),
            float(slab._z_max_field[None]),
            _field_layout(slab, ("_z_min_field", "_z_max_field", "_span_excess")),
            tuple(float(value) for value in slab.physics._zero_velocity[None]),
        )
        if not _primitive_key(key):
            return None

        def capture():
            tail = slab.last_tail
            state = (
                None
                if tail is None
                else {name: deepcopy(tail[name]) for name in _MATHEMATICAL_TAIL if name in tail}
            )
            if state is not None:
                state["mesh"] = {
                    name: deepcopy(tail.get("mesh", {})[name])
                    for name in _MATHEMATICAL_MESH
                    if name in tail.get("mesh", {})
                }
            return primary.capture_diagnostics(), float(slab._span_excess[None]), state

        def restore(state):
            primary_state, span, tail = state
            primary.restore_diagnostics(primary_state)
            slab._span_excess[None] = span
            if tail is not None:
                # Last actual work stays visible and is never represented as
                # a newly performed solve. Operational mesh/session counters
                # are neither rewound nor advanced by restoring this proof.
                actual = slab.last_tail
                if actual is not None and not actual.get("exact_stage_reuse", False):
                    slab.last_actual_mesh_work = deepcopy(actual)
                slab.last_tail = deepcopy(tail)
                slab.last_tail["exact_stage_reuse"] = True
                slab.last_tail["seconds"] = 0.0

        return InductionReuseContract(
            operator_key=key,
            autonomous=True,
            sources_are_read_only=True,
            complete_outputs_are_equivalent=True,
            diagnostics_are_complete=True,
            capture_diagnostics=capture,
            restore_diagnostics=restore,
            request_admission=self._admit_request,
        )
