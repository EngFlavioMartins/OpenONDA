"""Exact particle-stage reuse for the standard Gaussian slab.

Other radial kernels use the reflected-source FMM/slab conditions.
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
from ... import base as physics_base
from .. import gaussian_tail, slip_slab
from ..gaussian_tail import _interval
from ..gaussian_tail import error_bounds as tail_error_bounds
from ..reuse import InductionReuseConditions, _primitive_key
from ..reuse_backends import FMMReuseConditions, _field_layout, _standard_methods
from . import (
    blocked_fields,
    coordinates,
    correction,
    correction_distance,
    error_bounds,
    execution,
    fields,
    parameters,
    planning,
    runtime,
    session,
    stencil,
)

_MODULES = (
    slip_slab,
    session,
    parameters,
    fields,
    blocked_fields,
    runtime,
    correction,
    stencil,
    coordinates,
    planning,
    correction_distance,
    error_bounds,
    execution,
    gaussian_tail,
    tail_error_bounds,
    _interval,
    ieee,
)
_CLASSES = (
    session.GaussianSlabFieldSession,
    session.GaussianSlabSettings,
    parameters.GaussianMeshParameters,
    fields.GaussianImageFields,
    blocked_fields.GaussianBlockedCUDAFields,
    runtime.CUDAMemoryPool,
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
_MATHEMATICAL_TAIL = ("shell", "velocity", "gradient", "relative", "tail_error_method")
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
    "error_bound_scope",
)
_TRANSFER_METHODS = {
    name: inspect.getattr_static(physics_base.PhysicsBase, name)
    for name in (
        "_download_vector_field",
        "_download_scalar_field",
        "_extract_vec3_field_prefix",
        "_extract_scalar_field_prefix",
        "_host_transfer_buffer",
    )
}
_TRANSFER_CHUNK_SIZE = physics_base._HOST_TRANSFER_CHUNK_SIZE


def _standard_transfers(physics):
    return (
        physics is not None
        and physics_base._HOST_TRANSFER_CHUNK_SIZE is _TRANSFER_CHUNK_SIZE
        and all(
            name not in vars(physics)
            and inspect.getattr_static(type(physics), name, None) is standard
            for name, standard in _TRANSFER_METHODS.items()
        )
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


def _settings_key(value):
    if is_dataclass(value):
        return (
            type(value).__name__,
            tuple(
                (item.name, _settings_key(getattr(value, item.name)))
                for item in dataclass_fields(value)
            ),
        )
    return value


def _check_rounding():
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


def _check_session(slab):
    """Do not hide a dead/wrong-stream field behind a pure-result hit.

    An absent field or a changed role is allowed: no cached field depends on
    retaining that disposable GPU grid. No field is created or changed here.
    """
    current = slab._mesh_session
    if current is None:
        return
    current._check_thread()
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
        slab.gaussian_mesh_settings,
        current.execution_backend,
    )
    if current._configuration != expected:
        raise RuntimeError("Gaussian slab session does not match its bound operator")
    field = current._field
    if field is None:
        return
    if type(field) is execution.PortableGaussianImageFields:
        if field.closed or field.execution_backend != "cupy_cuda":
            raise RuntimeError("Gaussian slab reusable CUDA field is unavailable")
        field = field._implementation
    field._check_context()
    if type(field) is blocked_fields.GaussianBlockedCUDAFields:
        if field._leaf is not None or field._prepared_images is None:
            raise RuntimeError("Gaussian slab blocked resources are active or incomplete")
        return
    plans, local = field._plans, field._correction
    if plans is None or plans.closed or local is None or local.closed:
        raise RuntimeError("Gaussian slab private field resources are closed")
    plans.memory_pool.check_context()
    local._memory_pool.check_context()
    if field._prepared_images is None or field._compact_fields is None:
        raise RuntimeError("Gaussian slab private field resources are incomplete")


class GaussianSlabReuseConditions:
    """Whitelist for f32 CUDA Gaussian Slab + standard primary FMM only."""

    def __init__(self, backend):
        self.backend = backend
        self._base_backend = getattr(backend, "base", None)
        self._base_conditions = FMMReuseConditions(self._base_backend)

    def _supported(self):
        slab = self.backend
        if (
            not _standard_modules()
            or not _standard_methods(slab, slip_slab.SlipSlabInduction)
            or not _standard_transfers(getattr(slab, "physics", None))
            or slab.base is not self._base_backend
            or type(slab.gaussian_mesh_settings) is not session.GaussianSlabSettings
            or type(slab.gaussian_mesh_settings.mesh) is not parameters.GaussianMeshParameters
        ):
            return False
        current = slab._mesh_session
        if current is not None:
            if not _standard_instance(current, session.GaussianSlabFieldSession):
                return False
            field = current._field
            if field is not None:
                if type(field) is execution.PortableGaussianImageFields:
                    if (
                        not _standard_instance(field, execution.PortableGaussianImageFields)
                        or field.closed
                        or field.execution_backend != "cupy_cuda"
                    ):
                        return False
                    field = field._implementation
                if type(field) is blocked_fields.GaussianBlockedCUDAFields:
                    if not _standard_instance(field, blocked_fields.GaussianBlockedCUDAFields):
                        return False
                    if field._leaf is not None and not _standard_instance(
                        field._leaf, fields.GaussianImageFields
                    ):
                        return False
                elif not _standard_instance(field, fields.GaussianImageFields):
                    return False
                else:
                    for item, cls in (
                        (field._memory_pool, runtime.CUDAMemoryPool),
                        (field._plans, runtime.FFTPlanPair),
                        (field._correction, correction.GaussianCoreCorrectionGPU),
                    ):
                        if item is not None and not _standard_instance(item, cls):
                            return False
                    if field._correction is not None and not _standard_instance(
                        field._correction._memory_pool, runtime.CUDAMemoryPool
                    ):
                        return False
        return True

    def _check_request(self, **args):
        slab = self.backend
        if not slab._uses_mesh():
            raise RuntimeError("Gaussian slab stage reuse requires its bound mesh operator")
        if args["velocity_out"] is None or args["vortex_strength_rate_out"] is None:
            raise ValueError("particle mesh stages require velocity and strength-rate outputs")
        if type(args["strength_rate_enabled"]) is not bool:
            # The public mesh uses int(flag)==1, whereas private publication
            # uses a Boolean. Only the actual StageRHS Boolean flag is
            # checked; never silently reinterpret e.g. flag=2 on a hit.
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
        slab._mesh_count(args["count"], slab.gaussian_mesh_settings.max_sources)
        # Empty public stages return without tail-bound construction. Avoid
        # imposing a dependency on their otherwise dependency-free no-op.
        if args["count"]:
            _check_rounding()
            _check_session(slab)

    def __call__(self):
        if not self._supported():
            return None
        slab = self.backend
        primary = self._base_conditions()
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
            _settings_key(slab.gaussian_mesh_settings),
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

        return InductionReuseConditions(
            operator_key=key,
            autonomous=True,
            sources_are_read_only=True,
            complete_outputs_are_equivalent=True,
            diagnostics_are_complete=True,
            capture_diagnostics=capture,
            restore_diagnostics=restore,
            request_check=self._check_request,
        )
