"""Explicit reuse contracts for the standard FMM implementations.

This is a whitelist, not duck-typing of purportedly autonomous operators.
Only the exact FMM and SlipSlab-FMM classes, standard physics workspaces and
standard radial factories qualify. Source contents are checked separately by
``ExactContentInductionReuse``. Mutable numerical options and small device
configuration tables are read exactly; particle or hierarchy arrays are never
downloaded. Untracked custom methods, closures and subclasses decline reuse.

The backend's private hierarchy is *not* part of the cached result. A hit
invalidates hierarchy-readiness hints; a subsequent target query must rebuild
from its explicitly supplied sources. Work counters and timing records always
describe actual evaluations, never synthetic evaluations on cache hits.
"""

from functools import lru_cache
import inspect
from types import FunctionType

import taichi as ti

from ...kernels.base import RadialVortexKernel, make_device_vortex_kernels, make_vortex_kernel
from ..base import PhysicsBase
from ..engine import PhysicsEngine
from .base import _STRETCHING_MODES
from .fmm import device as fmm_module
from .fmm import target_geometry as geometry_module
from .fmm import targets as target_module
from .fmm.device import FMMDeviceWorkspace, FMMInduction
from .fmm.targets import FMMTargetEvaluator
from .reuse import InductionReuseContract, _canonical_key, _primitive_key
from .slip_slab import SlipSlabInduction
from .treecode.lbvh import TaichiTreecode

_RATE_DIAGNOSTICS = (
    "last_uncorrected_rate_defect",
    "last_strength_rate_norm",
    "last_relative_rate_defect",
)
_TAIL_OBSERVATIONS = ("shell", "block_start", "relative", "velocity", "gradient")
_FMM_TABLES = (
    "_coefficient_a",
    "_coefficient_b",
    "_coefficient_c",
    "_derivative_lookup",
    "_derivative_term_count",
    "_derivative_coefficient",
    "_derivative_exponent",
    "_derivative_radial_step",
)
_TARGET_TABLES = (
    "local_alpha",
    "local_inverse_factorial",
    "derivative_lookup",
    "derivative_term_count",
    "derivative_coeff",
    "derivative_exponent",
    "derivative_radial_steps",
)
_PHYSICS_METHODS = ("_copy_vec3", "_copy_mat3", "_zero_vec3_field")
_STANDARD_METHODS = {
    cls: {
        name: value
        for name, value in vars(cls).items()
        if callable(value) or isinstance(value, property)
    }
    for cls in (
        FMMInduction,
        FMMDeviceWorkspace,
        FMMTargetEvaluator,
        SlipSlabInduction,
        TaichiTreecode,
    )
}
_STANDARD_PHYSICS_METHODS = {
    name: inspect.getattr_static(PhysicsBase, name) for name in _PHYSICS_METHODS
}
_UNSUPPORTED = object()


def _standard_methods(instance, cls):
    """Reject monkey-patched numerical methods as well as subclasses."""
    return type(instance) is cls and all(
        name not in vars(instance) and inspect.getattr_static(cls, name) is method
        for name, method in _STANDARD_METHODS[cls].items()
        # Taichi installs per-instance __getattribute__ machinery itself.
        if not name.startswith("__")
    )


def _same_standard_function(actual, expected):
    """Compare standard factory code and immutable closure values, not names.

    Callable identity alone would admit a stateful custom closure. Both the
    Taichi wrapper code and its original Python code must be standard, and
    every captured value must recursively equal the reference factory value.
    Mutable captures and callable objects are intentionally unsupported.
    """
    if type(actual) is not FunctionType or type(expected) is not FunctionType:
        return False
    if actual.__code__ is not expected.__code__:
        return False
    original, reference = inspect.unwrap(actual), inspect.unwrap(expected)
    if original.__code__ is not reference.__code__:
        return False
    if original.__defaults__ != reference.__defaults__:
        return False
    left = tuple(cell.cell_contents for cell in (original.__closure__ or ()))
    right = tuple(cell.cell_contents for cell in (reference.__closure__ or ()))
    if len(left) != len(right):
        return False
    for value, standard in zip(left, right, strict=True):
        if type(standard) is FunctionType:
            if not _same_standard_function(value, standard):
                return False
        elif _primitive_key(standard):
            if not _primitive_key(value) or _canonical_key(value) != _canonical_key(standard):
                return False
        elif standard == ti.f32:
            if type(value) is not type(standard) or value != ti.f32:
                return False
        else:
            return False
    return True


@lru_cache(maxsize=4)
def _standard_functions(name):
    return make_device_vortex_kernels(name, ti.f32)


def _standard_tree_functions(tree):
    return all(
        _same_standard_function(getattr(tree, field), _standard_functions(kernel)[function])
        for field, kernel, function in (
            ("_gaussian_q", "GAUSSIAN", "q_"),
            ("_gaussian_zeta", "GAUSSIAN", "zeta_"),
            ("_gaussian_radial_factors", "GAUSSIAN", "radial_factors_"),
            ("_winckelmans_radial_factors", "WINCKELMANS", "radial_factors_"),
        )
    )


def _fields_alive(owner, name="_field_owner"):
    fields = getattr(owner, name, None)
    return fields is not None and fields.tree is not None


def _tables(owner, names):
    # These are small immutable-by-API operator tables, not particle data.
    # Exact bytes also detect unsupported direct edits instead of trusting a
    # field's identity or a probabilistic checksum.
    result = []
    for name in names:
        array = getattr(owner, name).to_numpy()
        result.append((name, array.dtype.str, array.shape, array.tobytes()))
    return tuple(result)


def _field_layout(owner, names):
    return tuple(
        (
            name,
            id(field),
            str(field.dtype),
            field.shape,
            getattr(field, "n", 1),
            getattr(field, "m", 1),
        )
        for name in names
        for field in (getattr(owner, name),)
    )


def _constants(module):
    return tuple(
        (name, value)
        for name, value in sorted(vars(module).items())
        if name.isupper() and _primitive_key(value)
    )


def _scalars(owner, names):
    values = tuple((name, getattr(owner, name)) for name in names)
    return values if _primitive_key(values) else _UNSUPPORTED


class StandardFMMReuseContract:
    """Callable contract provider, deliberately not installed by construction.

    A complete backend rebind/reallocation changes binding identities and
    safely causes a miss. The first call which grows scratch may therefore
    need one additional miss; no unsafe assumption about workspace lifetime
    is made to recover this cold-start optimization.

    Numerical module hot-patching while kernels are live is unsupported by
    Taichi itself. The whitelist also rejects instance/class method overrides
    introduced after this module's import. Public and actual device options
    are included independently, so inconsistent public/device values cannot
    obtain a hit from a previously valid configuration.
    """

    def __init__(self, backend):
        self.backend = backend
        self._runtime_program = ti.lang.impl.get_runtime().prog

    def __call__(self):
        if (
            self._runtime_program is None
            or ti.lang.impl.get_runtime().prog is not self._runtime_program
        ):
            return None
        slab = self.backend if type(self.backend) is SlipSlabInduction else None
        if slab is not None and (
            slab.physics is None or slab.physics.particle_kernel == "GAUSSIAN"
        ):
            # Gaussian images require their own source/tail/lifecycle guards.
            return None
        base = slab.base if slab is not None else self.backend
        if not _standard_methods(base, FMMInduction):
            return None
        physics, workspace = base.physics, base.workspace
        if (
            type(physics) not in (PhysicsBase, PhysicsEngine)
            or physics.accumulator_dtype != ti.f32
            or workspace is None
            or not _standard_methods(workspace, FMMDeviceWorkspace)
            or not _fields_alive(workspace)
            or not _standard_methods(workspace.tree, TaichiTreecode)
            or not _fields_alive(workspace.tree)
            or getattr(base, "_fixed_source_key", None) is not None
        ):
            return None
        if any(
            name in vars(physics) or inspect.getattr_static(type(physics), name) is not standard
            for name, standard in _STANDARD_PHYSICS_METHODS.items()
        ):
            return None
        kernel = base.kernel
        if type(kernel) is not RadialVortexKernel or kernel.name not in base.supported_kernels:
            return None
        if kernel != make_vortex_kernel(kernel.name) or physics.particle_kernel != kernel.name:
            return None
        radial = base._radial_factors
        if (
            radial is not physics._kernel_functions.get("radial_factors_")
            or workspace.radial_factors is not radial
            or not _same_standard_function(
                radial, _standard_functions(kernel.name)["radial_factors_"]
            )
            or not _standard_tree_functions(workspace.tree)
        ):
            return None
        tree = workspace.tree
        key = (
            "standard-fmm-reuse-v1",
            id(self._runtime_program),
            id(base),
            id(physics),
            id(kernel),
            id(radial),
            id(workspace),
            id(workspace._field_owner.tree),
            id(tree._field_owner.tree),
            str(physics.accumulator_dtype),
            _scalars(physics, ("particle_kernel", "max_n_particles", "max_evaluation_points")),
            _scalars(
                base,
                (
                    "stretching_scheme",
                    "_stretching_mode",
                    "method",
                    "max_n_particles",
                    "_velocity_tail_cutoff",
                    "_gradient_tail_cutoff",
                    "_max_evaluation_points",
                    "max_image_block_targets",
                    "max_image_geometry_bytes",
                ),
            ),
            _scalars(
                workspace,
                (
                    "kernel_name",
                    "velocity_tail_cutoff",
                    "gradient_tail_cutoff",
                    "max_n_particles",
                    "max_nodes",
                    "max_pairs",
                    "max_m2l_pairs",
                    "max_near_pairs",
                    "max_queue_pairs",
                    "m2l_batch_size",
                    "target_batch_capacity",
                ),
            ),
            _scalars(
                tree,
                (
                    "theta",
                    "theta_sq",
                    "max_leaf_size",
                    "kernel_type",
                    "max_stack_depth",
                    "max_tree_depth_guard",
                    "traversal_block_dim",
                    "device_sort_only",
                    "hierarchy_only",
                    "max_evaluation_points",
                    "default_cutoff_radius_factor",
                ),
            ),
            tuple(
                (name, getattr(tree, name)[None])
                for name in (
                    "kernel_type_id",
                    "regularization_tail_cutoff",
                    "multipole_order",
                    "sort_particle_targets",
                )
            ),
            _tables(workspace, _FMM_TABLES),
            _constants(fmm_module),
            tuple(sorted(_STRETCHING_MODES.items())),
        )
        geometry = base._image_geometry_cache
        geometry_key = None
        if not geometry_module.certified_geometry(geometry):
            return None
        if geometry is not None:
            storage = geometry.storage
            geometry_key = (
                id(geometry),
                geometry.max_bytes,
                None
                if storage is None
                else (
                    id(storage),
                    id(storage.owner.tree),
                    storage.capacity,
                    storage.bytes,
                    _field_layout(storage, geometry_module._STORAGE_FIELDS),
                ),
                _constants(geometry_module),
            )
        key += (geometry_key,)
        if slab is not None:
            # The other two kernels use PhysicsBase's direct target kernels,
            # whose extra function-closure contract is not certified here.
            if kernel.name not in {"GAUSSIAN", "WINCKELMANS"}:
                return None
            if not _standard_methods(slab, SlipSlabInduction) or slab.physics is not physics:
                return None
            if slab.kernel is not kernel:
                return None
            target = base._target_workspace
            target_key = None
            if target is not None:
                if (
                    not _standard_methods(target, FMMTargetEvaluator)
                    or target.source is not workspace
                    or not _fields_alive(target, "_fields")
                    or not _fields_alive(target, "_pair_fields")
                    or not _standard_methods(target.tree, TaichiTreecode)
                    or not _fields_alive(target.tree)
                ):
                    return None
                target_key = (
                    id(target),
                    id(target._fields.tree),
                    id(target._pair_fields.tree),
                    _scalars(
                        target,
                        (
                            "monopole_radial_kernel",
                            "source_separation",
                            "target_core_cutoff",
                            "max_targets",
                            "max_images",
                            "max_pairs",
                            "batch_capacity",
                            "target_path_capacity",
                        ),
                    ),
                    # These arrays are disposable geometry scratch, not part
                    # of the cached physical result.  Their bindings/layouts
                    # still belong to the operational backend contract: a
                    # hit must not hide a changed bounded ancestry workspace.
                    _field_layout(
                        target, ("target_path", "target_path_length", "target_path_error")
                    ),
                    _tables(target, _TARGET_TABLES),
                    _scalars(
                        target.tree,
                        (
                            "max_leaf_size",
                            "max_tree_depth_guard",
                            "device_sort_only",
                            "hierarchy_only",
                            "max_n_particles",
                            "max_nodes",
                        ),
                    ),
                )
            key += (
                id(slab),
                id(slab._z_min_field),
                id(slab._z_max_field),
                _scalars(
                    slab,
                    (
                        "z_min",
                        "z_max",
                        "tail_tolerance",
                        "max_shells",
                        "velocity_scale",
                        "gradient_scale",
                        "stretching_scheme",
                    ),
                ),
                float(slab._z_min_field[None]),
                float(slab._z_max_field[None]),
                bool(base.supports_image_blocks),
                tuple(float(value) for value in physics._zero_velocity[None]),
                _field_layout(
                    slab,
                    (
                        "_image_velocity",
                        "_image_gradient",
                        "_query_position",
                        "_image_shifts",
                        "_image_odd",
                        "_block_velocity",
                        "_block_gradient",
                        "_max_shell_velocity",
                        "_max_shell_gradient",
                        "_span_excess",
                        "_z_min_field",
                        "_z_max_field",
                    ),
                ),
                target_key,
                _constants(target_module),
            )
        if not _primitive_key(key):
            return None

        def capture():
            return (
                tuple(getattr(base.diagnostics, name) for name in _RATE_DIAGNOSTICS),
                None
                if slab is None or slab.last_tail is None
                else {name: slab.last_tail[name] for name in _TAIL_OBSERVATIONS},
                None if slab is None else float(slab._span_excess[None]),
            )

        def restore(state):
            rate, tail, span_excess = state
            for name, value in zip(_RATE_DIAGNOSTICS, rate, strict=True):
                setattr(base.diagnostics, name, value)
            # Do not rewind timing/count/decline records of real queries.
            # Existing operational entries remain last-actual-work values.
            if slab is not None and tail is not None:
                current = {} if slab.last_tail is None else dict(slab.last_tail)
                current.update(tail)
                slab.last_tail = current
                slab._span_excess[None] = span_excess
            base._last_tree_key = None
            base._source_moments_ready = False

        return InductionReuseContract(
            operator_key=key,
            autonomous=True,
            sources_are_read_only=True,
            complete_outputs_are_equivalent=True,
            diagnostics_are_complete=True,
            capture_diagnostics=capture,
            restore_diagnostics=restore,
        )


__all__ = ["StandardFMMReuseContract"]
