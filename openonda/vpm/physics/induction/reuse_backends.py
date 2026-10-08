"""Explicit reuse conditions for the standard FMM implementations.

This is a whitelist, not duck-typing of purportedly autonomous operators.
Only the exact FMM class, standard physics workspaces and standard radial
factories qualify; slip-slab images add their own mesh conditions. Source contents are checked separately by
``ExactContentInductionReuse``. Mutable numerical options and small device
configuration tables are read exactly; particle or hierarchy arrays are never
downloaded. Untracked custom methods, closures and subclasses decline reuse.

The backend's private hierarchy is *not* part of the cached result. A hit
invalidates hierarchy-readiness hints; a subsequent target query must rebuild
from its explicitly supplied sources. Work counters and timing records always
describe actual evaluations, never synthetic evaluations on cache hits.
"""

from collections.abc import Callable, Iterable
from functools import lru_cache
import inspect
from types import FunctionType, ModuleType

import taichi as ti

from ...kernels.base import RadialVortexKernel, make_device_vortex_kernels, make_vortex_kernel
from ..base import PhysicsBase
from ..engine import PhysicsEngine
from .base import _STRETCHING_MODES
from .fmm import device as fmm_module
from .fmm.device import FMMDeviceWorkspace, FMMInduction
from .reuse import InductionReuseConditions, _primitive_key, _typed_value_key
from .slip_slab import SlipSlabInduction
from .treecode.lbvh import TaichiTreecode

_RATE_DIAGNOSTICS = (
    "last_uncorrected_rate_defect",
    "last_strength_rate_norm",
    "last_relative_rate_defect",
)
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
        SlipSlabInduction,
        TaichiTreecode,
    )
}
_STANDARD_PHYSICS_METHODS = {
    name: inspect.getattr_static(PhysicsBase, name) for name in _PHYSICS_METHODS
}
_UNSUPPORTED = object()


def _standard_methods(instance: object, cls: type) -> bool:
    """Reject monkey-patched numerical methods as well as subclasses."""
    return type(instance) is cls and all(
        name not in vars(instance) and inspect.getattr_static(cls, name) is method
        for name, method in _STANDARD_METHODS[cls].items()
        # Taichi installs per-instance __getattribute__ machinery itself.
        if not name.startswith("__")
    )


def _same_standard_function(actual: object, expected: object) -> bool:
    """Compare standard factory code and immutable closure values, not names.

    Comparing function objects alone would accept a stateful custom closure. Both the
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
            if not _primitive_key(value) or _typed_value_key(value) != _typed_value_key(standard):
                return False
        elif standard == ti.f32:
            if type(value) is not type(standard) or value != ti.f32:
                return False
        else:
            return False
    return True


@lru_cache(maxsize=4)
def _standard_functions(name: str) -> dict[str, Callable]:
    return make_device_vortex_kernels(name, ti.f32)


def _standard_tree_functions(tree: object) -> bool:
    return all(
        _same_standard_function(getattr(tree, field), _standard_functions(kernel)[function])
        for field, kernel, function in (
            ("_gaussian_q", "GAUSSIAN", "q_"),
            ("_gaussian_zeta", "GAUSSIAN", "zeta_"),
            ("_gaussian_radial_factors", "GAUSSIAN", "radial_factors_"),
            ("_winckelmans_radial_factors", "WINCKELMANS", "radial_factors_"),
        )
    )


def _fields_alive(operator: object, name: str = "_device_fields") -> bool:
    fields = getattr(operator, name, None)
    return fields is not None and fields.tree is not None


def _tables(operator: object, names: Iterable[str]) -> tuple:
    # These are small immutable-by-API operator tables, not particle data.
    # Exact bytes also detect unsupported direct edits instead of trusting a
    # field's identity or a probabilistic checksum.
    result = []
    for name in names:
        array = getattr(operator, name).to_numpy()
        result.append((name, array.dtype.str, array.shape, array.tobytes()))
    return tuple(result)


def _field_layout(operator: object, names: Iterable[str]) -> tuple:
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
        for field in (getattr(operator, name),)
    )


def _constants(module: ModuleType) -> tuple:
    return tuple(
        (name, value)
        for name, value in sorted(vars(module).items())
        if name.isupper() and _primitive_key(value)
    )


def _scalars(operator: object, names: Iterable[str]) -> object:
    values = tuple((name, getattr(operator, name)) for name in names)
    return values if _primitive_key(values) else _UNSUPPORTED


class FMMReuseConditions:
    """Callable conditions provider, deliberately not installed by construction.

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

    def __init__(self, backend: "FMMInduction"):
        self.backend = backend
        self._runtime_program = ti.lang.impl.get_runtime().prog

    def __call__(self) -> "InductionReuseConditions | None":
        if (
            self._runtime_program is None
            or ti.lang.impl.get_runtime().prog is not self._runtime_program
        ):
            return None
        base = self.backend
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
            id(workspace._device_fields.tree),
            id(tree._device_fields.tree),
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
        if not _primitive_key(key):
            return None

        def capture() -> tuple:
            return tuple(getattr(base.diagnostics, name) for name in _RATE_DIAGNOSTICS)

        def restore(rate: tuple) -> None:
            for name, value in zip(_RATE_DIAGNOSTICS, rate, strict=True):
                setattr(base.diagnostics, name, value)
            base._last_tree_key = None
            base._source_moments_ready = False

        return InductionReuseConditions(
            operator_key=key,
            autonomous=True,
            sources_are_read_only=True,
            complete_outputs_are_equivalent=True,
            diagnostics_are_complete=True,
            capture_diagnostics=capture,
            restore_diagnostics=restore,
        )


__all__ = ["FMMReuseConditions"]
