"""Barnes--Hut induction evaluator behind the common VPM stage contract."""

from typing import Self

import numpy as np
import taichi as ti

from ....kernels.base import RadialVortexKernel, make_vortex_kernel
from ..base import _STRETCHING_MODES, normalize_stretching_scheme
from ..stretching import stretching_rate

_TREECODE_THETA = 0.1
_TREECODE_MULTIPOLE_ORDER = 1
_TREECODE_SORT_PARTICLE_TARGETS = False
_TREECODE_TRAVERSAL_BLOCK_DIM = 128


@ti.kernel
def _rate_from_gradient(
    gradient: ti.template(),
    strength: ti.template(),
    output: ti.template(),
    stretching_mode: ti.i32,
    count: ti.i32,
):
    """Contract the hierarchical velocity gradient using the selected scheme."""
    for i in range(count):
        output[i] = stretching_rate(gradient[i], strength[i], stretching_mode)


@ti.data_oriented
class TreecodeInduction:
    """Evaluate stage velocity and a consistent gradient-derived rate hierarchically.

    The LBVH workspace is rebuilt from every supplied stage state.  This keeps
    geometry and strength moments synchronized when an RK stage changes either
    position or vortex strength. The selected direct, transposed, or mixed
    stretching rate is contracted from the same hierarchical velocity
    gradient used for optional diagnostics; no direct pairwise rate fallback
    is hidden behind the treecode interface.
    """

    supported_kernels = frozenset({"GAUSSIAN", "WINCKELMANS"})
    supported_devices = frozenset({"AUTO", "CPU", "VULKAN", "CUDA", "METAL"})
    supports_gradient = True
    supports_variable_core_radius = True
    supports_target_fields = True
    # The LBVH fields are intentionally f32 to keep the device workspace
    # bounded; reject a nominal f64 case at the immutable configuration edge.
    supports_f64 = False
    device_resident = True

    def __init__(
        self,
        *,
        stretching_scheme: str = "TRANSPOSED",
        theta: float = _TREECODE_THETA,
        multipole_order: int = _TREECODE_MULTIPOLE_ORDER,
    ) -> None:
        """Create an unbound LBVH/Barnes--Hut induction evaluator.

        Parameters
        ----------
        stretching_scheme : {"DIRECT", "TRANSPOSED", "MIXED"}, default="TRANSPOSED"
            Formulation used to contract the hierarchical velocity gradient
            into ``dGamma/dt``. The tree traversal and induced velocity are
            unchanged by this choice.
        theta : float, default=0.1
            Barnes--Hut opening angle in ``(0, 2)``. Smaller values tighten
            the far-field approximation at increased traversal cost.
        multipole_order : {1, 2, 3}, default=1
            Order of the far-field expansion. Keep this choice fixed when
            comparing time integration or stabilization methods.

        Notes
        -----
        The production defaults use ``theta=0.1``, first-order multipoles, and
        a fixed f32 workspace. This backend supports Gaussian and Winckelmans
        kernels and rejects f64 configurations during case validation.
        Call :meth:`build` and then :meth:`bind` before evaluation.

        Raises
        ------
        ValueError
            If the stretching formulation or tree approximation is unsupported.
        """
        stretching_scheme = normalize_stretching_scheme(stretching_scheme)
        if not 0.0 < float(theta) < 2.0:
            raise ValueError("treecode theta must be in (0, 2)")
        if multipole_order not in (1, 2, 3):
            raise ValueError("treecode multipole_order must be 1, 2, or 3")
        self.method = "TREECODE"
        self.stretching_scheme = stretching_scheme
        self._stretching_mode = _STRETCHING_MODES[stretching_scheme]
        self.kernel = make_vortex_kernel("GAUSSIAN")
        self.physics = None
        self.theta = float(theta)
        self.multipole_order = int(multipole_order)
        self.sort_particle_targets = _TREECODE_SORT_PARTICLE_TARGETS
        self.traversal_block_dim = _TREECODE_TRAVERSAL_BLOCK_DIM
        self.max_n_particles = 1
        self.diagnostics = {
            "stretching_scheme": self.stretching_scheme,
            "stage_evaluations": 0,
            "gradient_evaluations": 0,
            "hierarchical_strength_rates": 0,
            "direct_strength_rate_fallbacks": 0,
        }

    @classmethod
    def _for_testing(
        cls,
        *,
        theta: float = _TREECODE_THETA,
        multipole_order: int = _TREECODE_MULTIPOLE_ORDER,
        sort_particle_targets: bool = _TREECODE_SORT_PARTICLE_TARGETS,
        traversal_block_dim: int = _TREECODE_TRAVERSAL_BLOCK_DIM,
        stretching_scheme: str = "TRANSPOSED",
    ) -> Self:
        """Construct a tuned evaluator for controlled qualification studies.

        Parameters
        ----------
        theta : float, default=0.1
            Barnes--Hut opening angle. Smaller values improve accuracy and
            increase traversal work; it must lie in ``(0, 2)``.
        multipole_order : {1, 2, 3}, default=1
            Truncated multipole order used by the tree workspace.
        sort_particle_targets : bool, default=False
            Whether particle targets are reordered for traversal locality.
        traversal_block_dim : int, default=128
            Device traversal block size; zero leaves the runtime default.
        stretching_scheme : {"DIRECT", "TRANSPOSED", "MIXED"}, default="TRANSPOSED"
            Strength-rate contraction formulation.

        Returns
        -------
        TreecodeInduction
            Unbound evaluator. This helper is private API intended for
            accuracy/performance tests, not case configuration.

        Raises
        ------
        ValueError
            If a tree parameter is outside its supported range.
        """
        if not 0.0 < float(theta) < 2.0:
            raise ValueError("treecode theta must be in (0, 2)")
        if int(multipole_order) not in (1, 2, 3):
            raise ValueError("treecode multipole_order must be 1, 2, or 3")
        if int(traversal_block_dim) < 0:
            raise ValueError("treecode traversal_block_dim must be non-negative")
        instance = cls(stretching_scheme=stretching_scheme)
        instance.theta = float(theta)
        instance.multipole_order = int(multipole_order)
        instance.sort_particle_targets = bool(sort_particle_targets)
        instance.traversal_block_dim = int(traversal_block_dim)
        return instance

    def build(self) -> Self:
        """Return a fresh unbound evaluator preserving tree settings."""
        return type(self)._for_testing(
            theta=self.theta,
            multipole_order=self.multipole_order,
            sort_particle_targets=self.sort_particle_targets,
            traversal_block_dim=self.traversal_block_dim,
            stretching_scheme=self.stretching_scheme,
        )

    def bind(self, physics: object, *, kernel: RadialVortexKernel | None = None) -> Self:
        """Bind this evaluator to a physics workspace and configure LBVH.

        Parameters
        ----------
        physics : PhysicsEngine
            Workspace owning f32 tree fields and device kernels.
        kernel : RadialVortexKernel or None, default=None
            Optional Gaussian or Winckelmans kernel selected for this run.

        Returns
        -------
        TreecodeInduction
            This bound evaluator.

        Notes
        -----
        Binding stores the workspace reference and applies the opening angle,
        multipole, sorting, and traversal settings to it. It does not build a
        tree; every stage rebuilds the tree from the supplied stage state.
        """
        self.physics = physics
        physics.configure_velocity(
            "TREECODE",
            self.theta,
            multipole_order=self.multipole_order,
            sort_particle_targets=self.sort_particle_targets,
            traversal_block_dim=self.traversal_block_dim,
        )
        if kernel is not None:
            self.kernel = kernel
        self.max_n_particles = physics.max_n_particles
        return self

    def evaluate_stage(
        self,
        *,
        position: object,
        vortex_strength: object,
        core_radius: object,
        count: int,
        velocity_out: object,
        vortex_strength_rate_out: object,
        velocity_gradient_out: object | None = None,
        strength_rate_enabled: bool = True,
        stage_time: float = 0.0,
    ) -> None:
        """Evaluate one complete particle stage with a rebuilt LBVH tree.

        Parameters
        ----------
        position : object
            Stage positions, logical shape ``(count, 3)``, in m.
        vortex_strength : object
            Stage particle-strength vectors ``Gamma``, shape ``(count, 3)``, in m³/s.
        core_radius : object
            Stage core radii, shape ``(count,)``, in m.
        count : int
            Active source/target prefix. Work is approximately O(count log N)
            for well-distributed particles, with accuracy controlled by
            ``theta``.
        velocity_out : object
            Output velocity field, shape ``(count, 3)``, in m/s.
        vortex_strength_rate_out : object
            Output stretching rate, shape ``(count, 3)``, in m³/s².
        velocity_gradient_out : object or None, default=None
            Optional gradient output, shape ``(count, 3, 3)``, in 1/s.
        strength_rate_enabled : bool, default=True
            If true, contract the same hierarchical gradient used by the
            backend. If false, explicitly zero the rate field.
        stage_time : float, default=0.0
            Stage time in seconds; accepted for the common contract and not
            used by the autonomous treecode.

        Raises
        ------
        RuntimeError
            If the evaluator is not bound.
        ValueError
            If ``count`` is outside the configured particle capacity.

        Notes
        -----
        The tree is rebuilt for each call so changed RK-stage positions,
        strengths, and core radii cannot reuse stale moments. The source fields
        are not mutated; output fields and backend diagnostics are.
        """
        del stage_time
        if self.physics is None:
            raise RuntimeError("TreecodeInduction must be bound before evaluation")
        count = int(count)
        if count < 0 or count > self.max_n_particles:
            raise ValueError(f"stage count {count} exceeds treecode capacity")
        if count == 0:
            return

        self.diagnostics["stage_evaluations"] += 1

        tree = self.physics._get_or_create_treecode(count, self.theta)
        tree.build(position, vortex_strength, core_radius, count)
        self.physics._target_tree_key = None
        self.physics.configure_velocity(
            "TREECODE",
            self.theta,
            multipole_order=self.multipole_order,
            sort_particle_targets=self.sort_particle_targets,
            traversal_block_dim=self.traversal_block_dim,
        )
        if strength_rate_enabled or velocity_gradient_out is not None:
            tree.compute_velocity_and_gradient_gpu()
            self.diagnostics["gradient_evaluations"] += 1
        else:
            tree.compute_velocities_gpu(background_field=self.physics._zero_velocity)
        self.physics._copy_vec3(tree.velocity, velocity_out, count)
        if velocity_gradient_out is not None:
            self.physics._copy_mat3(tree.velocity_gradient, velocity_gradient_out, count)

        if strength_rate_enabled:
            _rate_from_gradient(
                tree.velocity_gradient,
                tree.vortex_strength,
                vortex_strength_rate_out,
                self._stretching_mode,
                count,
            )
            self.diagnostics["hierarchical_strength_rates"] += 1
        else:
            self.physics._zero_vec3_field(vortex_strength_rate_out, count)

    def evaluate_targets(
        self,
        *,
        target_position,
        source_position,
        source_vortex_strength,
        source_core_radius,
        target_velocity,
        target_velocity_gradient,
        target_count: int,
        source_count: int,
        include_freestream: bool,
        background_velocity,
    ) -> None:
        """Evaluate target velocity and/or gradient with the LBVH operator.

        Parameters
        ----------
        target_position : object
            Target coordinates, shape ``(target_count, 3)``, in m.
        source_position, source_vortex_strength, source_core_radius : object
            Source fields with shapes ``(source_count, 3)``, ``(source_count,
            3)``, and ``(source_count,)`` in m, m³/s, and m.
        target_velocity : object or None
            Optional output, shape ``(target_count, 3)``, in m/s.
        target_velocity_gradient : object or None
            Optional output, shape ``(target_count, 3, 3)``, in 1/s.
        target_count, source_count : int
            Active target and source counts; inactive capacity entries are
            ignored.
        include_freestream : bool
            Add ``background_velocity`` to velocity output when true.
        background_velocity : object
            Three-vector freestream in m/s.

        Raises
        ------
        RuntimeError
            If the evaluator is not bound.

        Notes
        -----
        The source tree is rebuilt for every call. Target arrays are copied
        through host memory by the current implementation, so this operation
        is not fully device-resident even though the tree traversal is.
        """
        if self.physics is None:
            raise RuntimeError("TreecodeInduction must be bound before target evaluation")
        target_count = int(target_count)
        source_count = int(source_count)
        tree = self.physics._get_or_create_treecode(max(target_count, source_count), self.theta)
        tree.build(source_position, source_vortex_strength, source_core_radius, source_count)
        self.physics._target_tree_key = None
        target_np = self.physics._download_vector_field(target_position, target_count)
        background_np = None
        if include_freestream:
            background_np = np.asarray(
                [
                    background_velocity[None][0],
                    background_velocity[None][1],
                    background_velocity[None][2],
                ],
                dtype=np.float32,
            )
        if target_velocity is not None and target_velocity_gradient is not None:
            velocity, gradient = tree.compute_target_velocity_and_gradients(
                target_np, background_np
            )
            self.physics._upload_vector_array(velocity, target_velocity, target_count)
            self.physics._upload_matrix_array(gradient, target_velocity_gradient, target_count)
        elif target_velocity is not None:
            velocity = tree.compute_target_velocity(target_np, background_np)
            self.physics._upload_vector_array(velocity, target_velocity, target_count)
        elif target_velocity_gradient is not None:
            gradient = tree.compute_target_velocity_gradient(target_np)
            self.physics._upload_matrix_array(gradient, target_velocity_gradient, target_count)


__all__ = ["TreecodeInduction"]
