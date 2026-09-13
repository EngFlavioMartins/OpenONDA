"""Direct O(N²) induction adapter."""

from __future__ import annotations

from typing import Self

import taichi as ti

from ...kernels.base import RadialVortexKernel, make_vortex_kernel
from .base import _STRETCHING_MODES, normalize_stretching_scheme


@ti.data_oriented
class DirectInduction:
    """Adapt the existing exact direct kernels to :class:`InductionMethod`.

    The adapter owns no particle state.  It receives the complete temporary
    stage fields on every call and writes only the caller-provided outputs.
    Direct, transposed, and mixed stretching use the same pair walk as
    velocity evaluation. The backend choice does not select the formulation.
    """

    supported_kernels = frozenset(
        {"GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"}
    )
    supported_devices = frozenset({"AUTO", "CPU", "VULKAN", "CUDA", "METAL"})
    supports_gradient = True
    supports_variable_core_radius = True
    supports_f64 = True
    device_resident = True
    supports_target_fields = True

    def __init__(self, *, stretching_scheme: str = "TRANSPOSED") -> None:
        """Create an unbound exact-induction backend.

        Parameters
        ----------
        stretching_scheme : {"DIRECT", "TRANSPOSED", "MIXED"}, default="TRANSPOSED"
            Discrete formulation used for ``dGamma/dt``. ``DIRECT`` uses
            the velocity-gradient contraction ``G @ Gamma``;
            ``TRANSPOSED`` uses ``G.T @ Gamma``; and ``MIXED`` averages the
            two. The option changes strength stretching only, not the induced
            velocity.

        Notes
        -----
        Construction allocates no device fields and is independent of a
        particle container. Call :meth:`build` for a case-owned instance and
        :meth:`bind` before evaluating stages or targets.

        Raises
        ------
        ValueError
            If ``stretching_scheme`` is unsupported.
        """
        self.stretching_scheme = normalize_stretching_scheme(stretching_scheme)
        self._stretching_mode = _STRETCHING_MODES[self.stretching_scheme]
        self.method = "DIRECT"
        self.kernel = make_vortex_kernel("GAUSSIAN")
        self.physics = None
        self.max_n_particles = 1
        self._strain_rate = None

    def build(self) -> Self:
        """Return a fresh unbound evaluator with the same formulation.

        Returns
        -------
        DirectInduction
            Independent backend instance. No physics workspace is bound until
            :meth:`bind` is called.
        """
        return type(self)(stretching_scheme=self.stretching_scheme)

    def bind(self, physics: object, *, kernel: RadialVortexKernel | None = None) -> Self:
        """Bind the evaluator to one :class:`PhysicsEngine` workspace.

        Parameters
        ----------
        physics : PhysicsEngine
            Runtime workspace owning the Taichi kernels and accumulator dtype.
            Its ``max_n_particles`` determines the backend capacity.
        kernel : RadialVortexKernel or None, default=None
            Optional radial kernel. When omitted, the default Gaussian kernel
            remains active. The kernel must be compatible with the workspace's
            precision and device.

        Returns
        -------
        DirectInduction
            This bound evaluator, allowing fluent setup.

        Raises
        ------
        ValueError
            If the physics workspace has a non-positive particle capacity.

        Notes
        -----
        Binding mutates this runtime object: it stores the physics reference,
        selects direct velocity kernels, and allocates a per-particle strain
        tensor used when gradients are requested.
        """
        self.physics = physics
        physics.configure_velocity("DIRECT")
        if kernel is not None:
            self.kernel = kernel
        capacity = physics.max_n_particles
        if capacity < 1:
            raise ValueError("max_n_particles must be positive")
        self.max_n_particles = capacity
        self._strain_rate = ti.Matrix.field(
            3, 3, dtype=physics.accumulator_dtype, shape=(capacity,)
        )
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
        """Evaluate velocity, stretching, and optional gradient for one stage.

        Parameters
        ----------
        position : object
            Stage source positions, logical shape ``(count, 3)``, in m.
        vortex_strength : object
            Stage particle-strength vectors ``Gamma = omega V``, shape ``(count, 3)``,
            in m³/s.
        core_radius : object
            Stage core radii, shape ``(count,)``, in m.
        count : int
            Active prefix length. The direct walk costs O(count²).
        velocity_out : object
            Caller-owned output field, shape ``(count, 3)``, in m/s.
        vortex_strength_rate_out : object
            Caller-owned output field, shape ``(count, 3)``, in m³/s².
        velocity_gradient_out : object or None, default=None
            Optional caller-owned output tensor, shape ``(count, 3, 3)``, in
            1/s. Supplying it triggers a second direct gradient pass.
        strength_rate_enabled : bool, default=True
            If false, the rate field is explicitly zeroed while velocity and a
            requested gradient are still evaluated.
        stage_time : float, default=0.0
            Stage time in seconds. The direct kernel is autonomous; the value
            is accepted for backend compatibility and is not otherwise used.

        Raises
        ------
        RuntimeError
            If :meth:`bind` has not been called.
        ValueError
            If ``count`` is outside the bound capacity.

        Notes
        -----
        Only the supplied output fields and the backend's temporary strain
        workspace are mutated. The source stage fields are read-only here.
        """
        del stage_time
        if self.physics is None:
            raise RuntimeError("DirectInduction must be bound to a PhysicsEngine before evaluation")
        count = int(count)
        if count < 0 or count > self.max_n_particles:
            raise ValueError(
                f"stage count {count} exceeds induction capacity {self.max_n_particles}"
            )
        if count == 0:
            return

        if strength_rate_enabled and velocity_gradient_out is None:
            self.physics.compute_velocity_and_stretching_rate_kernel(
                position,
                vortex_strength,
                core_radius,
                velocity_out,
                vortex_strength_rate_out,
                self.physics._zero_velocity,
                self._stretching_mode,
                count,
            )
        else:
            self.physics.compute_velocities_kernel(
                position,
                vortex_strength,
                core_radius,
                velocity_out,
                self.physics._zero_velocity,
                count,
            )
        if not strength_rate_enabled:
            self.physics._zero_vec3_field(vortex_strength_rate_out, count)
        elif velocity_gradient_out is not None:
            for start in range(0, count, 4096):
                target_count = min(4096, count - start)
                self.physics.compute_stretching_rate_batch_kernel(
                    position,
                    vortex_strength,
                    core_radius,
                    vortex_strength_rate_out,
                    self._stretching_mode,
                    start,
                    target_count,
                    count,
                )
        if velocity_gradient_out is not None:
            self.physics.compute_velocity_gradients_kernel(
                position,
                vortex_strength,
                core_radius,
                velocity_gradient_out,
                self._strain_rate,
                count,
            )

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
        """Evaluate direct induction at arbitrary target points.

        Parameters
        ----------
        target_position : object
            Target coordinates, logical shape ``(target_count, 3)``, in m.
        source_position, source_vortex_strength, source_core_radius : object
            Source fields with shapes ``(source_count, 3)``, ``(source_count,
            3)``, and ``(source_count,)`` in m, m³/s, and m.
        target_velocity : object or None
            Optional output field of shape ``(target_count, 3)`` in m/s.
        target_velocity_gradient : object or None
            Optional output field of shape ``(target_count, 3, 3)`` in 1/s.
        target_count, source_count : int
            Active target/source prefix lengths. Cost is O(target_count ×
            source_count).
        include_freestream : bool
            Whether to add ``background_velocity`` to each velocity result.
        background_velocity : object
            Three-vector freestream in m/s.

        Raises
        ------
        RuntimeError
            If :meth:`bind` has not been called.
        """
        if self.physics is None:
            raise RuntimeError("DirectInduction must be bound before target evaluation")
        if target_velocity is not None:
            self.physics.compute_target_velocity_kernel(
                target_position,
                source_position,
                source_vortex_strength,
                source_core_radius,
                target_velocity,
                background_velocity if include_freestream else self.physics._zero_velocity,
                int(target_count),
                int(source_count),
            )
        if target_velocity_gradient is not None:
            self.physics.compute_target_velocity_gradient_kernel(
                target_position,
                source_position,
                source_vortex_strength,
                source_core_radius,
                target_velocity_gradient,
                int(target_count),
                int(source_count),
            )


__all__ = ["DirectInduction"]
