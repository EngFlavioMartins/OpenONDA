"""Infinite-span Gaussian induction for an explicitly two-dimensional flow.

Each particle is a straight infinite z filament represented over ``span`` metres.
Its stored strength is Gamma_z = circulation * span (m³/s), preserving the
ordinary particle storage contract. No finite-span or periodic-image truncation
is used. This model cannot represent three-dimensional cylinder wake modes.
"""

import math

import taichi as ti

from ...kernels.base import make_vortex_kernel


@ti.data_oriented
class PlanarInduction:
    """Gaussian Biot–Savart induction for an infinite, span-invariant z filament.

    Parameters
    ----------
    span : float, default=1.0
        Positive represented span L in m. Stored particle strength Gamma_z
        has units m³/s; filament circulation is Gamma_z/L in m²/s.
    plane_z : float, default=0.0
        Finite source-plane coordinate in m. Source positions must lie
        within 1e-6 m of this plane; target z coordinates are unrestricted.
    spanwise_tolerance : float, default=1e-3
        Positive dimensionless limit used by coupled FVM transfer for
        spanwise velocity and cell-stack variation, relative to its
        freestream/cell-stack velocity scale. This is not a core cutoff.

    Raises
    ------
    ValueError
        If span or tolerance is nonpositive/nonfinite, or plane_z is nonfinite.

    Notes
    -----
    For in-plane displacement r, each source induces
    u = Gamma_z/(2*pi*L) * (1-exp(-|r|²/sigma²))/|r|² * (-r_y,r_x,0).
    The Gaussian core radius sigma is in m. The analytic Jacobian uses
    G[i,j] = du_i/dx_j in s⁻¹. Vortex stretching is identically zero.
    Direct evaluation costs O(N_source*N_target) on the selected Taichi
    backend and uses its accumulator precision. Gaussian kernels alone
    are supported. Selecting this formulation also selects planar GBD
    and coupled renewal; it cannot represent finite ends or 3D wake modes.

    Examples
    --------
    >>> from openonda.vpm import PlanarInduction
    >>> induction = PlanarInduction(span=1.0, plane_z=0.0)
    >>> induction.planar_span
    1.0
    """

    supported_kernels = frozenset({"GAUSSIAN"})
    supported_devices = frozenset({"AUTO", "CPU", "VULKAN", "CUDA", "METAL"})
    supports_gradient = supports_variable_core_radius = supports_f64 = True
    supports_target_fields = device_resident = True
    stretching_scheme = "DIRECT"
    method = "PLANAR"

    def __init__(
        self, *, span: float = 1.0, plane_z: float = 0.0, spanwise_tolerance: float = 1e-3
    ):
        """Store the represented span, source plane and coupled invariance tolerance.

        See the class parameters for units and validation. Device fields are
        allocated only by ``bind``; construction does not initialize a device.
        """
        if not math.isfinite(span) or span <= 0 or not math.isfinite(plane_z):
            raise ValueError("Planar span must be positive and plane_z finite")
        self.planar_span = float(span)
        self.plane_z = float(plane_z)
        if not math.isfinite(spanwise_tolerance) or spanwise_tolerance <= 0:
            raise ValueError("spanwise_tolerance must be finite and positive")
        self.spanwise_tolerance = float(spanwise_tolerance)
        self.kernel = make_vortex_kernel("GAUSSIAN")
        self.physics = None

    def build(self):
        """Return an unbound copy with identical span, plane and tolerance.

        The returned instance owns no device workspace until it is bound.
        """
        return type(self)(
            span=self.planar_span, plane_z=self.plane_z, spanwise_tolerance=self.spanwise_tolerance
        )

    def validate_source_arrays(self, position, strength):
        """Check source geometry and strength without changing the input arrays.

        Parameters
        ----------
        position, strength : array_like, reshaped to (N, 3)
            World-frame positions in m and vector strengths in m³/s. The caller
            owns the common-length and finite-state validation.

        Raises
        ------
        ValueError
            If positions cannot be reshaped, any source lies more than 1e-6 m
            from plane_z, or either transverse strength component is nonzero.

        Notes
        -----
        Conversion may allocate host arrays. No span projection is performed.
        """
        import numpy as np

        position = np.asarray(position).reshape(-1, 3)
        strength = np.asarray(strength).reshape(-1, 3)
        if np.any(np.abs(position[:, 2] - self.plane_z) > 1e-6):
            raise ValueError("Planar source positions must lie at plane_z")
        if np.any(strength[:, :2] != 0):
            raise ValueError("Planar source vortex strength must have only a z component")

    def bind(self, physics, *, kernel=None):
        """Bind the physics owner and allocate diagnostic Taichi work fields.

        Parameters
        ----------
        physics : PhysicsBase
            Initialized owner supplying accumulator dtype and particle capacity.
        kernel : VortexKernel or None
            Optional kernel, which must be Gaussian when supplied.

        Returns
        -------
        PlanarInduction
            This instance, with fresh work fields in the owner's accumulator dtype.

        Raises
        ------
        ValueError
            If a non-Gaussian kernel is supplied.

        Side Effects
        ------------
        Replaces the physics reference and allocates velocity, gradient and
        per-particle integral buffers on the active Taichi device.
        """
        if kernel is not None and kernel.name != "GAUSSIAN":
            raise ValueError("PlanarInduction requires the Gaussian kernel")
        self.physics = physics
        self._dtype = physics.accumulator_dtype
        self.max_n_particles = physics.max_n_particles
        self._dummy_velocity = ti.Vector.field(3, self._dtype, shape=1)
        self._dummy_gradient = ti.Matrix.field(3, 3, self._dtype, shape=1)
        self._energy = ti.field(self._dtype, shape=self.max_n_particles)
        self._enstrophy = ti.field(self._dtype, shape=self.max_n_particles)
        self._filtered_enstrophy = ti.field(self._dtype, shape=self.max_n_particles)
        return self

    @ti.func
    def _pair(self, delta, strength, radius):
        """Return one source's velocity (m/s) and Jacobian (s⁻¹).

        Delta is target minus source in m; strength is Gamma in m³/s and
        radius is sigma in m. Only XY displacement and Gamma_z contribute.
        The small-radius Taylor branch preserves the finite source-centre curl.
        """
        dx, dy = delta[0], delta[1]
        r2 = dx * dx + dy * dy
        inv_sigma2 = 1.0 / (radius * radius)
        q = r2 * inv_sigma2
        # Taylor limits avoid cancellation and include the centre's finite curl.
        f = inv_sigma2 * (1.0 - 0.5 * q + q * q / 6.0 - q**3 / 24.0 + q**4 / 120.0 - q**5 / 720.0)
        df = (
            inv_sigma2
            * inv_sigma2
            * (-0.5 + q / 3.0 - q * q / 8.0 + q**3 / 30.0 - q**4 / 144.0 + q**5 / 840.0)
        )
        if q > 0.05:
            decay = ti.exp(-q)
            f = (1.0 - decay) / r2
            df = ((q + 1.0) * decay - 1.0) / (r2 * r2)
        c = strength[2] / (2.0 * math.pi * self.planar_span)
        velocity = ti.Vector([-c * dy * f, c * dx * f, 0.0])
        gradient = ti.Matrix.zero(self._dtype, 3, 3)
        gradient[0, 0] = -c * dy * 2.0 * dx * df
        gradient[0, 1] = -c * (f + 2.0 * dy * dy * df)
        gradient[1, 0] = c * (f + 2.0 * dx * dx * df)
        gradient[1, 1] = c * dx * 2.0 * dy * df
        return velocity, gradient

    @ti.kernel
    def _evaluate(
        self,
        target: ti.template(),
        source: ti.template(),
        strength: ti.template(),
        radius: ti.template(),
        velocity: ti.template(),
        gradient: ti.template(),
        nt: ti.i32,
        ns: ti.i32,
        do_velocity: ti.template(),
        do_gradient: ti.template(),
        background: ti.template(),
    ):
        """Sum active source fields into requested target device buffers.

        Read source[0:ns] and target[0:nt] in m, strengths in m³/s and core
        radii in m. Write requested velocity (m/s) and G[i,j]=du_i/dx_j (s⁻¹)
        for targets only; capacity tails remain untouched. Background is m/s.
        """
        for i in range(nt):
            u = ti.Vector.zero(self._dtype, 3)
            jac = ti.Matrix.zero(self._dtype, 3, 3)
            for j in range(ns):
                v, g = self._pair(target[i] - source[j], strength[j], radius[j])
                u += v
                if ti.static(do_gradient):
                    jac += g
            if ti.static(do_velocity):
                velocity[i] = u + ti.Vector(background, dt=self._dtype)
            if ti.static(do_gradient):
                gradient[i] = jac

    def evaluate_stage(
        self,
        *,
        position,
        vortex_strength,
        core_radius,
        count,
        velocity_out,
        vortex_strength_rate_out,
        velocity_gradient_out=None,
        strength_rate_enabled=True,
        stage_time=0.0,
    ):
        """Write induced particle velocity and zero planar stretching rates.

        Position, vortex_strength and core_radius are Taichi fields in m,
        m³/s and m, with ``count`` active entries. Velocity output is m/s;
        strength-rate output is m³/s² and is always set to zero. Optional
        velocity_gradient_out receives G[i,j]=du_i/dx_j in s⁻¹. All buffers
        are caller-owned and must have sufficient capacity. Inputs are read-only.

        The autonomous operator ignores stage_time (s) and strength_rate_enabled.
        It excludes freestream; the evolution owner adds that velocity separately.
        """
        del strength_rate_enabled, stage_time
        self.evaluate_targets(
            target_position=position,
            source_position=position,
            source_vortex_strength=vortex_strength,
            source_core_radius=core_radius,
            target_velocity=velocity_out,
            target_velocity_gradient=velocity_gradient_out,
            target_count=count,
            source_count=count,
            include_freestream=False,
            background_velocity=(0.0, 0.0, 0.0),
        )
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
        target_count,
        source_count,
        include_freestream,
        background_velocity,
    ):
        """Evaluate planar velocity and/or its Jacobian into caller-owned fields.

        Parameters
        ----------
        target_position, source_position : Taichi vector fields, shape (capacity, 3)
            World coordinates in m; active prefixes are set by the counts.
        source_vortex_strength : Taichi vector field, shape (capacity, 3)
            Validated source Gamma in m³/s, with only z components.
        source_core_radius : Taichi scalar field, shape (capacity,)
            Positive source Gaussian radii in m. No target-core averaging is used.
        target_velocity : Taichi vector field or None
            Output velocity in m/s; None omits this calculation.
        target_velocity_gradient : Taichi matrix field or None
            Output G[i,j]=du_i/dx_j in s⁻¹; None omits this calculation.
        target_count, source_count : int
            Active target/source extents, excluding unused capacity.
        include_freestream : bool
            Include background_velocity in velocity only.
        background_velocity : iterable of 3 floats or scalar Taichi vector field
            Uniform velocity in m/s. Its z component must be zero when included.

        Raises
        ------
        RuntimeError
            If no physics owner has been bound.
        ValueError
            If counts are invalid, neither output is requested, or the included
            background has a nonzero spanwise component.

        Side Effects
        ------------
        Overwrites requested output prefixes in bound accumulator precision.
        Source fields and inactive output capacity are unchanged.
        """
        if self.physics is None:
            raise RuntimeError("PlanarInduction must be bound before evaluation")
        if not 0 <= source_count <= self.max_n_particles or target_count < 0:
            raise ValueError("Invalid planar induction counts")
        if target_velocity is None and target_velocity_gradient is None:
            raise ValueError("At least one planar output is required")
        background = (
            (
                tuple(background_velocity[None])
                if hasattr(background_velocity, "to_numpy")
                else tuple(background_velocity)
            )
            if include_freestream
            else (0.0, 0.0, 0.0)
        )
        background = tuple(float(value) for value in background)
        if abs(background[2]) > 1e-14:
            raise ValueError("Planar induction requires zero spanwise freestream")
        self._evaluate(
            target_position,
            source_position,
            source_vortex_strength,
            source_core_radius,
            target_velocity if target_velocity is not None else self._dummy_velocity,
            target_velocity_gradient
            if target_velocity_gradient is not None
            else self._dummy_gradient,
            target_count,
            source_count,
            target_velocity is not None,
            target_velocity_gradient is not None,
            background,
        )

    @ti.kernel
    def evaluate_vorticity(
        self,
        target: ti.template(),
        source: ti.template(),
        strength: ti.template(),
        radius: ti.template(),
        out: ti.template(),
        nt: ti.i32,
        ns: ti.i32,
    ):
        """Write Gaussian filament curl for nt active target positions.

        Source and target positions are device vectors in m. Read ns source
        strengths in m³/s and core radii in m. Write out[0:nt] in s⁻¹, with
        zero x/y components, using omega_z=Gamma_z*exp(-r²/sigma²)/(pi*L*sigma²).
        The bound owner's accumulator dtype controls arithmetic and output.
        """
        for i in range(nt):
            omega = ti.cast(0.0, self._dtype)
            for j in range(ns):
                dx, dy = target[i][0] - source[j][0], target[i][1] - source[j][1]
                sigma2 = radius[j] * radius[j]
                omega += (
                    strength[j][2]
                    * ti.exp(-(dx * dx + dy * dy) / sigma2)
                    / (math.pi * self.planar_span * sigma2)
                )
            out[i] = ti.Vector([0.0, 0.0, omega])

    @ti.func
    def _exp1(self, x):
        """Approximate E1(x) for positive dimensionless x on the Taichi device.

        Use a power series at x<=1 and a continued fraction above one.
        The integral kernel calls this only for 1e-5<x<40.
        """
        value = ti.cast(0.0, self._dtype)
        if x <= 1.0:
            term = -x
            series = term
            for k in range(2, 25):
                term *= -x / k
                series += term / k
            value = -0.5772156649015329 - ti.log(x) - series
        else:
            b, c = x + 1.0, ti.cast(1.0e30, self._dtype)
            d = 1.0 / b
            fraction = d
            for k in range(1, 45):
                a = -1.0 * k * k
                b += 2.0
                d = 1.0 / (a * d + b)
                c = b + a / c
                fraction *= c * d
            value = fraction * ti.exp(-x)
        return value

    @ti.kernel
    def _integrals(
        self, position: ti.template(), strength: ti.template(), radius: ti.template(), count: ti.i32
    ):
        """Write active per-particle Gaussian pair integrals over XY times span.

        Read count positions (m), strengths (m³/s) and radii (m) from device
        fields. Energy entries have units m⁵/s²; enstrophy and test-filtered
        enstrophy entries have units m³/s², without a one-half factor.
        Energy entries are gauge terms whose sum is physical only for zero
        net circulation. The diagnostic owner handles divergent total energy.
        """
        for i in range(count):
            energy = ti.cast(0.0, self._dtype)
            enstrophy = ti.cast(0.0, self._dtype)
            filtered_enstrophy = ti.cast(0.0, self._dtype)
            for j in range(count):
                dx, dy = position[i][0] - position[j][0], position[i][1] - position[j][1]
                r2 = dx * dx + dy * dy
                sigma2 = radius[i] * radius[i] + radius[j] * radius[j]
                q = r2 / sigma2
                potential = 0.5 * (ti.log(sigma2) - 0.5772156649015329)
                if q > 1.0e-5:
                    potential = 0.5 * ti.log(r2)
                    if q < 40.0:
                        potential += 0.5 * self._exp1(q)
                else:
                    potential += 0.5 * q - 0.125 * q * q
                pair = strength[i][2] * strength[j][2] / self.planar_span
                energy -= pair * potential / (4.0 * math.pi)
                enstrophy += pair * ti.exp(-q) / (math.pi * sigma2)
                filtered_enstrophy += pair * ti.exp(-0.5 * q) / (2.0 * math.pi * sigma2)
            self._energy[i] = energy
            self._enstrophy[i] = enstrophy
            self._filtered_enstrophy[i] = filtered_enstrophy

    def particle_integrals(self, particles):
        """Return Gaussian pair-energy terms and enstrophy over XY times span.

        Parameters
        ----------
        particles : Particles
            Active positions in m, strengths in m³/s and Gaussian core radii in m.

        Returns
        -------
        energy, enstrophy : numpy.ndarray, shape (N,)
            Independent host arrays in the bound accumulator dtype. Energy
            terms are m⁵/s²; enstrophy terms are m³/s² without a one-half factor.

        Notes
        -----
        Energy entries are finite gauge terms, not individually physical
        energies. Their sum gives unbounded-domain energy only for zero net
        circulation. The calling diagnostics owner reports infinity for
        nonzero net circulation; this low-level method does not perform that test.
        Evaluation costs O(N²), overwrites device diagnostic buffers and does
        not mutate particle state.
        """
        count = len(particles)
        self._integrals(particles.position, particles.vortex_strength, particles.core_radius, count)
        return self._energy.to_numpy()[:count], self._enstrophy.to_numpy()[:count]
