"""Gaussian planar filaments between two parallel impermeable slip walls."""

import math

import numpy as np
import taichi as ti

from .planar import PlanarInduction


@ti.data_oriented
class PlanarChannelInduction(PlanarInduction):
    """Infinite-span Gaussian induction in ``-half_width <= y <= half_width``.

    ``half_width``, ``span`` and ``plane_z`` are in metres. The channel is
    unbounded in x. Analytic infinite point-vortex image families supply the
    harmonic wall correction; the original Gaussian source core is retained.
    Sources must remain at least nine core radii from either wall. Under this
    checked restriction the omitted Gaussian image-core tails are below
    floating-point precision. Targets may lie on the walls.

    Velocity and Jacobian use the parent planar conventions and units. The
    induced normal wall velocity and induced channel volume flux are zero.
    Uniform freestream is added separately, including any prescribed uniform
    transverse startup velocity. Channel energy is finite for nonzero net
    circulation and includes the image contribution. Gaussian enstrophy
    differs from its unbounded integral only by the negligible core tails.
    """

    method = "PLANAR_CHANNEL"

    def __init__(self, *, half_width, span=1.0, plane_z=0.0, spanwise_tolerance=1e-3):
        super().__init__(span=span, plane_z=plane_z, spanwise_tolerance=spanwise_tolerance)
        if not math.isfinite(half_width) or half_width <= 0:
            raise ValueError("Planar channel half width must be finite and positive")
        self.channel_half_width = float(half_width)
        self._image_wave_number = math.pi / (4 * self.channel_half_width)

    def build(self):
        return type(self)(
            half_width=self.channel_half_width,
            span=self.planar_span,
            plane_z=self.plane_z,
            spanwise_tolerance=self.spanwise_tolerance,
        )

    def validate_source_arrays(self, position, strength):
        super().validate_source_arrays(position, strength)
        if np.any(np.abs(np.asarray(position).reshape(-1, 3)[:, 1]) >= self.channel_half_width):
            raise ValueError("Planar channel sources must lie strictly between the slip walls")

    def bind(self, physics, *, kernel=None):
        super().bind(physics, kernel=kernel)
        self._invalid_geometry = ti.field(ti.i32, shape=())
        return self

    @ti.kernel
    def _check_geometry(
        self,
        target: ti.template(),
        source: ti.template(),
        radius: ti.template(),
        nt: ti.i32,
        ns: ti.i32,
    ):
        self._invalid_geometry[None] = 0
        for j in range(ns):
            if self.channel_half_width - ti.abs(source[j][1]) < 9 * radius[j]:
                ti.atomic_max(self._invalid_geometry[None], 1)
        for i in range(nt):
            if ti.abs(target[i][1]) > self.channel_half_width + 1e-6:
                ti.atomic_max(self._invalid_geometry[None], 2)

    def _validate_geometry(self, target, source, radius, nt, ns):
        if not 0 <= ns <= self.max_n_particles or nt < 0:
            raise ValueError("Invalid planar channel induction counts")
        self._check_geometry(target, source, radius, nt, ns)
        invalid = self._invalid_geometry[None]
        if invalid == 1:
            raise ValueError(
                "Planar channel sources must remain nine Gaussian core radii from slip walls"
            )
        if invalid == 2:
            raise ValueError("Planar channel evaluation targets must lie between the slip walls")

    @ti.func
    def _multiply(self, first, second):
        return ti.Vector(
            [
                first[0] * second[0] - first[1] * second[1],
                first[0] * second[1] + first[1] * second[0],
            ]
        )

    @ti.func
    def _coth(self, real, imaginary):
        value = ti.Vector([1.0, 0.0], dt=self._dtype)
        if real < -15.0:
            value[0] = -1.0
        elif real < 15.0:
            positive, negative = ti.exp(2 * real), ti.exp(-2 * real)
            denominator = 0.5 * (positive + negative) - ti.cos(2 * imaginary)
            value = ti.Vector(
                [
                    0.5 * (positive - negative) / denominator,
                    -ti.sin(2 * imaginary) / denominator,
                ]
            )
        return value

    @ti.func
    def _image_pair(self, target, source):
        dx, dy = target[0] - source[0], target[1] - source[1]
        k = self._image_wave_number
        scaled = ti.Vector([k * dx, k * dy], dt=self._dtype)
        positive = ti.Vector.zero(self._dtype, 2)
        derivative = ti.Vector.zero(self._dtype, 2)
        unit = ti.Vector([1.0, 0.0], dt=self._dtype)
        if scaled.norm_sqr() < 1e-6:
            square = self._multiply(scaled, scaled)
            fourth = self._multiply(square, square)
            positive = k * self._multiply(scaled, unit / 3 - square / 45 + 2 * fourth / 945)
            derivative = k * k * (unit / 3 - square / 15 + 2 * fourth / 189)
        else:
            value = self._coth(scaled[0], scaled[1])
            reciprocal = ti.Vector([dx, -dy], dt=self._dtype) / (dx * dx + dy * dy)
            positive = k * value - reciprocal
            derivative = k * k * (unit - self._multiply(value, value)) + self._multiply(
                reciprocal, reciprocal
            )
        mirror = self._coth(k * dx, k * (target[1] - 2 * self.channel_half_width + source[1]))
        return (
            positive - k * mirror,
            derivative - k * k * (unit - self._multiply(mirror, mirror)),
        )

    @ti.kernel
    def _add_images(
        self,
        target: ti.template(),
        source: ti.template(),
        strength: ti.template(),
        velocity: ti.template(),
        gradient: ti.template(),
        nt: ti.i32,
        ns: ti.i32,
        do_velocity: ti.template(),
        do_gradient: ti.template(),
    ):
        for i in range(nt):
            value = ti.Vector.zero(self._dtype, 2)
            derivative = ti.Vector.zero(self._dtype, 2)
            for j in range(ns):
                pair, pair_derivative = self._image_pair(target[i], source[j])
                coefficient = strength[j][2] / (2 * math.pi * self.planar_span)
                value += coefficient * pair
                if ti.static(do_gradient):
                    derivative += coefficient * pair_derivative
            if ti.static(do_velocity):
                velocity[i] += ti.Vector([value[1], value[0], 0.0], dt=self._dtype)
            if ti.static(do_gradient):
                gradient[i] += ti.Matrix(
                    [
                        [derivative[1], derivative[0], 0.0],
                        [derivative[0], -derivative[1], 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dt=self._dtype,
                )

    def evaluate_targets(self, **values):
        if self.physics is not None:
            self._validate_geometry(
                values["target_position"],
                values["source_position"],
                values["source_core_radius"],
                values["target_count"],
                values["source_count"],
            )
        super().evaluate_targets(**values)
        self._add_images(
            values["target_position"],
            values["source_position"],
            values["source_vortex_strength"],
            values["target_velocity"]
            if values["target_velocity"] is not None
            else self._dummy_velocity,
            values["target_velocity_gradient"]
            if values["target_velocity_gradient"] is not None
            else self._dummy_gradient,
            values["target_count"],
            values["source_count"],
            values["target_velocity"] is not None,
            values["target_velocity_gradient"] is not None,
        )

    @ti.func
    def _log_sinh_magnitude(self, real, imaginary):
        value = ti.abs(real) - 0.6931471805599453
        if ti.abs(real) < 15:
            value = 0.5 * ti.log(
                0.25 * (ti.exp(2 * real) + ti.exp(-2 * real) - 2 * ti.cos(2 * imaginary))
            )
        return value

    @ti.func
    def _image_potential(self, target, source):
        dx, dy = target[0] - source[0], target[1] - source[1]
        k = self._image_wave_number
        scaled = ti.Vector([k * dx, k * dy], dt=self._dtype)
        positive = ti.log(k)
        if scaled.norm_sqr() < 1e-6:
            square = self._multiply(scaled, scaled)
            fourth = self._multiply(square, square)
            sixth = self._multiply(square, fourth)
            positive += square[0] / 6 - fourth[0] / 180 + sixth[0] / 2835
        else:
            positive = self._log_sinh_magnitude(scaled[0], scaled[1]) - 0.5 * ti.log(
                dx * dx + dy * dy
            )
        mirror = self._log_sinh_magnitude(
            k * dx, k * (target[1] - 2 * self.channel_half_width + source[1])
        )
        return positive - mirror

    @ti.kernel
    def _add_image_energy(self, position: ti.template(), strength: ti.template(), count: ti.i32):
        for i in range(count):
            energy = ti.cast(0.0, self._dtype)
            for j in range(count):
                energy -= (
                    strength[i][2]
                    * strength[j][2]
                    / self.planar_span
                    * self._image_potential(position[i], position[j])
                    / (4 * math.pi)
                )
            self._energy[i] += energy

    def particle_integrals(self, particles):
        count = len(particles)
        self._validate_geometry(
            particles.position, particles.position, particles.core_radius, count, count
        )
        self._integrals(particles.position, particles.vortex_strength, particles.core_radius, count)
        self._add_image_energy(particles.position, particles.vortex_strength, count)
        return self._energy.to_numpy()[:count], self._enstrophy.to_numpy()[:count]
