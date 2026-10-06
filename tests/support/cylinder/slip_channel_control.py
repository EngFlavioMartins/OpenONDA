"""Device-resident harmonic slip-wall images for autonomous force controls."""

from contextlib import contextmanager
import math

import taichi as ti


@ti.data_oriented
class SlipChannelImages:
    """Add the infinite point-vortex image families at y=+/-H.

    In these experiments sources stay within +/-5 m, H=10 m and Gaussian
    radii are at most 0.04 m. Image-core corrections are below floating-point
    precision. The original free-space Gaussian core is retained exactly.
    """

    def __init__(self, induction, half_width, *, inlet_x=None):
        self.induction = induction
        self.half_width = half_width
        self.wave_number = math.pi / (4 * half_width)
        self.dtype = induction._dtype
        self.has_inlet = inlet_x is not None
        self.inlet_x = 0.0 if inlet_x is None else float(inlet_x)

    @ti.func
    def multiply(self, first, second):
        return ti.Vector(
            [
                first[0] * second[0] - first[1] * second[1],
                first[0] * second[1] + first[1] * second[0],
            ]
        )

    @ti.func
    def coth(self, real, imaginary):
        value = ti.Vector([1.0, 0.0], dt=self.dtype)
        if real < -15.0:
            value[0] = -1.0
        elif real < 15.0:
            positive, negative = ti.exp(2 * real), ti.exp(-2 * real)
            denominator = 0.5 * (positive + negative) - ti.cos(2 * imaginary)
            value = ti.Vector(
                [0.5 * (positive - negative) / denominator, -ti.sin(2 * imaginary) / denominator]
            )
        return value

    @ti.func
    def pair(self, target, source):
        dx, dy = target[0] - source[0], target[1] - source[1]
        k = self.wave_number
        scaled = ti.Vector([k * dx, k * dy], dt=self.dtype)
        positive = ti.Vector.zero(self.dtype, 2)
        derivative = ti.Vector.zero(self.dtype, 2)
        unit = ti.Vector([1.0, 0.0], dt=self.dtype)
        if scaled.norm_sqr() < 1e-6:
            square = self.multiply(scaled, scaled)
            fourth = self.multiply(square, square)
            positive = k * self.multiply(scaled, unit / 3 - square / 45 + 2 * fourth / 945)
            derivative = k * k * (unit / 3 - square / 15 + 2 * fourth / 189)
        else:
            value = self.coth(scaled[0], scaled[1])
            reciprocal = ti.Vector([dx, -dy], dt=self.dtype) / (dx * dx + dy * dy)
            positive = k * value - reciprocal
            derivative = k * k * (unit - self.multiply(value, value)) + self.multiply(
                reciprocal, reciprocal
            )
        mirror = self.coth(k * dx, k * (target[1] - 2 * self.half_width + source[1]))
        correction = positive - k * mirror
        correction_derivative = derivative - k * k * (unit - self.multiply(mirror, mirror))
        if ti.static(self.has_inlet):
            reflected_dx = target[0] - 2 * self.inlet_x + source[0]
            first = self.coth(k * reflected_dx, k * dy)
            second = self.coth(k * reflected_dx, k * (target[1] - 2 * self.half_width + source[1]))
            correction -= k * (first - second)
            correction_derivative -= (
                k * k * (self.multiply(second, second) - self.multiply(first, first))
            )
        return correction, correction_derivative

    @ti.kernel
    def add_images(
        self,
        targets: ti.template(),
        sources: ti.template(),
        strengths: ti.template(),
        velocities: ti.template(),
        gradients: ti.template(),
        target_count: ti.i32,
        source_count: ti.i32,
        do_velocity: ti.template(),
        do_gradient: ti.template(),
    ):
        for target in range(target_count):
            value = ti.Vector.zero(self.dtype, 2)
            derivative = ti.Vector.zero(self.dtype, 2)
            for source in range(source_count):
                pair, pair_derivative = self.pair(targets[target], sources[source])
                coefficient = strengths[source][2] / (2 * math.pi * self.induction.planar_span)
                value += coefficient * pair
                if ti.static(do_gradient):
                    derivative += coefficient * pair_derivative
            if ti.static(do_velocity):
                velocities[target] += ti.Vector([value[1], value[0], 0.0], dt=self.dtype)
            if ti.static(do_gradient):
                gradients[target] += ti.Matrix(
                    [
                        [derivative[1], derivative[0], 0.0],
                        [derivative[0], -derivative[1], 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dt=self.dtype,
                )

    def evaluate(self, values):
        velocity, gradient = values["target_velocity"], values["target_velocity_gradient"]
        self.add_images(
            values["target_position"],
            values["source_position"],
            values["source_vortex_strength"],
            velocity if velocity is not None else self.induction._dummy_velocity,
            gradient if gradient is not None else self.induction._dummy_gradient,
            values["target_count"],
            values["source_count"],
            velocity is not None,
            gradient is not None,
        )


@contextmanager
def slip_channel_induction(owner, half_width, measurements, *, inlet_x=None):
    induction = owner.vpm_solver.induction
    original = induction.evaluate_targets
    images = SlipChannelImages(induction, half_width, inlet_x=inlet_x)
    measurements.update(
        half_width_m=half_width,
        calls=0,
        inlet_normal_velocity_plane_m=inlet_x,
        image_core_assumption="Sources in +/-5 m and Gaussian radius <=0.04 m; images outside +/-10 m",
    )

    def evaluate(**values):
        original(**values)
        images.evaluate(values)
        measurements["calls"] += 1

    induction.evaluate_targets = evaluate
    try:
        yield
    finally:
        induction.evaluate_targets = original
