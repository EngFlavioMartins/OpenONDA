"""Private cancellation-safe channel image control for f32 arithmetic."""

from contextlib import contextmanager
import math

import taichi as ti

from .slip_channel_control import SlipChannelImages


@ti.data_oriented
class StableChannelImages(SlipChannelImages):
    def __init__(self, induction, half_width, *, accurate_summation=False):
        super().__init__(induction, half_width)
        self.sum_dtype = ti.f64 if accurate_summation else self.dtype

    @ti.func
    def coth(self, real, imaginary):
        value = ti.Vector([1.0, 0.0], dt=self.dtype)
        if real < -15:
            value[0] = -1
        elif real < 15:
            sinh = ti.cast(0.0, self.dtype)
            cosh = ti.cast(0.0, self.dtype)
            if ti.abs(real) < 0.1:
                square = real * real
                sinh = real * (1 + square / 6 + square * square / 120 + square**3 / 5040)
                cosh = 1 + square / 2 + square * square / 24 + square**3 / 720 + square**4 / 40320
            else:
                positive, negative = ti.exp(real), ti.exp(-real)
                sinh, cosh = 0.5 * (positive - negative), 0.5 * (positive + negative)
            sine = ti.sin(imaginary)
            denominator = 2 * (sinh * sinh + sine * sine)
            value = ti.Vector([2 * sinh * cosh / denominator, -ti.sin(2 * imaginary) / denominator])
        return value

    @ti.func
    def pair(self, target, source):
        dx, dy = target[0] - source[0], target[1] - source[1]
        k = self.wave_number
        scaled = ti.Vector([k * dx, k * dy], dt=self.dtype)
        positive = ti.Vector.zero(self.dtype, 2)
        derivative = ti.Vector.zero(self.dtype, 2)
        unit = ti.Vector([1.0, 0.0], dt=self.dtype)
        if scaled.norm_sqr() < 0.25:
            square = self.multiply(scaled, scaled)
            fourth = self.multiply(square, square)
            sixth = self.multiply(square, fourth)
            eighth = self.multiply(fourth, fourth)
            positive = k * self.multiply(
                scaled,
                unit / 3 - square / 45 + 2 * fourth / 945 - sixth / 4725 + 2 * eighth / 93555,
            )
            derivative = (
                k
                * k
                * (unit / 3 - square / 15 + 2 * fourth / 189 - sixth / 675 + 2 * eighth / 10395)
            )
        else:
            value = self.coth(scaled[0], scaled[1])
            reciprocal = ti.Vector([dx, -dy], dt=self.dtype) / (dx * dx + dy * dy)
            positive = k * value - reciprocal
            derivative = k * k * (unit - self.multiply(value, value)) + self.multiply(
                reciprocal, reciprocal
            )
        mirror = self.coth(k * dx, k * (target[1] - 2 * self.half_width + source[1]))
        return positive - k * mirror, derivative - k * k * (unit - self.multiply(mirror, mirror))

    @ti.kernel
    def add_images(
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
            value = ti.Vector.zero(self.sum_dtype, 2)
            derivative = ti.Vector.zero(self.sum_dtype, 2)
            for j in range(ns):
                pair, pair_derivative = self.pair(target[i], source[j])
                coefficient = strength[j][2] / (2 * math.pi * self.induction.planar_span)
                value += ti.cast(coefficient * pair, self.sum_dtype)
                if ti.static(do_gradient):
                    derivative += ti.cast(coefficient * pair_derivative, self.sum_dtype)
            if ti.static(do_velocity):
                velocity[i] += ti.cast(ti.Vector([value[1], value[0], 0.0]), velocity.dtype)
            if ti.static(do_gradient):
                gradient[i] += ti.cast(
                    ti.Matrix(
                        [
                            [derivative[1], derivative[0], 0.0],
                            [derivative[0], -derivative[1], 0.0],
                            [0.0, 0.0, 0.0],
                        ]
                    ),
                    gradient.dtype,
                )


@contextmanager
def stable_channel_images(owner, measurements, *, accurate_summation=False):
    induction = owner.vpm_solver.induction
    images = StableChannelImages(
        induction, induction.channel_half_width, accurate_summation=accurate_summation
    )
    original = induction._add_images
    induction._add_images = images.add_images
    measurements.update(
        pair_arithmetic=owner.vpm_solver.precision,
        taylor_radius_dimensionless=0.5,
        summation="f64" if accurate_summation else owner.vpm_solver.precision,
    )
    try:
        yield
    finally:
        induction._add_images = original
