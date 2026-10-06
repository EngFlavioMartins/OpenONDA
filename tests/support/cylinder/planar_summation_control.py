"""Separate planar pair arithmetic from summation error in private controls."""

from contextlib import contextmanager
import math

import taichi as ti


@ti.data_oriented
class AccuratePlanarSums:
    def __init__(self, induction):
        self.induction = induction

    @ti.kernel
    def evaluate(
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
        background: ti.types.vector(3, ti.f64),
    ):
        for i in range(nt):
            u = ti.Vector.zero(ti.f64, 3)
            jacobian = ti.Matrix.zero(ti.f64, 3, 3)
            for j in range(ns):
                pair, derivative = self.induction._pair(
                    target[i] - source[j], strength[j], radius[j]
                )
                u += ti.cast(pair, ti.f64)
                if ti.static(do_gradient):
                    jacobian += ti.cast(derivative, ti.f64)
            if ti.static(do_velocity):
                velocity[i] = ti.cast(u + background, velocity.dtype)
            if ti.static(do_gradient):
                gradient[i] = ti.cast(jacobian, gradient.dtype)

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
            value = ti.Vector.zero(ti.f64, 2)
            derivative = ti.Vector.zero(ti.f64, 2)
            for j in range(ns):
                pair, pair_derivative = self.induction._image_pair(target[i], source[j])
                coefficient = strength[j][2] / (2 * math.pi * self.induction.planar_span)
                value += ti.cast(coefficient * pair, ti.f64)
                if ti.static(do_gradient):
                    derivative += ti.cast(coefficient * pair_derivative, ti.f64)
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
def accurate_planar_summation(owner, measurements):
    induction = owner.vpm_solver.induction
    evaluator = AccuratePlanarSums(induction)
    original = induction._evaluate
    images = getattr(induction, "_add_images", None)
    induction._evaluate = evaluator.evaluate
    if images is not None:
        induction._add_images = evaluator.add_images
    measurements.update(
        pair_arithmetic=owner.vpm_solver.precision,
        summation="f64",
        output_storage=owner.vpm_solver.precision,
    )
    try:
        yield
    finally:
        induction._evaluate = original
        if images is not None:
            induction._add_images = images
