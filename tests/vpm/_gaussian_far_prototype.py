"""Unwired Gaussian arithmetic-underflow shortcut and same-pair benchmark.

The bounds are about floating-point arithmetic, not a blob approximation.
No production factory or induction method imports this prototype.
"""

import math

import numpy as np
import taichi as ti

from source.solvers.vpm.kernels.gaussian import create_gaussian_kernels


def shortcut_functions(dtype):
    original = create_gaussian_kernels(dtype)
    original_q, original_zeta = original["q_"], original["zeta_"]
    threshold = 11.0 if dtype == ti.f32 else 28.0
    # Keep (2/sqrt(pi))*rho finite in the original multiplication order.
    # Merely requiring finite rho would incorrectly turn some NaNs into q_inf.
    # The same exact binary constant also survives a f32-default runtime
    # evaluating explicit f64 fields. A Python 2**1023 literal can otherwise
    # become infinity before promotion and accidentally admit infinite rho.
    largest_safe = 2.0**120
    q_infinity = 1.0 / (4.0 * math.pi)

    @ti.func
    def q(density):
        result = ti.cast(0.0, dtype)
        if density >= threshold and density <= largest_safe:
            result = ti.cast(q_infinity, dtype)
        else:
            result = original_q(density)
        return result

    @ti.func
    def zeta(density):
        result = ti.cast(0.0, dtype)
        if density >= threshold and density <= largest_safe:
            result = ti.cast(0.0, dtype)
        else:
            result = original_zeta(density)
        return result

    return original_q, original_zeta, q, zeta


def adversarial_densities(numpy_dtype):
    info = np.finfo(numpy_dtype)
    threshold = numpy_dtype(11 if numpy_dtype == np.float32 else 28)
    upper = numpy_dtype(2.0 ** (127 if numpy_dtype == np.float32 else 1023))
    values = [0.0, -0.0, 1.0, -1.0, 6.0, -6.0, np.inf, -np.inf, np.nan, -np.nan]
    for centre in (
        threshold,
        numpy_dtype(2.0**120),
        upper,
        numpy_dtype(float(info.max) / (2.0 / math.sqrt(math.pi))),
        numpy_dtype(info.max),
        numpy_dtype(info.tiny),
    ):
        below = above = centre
        for _ in range(32):
            values.extend((below, above, -below, -above))
            with np.errstate(over="ignore", invalid="ignore"):
                below = np.nextafter(below, numpy_dtype(-np.inf), dtype=numpy_dtype)
                above = np.nextafter(above, numpy_dtype(np.inf), dtype=numpy_dtype)
    rng = np.random.default_rng(20261002)
    integer_dtype = np.uint32 if numpy_dtype == np.float32 else np.uint64
    random_bits = rng.integers(0, np.iinfo(integer_dtype).max, size=4096, dtype=integer_dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.concatenate(
            (
                np.asarray(values, dtype=numpy_dtype),
                random_bits.view(numpy_dtype),
                np.linspace(0, 2 * float(threshold), 4096, dtype=numpy_dtype),
            )
        )


def assert_same_values(actual, expected):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    selected = ~np.isnan(expected)
    integer_dtype = np.uint32 if expected.dtype == np.float32 else np.uint64
    np.testing.assert_array_equal(
        actual.view(integer_dtype)[selected], expected.view(integer_dtype)[selected]
    )


@ti.data_oriented
class GaussianArithmeticProbe:
    def __init__(self, count, dtype):
        self.dtype = dtype
        self.original_q, self.original_zeta, self.fast_q, self.fast_zeta = shortcut_functions(dtype)
        self.density = ti.field(dtype=dtype, shape=count)
        self.original = ti.Vector.field(2, dtype=dtype, shape=count)
        self.candidate = ti.Vector.field(2, dtype=dtype, shape=count)

    @ti.kernel
    def evaluate(self):
        for i in self.density:
            rho = self.density[i]
            self.original[i] = ti.Vector([self.original_q(rho), self.original_zeta(rho)])
            self.candidate[i] = ti.Vector([self.fast_q(rho), self.fast_zeta(rho)])


@ti.data_oriented
class GaussianPairBenchmark:
    """Same displacement/core/strength loads and monopole arithmetic both ways.

    This isolates a potential kernel-level improvement; it is not an induction
    traversal or whole-solver speedup. q/zeta are evaluated together so compiler
    CSE of the original exponential is allowed, exactly as in the hot pair.
    """

    def __init__(self, targets, pairs, dtype=ti.f32):
        self.dtype = dtype
        self.pairs = pairs
        self.original_q, self.original_zeta, self.fast_q, self.fast_zeta = shortcut_functions(dtype)
        self.displacement = ti.Vector.field(3, dtype=dtype, shape=(targets, pairs))
        self.strength = ti.Vector.field(3, dtype=dtype, shape=(targets, pairs))
        self.core = ti.field(dtype=dtype, shape=(targets, pairs))
        self.velocity = ti.Vector.field(3, dtype=dtype, shape=targets)
        self.gradient = ti.Matrix.field(3, 3, dtype=dtype, shape=targets)

    @ti.kernel
    def evaluate(self, optimized: ti.template()):
        for i in self.velocity:
            velocity = ti.Vector.zero(self.dtype, 3)
            gradient = ti.Matrix.zero(self.dtype, 3, 3)
            for j in range(self.pairs):
                displacement, strength, core = (
                    self.displacement[i, j],
                    self.strength[i, j],
                    self.core[i, j],
                )
                radius = displacement.norm()
                rho = radius / core
                q, zeta = ti.cast(0.0, self.dtype), ti.cast(0.0, self.dtype)
                if ti.static(optimized):
                    q, zeta = self.fast_q(rho), self.fast_zeta(rho)
                else:
                    q, zeta = self.original_q(rho), self.original_zeta(rho)
                zeta /= core * core * core
                r2 = radius * radius
                r3 = r2 * radius
                r5 = r3 * r2
                cross = displacement.cross(strength)
                skew = ti.Matrix(
                    [
                        [0.0, -strength[2], strength[1]],
                        [strength[2], 0.0, -strength[0]],
                        [-strength[1], strength[0], 0.0],
                    ]
                )
                velocity += -q * cross / r3
                gradient += q / r3 * skew + (3.0 * q / r5 - zeta / r2) * cross.outer_product(
                    displacement
                )
            self.velocity[i], self.gradient[i] = velocity, gradient


def pair_inputs(targets, pairs, pattern):
    rng = np.random.default_rng(1234)
    shape = (targets, pairs)
    displacement = rng.normal(size=shape + (3,)).astype(np.float32)
    displacement += np.array([2.0, 0.0, 0.0], np.float32)
    radius = np.linalg.norm(displacement, axis=-1)
    if pattern == "far":
        rho = rng.uniform(12, 3000, size=shape)
    elif pattern == "near":
        rho = rng.uniform(0.1, 8, size=shape)
    elif pattern == "mixed":
        rho = rng.uniform(12, 3000, size=shape)
        rho[:, ::4] = rng.uniform(0.1, 8, size=(targets, (pairs + 3) // 4))
    else:
        raise ValueError(pattern)
    return (
        displacement,
        rng.normal(size=shape + (3,)).astype(np.float32),
        (radius / rho).astype(np.float32),
    )
