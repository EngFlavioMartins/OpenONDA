"""Independent quadrature checks of Gaussian values and crossover derivatives."""

import math

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.kernels.gaussian import create_gaussian_kernels
from source.solvers.vpm.kernels.high_order_gaussian import create_high_order_gaussian_kernels
from source.solvers.vpm.kernels.super_gaussian import create_super_gaussian_kernels
from source.solvers.vpm.physics.induction.treecode.lbvh import TaichiTreecode


@ti.data_oriented
class _Sampler:
    def __init__(self, radii, dtype, *, tree=False, factory=create_gaussian_kernels):
        functions = factory(dtype)
        self.q_func = functions["q_"]
        self.zeta_func = functions["zeta_"]
        self.g_func = functions["g_"]
        if tree:
            self.tree = TaichiTreecode(max_n_particles=1, max_nodes=2, kernel_type="GAUSSIAN")
            self.q_func = self.tree.q_kernel
            self.zeta_func = self.tree.zeta_kernel
        self.rho = ti.field(dtype, shape=len(radii))
        self.q = ti.field(dtype, shape=len(radii))
        self.zeta = ti.field(dtype, shape=len(radii))
        self.g = ti.field(dtype, shape=len(radii))
        self.rho.from_numpy(radii)

    @ti.kernel
    def evaluate(self):
        for i in self.rho:
            self.q[i] = self.q_func(self.rho[i])
            self.zeta[i] = self.zeta_func(self.rho[i])
            self.g[i] = self.g_func(self.rho[i])


def _reference(rho):
    # q = pi^-3/2 rho^3 integral_0^1 s^2 exp(-rho^2 s^2) ds.
    # This avoids both the implementation's series and closed-form cancellation.
    nodes, weights = np.polynomial.legendre.leggauss(128)
    s = (nodes + 1.0) / 2.0
    w = weights / 2.0
    exponential = np.exp(-((rho[:, None] * s) ** 2))
    q = math.pi**-1.5 * rho**3 * (exponential @ (w * s**2))
    g = np.array(
        [math.erf(float(r)) / (4.0 * math.pi * r) if r else 0.5 * math.pi**-1.5 for r in rho]
    )
    zeta = math.pi**-1.5 * np.exp(-(rho**2))
    return q, zeta, g


@pytest.mark.parametrize("precision", ("f32", "f64"))
def test_gaussian_values_match_independent_quadrature_at_both_splices(precision, tmp_path):
    dtype, numpy_dtype = (ti.f32, np.float32) if precision == "f32" else (ti.f64, np.float64)
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=dtype,
        cpu_max_num_threads=1,
        offline_cache=False,
        offline_cache_file_path=str(tmp_path / "taichi"),
    )
    try:
        # Include adjacent representable values at the old and new branch points.
        centres = np.array([0.2, 1.0, 6.0], dtype=numpy_dtype)
        rho = np.unique(
            np.concatenate(
                (
                    [0.0],
                    np.geomspace(1e-8, 10.0, 300),
                    centres,
                    np.nextafter(centres, numpy_dtype(0.0)),
                    np.nextafter(centres, numpy_dtype(np.inf)),
                )
            ).astype(numpy_dtype)
        )
        sampler = _Sampler(rho, dtype)
        sampler.evaluate()
        expected = _reference(rho.astype(np.float64))
        rtol = 6e-7 if precision == "f32" else 3e-14
        for actual, reference in ((sampler.q, expected[0]), (sampler.g, expected[2])):
            np.testing.assert_allclose(actual.to_numpy(), reference, rtol=rtol, atol=1e-44)
        # Squaring rho magnifies relative error in the exponential tail, and
        # Taichi f32 flushes subnormal values. These are distinct from q's
        # small-radius cancellation and require the exponential's own scale.
        np.testing.assert_allclose(
            sampler.zeta.to_numpy(),
            expected[1],
            rtol=4e-6 if precision == "f32" else 3e-14,
            atol=np.finfo(numpy_dtype).tiny,
        )
        np.testing.assert_allclose(
            make_vortex_kernel("GAUSSIAN").q(rho), expected[0], rtol=3e-14, atol=0.0
        )
        assert sampler.q.to_numpy()[0] == 0.0
        if precision == "f32":
            tree = _Sampler(rho, dtype, tree=True)
            tree.evaluate()
            np.testing.assert_array_equal(tree.q.to_numpy(), sampler.q.to_numpy())
            np.testing.assert_array_equal(tree.zeta.to_numpy(), sampler.zeta.to_numpy())
    finally:
        ti.reset()


def test_device_gaussian_derivative_matches_mollifier_across_splices(tmp_path):
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
        offline_cache_file_path=str(tmp_path / "taichi"),
    )
    try:
        centres = np.array([0.199999, 0.2, 0.200001, 0.999999, 1.0, 1.000001])
        h = 2e-5
        radii = np.concatenate((centres - 2 * h, centres - h, centres + h, centres + 2 * h))
        sampler = _Sampler(radii, ti.f64)
        sampler.evaluate()
        qm2, qm1, qp1, qp2 = sampler.q.to_numpy().reshape(4, -1)
        derivative = (qm2 - 8 * qm1 + 8 * qp1 - qp2) / (12 * h)
        exact = centres**2 * math.pi**-1.5 * np.exp(-(centres**2))
        np.testing.assert_allclose(derivative, exact, rtol=2e-10, atol=1e-13)
    finally:
        ti.reset()


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize(
    "name,factory,scale",
    [
        ("HIGH_ORDER_GAUSSIAN", create_high_order_gaussian_kernels, 1.0),
        ("SUPER_GAUSSIAN", create_super_gaussian_kernels, math.sqrt(2.0)),
    ],
)
def test_corrected_gaussians_match_density_quadrature_and_potential(
    precision, name, factory, scale
):
    """Integrate the declared density, independently of the q implementation."""
    dtype, np_dtype = (ti.f32, np.float32) if precision == "f32" else (ti.f64, np.float64)
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=dtype, cpu_max_num_threads=1, offline_cache=False)
    try:
        rho = np.unique(
            np.r_[0.0, np.geomspace(1e-9, 10.0, 250), 0.499999, 0.5, 0.500001, 1.0, scale]
        ).astype(np_dtype)
        r = rho.astype(np.float64)
        nodes, weights = np.polynomial.legendre.leggauss(128)
        x, w = (nodes + 1) / 2, weights / 2
        s = r[:, None] * x / scale
        density = (2.5 - s**2) * np.exp(-(s**2)) / (math.pi**1.5 * scale**3)
        expected_q = r**3 * (density @ (w * x**2))
        y = r / scale
        expected_g = np.array(
            [
                (math.erf(v) / v if v else 2 / math.sqrt(math.pi))
                + math.exp(-v * v) / math.sqrt(math.pi)
                for v in y
            ]
        ) / (4 * math.pi * scale)
        expected_zeta = (2.5 - y * y) * np.exp(-y * y) / (math.pi**1.5 * scale**3)
        sampler = _Sampler(rho, dtype, factory=factory)
        sampler.evaluate()
        tol = 9e-7 if precision == "f32" else 5e-14
        np.testing.assert_allclose(sampler.q.to_numpy(), expected_q, rtol=tol, atol=1e-44)
        np.testing.assert_allclose(sampler.g.to_numpy(), expected_g, rtol=tol, atol=1e-44)
        np.testing.assert_allclose(
            sampler.zeta.to_numpy(),
            expected_zeta,
            rtol=1e-5 if precision == "f32" else 1e-13,
            atol=np.finfo(np_dtype).tiny,
        )
        np.testing.assert_allclose(make_vortex_kernel(name).q(r), expected_q, rtol=5e-14, atol=0.0)
    finally:
        ti.reset()
