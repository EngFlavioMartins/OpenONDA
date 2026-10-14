"""Every registered particle kernel satisfies the equations its definition declares.

A new kernel registered in ``kernels/registry.py`` is checked here without
editing this file: host against device functions, normalization and far
field, the second moment, and the slab-image residual and defect envelopes.
"""

import math
import typing

import numpy as np
import pytest
import taichi as ti

from openonda.vpm.config.case import Numerics
from openonda.vpm.kernels.definition import AlgebraicDefectEnvelope
from openonda.vpm.kernels.gaussian import radial_factors_host as gaussian_factors
from openonda.vpm.kernels.registry import KERNELS

NAMES = tuple(KERNELS)
INVERSE_FOUR_PI = 1.0 / (4.0 * math.pi)
NODES, WEIGHTS = np.polynomial.legendre.leggauss(160)


def _radial_integral(function, upper):
    """Gauss-Legendre integral of ``function`` over [0, upper], piecewise."""
    edges = np.linspace(0.0, upper, 41)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        x = low + (high - low) * (NODES + 1) / 2
        total += (high - low) / 2 * float(np.sum(WEIGHTS * function(x)))
    return total


@pytest.fixture(scope="module", autouse=True)
def cpu_f64():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=1)
    yield
    ti.reset()


@ti.data_oriented
class _DeviceSampler:
    def __init__(self, name, rho):
        functions = KERNELS[name].device_factory(ti.f64)
        self.q_function = functions["q_"]
        self.zeta_function = functions["zeta_"]
        self.factors_function = functions["radial_factors_"]
        self.second_moment_function = functions["angular_impulse_correction_constant_"]
        self.rho = ti.field(ti.f64, shape=len(rho))
        self.q = ti.field(ti.f64, shape=len(rho))
        self.zeta = ti.field(ti.f64, shape=len(rho))
        self.a = ti.field(ti.f64, shape=len(rho))
        self.b = ti.field(ti.f64, shape=len(rho))
        self.second_moment = ti.field(ti.f64, shape=())
        self.rho.from_numpy(np.asarray(rho, dtype=np.float64))

    @ti.kernel
    def evaluate(self):
        for i in self.rho:
            self.q[i] = self.q_function(self.rho[i])
            self.zeta[i] = self.zeta_function(self.rho[i])
            factors = self.factors_function(self.rho[i], ti.cast(1.0, ti.f64), True)
            self.a[i] = factors[0]
            self.b[i] = factors[1]
        self.second_moment[None] = self.second_moment_function()


def test_configuration_literal_lists_every_registered_kernel():
    literal = typing.get_type_hints(Numerics)["particle_kernel"]
    assert set(typing.get_args(literal)) == set(KERNELS)


@pytest.mark.parametrize("name", NAMES)
def test_device_functions_follow_the_host_equations(name):
    definition = KERNELS[name]
    rho = np.r_[0.0, np.geomspace(1e-6, 12.0, 300)]
    sampler = _DeviceSampler(name, rho)
    sampler.evaluate()
    np.testing.assert_allclose(sampler.q.to_numpy(), definition.q(rho), rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(
        sampler.zeta.to_numpy(), definition.zeta(rho), rtol=1e-12, atol=1e-16
    )
    # a = q/r^3 and b = (3q - rho^3 zeta)/r^5 at unit core; away from zero
    # from the host equations, and a(0) = zeta(0)/3 at the core centre.
    positive = rho > 1e-2
    q, zeta = definition.q(rho[positive]), definition.zeta(rho[positive])
    r = rho[positive]
    np.testing.assert_allclose(sampler.a.to_numpy()[positive], q / r**3, rtol=1e-10)
    np.testing.assert_allclose(
        sampler.b.to_numpy()[positive], (3 * q - r**3 * zeta) / r**5, rtol=1e-8, atol=1e-14
    )
    np.testing.assert_allclose(sampler.a.to_numpy()[0], definition.zeta(0.0) / 3, rtol=1e-12)
    assert sampler.second_moment[None] == definition.second_moment


@pytest.mark.parametrize("name", NAMES)
def test_density_is_normalized_and_integrates_to_q(name):
    definition = KERNELS[name]
    # dq/drho = rho^2 zeta, q(0) = 0, so q is the enclosed density integral.
    for rho in (0.3, 1.0, 2.5, 6.0):
        enclosed = _radial_integral(lambda x: x * x * definition.zeta(x), rho)
        assert definition.q(rho) == pytest.approx(enclosed, rel=1e-11, abs=1e-15)
    far = 400.0 if name == "winckelmans" else 30.0
    total = _radial_integral(lambda x: x * x * definition.zeta(x), far)
    tail = 15.0 / 8.0 / (4 * math.pi * far**4) if name == "winckelmans" else 0.0
    assert total + tail == pytest.approx(INVERSE_FOUR_PI, rel=1e-9)
    assert definition.q(far) == pytest.approx(INVERSE_FOUR_PI - tail, rel=1e-9)


@pytest.mark.parametrize("name", NAMES)
def test_second_moment_is_the_density_moment(name):
    definition = KERNELS[name]
    if name == "winckelmans":
        # 4 pi int x^4 zeta converges like x^-2; integrate x in (0, inf) via x = t/(1-t).
        def integrand(t):
            x = t / (1 - t)
            return 4 * math.pi * x**4 * definition.zeta(x) / (1 - t) ** 2

        moment = _radial_integral(integrand, 1 - 1e-9)
    else:
        moment = _radial_integral(lambda x: 4 * math.pi * x**4 * definition.zeta(x), 30.0)
    assert moment == pytest.approx(definition.second_moment, abs=1e-6)
    assert definition.angular_core_coefficient == 2 * definition.second_moment / 9
    assert definition.supports_core_spreading == (definition.second_moment > 0)


def _defects(name, x):
    """Core defect T and gradient defect 3T + rho^3 zeta at partner ratio x.

    Written without the cancellation of 1/(4 pi) - q, independently of the
    residual and envelope code, so that tiny tails are compared accurately.
    """
    x = np.asarray(x, dtype=np.float64)
    erfc = np.vectorize(math.erfc)
    gaussian = (erfc(x) + 2 / math.sqrt(math.pi) * x * np.exp(-x * x)) * INVERSE_FOUR_PI
    if name == "gaussian":
        return gaussian, 3 * gaussian + x**3 * np.exp(-x * x) * math.pi**-1.5
    if name in {"high_order_gaussian", "super_gaussian"}:
        # Both are the corrected Gaussian in the partner coordinate.
        defect = gaussian - 0.5 * math.pi**-1.5 * x**3 * np.exp(-x * x)
        zeta = (2.5 - x * x) * np.exp(-x * x) * math.pi**-1.5
        return defect, 3 * defect + x**3 * zeta
    assert name == "winckelmans"
    u = x / np.sqrt(x * x + 1)
    defect = INVERSE_FOUR_PI * u**4 * (1 + 2 * u + 3 * u * u + 1.5 * u**3) / ((1 + u) ** 2 * x**4)
    return defect, 3 * defect + INVERSE_FOUR_PI * 7.5 * x**3 / (x * x + 1) ** 3.5


@pytest.mark.parametrize("name", NAMES)
def test_closed_form_defects_match_the_host_equations(name):
    definition = KERNELS[name]
    ratio = definition.slab_split.partner_ratio
    x = np.linspace(0.2, 3.0, 50)
    defect, gradient = _defects(name, x)
    rho = ratio * x
    q, zeta = definition.q(rho), definition.zeta(rho)
    np.testing.assert_allclose(defect, INVERSE_FOUR_PI - q, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(gradient, 3 * (INVERSE_FOUR_PI - q) + rho**3 * zeta, atol=1e-14)


@pytest.mark.parametrize("name", NAMES)
def test_residual_is_the_kernel_minus_its_gaussian_partner(name):
    definition = KERNELS[name]
    split = definition.slab_split
    if split.residual is None:
        assert name == "gaussian" and split.partner_ratio == 1.0
        return
    width = 0.7  # partner width s
    sigma = width / split.partner_ratio
    for radius in np.r_[0.0, np.geomspace(1e-6, 3.5, 120) * width]:
        a, b = split.residual(radius, width)
        a_python, b_python = split.residual.py_func(radius, width)
        assert (a, b) == pytest.approx((a_python, b_python), rel=1e-14, abs=1e-300)
        a_g, b_g = gaussian_factors(radius, width)
        if radius < 1e-3 * width:
            # Origin limits a = zeta(0)/(3 sigma^3), b = -zeta'' limit, from the series.
            continue
        rho = radius / sigma
        q, zeta = definition.q(rho), definition.zeta(rho)
        a_kernel = q / radius**3
        b_kernel = (3 * q - rho**3 * zeta) / radius**5
        assert a + a_g == pytest.approx(a_kernel, rel=1e-10)
        assert b + b_g == pytest.approx(b_kernel, rel=1e-8, abs=1e-12 / width**5)
    # Far beyond the partner core the residual is the defect difference.
    for x in (5.0, 20.0, 200.0):
        a, b = split.residual(x * width, width)
        defect, gradient = _defects(name, x)
        gaussian_defect, gaussian_gradient = _defects("gaussian", x)
        r = x * width
        assert a == pytest.approx((gaussian_defect - defect) / r**3, rel=1e-12, abs=1e-300)
        assert b == pytest.approx((gaussian_gradient - gradient) / r**5, rel=1e-12, abs=1e-300)


@pytest.mark.parametrize("name", NAMES)
def test_defect_envelopes_dominate_and_decay_like_inverse_fourth_powers(name):
    envelope = KERNELS[name].slab_split.defect_envelope
    if envelope is None:
        return
    upper = 1e4 if isinstance(envelope, AlgebraicDefectEnvelope) else 25.0
    x = np.geomspace(envelope.minimum_ratio, upper, 4000)
    defect, gradient = _defects(name, x)
    velocity_bound, gradient_bound = envelope.evaluate(x)
    assert np.all(np.abs(defect) <= velocity_bound)
    assert np.all(np.abs(gradient) <= gradient_bound)
    # x^4 E must not increase, so omitted image rows sum in closed form.
    for bound in (velocity_bound, gradient_bound):
        scaled = x**4 * bound
        assert np.all(np.diff(scaled) <= 1e-12 * scaled[:-1])
    if isinstance(envelope, AlgebraicDefectEnvelope):
        # The envelope is the exact asymptote, not a loose constant.
        tight_velocity, tight_gradient = envelope.evaluate(x[-1])
        assert abs(defect[-1]) == pytest.approx(tight_velocity, rel=1e-6)
        assert abs(gradient[-1]) == pytest.approx(tight_gradient, rel=1e-6)
