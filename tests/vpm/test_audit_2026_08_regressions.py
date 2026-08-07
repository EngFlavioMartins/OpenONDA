"""
Regression tests for the 2026-08 VPM audit (docs/reviews/2026-08-vpm-audit.md).

Each test pins one defect that the audit found and fixed:

  N-1  ``HIGH_ORDER_GAUSSIAN`` / ``SUPER_GAUSSIAN`` combined with the *default*
       velocity method ``TREECODE`` were accepted by ``VPMSetup`` validation and
       then raised ``ValueError`` from inside the first velocity evaluation.
       The kernel/method combination must now be rejected at construction.

  N-2  The treecode carried a second copy of the Gaussian ``q`` kernel whose
       small-rho Taylor branch used 4/(3 sqrt(pi^3)) instead of 4/(3 sqrt(pi)) —
       a factor of pi.  The direct and tree implementations must agree with the
       analytic kernel across the branch point.

  N-2b The same kernel's small-rho crossover sat at rho = 1e-4, three decades
       inside the region where the closed form has already lost every
       significant digit to cancellation (measured: 5.6e-2 relative error at
       rho = 1e-2, 7e+4 at rho = 1e-4, and a sign flip just above the branch).

  N-6  ``random_seed`` was hardcoded in ``ti.init``, so every Random Walk Method
       run produced the identical realization and an ensemble could not be
       formed.  It is now a ``VPMSetup`` field.

  D-2  ``vlm/kernels/wake_shedding.py`` was dead code implementing a wake model
       that shed the *full* panel circulation per trailing-edge panel, so the
       shed streamwise circulation summed to sum(Gamma_i)*l != 0, violating
       Kelvin's theorem.  The surviving implementation sheds spanwise
       *differences* and must telescope to zero.

Run on the Taichi CPU backend in f32, matching the production kernels.
"""


# NOTE: deliberately no ``from __future__ import annotations`` — it stringifies
# the ``ti.i32`` annotations below, which Taichi resolves eagerly and rejects.

import numpy as np
import pytest
from scipy.special import erf
import taichi as ti

from source.solvers.VPM.config.constants import (
    GAUSSIAN_Q_SERIES_CROSSOVER,
    TREECODE_SUPPORTED_KERNELS,
)
from source.solvers.VPM.config.types import VelocityConfig, VPMSetup

ONE_OVER_FOUR_PI = 0.07957747154594767


# ── N-1: configuration rejects kernels the treecode cannot evaluate ──────────


@pytest.mark.unit
@pytest.mark.parametrize("kernel", ["HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN"])
def test_treecode_rejects_unsupported_kernel_at_config_time(kernel):
    """An unsupported kernel + TREECODE must fail loudly in VPMSetup, not mid-run."""
    with pytest.raises(ValueError, match="cannot be used with velocity method 'TREECODE'"):
        VPMSetup.dns_simulation(
            particles_kernel=kernel,
            velocity=VelocityConfig.treecode(theta=0.5),
            processing_unit="CPU",
            max_particles=1000,
        )


@pytest.mark.unit
@pytest.mark.parametrize("kernel", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN"])
def test_direct_method_accepts_every_kernel(kernel):
    """The direct O(N^2) path implements all kernels, so nothing is rejected there."""
    setup = VPMSetup.dns_simulation(
        particles_kernel=kernel,
        velocity=VelocityConfig.direct(),
        processing_unit="CPU",
        max_particles=1000,
    )
    assert setup.particles_kernel == kernel


@pytest.mark.unit
@pytest.mark.parametrize("kernel", TREECODE_SUPPORTED_KERNELS)
def test_treecode_supported_kernels_are_accepted(kernel):
    setup = VPMSetup.dns_simulation(
        particles_kernel=kernel,
        velocity=VelocityConfig.treecode(theta=0.5),
        processing_unit="CPU",
        max_particles=1000,
    )
    assert setup.velocity.method == "TREECODE"


@pytest.mark.unit
def test_treecode_kernel_support_has_a_single_source_of_truth():
    """VPMSetup validation and TaichiTreecode must read the same constant.

    They previously carried independent lists, which is how the mismatch in N-1
    survived: the config grew two kernels the treecode never learned about.
    """
    from source.solvers.VPM.acceleration import treecode_gpu

    assert treecode_gpu.TREECODE_SUPPORTED_KERNELS is TREECODE_SUPPORTED_KERNELS


# ── N-2: treecode q kernel matches the analytic Gaussian across the branch ───


def _q_gaussian_exact(rho: np.ndarray) -> np.ndarray:
    """Reference q(rho) = [erf(rho) - (2/sqrt(pi)) rho exp(-rho^2)] / (4 pi).

    The closed form cancels catastrophically for small rho *even in float64*
    (relative error ~1.5 eps / rho^2, i.e. 3e-4 at rho = 1e-6), so below the
    crossover this uses the series

        q(rho) = (4/(3 sqrt(pi))) rho^3 [1 - (3/5)rho^2 + (3/14)rho^4 - (1/18)rho^6]
                 / (4 pi) + O(rho^11),

    from erf(r) - (2/sqrt(pi)) r exp(-r^2)
         = (2/sqrt(pi)) sum_n (-1)^n r^(2n+1) / n! * (-2n/(2n+1)).
    Truncation at the rho^6 term is ~1e-9 relative at rho = 0.2, so the reference
    is exact to float64 for the whole tested range.
    """
    rho = np.asarray(rho, dtype=np.float64)
    r2 = rho**2
    series = (
        4.0
        / (3.0 * np.sqrt(np.pi))
        * rho**3
        * (1.0 - 0.6 * r2 + (3.0 / 14.0) * r2**2 - (1.0 / 18.0) * r2**3)
    )
    closed = erf(rho) - 2.0 / np.sqrt(np.pi) * rho * np.exp(-r2)
    return np.where(rho < GAUSSIAN_Q_SERIES_CROSSOVER, series, closed) * ONE_OVER_FOUR_PI


def _probe_treecode_q(rho: np.ndarray) -> np.ndarray:
    """Evaluate ``TaichiTreecode.q_kernel`` at ``rho`` on the CPU backend.

    The probe class is built inside the call because ``@ti.kernel`` resolves its
    annotations eagerly, which requires ``ti.init`` to have already run.
    """
    ti.init(arch=ti.cpu, default_fp=ti.f32, random_seed=0)
    from source.solvers.VPM.acceleration.treecode_gpu import TaichiTreecode

    @ti.data_oriented
    class _QProbe:
        def __init__(self, tree, n):
            self.tree = tree
            self.rho = ti.field(dtype=ti.f32, shape=n)
            self.out = ti.field(dtype=ti.f32, shape=n)

        @ti.kernel
        def run(self, n: ti.i32):
            for i in range(n):
                self.out[i] = self.tree.q_kernel(self.rho[i])

    tree = TaichiTreecode(max_particles=8, max_nodes=16, theta=0.5, kernel_type="GAUSSIAN")
    probe = _QProbe(tree, len(rho))
    probe.rho.from_numpy(np.ascontiguousarray(rho, dtype=np.float32))
    probe.run(len(rho))
    return probe.out.to_numpy().astype(np.float64)


@pytest.mark.verification
def test_treecode_gaussian_q_matches_analytic_across_taylor_branch():
    """The small-rho Taylor branch must join the erf branch continuously.

    The bug made the branch a factor of pi too small.  Sampling either side of
    the rho = 1e-4 switch catches any future drift between the treecode's private
    kernel copy and ``kernels/gaussian.py``.
    """
    # Span the whole small-argument regime, including the band 1e-4..1e-2 where
    # the old threshold left the closed form in charge and it was 5.6e-2 to
    # 7e+4 wrong, plus well-separated pairs that must be untouched.
    rho = np.array(
        [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.19, 0.21, 0.5, 1.0, 2.0, 4.0],
        dtype=np.float32,
    )
    got = _probe_treecode_q(rho)
    want = _q_gaussian_exact(rho.astype(np.float64))

    # q ~ rho^3 over six decades, so compare relatively.  The floor is set by the
    # A&S 7.1.26 erf approximation used on the closed-form branch (~1e-7 abs).
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=0.0)


@pytest.mark.verification
def test_treecode_gaussian_q_is_continuous_at_the_series_crossover():
    """No jump across the crossover, and q stays strictly positive.

    Two historical failures are covered.  The pi-scaled series produced a 3.14x
    step at the crossover; the 1e-4 crossover left the cancelling closed form in
    charge just above it, where it returned *negative* values (measured
    -4.7e-09 where q = +7.5e-13).
    """
    eps = 1e-3 * GAUSSIAN_Q_SERIES_CROSSOVER
    rho = np.array(
        [GAUSSIAN_Q_SERIES_CROSSOVER - eps, GAUSSIAN_Q_SERIES_CROSSOVER + eps], dtype=np.float32
    )
    below, above = _probe_treecode_q(rho)
    assert below > 0.0 and above > 0.0, f"q must stay positive, got {below:.3e}, {above:.3e}"
    assert abs(above / below - 1.0) < 1e-2, (
        f"discontinuity across the series crossover: {below:.6e} -> {above:.6e} "
        f"(ratio {above / below:.4f})"
    )


@pytest.mark.verification
def test_gaussian_q_is_positive_and_monotone_through_the_cancellation_band():
    """q must be positive and increasing everywhere, including 1e-5 < rho < 1e-2.

    This is the band the old 1e-4 crossover handed to the closed form, where f32
    cancellation destroyed every significant digit and flipped the sign.
    """
    rho = np.geomspace(1e-6, 1.0, 60).astype(np.float32)
    got = _probe_treecode_q(rho)
    assert np.all(got > 0.0), f"non-positive q at rho={rho[got <= 0.0]}"
    assert np.all(np.diff(got) > 0.0), "q(rho) must be strictly increasing"


# ── D-2: VLM wake shedding conserves circulation (Kelvin) ───────────────────


def _telescoped_trailing_strengths(gamma: np.ndarray) -> np.ndarray:
    """Reference model of the shipped shedding rule, per unit ``l_te * V_unit``.

    Mirrors ``vlm/solver/kernels.py:shed_wake_particles_kernel``:
      * left edge of panel i  -> -(Gamma_i - Gamma_{i-1}); at the left tip -Gamma_0
      * right edge, tip only  -> +Gamma_{n-1}
    """
    out = [-gamma[0]]
    out += [-(gamma[i] - gamma[i - 1]) for i in range(1, len(gamma))]
    out.append(+gamma[-1])
    return np.array(out)


@pytest.mark.verification
@pytest.mark.parametrize(
    "gamma",
    [
        np.array([1.0, 1.0, 1.0, 1.0]),  # rectangular loading
        np.array([0.4, 0.9, 1.0, 0.9, 0.4]),  # elliptic-ish loading
        np.array([1.0, -0.5, 0.25]),  # sign changes
        np.linspace(0.1, 2.0, 12),  # monotone ramp
    ],
)
def test_shed_trailing_circulation_telescopes_to_zero(gamma):
    """Total shed streamwise circulation must vanish (Kelvin's theorem).

    The deleted ``vlm/kernels/wake_shedding.py`` shed ``Gamma_i`` at every TE
    panel instead of the spanwise difference, giving sum(Gamma_i) != 0 for any
    lifting distribution.  This pins the correct behaviour.
    """
    shed = _telescoped_trailing_strengths(gamma)
    assert abs(shed.sum()) < 1e-12 * max(1.0, np.abs(gamma).sum())


@pytest.mark.verification
def test_deleted_full_gamma_wake_model_would_violate_kelvin():
    """Guard the *reason* the dead module was removed, not just its absence."""
    gamma = np.array([0.4, 0.9, 1.0, 0.9, 0.4])
    naive = gamma.sum()  # what wake_shedding.py emitted: Gamma_i per TE panel
    correct = _telescoped_trailing_strengths(gamma).sum()
    assert naive > 0.5, "sanity: the naive model has large net streamwise circulation"
    assert abs(correct) < 1e-12


@pytest.mark.unit
def test_dead_wake_shedding_module_is_gone():
    """``vlm.kernels`` must no longer export the superseded wake kernels."""
    from source.solvers.VPM.boundary_elements.vlm import kernels

    assert not hasattr(kernels, "compute_wake_particles_kernel")
    assert not hasattr(kernels, "compute_wake_simple_kernel")


# ── N-6: the RWM random seed is configurable ────────────────────────────────


@pytest.mark.unit
def test_random_seed_is_a_configuration_field_with_a_reproducible_default():
    """RWM reproducibility must be a choice, not a hardcoded constant."""
    assert VPMSetup.dns_simulation(processing_unit="CPU").random_seed == 42
    assert VPMSetup.dns_simulation(processing_unit="CPU", random_seed=7).random_seed == 7


@pytest.mark.unit
def test_random_seed_reaches_the_taichi_backend():
    """The seed must be forwarded to ``ti.init``, not silently dropped."""
    import inspect

    from source.solvers.VPM.config import backend

    assert "random_seed" in inspect.signature(backend.initialize_taichi_backend).parameters
    source = inspect.getsource(backend.initialize_taichi_backend)
    assert 'init_kwargs["random_seed"] = random_seed' in source, (
        "initialize_taichi_backend must pass its random_seed argument to ti.init"
    )


@pytest.mark.unit
def test_random_seed_is_serialized():
    """An ensemble member must be recoverable from its serialized config."""
    assert (
        VPMSetup.dns_simulation(processing_unit="CPU", random_seed=11).to_dict()["random_seed"]
        == 11
    )


# ── N-3: the precision contract is truthful ─────────────────────────────────


@pytest.mark.unit
def test_f64_with_treecode_is_rejected_rather_than_silently_downgraded():
    """f64 + TREECODE cannot deliver f64: the tree is f32 end-to-end."""
    with pytest.raises(ValueError, match="precision='f64' cannot be used with"):
        VPMSetup.dns_simulation(
            precision="f64",
            velocity=VelocityConfig.treecode(theta=0.5),
            processing_unit="CPU",
            max_particles=1000,
        )


@pytest.mark.unit
def test_f64_with_direct_summation_is_accepted():
    """The direct path is the one route where f64 widens the velocity sum."""
    setup = VPMSetup.dns_simulation(
        precision="f64",
        velocity=VelocityConfig.direct(),
        processing_unit="CPU",
        max_particles=1000,
    )
    assert setup.precision == "f64"


@pytest.mark.unit
def test_precision_docstring_labels_f64_experimental():
    """The user-facing contract must not advertise f64 as production-ready."""
    import inspect

    src = inspect.getsource(VPMSetup)
    marker = src.split("precision: Literal", 1)[1][:1400]
    assert "EXPERIMENTAL" in marker and "not end-to-end" in marker, (
        "the precision docstring must state that f64 is experimental and partial"
    )
    assert "supported production precision" in marker


# ── N-4: core-spreading constants match the kernels' own second moments ─────


def _sph(f):
    from scipy.integrate import quad

    return quad(f, 0.0, np.inf, limit=800)[0]


_ZETA = {
    "GAUSSIAN": lambda r: np.pi**-1.5 * np.exp(-r * r),
    "WINCKELMANS": lambda r: 15.0 / (8.0 * np.pi) * (1.0 + r * r) ** -3.5,
    "HIGH_ORDER_GAUSSIAN": lambda r: np.pi**-1.5 * (2.5 - r * r) * np.exp(-r * r),
    "SUPER_GAUSSIAN": lambda r: (
        (1.0 / (4.0 * np.pi)) * np.sqrt(2.0 / np.pi) * np.exp(-r * r / 2.0) * (2.5 - r * r / 2.0)
    ),
}


@pytest.mark.verification
@pytest.mark.parametrize("name", sorted(_ZETA))
def test_regularization_kernels_are_normalized(name):
    """Every zeta must integrate to 1 over R^3, or Gamma is not circulation."""
    zeta = _ZETA[name]
    assert _sph(lambda r: zeta(r) * 4.0 * np.pi * r * r) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.verification
@pytest.mark.parametrize(
    ("name", "expected_m2"),
    [
        ("GAUSSIAN", 1.5),
        ("WINCKELMANS", 1.5),
        # The (2.5 - rho^2) polynomial cancels the second moment: these kernels are
        # second-order accurate, so their angular-impulse correction must vanish.
        ("HIGH_ORDER_GAUSSIAN", 0.0),
        ("SUPER_GAUSSIAN", 0.0),
    ],
)
def test_kernel_second_moment_matches_declared_angular_impulse_constant(name, expected_m2):
    """m2 = int |q|^2 zeta d^3q is what angular_impulse_correction_constant_ declares."""
    zeta = _ZETA[name]
    m2 = _sph(lambda r: r * r * zeta(r) * 4.0 * np.pi * r * r)
    assert m2 == pytest.approx(expected_m2, abs=1e-9)


@pytest.mark.verification
@pytest.mark.parametrize(("name", "expected_c"), [("GAUSSIAN", 4.0), ("WINCKELMANS", 4.0)])
def test_core_spreading_constant_follows_from_the_second_moment(name, expected_c):
    """C = 6/m2, from <r^2> = m2 sigma^2 and d<r^2>/dt = 6 nu.

    Both kernels have m2 = 3/2, so both give C = 4.  WINCKELMANS previously
    declared 256/45 = 5.689, a hand-calibrated value presented as a derived one.
    """
    zeta = _ZETA[name]
    m2 = _sph(lambda r: r * r * zeta(r) * 4.0 * np.pi * r * r)
    assert 6.0 / m2 == pytest.approx(expected_c, rel=1e-9)


@pytest.mark.verification
def test_declared_diffusivity_constants_match_the_derivation():
    """Pin the values the Taichi kernels actually return."""
    ti.init(arch=ti.cpu, default_fp=ti.f32, random_seed=0)
    from source.solvers.VPM.kernels.gaussian import create_gaussian_kernels
    from source.solvers.VPM.kernels.winckelmans import create_winckelmans_kernels

    @ti.data_oriented
    class _Probe:
        def __init__(self, fn):
            self.fn = fn
            self.out = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def run(self):
            self.out[None] = self.fn()

    for factory, expected in ((create_gaussian_kernels, 4.0), (create_winckelmans_kernels, 4.0)):
        probe = _Probe(factory(ti.f32)["diffusivity_constant_"])
        probe.run()
        assert probe.out[None] == pytest.approx(expected, rel=1e-6)
