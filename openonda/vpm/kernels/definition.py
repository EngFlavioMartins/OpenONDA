"""The governing equations of one radial vortex-particle kernel.

Each kernel module ends with one :class:`KernelDefinition`, and
``kernels.registry`` collects them. Every consumer (induction backends,
diagnostics, configuration checks and slip-slab images) reads a kernel from
the registry, so a kernel is written down in exactly one module.

Adding a particle kernel
------------------------
Write a module next to ``gaussian.py`` that provides:

1. host ``q(rho)`` and ``zeta(rho)`` in float64 NumPy: the enclosed
   circulation factor, including 1/(4 pi), and the normalized radial
   vorticity density, with ``dq/drho = rho**2 zeta`` and ``q(inf) = 1/(4 pi)``;
2. a Taichi factory returning ``q_``, ``zeta_``, ``radial_factors_``, ``g_``
   and the diffusivity, energy-equivalence, volume and angular-impulse
   constants, with series wherever the closed forms cancel;
3. ``second_moment``, the m2 = int |x|^2 zeta d^3x of the density;
4. a :class:`SlabImageSplit` for free-slip slab images: the kernel written as
   a Gaussian partner of width ``partner_ratio * sigma`` plus a residual, the
   residual's radial factors (Python and CUDA), and an envelope of the
   kernel's core defect 1/(4 pi) - q beyond ``minimum_ratio`` partner widths.

Register the definition in ``kernels/registry.py``. If its residual belongs to
a new family, add that family to ``residual_family`` dispatch in
``physics/induction/gaussian_mesh/correction.py``.
``tests/vpm/test_kernel_conformance.py`` checks every registered kernel: host
against device functions, normalization, far field, second moment, residual
against the kernel minus its partner, and the envelopes against the defect.
"""

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np

ArrayFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class AlgebraicDefectEnvelope:
    """Core defect bounded by a power law in the partner coordinate x = r/s.

    4 pi |T(x)| <= velocity / x**power and
    4 pi |3 T(x) + x**3 zeta(x)| <= gradient / x**power for x >= minimum_ratio,
    where T = 1/(4 pi) - q is the kernel's deviation from Biot-Savart.
    """

    velocity: float
    gradient: float
    power: int
    minimum_ratio: float

    def __post_init__(self) -> None:
        if not (self.velocity > 0 and self.gradient > 0 and self.minimum_ratio > 0):
            raise ValueError("algebraic defect envelope needs positive coefficients and ratio")
        if self.power < 4:
            raise ValueError("x**4 times an algebraic envelope must not increase")

    def evaluate(self, x: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Float64 velocity and gradient defect envelopes at partner ratios ``x``."""
        x = np.asarray(x, dtype=np.float64)
        scale = 1.0 / (4.0 * math.pi * x**self.power)
        return self.velocity * scale, self.gradient * scale


@dataclass(frozen=True)
class GaussianDefectEnvelope:
    """Core defect bounded by a polynomial times exp(-x**2), x = r/s.

    |T(x)| <= pi**-1.5 exp(-x**2) sum_k c_k x**k over ``velocity`` (k, c_k)
    pairs, and the same with ``gradient`` for |3 T(x) + x**3 zeta(x)|, for
    x >= minimum_ratio. ``minimum_ratio`` must make x**4 times each envelope
    non-increasing, so the omitted image rows can be summed in closed form.
    """

    velocity: tuple[tuple[int, float], ...]
    gradient: tuple[tuple[int, float], ...]
    minimum_ratio: float

    def __post_init__(self) -> None:
        for terms in (self.velocity, self.gradient):
            if not terms or any(coefficient <= 0 for _, coefficient in terms):
                raise ValueError("Gaussian defect envelope needs positive coefficients")
            # x**(4+k) exp(-x**2) decreases once x**2 >= (4+k)/2.
            if self.minimum_ratio**2 < (4 + max(power for power, _ in terms)) / 2:
                raise ValueError("x**4 times the Gaussian envelope increases beyond minimum_ratio")

    def evaluate(self, x: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Float64 velocity and gradient defect envelopes at partner ratios ``x``."""
        x = np.asarray(x, dtype=np.float64)
        weight = np.exp(-x * x) / math.pi**1.5
        return tuple(
            weight * sum(coefficient * x**power for power, coefficient in terms)
            for terms in (self.velocity, self.gradient)
        )


@dataclass(frozen=True)
class SlabImageSplit:
    """A kernel written as a Gaussian partner plus a residual, for slab images.

    K_sigma = G_s + R_s with partner width s = partner_ratio * sigma. The slab
    image mesh carries the Gaussian partner; ``residual(r, s)`` returns the
    residual radial factors (a, b) in m^-3 and m^-5, in the convention of the
    Taichi ``radial_factors_``. ``residual_cuda`` defines the same function as
    ``__device__ void residual(Real r, Real s, Real &a, Real &b)``. A pure
    Gaussian has no residual and no envelope; its defect bound is the
    Gaussian one of the mesh.
    """

    partner_ratio: float = 1.0
    residual: Callable[[float, float], tuple[float, float]] | None = None
    residual_cuda: str | None = None
    defect_envelope: AlgebraicDefectEnvelope | GaussianDefectEnvelope | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.partner_ratio) or self.partner_ratio < 1.0:
            raise ValueError("partner Gaussian must be at least as wide as the kernel core")
        pieces = (self.residual, self.residual_cuda, self.defect_envelope)
        if any(piece is None for piece in pieces) and any(piece is not None for piece in pieces):
            raise ValueError("a residual needs Python and CUDA factors and a defect envelope")


@dataclass(frozen=True)
class KernelDefinition:
    """Everything the solver needs to know about one radial vortex kernel."""

    name: str
    q: ArrayFunction
    zeta: ArrayFunction
    device_factory: Callable
    second_moment: float
    slab_split: SlabImageSplit
    treecode_kernel_id: int | None = None

    def __post_init__(self) -> None:
        if self.name != self.name.lower() or not self.name:
            raise ValueError("kernel names are nonempty upper-case identifiers")
        if not math.isfinite(self.second_moment):
            raise ValueError("kernel second moment must be finite")

    @property
    def angular_core_coefficient(self) -> float:
        """(2/9) m2, the core term of the angular impulse of one particle."""
        return 2.0 * self.second_moment / 9.0

    @property
    def supports_core_spreading(self) -> bool:
        """Core spreading needs a positive second moment, d(sigma^2)/dt = 6 nu/m2."""
        return self.second_moment > 0.0
