"""Vortex-stretching configuration for the VPM solver.
Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StretchingConfig:
    """
    Configuration for vortex stretching schemes.

    The formulations may use either direct evaluation or a controlled treecode
    approximation.  ``StabilizationConfig`` is only a particle-retention policy
    and never modifies vortex strength.

    Modes:
        - DIRECT: dΓ/dt = (Γ·∇)u, the standard direct formulation
        - TRANSPOSED: dΓ/dt = (∇u)ᵀΓ, conserves vector vortex strength
        - MIXED: Strain-based symmetric direct formulation

    Examples:
          # Transposed stretching (conserves vortex strength, recommended)
          stretching = StretchingConfig.transposed()

          # Direct stretching
          stretching = StretchingConfig.direct()

          # Mixed/strain stretching
          stretching = StretchingConfig.mixed()

          # Disabled stretching
          stretching = StretchingConfig.disabled()
    """

    mode: Literal["DIRECT", "TRANSPOSED", "MIXED"] = "TRANSPOSED"
    """Stretching formulation mode."""

    scheme: Literal["EULER", "RK2", "RK3", "RK4"] = "RK3"
    """Time integration scheme for the stretching substep (dΓ/dt).

    Options:
      - 'EULER': forward Euler (1st order)
      - 'RK2':   Heun's method, 2nd-order Runge–Kutta
      - 'RK3':   SSP-RK3, 3rd-order strong-stability-preserving (default)
      - 'RK4':   classical 4th-order Runge–Kutta
    """

    enabled: bool = True
    """Whether vortex stretching is enabled."""

    use_treecode: bool = False
    """Compute the stretching rate from the treecode velocity gradient instead
    of the direct O(N²) pairwise kernel.

    The stretching rate dΓ/dt = (Γ·∇)u (DIRECT), (∇u)ᵀ·Γ (TRANSPOSED) or the
    symmetric S·Γ (MIXED) is an exact local contraction of the velocity-gradient
    tensor ∇u.  The direct kernel forms ∇u·Γ implicitly with an O(N²) pair sum;
    with ``use_treecode=True`` the same ∇u is evaluated by the Barnes–Hut
    treecode (O(N log N)) and contracted locally.  The contraction itself is
    exact (relL2 ≈ 1e-6 against the direct rate when contracted with the
    directly-computed gradient); through the treecode gradient the rate differs
    from direct by the Barnes–Hut opening-angle tolerance (measured relL2
    ≈ 4e-2 at θ=0.2 on a random cloud — the same error class the advection
    velocity already carry).

    IMPORTANT — measured tradeoff (RTX 3060, 2026-07): the treecode *gradient*
    traversal (9 tensor components, deep walks) is intrinsically expensive, so
    the O(N²) direct kernel is actually FASTER up to at least N ≈ 250k (direct
    0.80× the treecode wall time at 249k).  The crossover is beyond a 6 GB card.
    The physics is preserved either way (vortex strength matches to ~4e-5 after
    several steps).  Enable this only above the crossover, or once the treecode
    traversal itself is made cheaper (higher-order multipoles → larger θ).
    Default False keeps the faster exact-direct rate."""

    treecode_theta: float = 0.3
    """Barnes–Hut opening angle for the treecode stretching gradient
    (only used when ``use_treecode=True``).  Smaller = more accurate/slower."""

    conserve_moments: bool = False
    """Project each coupled stretching rate onto the closed-flow moment manifold.

    The standard ``DIRECT`` particle contraction is consistent with
    ``(omega . grad) u`` but, at finite particle resolution, its quadrature
    does not make ``sum(vortex_strength)`` or linear impulse algebraic invariants.  A
    treecode adds a second, controlled source of moment error.  When enabled,
    the minimum-L2 global correction is applied at every Runge--Kutta stage so
    vector vortex strength, linear impulse, and kernel-corrected angular impulse
    have zero inviscid rate.  Viscous core spreading is left untouched.  This
    option requires ``COUPLED`` time integration because both impulse rates
    depend on position and strength rates from the same stage.
    """

    conserve_energy: bool = False
    """Make the inviscid rate of the discrete blob energy vanish.

    The correction is included in the same minimum-norm stage projection as
    the closed-flow moments.  It uses the exact gradient of the quadratic
    regularized-particle energy, including the position-rate contribution.
    Viscous core spreading remains outside the projection, so the only energy
    change retained by the split update is the modeled viscous/SGS sink.  This
    option requires ``COUPLED`` time integration.
    """

    reformulated: bool = False
    """Advance particle core size consistently with inviscid stretching.

    Reformulated VPM restores local vortex-tube mass and angular-momentum
    conservation by evolving ``vortex_strength`` and ``core_radius`` at the
    same Runge--Kutta stages.  The classic fixed-core formulation remains the
    default for backward compatibility.
    """

    def __post_init__(self) -> None:
        mode = self.mode.upper()
        scheme = self.scheme.upper()
        if mode not in ("DIRECT", "TRANSPOSED", "MIXED"):
            raise ValueError(
                f"stretching mode must be DIRECT, TRANSPOSED, or MIXED, got {self.mode!r}"
            )
        if scheme not in ("EULER", "RK2", "RK3", "RK4"):
            raise ValueError(
                f"stretching scheme must be EULER, RK2, RK3, or RK4, got {self.scheme!r}"
            )
        if not 0.0 < self.treecode_theta < 2.0:
            raise ValueError(f"treecode_theta must be in (0, 2), got {self.treecode_theta!r}")
        if self.conserve_energy and not self.conserve_moments:
            raise ValueError("conserve_energy requires conserve_moments")
        if self.reformulated and self.conserve_energy:
            raise ValueError(
                "reformulated stretching does not yet support energy-rate projection"
            )
        if mode != self.mode:
            object.__setattr__(self, "mode", mode)
        if scheme != self.scheme:
            object.__setattr__(self, "scheme", scheme)

    @staticmethod
    def direct(
        scheme: str = "RK3",
        use_treecode: bool = False,
        treecode_theta: float = 0.3,
        conserve_moments: bool = False,
        conserve_energy: bool = False,
        reformulated: bool = False,
    ):
        """Direct scheme: dalpha/dt = (alpha*grad)u.

        Options for `scheme`:
          - 'EULER': forward Euler (1st order)
          - 'RK2':   Heun's method, 2nd-order Runge–Kutta
          - 'RK3':   SSP-RK3, 3rd-order strong-stability-preserving (default)
          - 'RK4':   classical 4th-order Runge–Kutta

        Set ``use_treecode=True`` to evaluate the rate from the O(N log N)
        treecode gradient instead of the O(N²) pairwise kernel (large N).
        """
        return StretchingConfig(
            mode="DIRECT",
            scheme=scheme,
            use_treecode=use_treecode,
            treecode_theta=treecode_theta,
            conserve_moments=conserve_moments,
            conserve_energy=conserve_energy,
            reformulated=reformulated,
        )

    @staticmethod
    def transposed(
        scheme: str = "RK3",
        use_treecode: bool = False,
        treecode_theta: float = 0.3,
        conserve_moments: bool = False,
        conserve_energy: bool = False,
        reformulated: bool = False,
    ):
        """Transposed scheme: dalpha/dt = grad(u)^T alpha; conserves Σalpha.

        Options for `scheme`:
          - 'EULER': forward Euler (1st order)
          - 'RK2':   Heun's method, 2nd-order Runge–Kutta
          - 'RK3':   SSP-RK3, 3rd-order strong-stability-preserving (default)
          - 'RK4':   classical 4th-order Runge–Kutta

        Set ``use_treecode=True`` to evaluate the rate from the O(N log N)
        treecode gradient instead of the O(N²) pairwise kernel (large N).
        """
        return StretchingConfig(
            mode="TRANSPOSED",
            scheme=scheme,
            use_treecode=use_treecode,
            treecode_theta=treecode_theta,
            conserve_moments=conserve_moments,
            conserve_energy=conserve_energy,
            reformulated=reformulated,
        )

    @staticmethod
    def mixed(
        scheme: str = "RK3",
        use_treecode: bool = False,
        treecode_theta: float = 0.3,
        conserve_moments: bool = False,
        conserve_energy: bool = False,
        reformulated: bool = False,
    ):
        """Mixed/strain scheme: symmetric formulation

        Options for `scheme`:
          - 'EULER': forward Euler (1st order)
          - 'RK2':   Heun's method, 2nd-order Runge–Kutta
          - 'RK3':   SSP-RK3, 3rd-order strong-stability-preserving (default)
          - 'RK4':   classical 4th-order Runge–Kutta

        Set ``use_treecode=True`` to evaluate the rate from the O(N log N)
        treecode gradient instead of the O(N²) pairwise kernel (large N).
        """
        return StretchingConfig(
            mode="MIXED",
            scheme=scheme,
            use_treecode=use_treecode,
            treecode_theta=treecode_theta,
            conserve_moments=conserve_moments,
            conserve_energy=conserve_energy,
            reformulated=reformulated,
        )

    @staticmethod
    def disabled():
        return StretchingConfig(enabled=False)


# VLM definitions are independent of the runtime solver.


# =========================================================
# TURBULENCE CONFIGURATION
# =========================================================
