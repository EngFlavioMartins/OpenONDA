"""Central stage right-hand side for coupled VPM evolution."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

import taichi as ti

from openonda.core.profiling import Profiler

from .induction.base import (
    _STRETCHING_MODES,
    InductionMethod,
    StageRates,
    StageState,
    normalize_stretching_scheme,
)
from .induction.reuse import AcceptedStateInduction, InductionReuseStatistics

if TYPE_CHECKING:
    from ..boundary_elements.vlm.solver.vlm_solver import VLMSolver
    from ..numerics.rk_tableaux import RKTableau
    from ..particles.container import Particles
    from .base import PhysicsBase


class ExternalStageContribution(Protocol):
    """Add explicitly modelled external rates at one RK stage.

    A provider may add velocity only, velocity plus a gradient, or a direct
    vortex-strength rate. It must use the supplied :class:`StageState`; the
    accepted particle fields are not part of this protocol.
    """

    def add_stage_rates(
        self, stage_state: StageState, stage_time: float, stage_rates: StageRates
    ) -> None:
        """Accumulate external velocity/rate contributions into stage outputs."""


class VLMStageContribution:
    """Add the VLM bound field and its strip reaction at RK stage positions.

    The accepted circulation is held fixed while the stage targets move.
    Vortex strength is exchanged with the bound strips only through the
    accepted RK integration step; no stage query emits wake particles or
    commits VLM history.
    """

    def __init__(self, vlm_solver: "VLMSolver", physics: "PhysicsBase") -> None:
        """Create a stage provider backed by a solved VLM object.

        Parameters
        ----------
        vlm_solver : VLMSolver
            Solver that evaluates its bound field at the temporary stage
            positions supplied later.
        physics : PhysicsEngine
            Source of the stretching scheme applied to the VLM velocity
            gradient; its induction backend must be bound.
        """
        if physics.induction is None:
            raise RuntimeError("VLM stage coupling requires a bound induction backend")
        self.vlm_solver = vlm_solver
        self._stretching_mode = _STRETCHING_MODES[
            normalize_stretching_scheme(physics.induction.stretching_scheme)
        ]
        self._stage_weights: tuple[float, ...] | None = None
        self._consumed_stage_indices: set[int] = set()

    @contextmanager
    def integration_step(
        self, tableau: "RKTableau", time_step_size: float, strength_enabled: bool
    ) -> Iterator[None]:
        """Retain only accepted RK contributions, excluding diagnostic probes.

        Exchange weights are indexed by the explicit RK stage identity supplied
        in :class:`StageState`.  A repeated field query at the same stage can
        still evaluate the VLM field, but it cannot publish a second exchange
        contribution. Published exchange requires an explicit stage index.
        """
        scale = time_step_size if strength_enabled else 0.0
        self._stage_weights = tuple(scale * weight for weight in tableau.b)
        self._consumed_stage_indices = set()
        try:
            with self.vlm_solver.particle_transport_step():
                yield
        finally:
            self._stage_weights = None
            self._consumed_stage_indices.clear()

    def _weight_for_stage(self, stage_state: StageState, *, publish: bool) -> float:
        """Return one quadrature weight for a stage, at most once per trial."""
        if not publish or self._stage_weights is None:
            return 0.0
        stage_index = stage_state.stage_index
        if stage_index is None:
            raise ValueError("VLM exchange requires an explicit RK stage_index")
        stage_index = int(stage_index)
        if stage_index < 0 or stage_index >= len(self._stage_weights):
            return 0.0
        if stage_index in self._consumed_stage_indices:
            return 0.0
        self._consumed_stage_indices.add(stage_index)
        return float(self._stage_weights[stage_index])

    def add_stage_rates(
        self, stage_state: StageState, stage_time: float, stage_rates: StageRates
    ) -> None:
        """Add VLM-induced velocity and stretching rates.

        The VLM solution is held fixed during the RK step, but targets are the
        current temporary stage positions.  ``stage_rates.velocity`` and the
        optional gradient/rate fields are mutated in place.
        """
        weight = self._weight_for_stage(
            stage_state,
            publish=bool(stage_rates.strength_rate_enabled),
        )
        self.vlm_solver.add_stage_rates(
            stage_state,
            stage_time,
            stage_rates,
            self._stretching_mode,
            weight if stage_rates.strength_rate_enabled else 0.0,
        )


class StageRHS:
    """Combine selected self-induced induction and external stage providers.

    The object has no accepted-state access.  Every provider receives the exact
    temporary position/strength fields and physical time of the current RK
    stage.  Providers that only model velocity leave the strength-rate field
    untouched; providers that model external stretching or forcing explicitly
    add the corresponding rate.  The self-induced field of one accepted
    particle revision is stored, so the post-step state refresh and the next
    step's first RK stage evaluate the induction once.
    """

    def __init__(
        self,
        induction: InductionMethod,
        providers: tuple[ExternalStageContribution, ...] = (),
        *,
        strength_enabled: bool = True,
        reuse_induction: bool = True,
        retain_strength_rate: bool = True,
        profiler: Profiler | None = None,
    ) -> None:
        """Create a central RHS evaluator for one RK stage.

        Parameters
        ----------
        induction : InductionMethod
            Self-induced velocity, velocity-gradient, and vortex-strength-rate
            provider.
        providers : tuple of ExternalStageContribution, default=()
            Additional stage-consistent providers, evaluated in order after
            self induction.  Their rates are accumulated into the same output
            fields.
        strength_enabled : bool, default=True
            Global switch for vortex-strength evolution.  False zeros the
            strength-rate output after all providers run, while velocity and
            diagnostic gradients remain evaluated.
        reuse_induction : bool, default=True
            Reuse the self-induced field of an accepted particle state for
            later requests at the same ``StageState.accepted_revision``.
            Providers are evaluated for every request. False evaluates the
            induction for every request.
        retain_strength_rate : bool, default=True
            Also evaluate the self-induced strength rate when an accepted
            state is evaluated without it, so the next step's first RK stage
            can reuse that state. Set False when a split operator always
            changes the sources before the first stage, as core spreading does.
        profiler : Profiler or None, default=None
            Step profiler that receives the phases ``"Induction"``,
            ``"Wall projection"`` and ``"External stage rates"``. ``None``
            times nothing.
        """
        self.induction = induction
        self.providers = tuple(providers)
        self.strength_enabled = bool(strength_enabled)
        self.reuse_induction = bool(reuse_induction)
        self.retain_strength_rate = bool(retain_strength_rate)
        self.profiler = Profiler(enabled=False) if profiler is None else profiler
        self._accepted_state = AcceptedStateInduction()
        self._reuse_statistics = InductionReuseStatistics()
        self._closed = False
        # A position guard projects the stage positions in place and returns
        # True when it moved any of them.
        self.position_guard: Callable[[StageState], bool] | None = None
        self.accepted_position_projector: Callable[..., None] | None = None

    @property
    def induction_reuse_statistics(self) -> InductionReuseStatistics:
        """Cumulative counts of induction evaluations and reused accepted states."""
        return replace(self._reuse_statistics)

    def close(self) -> None:
        """Release owned private fields before the Taichi runtime is reset."""
        if self._closed:
            return
        self._closed = True
        self._accepted_state.close()

    def _reuse_accepted_state(self, stage_state: StageState, stage_rates: StageRates) -> bool:
        """Publish the stored self-induced field of an unchanged accepted state."""
        revision = stage_state.accepted_revision
        if (
            not self.reuse_induction
            or revision is None
            or not self._accepted_state.matches(
                self.induction,
                revision,
                int(stage_state.count),
                stage_rates.velocity_gradient is not None,
                bool(stage_rates.strength_rate_enabled),
            )
        ):
            return False
        self._accepted_state.publish(
            velocity_out=stage_rates.velocity,
            vortex_strength_rate_out=stage_rates.vortex_strength_rate,
            velocity_gradient_out=stage_rates.velocity_gradient,
            strength_rate_enabled=bool(stage_rates.strength_rate_enabled),
        )
        self._reuse_statistics.reuses += 1
        return True

    def evaluate_induction(
        self, stage_state: "StageState", stage_time: float, stage_rates: "StageRates"
    ) -> None:
        """Evaluate the self-induced field of one stage with the induction backend.

        Every backend evaluation passes through this method and is timed as
        the phase ``"Induction"``; reused accepted states do not.
        An accepted state (``stage_state.accepted_revision`` set) is stored
        for later requests at the same revision. No guard, provider, or
        particle state check is part of this evaluation.
        """
        if self._closed:
            raise RuntimeError("StageRHS is closed")
        self._reuse_statistics.evaluations += 1
        with self.profiler.section("Induction"):
            self._evaluate_induction(stage_state, stage_time, stage_rates)

    def _evaluate_induction(
        self, stage_state: "StageState", stage_time: float, stage_rates: "StageRates"
    ) -> None:
        count = int(stage_state.count)
        revision = stage_state.accepted_revision
        if not self.reuse_induction or revision is None or count == 0:
            self.induction.evaluate_stage(
                position=stage_state.position,
                vortex_strength=stage_state.vortex_strength,
                core_radius=stage_state.core_radius,
                count=stage_state.count,
                velocity_out=stage_rates.velocity,
                vortex_strength_rate_out=stage_rates.vortex_strength_rate,
                velocity_gradient_out=stage_rates.velocity_gradient,
                strength_rate_enabled=stage_rates.strength_rate_enabled,
                stage_time=stage_time,
            )
            return
        strength_rate = bool(stage_rates.strength_rate_enabled)
        self._accepted_state.evaluate(
            self.induction,
            revision,
            position=stage_state.position,
            vortex_strength=stage_state.vortex_strength,
            core_radius=stage_state.core_radius,
            count=count,
            velocity_out=stage_rates.velocity,
            gradient=stage_rates.velocity_gradient is not None,
            strength_rate=strength_rate or self.retain_strength_rate,
            stage_time=stage_time,
        )
        self._accepted_state.publish(
            velocity_out=stage_rates.velocity,
            vortex_strength_rate_out=stage_rates.vortex_strength_rate,
            velocity_gradient_out=stage_rates.velocity_gradient,
            strength_rate_enabled=strength_rate,
        )

    @contextmanager
    def integration_step(self, tableau: "RKTableau", time_step_size: float) -> Iterator[None]:
        """Commit provider exchange increments with the common particle RK step."""
        with ExitStack() as stack:
            for provider in self.providers:
                context = getattr(provider, "integration_step", None)
                if context is not None:
                    stack.enter_context(context(tableau, time_step_size, self.strength_enabled))
            yield

    @property
    def requires_velocity_gradient(self) -> bool:
        """Whether a provider needs the self-induced Jacobian at every stage.

        Most direct induction calls only need velocity and stretching, and can
        use their fused velocity/rate path, so RK pays for a separate gradient
        traversal only when a provider asks for it.
        """
        return any(
            bool(getattr(provider, "requires_velocity_gradient", False))
            for provider in self.providers
        )

    def evaluate(self, stage_state: StageState, stage_time: float, stage_rates: StageRates) -> None:
        """Evaluate and accumulate all rates for one common RK stage.

        Parameters
        ----------
        stage_state : StageState
            Temporary stage positions, strengths, core radii, active count,
            and stage time.  No accepted particle field is consulted.
        stage_time : float
            Physical stage time in seconds.
        stage_rates : StageRates
            Writable output fields.  Velocity is m/s, velocity gradient is
            1/s, and vortex-strength rate is m³/s².

        Notes
        -----
        The output fields are mutated in place.  Providers see the same stage
        state and are evaluated sequentially; disabling strength evolution
        zeros the final rate after all provider callbacks complete.  When the
        stage is an accepted state whose self-induced field is stored at the
        same revision, that field is published instead of evaluated again;
        the position guard and every provider still run.
        """
        if self._closed:
            raise RuntimeError("StageRHS is closed")
        if self.position_guard is not None:
            with self.profiler.section("Wall projection"):
                moved = self.position_guard(stage_state)
            if moved and stage_state.accepted_revision is not None:
                # The projected positions are no longer the accepted sources.
                self._accepted_state.invalidate()
                stage_state = replace(stage_state, accepted_revision=None)
        if not self._reuse_accepted_state(stage_state, stage_rates):
            self.evaluate_induction(stage_state, stage_time, stage_rates)
        if self.providers:
            with self.profiler.section("External stage rates"):
                for provider in self.providers:
                    provider.add_stage_rates(stage_state, stage_time, stage_rates)
        if not self.strength_enabled or not stage_rates.strength_rate_enabled:
            _zero_stage_field(stage_rates.vortex_strength_rate, stage_state.count)


@ti.kernel
def _zero_stage_field(field: ti.template(), count: ti.i32):  # type: ignore[invalid-annotation]
    for i in range(count):
        field[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.data_oriented
class ParticleExternalStageContribution:
    """Add the uniform background velocity to the stage velocity.

    Self-induced induction belongs to the selected induction object; the
    freestream is the only external term evaluated at the temporary stage
    positions.
    """

    def __init__(self, particles: "Particles") -> None:
        self.particles = particles

    @ti.kernel
    def _add_background(self, velocity: ti.template(), count: ti.i32):  # type: ignore[invalid-annotation]
        for i in range(count):
            velocity[i] += self.particles.velocity_background[None]

    def add_stage_rates(
        self, stage_state: StageState, stage_time: float, stage_rates: StageRates
    ) -> None:
        """Add the background velocity to the stage velocity."""
        del stage_time
        if stage_state.count:
            self._add_background(stage_rates.velocity, stage_state.count)


__all__ = [
    "ExternalStageContribution",
    "ParticleExternalStageContribution",
    "StageRHS",
    "VLMStageContribution",
]
