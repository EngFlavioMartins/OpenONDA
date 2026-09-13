"""Contracts shared by VPM induction backends.

The contract deliberately describes rates rather than a particular numerical
algorithm.  A caller supplies the complete temporary RK stage state and owns
the output fields.  Implementations must read only that supplied state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, cast

StretchingScheme = Literal["DIRECT", "TRANSPOSED", "MIXED"]
_STRETCHING_MODES = {"DIRECT": 0, "TRANSPOSED": 1, "MIXED": 2}


def normalize_stretching_scheme(scheme: str) -> StretchingScheme:
    """Normalize and validate the vortex-strength stretching formulation.

    Parameters
    ----------
    scheme : str
        Case-insensitive spelling of ``"DIRECT"``, ``"TRANSPOSED"``, or
        ``"MIXED"``. The value selects the discrete representation of
        ``dGamma/dt``; it does not select a velocity-induction backend.

    Returns
    -------
    StretchingScheme
        The upper-case canonical literal used by backend constructors.

    Raises
    ------
    ValueError
        If ``scheme`` is not one of the three supported formulations.
    """
    normalized = str(scheme).upper()
    if normalized not in _STRETCHING_MODES:
        raise ValueError(
            f"stretching_scheme must be 'direct', 'transposed', or 'mixed'; got {scheme!r}"
        )
    return cast(StretchingScheme, normalized)


@dataclass(frozen=True, slots=True)
class StageState:
    """Immutable view of the particle source state at one RK stage.

    Attributes
    ----------
    position : object
        Taichi vector field with logical shape ``(count, 3)`` in metres.
    vortex_strength : object
        Taichi vector field with logical shape ``(count, 3)`` in m³/s. This
        is the particle-strength vector ``Gamma = omega * V``.
    core_radius : object
        Taichi scalar field with logical shape ``(count,)`` in metres.
    count : int
        Number of active entries in the fixed-capacity fields. Only the prefix
        ``[0:count)`` may be read or written by an evaluator.
    time : float, default=0.0
        Physical stage time in seconds, used by time-dependent closures or
        diagnostics. Induction itself is otherwise autonomous.
    stage_index : int or None, default=None
        Explicit zero-based RK stage identity within the current trial.  The
        value is ``None`` for ad-hoc field queries outside an integrator.  It
        is intentionally separate from physical time because equal-time RK
        stages may carry different temporary particle states.

    Notes
    -----
    These fields are temporary RK-stage values and are not necessarily the
    solver's accepted particle state. The caller retains ownership and must
    not assume that an evaluator copies them.
    """

    position: object
    vortex_strength: object
    core_radius: object
    count: int
    time: float = 0.0
    stage_index: int | None = None


@dataclass(frozen=True, slots=True)
class StageRates:
    """Device fields receiving rates from one induction evaluation.

    Attributes
    ----------
    velocity : object
        Output field with logical shape ``(count, 3)`` in m/s.
    vortex_strength_rate : object
        Output field with logical shape ``(count, 3)`` in m³/s². It is the
        selected stretching formulation's rate for particle strength.
    velocity_gradient : object or None, default=None
        Optional output tensor field with logical shape ``(count, 3, 3)`` in
        1/s. ``None`` means no gradient is requested.
    strength_rate_enabled : bool, default=True
        Whether the caller expects a non-zero strength rate. Backends must
        write zeros when this is false so stale stage data cannot leak into an
        RK update.
    """

    velocity: object
    vortex_strength_rate: object
    velocity_gradient: object | None = None
    strength_rate_enabled: bool = True


class InductionMethod(Protocol):
    """Protocol implemented by every VPM velocity-induction backend.

    A backend reads only the explicitly supplied source stage and writes only
    the caller-owned output fields. ``evaluate_stage`` is used for particles
    advected by the RK integrator; ``evaluate_targets`` is used for probes,
    coupling points, and boundary elements. Backend capability flags are
    class attributes so case validation can reject unsupported combinations
    before a run begins.
    """

    supported_kernels: ClassVar[frozenset[str]]
    supports_gradient: ClassVar[bool]
    supports_variable_core_radius: ClassVar[bool]
    supports_f64: ClassVar[bool]
    supports_target_fields: ClassVar[bool]
    device_resident: ClassVar[bool]
    stretching_scheme: StretchingScheme

    def build(self) -> InductionMethod:
        """Return an evaluator bound to a solver's device/runtime state.

        Returns
        -------
        InductionMethod
            Backend instance ready for :meth:`evaluate_stage` and, when
            supported, :meth:`evaluate_targets`. The returned object may own
            acceleration structures or device allocations.
        """

    def evaluate_stage(
        self,
        *,
        position: object,
        vortex_strength: object,
        core_radius: object,
        count: int,
        velocity_out: object,
        vortex_strength_rate_out: object,
        velocity_gradient_out: object | None = None,
        strength_rate_enabled: bool = True,
        stage_time: float = 0.0,
    ) -> None:
        """Evaluate particle velocity and stretching rates for one stage.

        The Jacobian includes the finite skew contribution at every regularized
        source centre, including self. Self velocity and self stretching vanish;
        the curl used by vorticity diagnostics and relaxation does not.

        Parameters
        ----------
        position, vortex_strength, core_radius : object
            Caller-owned Taichi fields containing the stage sources with
            logical shapes ``(count, 3)``, ``(count, 3)``, and ``(count,)``;
            units are m, m³/s, and m.
        count : int
            Active source count. Capacity entries after this prefix are
            ignored and must not affect the result.
        velocity_out : object
            ``(count, 3)`` output field in m/s.
        vortex_strength_rate_out : object
            ``(count, 3)`` output field in m³/s².
        velocity_gradient_out : object or None, default=None
            Optional ``(count, 3, 3)`` output in 1/s. Backends that do not
            support gradients must reject a non-``None`` request during case
            validation rather than silently returning unrelated data.
        strength_rate_enabled : bool, default=True
            If false, write a zero strength-rate field while still computing
            velocity and any requested gradient.
        stage_time : float, default=0.0
            Physical stage time in seconds.

        Notes
        -----
        The method mutates only the supplied output fields and any documented
        backend cache. It must not mutate the source stage.

        Raises
        ------
        ValueError
            If counts, field shapes, precision, kernel, or device settings
            violate the backend's declared capabilities.
        """

    def evaluate_targets(
        self,
        *,
        target_position: object,
        source_position: object,
        source_vortex_strength: object,
        source_core_radius: object,
        target_velocity: object | None,
        target_velocity_gradient: object | None,
        target_count: int,
        source_count: int,
        include_freestream: bool,
        background_velocity: object,
    ) -> None:
        """Evaluate source-induced velocity/gradient at arbitrary targets.

        Parameters
        ----------
        target_position : object
            Target coordinates with logical shape ``(target_count, 3)`` in m.
        source_position, source_vortex_strength, source_core_radius : object
            Source fields with logical shapes ``(source_count, 3)``,
            ``(source_count, 3)``, and ``(source_count,)`` in m, m³/s, and m.
        target_velocity : object or None
            Optional ``(target_count, 3)`` output in m/s.
        target_velocity_gradient : object or None
            Optional ``(target_count, 3, 3)`` output in 1/s.
        target_count, source_count : int
            Active target and source counts. Fixed-capacity tails are ignored.
        include_freestream : bool
            Add ``background_velocity`` to velocity output when true.
        background_velocity : object
            Three-vector freestream in m/s; ignored when
            ``include_freestream`` is false.

        Raises
        ------
        ValueError
            If no output is requested, counts are invalid, or the backend does
            not support arbitrary targets or the requested gradient.
        """


__all__ = ["InductionMethod", "StageRates", "StageState", "StretchingScheme"]
