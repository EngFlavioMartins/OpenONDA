"""Shared types and helpers for analytical initial-condition models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from ..data import ParticleDistribution, VortexParticleSet


class DistributionBuilder(Protocol):
    """Immutable geometry builder for analytical flows."""

    def build(self) -> ParticleDistribution:
        """Build particle geometry and quadrature."""


class InitialCondition(Protocol):
    """Typed builder consumed by ``VPMCase.initial_conditions``."""

    def build(self, distribution: ParticleDistribution | None = None) -> VortexParticleSet:
        """Attribute a flow to supplied or configured particle geometry."""


class InitialVelocity(StrEnum):
    """How an initializer populates its particle velocity field."""

    ZERO = "zero"
    ANALYTICAL = "analytical"


@dataclass(frozen=True, slots=True)
class ParticleCoreCompensation:
    """Kernel-aware physical-core correction for blob representations.

    ``kernel_diffusivity`` is the positive diffusion coefficient used by the
    selected kernel model. Omit this correction to leave the requested physical
    vortex-core radius uncorrected.
    """

    kernel_diffusivity: float = 4.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.kernel_diffusivity) or self.kernel_diffusivity <= 0.0:
            raise ValueError("kernel_diffusivity must be finite and positive")


DistributionSource = ParticleDistribution | DistributionBuilder | None


def constant_group_id(group_id: int | None, count: int) -> np.ndarray | None:
    """Return a validated constant int32 group field for a particle set."""
    if group_id is None:
        return None
    if isinstance(group_id, bool) or not isinstance(group_id, (int, np.integer)):
        raise ValueError("group_id must be an integer")
    info = np.iinfo(np.int32)
    if group_id < info.min or group_id > info.max:
        raise ValueError("group_id must fit in int32")
    return np.full(count, group_id, dtype=np.int32)


def resolve_distribution(
    supplied: ParticleDistribution | None, configured: DistributionSource
) -> ParticleDistribution:
    """Resolve an explicit build argument or the model's configured geometry."""
    distribution = supplied if supplied is not None else configured
    if isinstance(distribution, ParticleDistribution):
        return distribution
    if distribution is None:
        raise ValueError(
            "distribution is required; pass build(distribution) or configure distribution="
        )
    return distribution.build()
