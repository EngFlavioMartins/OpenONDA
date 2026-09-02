"""FMM runtime diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FMMDiagnostics:
    """Counters recorded for one evaluator instance."""

    strength_rate_mode: str = "HIERARCHICAL_GRADIENT"

    stage_evaluations: int = 0
    hierarchy_builds: int = 0
    p2p_interactions: int = 0
    m2l_interactions: int = 0
    p2m_operations: int = 0
    m2m_operations: int = 0
    l2l_operations: int = 0
    nonzero_l2l_operations: int = 0
    l2p_evaluations: int = 0
    gradient_evaluations: int = 0
    hierarchical_strength_rates: int = 0
    direct_strength_rate_fallbacks: int = 0
    host_particle_transfers: int = 0
    last_uncorrected_rate_defect: float = 0.0
    last_strength_rate_norm: float = 0.0
    last_relative_rate_defect: float = 0.0
    peak_node_count: int = 0
    peak_interaction_list_count: int = 0
    device_memory_estimate_bytes: int = 0
    last_tree_build_seconds: float = 0.0
    last_upward_pass_seconds: float = 0.0
    last_interaction_list_seconds: float = 0.0
    last_m2l_seconds: float = 0.0
    last_downward_pass_seconds: float = 0.0
    last_near_field_seconds: float = 0.0
    last_strength_rate_seconds: float = 0.0


__all__ = ["FMMDiagnostics"]
