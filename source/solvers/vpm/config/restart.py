"""Shared compatibility rules for standalone and coupled VPM restarts."""

from copy import deepcopy
from typing import Any


def _configuration_mismatches(
    expected: Any,
    found: Any,
    path: str = "",
) -> list[str]:
    """Return incompatible paths, allowing more storage for a fixed algorithm."""
    if isinstance(expected, dict) and isinstance(found, dict):
        paths: list[str] = []
        for key in sorted(set(expected) | set(found)):
            child_path = f"{path}.{key}" if path else key
            if child_path in {"compute_device", "device_memory_fraction"}:
                # Device selection and retired pool sizing are operational.
                # Precision, kernels, and every physical setting still match.
                continue
            if child_path == "max_n_particles":
                old_capacity, new_capacity = found.get(key), expected.get(key)
                if (
                    type(old_capacity) is int
                    and type(new_capacity) is int
                    and new_capacity > old_capacity
                ):
                    # Population is bounded strictly by capacity: no
                    # active operator truncates a field or changes algorithms
                    # when a proposal exceeds it. A larger allocation can
                    # resume the same configuration after an exhausted one.
                    continue
            if key not in expected or key not in found:
                paths.append(child_path)
            else:
                paths.extend(_configuration_mismatches(expected[key], found[key], child_path))
        return paths
    if isinstance(expected, list) and isinstance(found, list):
        if len(expected) != len(found):
            return [path]
        paths = []
        for index, (expected_item, found_item) in enumerate(zip(expected, found, strict=True)):
            paths.extend(_configuration_mismatches(expected_item, found_item, f"{path}[{index}]"))
        return paths
    return [] if expected == found else [path]


def _normalize_capacity_aliases(configuration: dict[str, Any]) -> None:
    """Canonicalize optional caps that resolve to the container capacity."""
    particle_capacity = configuration.get("max_n_particles")
    if type(particle_capacity) is not int:
        return

    viscous = configuration.get("viscous")
    if isinstance(viscous, dict):
        for name in ("dvh_max_nodes", "gbd_max_nodes"):
            # Retired operational caps do not alter the restored saved state.
            # Future regeneration uses only the hard solver particle capacity.
            viscous.pop(name, None)

    stabilization = configuration.get("stabilization")
    if not isinstance(stabilization, dict):
        return
    divergence_relaxation = stabilization.get("divergence_relaxation")
    if isinstance(divergence_relaxation, dict):
        # This retired allocation ceiling never changed an accepted field.
        divergence_relaxation.pop("max_grid_nodes", None)
    feedback_inactive = (
        stabilization.get("selective_eddy_viscosity_coefficient", 0.0) == 0.0
        or stabilization.get("selective_eddy_viscosity_feedback_gain", 0.0) == 0.0
    )
    if feedback_inactive:
        for name in (
            "selective_eddy_viscosity_feedback_gain",
            "selective_eddy_viscosity_feedback_interval_steps",
            "selective_eddy_viscosity_feedback_growth_limit",
            "selective_eddy_viscosity_max_coefficient",
        ):
            stabilization.pop(name, None)
    if (
        stabilization.get("regularization_interval_steps", 0) == 0
        or stabilization.get("regularization_max_events") is None
    ):
        stabilization.pop("regularization_max_events", None)

    inactive = stabilization.get("regularization_interval_steps") == 0
    if inactive or stabilization.get("regularization_max_particles") in (
        None,
        particle_capacity,
    ):
        stabilization.pop("regularization_max_particles", None)
    # Removed capacity-specific settings are compatible only when inactive or
    # equivalent to the ordinary remeshing configuration.
    standard_limit = stabilization.get("regularization_max_particles") or particle_capacity
    for name in tuple(stabilization):
        if not name.startswith("regularization_capacity_"):
            continue
        value = stabilization[name]
        if name == "regularization_capacity_fraction":
            equivalent = value == 1.0
        elif name == "regularization_capacity_max_particles":
            equivalent = value is None or value == standard_limit
        elif name == "regularization_capacity_energy_rate_trigger":
            equivalent = value is None
        else:
            ordinary_name = name.replace("regularization_capacity_", "regularization_")
            equivalent = value is None or value == stabilization.get(ordinary_name)
        if inactive or equivalent:
            stabilization.pop(name)

    refinement = stabilization.get("filament_refinement")
    if isinstance(refinement, dict):
        for name, default in (
            ("late_interval_steps", None),
            ("late_start_step", None),
            ("late_absolute_only", False),
            ("end_step", None),
        ):
            if refinement.get("interval_steps", 0) == 0 or refinement.get(name, default) == default:
                refinement.pop(name, None)
    if isinstance(refinement, dict) and refinement.get("max_n_particles") in (
        None,
        particle_capacity,
    ):
        refinement.pop("max_n_particles", None)


def canonical_restart_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    """Return a compatibility copy, leaving authenticated input untouched."""
    result = deepcopy(configuration)
    _normalize_capacity_aliases(result)
    result.pop("compute_device", None)
    result.pop("device_memory_fraction", None)
    result.pop("max_evaluation_points", None)
    induction = result.get("induction")
    if isinstance(induction, dict):
        policy = induction.get("gaussian_mesh_policy")
        if (isinstance(policy, dict)
                and policy.get("tail_contract") == "gaussian_interval_remainder_v1"
                and policy.get("backend") in {"auto", "cpu", "cupy_cuda"}):
            # The same finite Gaussian/cardinal operator can execute with
            # CUDA FFTs or bounded host FFT blocks. Authenticate the saved
            # mapping first; only execution placement is portable on restart.
            # Mesh, cores, images, tail gates and precision remain exact.
            policy.pop("backend")
    return result
