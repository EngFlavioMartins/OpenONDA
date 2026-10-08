"""Current numerical configuration for standalone and coupled VPM restarts."""

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
            if child_path == "compute_device":
                # Execution placement is operational.
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
    return [] if type(expected) is type(found) and expected == found else [path]


def restart_configuration_values(configuration: dict[str, Any]) -> dict[str, Any]:
    """Return the physical configuration, leaving validated input untouched."""
    result = deepcopy(configuration)
    result.pop("compute_device", None)
    induction = result.get("induction")
    if isinstance(induction, dict):
        settings = induction.get("gaussian_mesh_settings")
        if (
            isinstance(settings, dict)
            and settings.get("tail_error_method") == "gaussian_interval_remainder_v1"
            and settings.get("backend") in {"auto", "cpu", "cupy_cuda"}
        ):
            # The same finite Gaussian/cardinal operator can execute with
            # CUDA FFTs or bounded host FFT blocks. Validate the saved
            # mapping first; only execution placement is portable on restart.
            # Mesh, cores, images, tail checks and precision remain exact.
            settings.pop("backend")
    return result
