"""The digest that ties a restart backup to the configuration it was written with.

Every backup format stores :func:`configuration_digest` of its configuration:
the FVM backup, the VPM backup, the VLM state inside it and the coupled
backup. A restart recomputes the digest from the current configuration and
refuses a backup whose digest differs.

What a configuration digest covers
----------------------------------
The digest covers the settings that change the computed solution, as a
mapping of JSON values: numerical schemes, time step and its control,
transport and turbulence models, boundary conditions, mesh motion, the
particle kernel and precision, and the coupling settings. It leaves out what
a continued run may change: the clock end, output, logging, backup and
sampler settings, the execution backend, linear-solver options, the core
count, and the warning and stop limits. The initial fields are not covered
because the saved state replaces them. The mesh, the particles and every
other array of the state are identified by the backup's own checks, not by
this digest.

Two mappings have the same digest when they have the same keys and values.
Key order, and tuple against list, do not matter. A NumPy array is identified
by its data type, shape and the SHA-256 of its bytes, a NumPy scalar by its
Python value, and a function by its module and qualified name only: two
different functions with one name have one digest.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import numpy as np

__all__ = ["canonical_json", "configuration_digest"]


def _encode(value: object) -> object:
    """Return the JSON value that stands for an object json cannot write itself."""
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if callable(value) and hasattr(value, "__qualname__"):
        return {"callable": f"{getattr(value, '__module__', '')}.{value.__qualname__}"}
    raise TypeError(f"Cannot digest a configuration value of type {type(value).__name__}")


def canonical_json(configuration: Mapping[str, object]) -> str:
    """Write a configuration as compact JSON with sorted keys.

    Parameters
    ----------
    configuration : mapping
        JSON values, NumPy arrays and scalars, and functions; see the module
        description.

    Returns
    -------
    str
        The text that :func:`configuration_digest` hashes.

    Raises
    ------
    TypeError
        If a value is none of the above.
    ValueError
        If a float is NaN or infinite.
    """
    return json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_encode
    )


def configuration_digest(configuration: Mapping[str, object]) -> str:
    """Return the SHA-256 digest of a configuration as a hexadecimal string.

    Parameters
    ----------
    configuration : mapping
        The settings that change the solution; see the module description for
        what belongs in it and how its values are identified.

    Returns
    -------
    str
        SHA-256 of the UTF-8 text of :func:`canonical_json`.

    Raises
    ------
    TypeError
        If a value cannot be written.
    ValueError
        If a float is NaN or infinite.
    """
    return hashlib.sha256(canonical_json(configuration).encode("utf-8")).hexdigest()
