"""Every radial vortex kernel the solver can run, by name.

Each kernel module defines its equations once, in a ``KernelDefinition``;
this registry is the only list of kernels. Configuration checks, induction
backends, diagnostics and slip-slab images all read it.
"""

from . import gaussian, high_order_gaussian, super_gaussian, winckelmans
from .definition import KernelDefinition

KERNELS: dict[str, KernelDefinition] = {
    definition.name: definition
    for definition in (
        gaussian.DEFINITION,
        high_order_gaussian.DEFINITION,
        super_gaussian.DEFINITION,
        winckelmans.DEFINITION,
    )
}


def kernel_definition(name: str) -> KernelDefinition:
    """Return the registered definition of a case-insensitive kernel name."""
    try:
        return KERNELS[str(name).upper()]
    except KeyError as exc:
        raise ValueError(
            f"unsupported vortex kernel {name!r}; registered kernels: {sorted(KERNELS)}"
        ) from exc


def kernel_names() -> frozenset[str]:
    """Names of every registered kernel."""
    return frozenset(KERNELS)


def treecode_kernel_names() -> tuple[str, ...]:
    """Kernels with treecode far-field expansions, ordered by their treecode id."""
    tree = [d for d in KERNELS.values() if d.treecode_kernel_id is not None]
    return tuple(d.name for d in sorted(tree, key=lambda d: d.treecode_kernel_id))
