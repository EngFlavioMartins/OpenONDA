"""Declarative boundary capabilities shared by all FVM operators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryOperator:
    """Boundary condition name and the discrete operators that implement it."""

    name: str
    fields: frozenset[str]
    operators: frozenset[str]
    coupling_only: bool = False


_ALL_OPERATORS = frozenset(
    {"gradient", "convection", "diffusion", "pressure", "flux", "ghost", "diagnostics"}
)


class BoundaryRegistry:
    """Single capability registry used for early boundary validation."""

    def __init__(self) -> None:
        self._entries: dict[str, BoundaryOperator] = {}

    def register(self, entry: BoundaryOperator) -> None:
        if entry.name in self._entries:
            raise ValueError(f"Boundary operator {entry.name!r} is already registered")
        self._entries[entry.name] = entry

    def require(self, name: str, field: str, operator: str) -> BoundaryOperator:
        entry = self._entries.get(name)
        if entry is None or field not in entry.fields or operator not in entry.operators:
            supported = sorted(
                key
                for key, candidate in self._entries.items()
                if field in candidate.fields and operator in candidate.operators
            )
            raise ValueError(
                f"Boundary {name!r} does not implement {operator} for {field}; "
                f"supported: {supported}"
            )
        return entry

    def names_for(self, field: str) -> set[str]:
        return {name for name, entry in self._entries.items() if field in entry.fields}


BOUNDARIES = BoundaryRegistry()
for _name, _fields in (
    ("fixedValue", frozenset({"U", "p"})),
    ("zeroGradient", frozenset({"U", "p"})),
    ("empty", frozenset({"U", "p"})),
    ("noSlip", frozenset({"U"})),
    ("inletOutlet", frozenset({"U"})),
    ("slip", frozenset({"U"})),
    ("symmetry", frozenset({"U"})),
):
    BOUNDARIES.register(BoundaryOperator(_name, _fields, _ALL_OPERATORS))

BOUNDARIES.register(
    BoundaryOperator("directionMixed", frozenset({"U"}), _ALL_OPERATORS, coupling_only=True)
)
