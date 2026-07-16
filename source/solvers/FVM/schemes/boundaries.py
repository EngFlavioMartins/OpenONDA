"""Declarative boundary capabilities shared by all FVM operators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class BoundaryStrategy(Enum):
    """Canonical behavior selected by the boundary registry."""

    FIXED_VALUE = auto()
    ZERO_GRADIENT = auto()
    EMPTY = auto()
    NO_SLIP = auto()
    INLET_OUTLET = auto()
    SLIP = auto()
    SYMMETRY = auto()
    CYCLIC = auto()
    FREESTREAM = auto()
    DIRECTION_MIXED = auto()


@dataclass(frozen=True)
class BoundaryOperator:
    """Boundary condition name and the discrete operators that implement it."""

    name: str
    fields: frozenset[str]
    operators: frozenset[str]
    strategy: BoundaryStrategy
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

    def strategy(self, name: str, field: str, operator: str) -> BoundaryStrategy:
        """Resolve validated operator behavior without per-module name fallback."""
        return self.require(name, field, operator).strategy


BOUNDARIES = BoundaryRegistry()
for _name, _fields, _strategy in (
    ("fixedValue", frozenset({"U", "p", "scalar"}), BoundaryStrategy.FIXED_VALUE),
    ("zeroGradient", frozenset({"U", "p", "scalar"}), BoundaryStrategy.ZERO_GRADIENT),
    ("empty", frozenset({"U", "p", "scalar"}), BoundaryStrategy.EMPTY),
    ("noSlip", frozenset({"U"}), BoundaryStrategy.NO_SLIP),
    ("inletOutlet", frozenset({"U"}), BoundaryStrategy.INLET_OUTLET),
    ("slip", frozenset({"U"}), BoundaryStrategy.SLIP),
    ("symmetry", frozenset({"U"}), BoundaryStrategy.SYMMETRY),
    ("cyclic", frozenset({"U", "p", "scalar"}), BoundaryStrategy.CYCLIC),
    ("freestream", frozenset({"U", "p", "scalar"}), BoundaryStrategy.FREESTREAM),
):
    BOUNDARIES.register(BoundaryOperator(_name, _fields, _ALL_OPERATORS, _strategy))

BOUNDARIES.register(
    BoundaryOperator(
        "directionMixed",
        frozenset({"U"}),
        _ALL_OPERATORS,
        BoundaryStrategy.DIRECTION_MIXED,
        coupling_only=True,
    )
)
