"""Declarative boundary capabilities shared by all FVM operators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class BoundaryStrategy(Enum):
    """Enumeration of canonical boundary-condition behaviours.

    Each member corresponds to one of the supported OpenFOAM boundary types
    and determines how the discrete operators (gradient, convection,
    diffusion, flux) treat the boundary faces.
    """

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
    FIXED_FLUX_PRESSURE = auto()
    FIXED_GRADIENT = auto()


@dataclass(frozen=True)
class BoundaryOperator:
    """Registration record binding a boundary type to its discrete operators.

    Associates a boundary-condition name (e.g. ``"fixedValue"``) with the
    set of fields it applies to, the set of discrete operators it
    implements, and the :class:`BoundaryStrategy` that controls the
    numerical treatment.

    Attributes
    ----------
    name : str
        OpenFOAM boundary-condition name (e.g. ``"fixedValue"``).
    fields : frozenset[str]
        Fields this BC applies to (e.g. ``{"U", "p", "scalar"}``).
    operators : frozenset[str]
        Operators implemented (e.g. ``{"gradient", "convection"}``).
    strategy : BoundaryStrategy
        Canonical behaviour enum value.
    coupling_only : bool
        Whether this BC is only available through the FVM–VPM coupler.
    """

    name: str
    fields: frozenset[str]
    operators: frozenset[str]
    strategy: BoundaryStrategy
    coupling_only: bool = False


_ALL_OPERATORS = frozenset(
    {"gradient", "convection", "diffusion", "pressure", "flux", "ghost", "diagnostics"}
)


class BoundaryRegistry:
    """Registry of all supported boundary-condition operators.

    Enforces uniqueness (each name can be registered only once) and provides
    lookups that validate whether a given boundary name implements the
    required operator for a given field.  Use :meth:`require` before applying
    a boundary treatment; it raises a clear error if the combination is not
    supported.

    The module-level singleton :data:`BOUNDARIES` is pre-populated with all
    standard OpenFOAM boundary types.
    """

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

# Pressure partner for a prescribed-velocity boundary.  The absolute pressure
# carries a momentum-compatible normal gradient, while the pressure correction
# is homogeneous Neumann so that the prescribed boundary flux is not changed.
BOUNDARIES.register(
    BoundaryOperator(
        "fixedFluxPressure",
        frozenset({"p"}),
        _ALL_OPERATORS,
        BoundaryStrategy.FIXED_FLUX_PRESSURE,
        coupling_only=True,
    )
)

BOUNDARIES.register(
    BoundaryOperator(
        "fixedGradient",
        frozenset({"p"}),
        _ALL_OPERATORS,
        BoundaryStrategy.FIXED_GRADIENT,
        coupling_only=True,
    )
)
