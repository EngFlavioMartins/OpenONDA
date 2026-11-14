"""Exception classes shared by the solver packages."""


class NonFiniteSolutionError(FloatingPointError):
    """A field, a diagnostic or a model output holds a NaN or an infinity."""
