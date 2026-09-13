"""Centred and one-sided normal derivatives at two spatial step sizes."""

import numpy as np


def normal_derivative_estimates(values, step):
    """Second-order estimates from offsets 0, +/-h, +/-2h and +/-h/2."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("A finite positive spatial step is required")
    centre, plus, minus, plus2, minus2, ph, mh = (np.asarray(values[key]) for key in
                                                ("boundary", "plus", "minus", "plus2", "minus2", "plus_half", "minus_half"))
    return {"centred": (plus-minus)/(2*step), "centred_half": (ph-mh)/step,
            "exterior": (-3*centre+4*plus-plus2)/(2*step),
            "interior": (3*centre-4*minus+minus2)/(2*step),
            "exterior_half": (-3*centre+4*ph-plus)/step,
            "interior_half": (3*centre-4*mh+minus)/step}
