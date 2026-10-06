"""Test linear prediction of the accepted interface renewal correction."""

from contextlib import contextmanager

import numpy as np

from source.coupler.interface_prediction import _same_prediction_inputs


@contextmanager
def linearly_predicted_interface(owner, measurements):
    """Change the initial Picard estimate, retaining native acceptance checks."""
    history = owner.interface_predictor
    original = history.prepare
    older = None
    measurements.update(extrapolated_seeds=0, original_seeds=0)

    def prepare(coupler, geometry, old, raw):
        nonlocal older
        previous = history._history
        seed, information = original(coupler, geometry, old, raw)
        if seed is not None:
            measurements["original_seeds"] += 1
            if (
                older is not None
                and previous is not None
                and _same_prediction_inputs(older["inputs"], previous["inputs"])
                and abs(previous["clock"][1] - older["clock"][1] - coupler.vpm_time_step_size)
                < 1e-9
            ):
                correction = previous["correction"]
                preceding = older["correction"]
                velocity = seed[0] + correction[0] - preceding[0]
                normal = np.einsum("ij,ij->i", velocity, geometry[1])
                gradient = seed[2] + correction[2] - preceding[2]
                seed = velocity, normal, gradient
                measurements["extrapolated_seeds"] += 1
                information["reason"] = "linearly_predicted_accepted_correction"
        older = previous if seed is not None else None
        return seed, information

    history.prepare = prepare
    try:
        yield
    finally:
        history.prepare = original
