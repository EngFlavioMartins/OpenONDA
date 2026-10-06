"""Quantify native scatter damping and phase error at resolved wake scales."""

import argparse
import json
from pathlib import Path

import numpy as np

from source.coupler.stable_renewal import _m4_prime_scalar
from source.solvers.vpm.physics.diffusion.grid import _lagrange6_weights


def measurements():
    rows = []
    viscosity = 1 / 150
    velocity = 0.68
    for spacing, interval in ((0.04, 0.04), (0.04, 0.008), (0.02, 0.04)):
        displacement = velocity * interval / spacing
        integer = np.floor(displacement)
        fraction = displacement - integer
        for stencil in (4, 6):
            offsets = np.arange(stencil) - (1 if stencil == 4 else 2)
            weights = (
                np.array([_m4_prime_scalar(fraction - index) for index in offsets])
                if stencil == 4
                else _lagrange6_weights(np.array([fraction]))[0]
            )
            for wavelength in (0.16, 0.24, 0.32, 0.64, 1.28):
                wave_number = 2 * np.pi / wavelength
                symbol = np.sum(weights * np.exp(-1j * wave_number * spacing * (integer + offsets)))
                phase_error = np.angle(symbol * np.exp(1j * wave_number * velocity * interval))
                artificial_viscosity = -np.log(abs(symbol)) / (wave_number**2 * interval)
                rows.append(
                    {
                        "particle_spacing_m": spacing,
                        "remeshing_interval_s": interval,
                        "advection_velocity_m_per_s": velocity,
                        "stencil_nodes_per_axis": stencil,
                        "wavelength_m": wavelength,
                        "amplitude_retained_per_remesh": float(abs(symbol)),
                        "amplitude_retained_after_2s": float(abs(symbol) ** (2 / interval)),
                        "equivalent_artificial_viscosity_m2_per_s": float(artificial_viscosity),
                        "artificial_to_molecular_viscosity_ratio": float(
                            artificial_viscosity / viscosity
                        ),
                        "relative_advection_velocity_error": float(
                            -phase_error / (wave_number * interval * velocity)
                        ),
                    }
                )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    arguments.output.write_text(
        json.dumps(
            {
                "scope": "Exact Fourier symbol of native one-dimensional scatter weights under constant translation. Molecular diffusion, renewal and nonlinear wake feedback are excluded; this quantifies an error mechanism, not autonomous force recovery.",
                "measurements": measurements(),
            },
            indent=2,
        )
        + "\n"
    )
