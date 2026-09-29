"""Volume-weighted errors against exact Taylor–Green viscous decay."""

import numpy as np


def relative_l2(numerical: np.ndarray, analytic: np.ndarray, cell_volume: np.ndarray) -> float:
    numerator = np.sum(cell_volume[:, None] * (numerical - analytic) ** 2)
    denominator = np.sum(cell_volume[:, None] * analytic**2)
    return float(np.sqrt(numerator / denominator))


def flow_integrals(fields) -> tuple[float, float]:
    """Energy and enstrophy integrated over every physical cell exactly once."""
    energy = 0.5 * np.sum(fields.cell_volume * np.sum(fields.velocity**2, axis=1))
    enstrophy = 0.5 * np.sum(fields.cell_volume * np.sum(fields.vorticity**2, axis=1))
    return float(energy), float(enstrophy)


def history_row(
    fields, exact_velocity, kinematic_viscosity, initial_total_kinetic_energy, initial_enstrophy
) -> dict[str, float | int]:
    centres = fields.cell_centre
    cell_volume = fields.cell_volume
    analytic = exact_velocity(centres, fields.time, kinematic_viscosity)
    total_kinetic_energy, total_enstrophy = flow_integrals(fields)
    analytic_total_kinetic_energy = initial_total_kinetic_energy * np.exp(
        -4.0 * kinematic_viscosity * fields.time
    )
    analytic_total_enstrophy = initial_enstrophy * np.exp(-4.0 * kinematic_viscosity * fields.time)
    return {
        "step": fields.step,
        "time": fields.time,
        "total_kinetic_energy": total_kinetic_energy,
        "analytic_total_kinetic_energy": analytic_total_kinetic_energy,
        "total_kinetic_energy_relative_error": abs(
            total_kinetic_energy - analytic_total_kinetic_energy
        )
        / analytic_total_kinetic_energy,
        "velocity_l2_error": relative_l2(fields.velocity, analytic, cell_volume),
        "total_enstrophy": total_enstrophy,
        "analytic_total_enstrophy": analytic_total_enstrophy,
        "total_enstrophy_relative_error": abs(total_enstrophy - analytic_total_enstrophy)
        / analytic_total_enstrophy,
        "max_continuity_error": fields.max_continuity_error,
        "max_courant_number": fields.max_courant_number,
    }
