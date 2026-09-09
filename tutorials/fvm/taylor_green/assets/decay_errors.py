"""Volume-weighted errors against exact Taylor–Green viscous decay."""

import numpy as np

import openonda.fvm as fvm


def relative_l2(numerical: np.ndarray, analytic: np.ndarray, cell_volume: np.ndarray) -> float:
    numerator = np.sum(cell_volume[:, None] * (numerical - analytic) ** 2)
    denominator = np.sum(cell_volume[:, None] * analytic**2)
    return float(np.sqrt(numerator / denominator))


def history_row(
    fvm_solver, exact_velocity, kinematic_viscosity, initial_total_kinetic_energy, initial_enstrophy
) -> dict[str, float | int]:
    centres = fvm_solver.geo_data["cell_centre"]
    cell_volume = fvm_solver.geo_data["cell_volume"]
    analytic = exact_velocity(centres, fvm_solver.time, kinematic_viscosity)
    total_kinetic_energy = fvm.compute_kinetic_energy(fvm_solver.velocity, fvm_solver.geo_data)
    analytic_total_kinetic_energy = initial_total_kinetic_energy * np.exp(
        -4.0 * kinematic_viscosity * fvm_solver.time
    )
    total_enstrophy = fvm.compute_enstrophy(
        fvm_solver.velocity, fvm_solver.mesh_data, fvm_solver.geo_data
    )
    analytic_total_enstrophy = initial_enstrophy * np.exp(
        -4.0 * kinematic_viscosity * fvm_solver.time
    )
    continuity = fvm.compute_continuity_error(
        fvm_solver.volumetric_face_flux,
        fvm_solver.mesh_data,
        fvm_solver.geo_data,
    )
    max_continuity_error = np.max(np.abs(continuity) / (cell_volume + 1e-30))
    return {
        "step": fvm_solver.step,
        "time": fvm_solver.time,
        "total_kinetic_energy": total_kinetic_energy,
        "analytic_total_kinetic_energy": analytic_total_kinetic_energy,
        "total_kinetic_energy_relative_error": abs(
            total_kinetic_energy - analytic_total_kinetic_energy
        )
        / analytic_total_kinetic_energy,
        "velocity_l2_error": relative_l2(
            fvm_solver.velocity[: len(cell_volume)], analytic, cell_volume
        ),
        "total_enstrophy": total_enstrophy,
        "analytic_total_enstrophy": analytic_total_enstrophy,
        "total_enstrophy_relative_error": abs(total_enstrophy - analytic_total_enstrophy)
        / analytic_total_enstrophy,
        "max_continuity_error": float(max_continuity_error),
        "max_courant_number": fvm_solver.max_courant_number,
    }
