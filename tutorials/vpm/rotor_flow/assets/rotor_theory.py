"""
Rotor reference theory for VLM+VPM validation.

Provides a BEM solver matched to the flat-plate VLM polar (lift_coefficient = 2π sinα)
so the comparison is apples-to-apples, plus actuator-disk momentum theory
for far-wake velocity-deficit validation.

Functions
---------
solve_blade_element_momentum(radial_position, chord, twist_angle_radians,
                             n_blades, rotor_radius, freestream_speed,
                             angular_velocity, lift_curve_slope,
                             max_iterations, convergence_tolerance)
    Classic iterative BEM with Glauert-corrected Prandtl tip-loss.
    Returns DataFrame with spanwise aerodynamic quantities and thrust_coefficient, power_coefficient.

actuator_disk_velocity_ratio(axial_induction_factor, normalized_radial_position)
    Far-wake axial velocity from actuator-disk momentum theory.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def solve_blade_element_momentum(
    radial_position: np.ndarray,
    chord: np.ndarray,
    twist_angle_radians: np.ndarray,
    n_blades: int,
    rotor_radius: float,
    freestream_speed: float,
    angular_velocity: float,
    lift_curve_slope: float = 2.0 * np.pi,
    constant_drag_coefficient: float = 0.0,
    max_iterations: int = 200,
    convergence_tolerance: float = 1e-7,
) -> pd.DataFrame:
    """Iterative BEM with Prandtl tip-loss, matched to a flat-plate lift polar.

    Parameters
    ----------
    radial_position: radial stations [m], strictly inside the rotor disk
    chord      : local chord at each station [m], shape (N,)
    twist_angle_radians  : local geometric pitch angle [rad] (positive → LE into wind)
    n_blades   : number of blades
    rotor_radius: rotor tip radius [m]
    freestream_speed      : axial freestream velocity [m/s]
    angular_velocity: rotor angular velocity [rad/s]
    lift_curve_slope   : section lift-curve slope [1/rad] (default 2π for thin plate)
    constant_drag_coefficient   : section drag coefficient (constant; default 0 for flat-plate VLM)
    max_iterations     : max BEM iterations per station
    convergence_tolerance: convergence tolerance on axial induction factor

    Returns
    -------
    pd.DataFrame with columns:
        radial_position, normalized_radial_position, chord, twist_angle_degrees,
        inflow_angle_degrees, angle_of_attack_degrees, lift_coefficient,
        drag_coefficient, axial_induction_factor, tangential_induction_factor,
        thrust_per_radius [N/m], torque_per_radius [N·m/m],
        circulation [m²/s]      (bound circulation Γ = ½ c lift_coefficient relative_speed)
    And scalar attributes stored as df.attrs:
        thrust_coefficient, power_coefficient, thrust [N], torque [N·m], power [W]
    """
    radial_position = np.asarray(radial_position, dtype=float)
    chord = np.asarray(chord, dtype=float)
    twist_angle_radians = np.asarray(twist_angle_radians, dtype=float)
    n_radial_stations = len(radial_position)

    axial_induction_factor_array = np.zeros(n_radial_stations)
    tangential_induction_factor_array = np.zeros(n_radial_stations)
    inflow_angle_array = np.zeros(n_radial_stations)
    angle_of_attack_array = np.zeros(n_radial_stations)
    lift_coefficient_array = np.zeros(n_radial_stations)
    drag_coefficient_array = np.full(n_radial_stations, constant_drag_coefficient)
    thrust_per_radius_array = np.zeros(n_radial_stations)
    torque_per_radius_array = np.zeros(n_radial_stations)
    circulation_array = np.zeros(n_radial_stations)

    for station_index in range(n_radial_stations):
        station_radius = radial_position[station_index]
        station_chord = chord[station_index]
        station_twist_angle = twist_angle_radians[station_index]
        normalized_radius = station_radius / rotor_radius

        axial_induction_factor = 1.0 / 3.0
        tangential_induction_factor = 0.0

        for _ in range(max_iterations):
            # Relative flow angle
            axial_velocity = freestream_speed * (1.0 - axial_induction_factor)
            tangential_velocity = (
                angular_velocity * station_radius * (1.0 + tangential_induction_factor)
            )
            if abs(axial_velocity) < 1e-12 and abs(tangential_velocity) < 1e-12:
                break
            inflow_angle = np.arctan2(axial_velocity, tangential_velocity)

            # Prandtl tip-loss factor (Glauert formulation)
            tip_loss_exponent = (
                (n_blades / 2.0)
                * (1.0 - normalized_radius)
                / (normalized_radius * abs(np.sin(inflow_angle)) + 1e-14)
            )
            tip_loss_exponent = np.clip(tip_loss_exponent, 0.0, 50.0)
            tip_loss_factor = (2.0 / np.pi) * np.arccos(np.exp(-tip_loss_exponent))
            tip_loss_factor = max(tip_loss_factor, 1e-6)

            # Section aerodynamics (flat-plate polar)
            angle_of_attack = inflow_angle - station_twist_angle
            lift_coefficient = lift_curve_slope * np.sin(angle_of_attack)
            drag_coefficient = constant_drag_coefficient

            # Solidity at this station
            solidity = n_blades * station_chord / (2.0 * np.pi * station_radius)

            # Thrust and torque coefficients (Glauert BEM)
            normal_force_coefficient = lift_coefficient * np.cos(
                inflow_angle
            ) + drag_coefficient * np.sin(inflow_angle)
            tangential_force_coefficient = lift_coefficient * np.sin(
                inflow_angle
            ) - drag_coefficient * np.cos(inflow_angle)

            axial_denominator = 4.0 * tip_loss_factor * np.sin(inflow_angle) ** 2
            updated_axial_induction_factor = 1.0 / (
                axial_denominator / (solidity * normal_force_coefficient) + 1.0
            )

            tangential_denominator = (
                4.0 * tip_loss_factor * np.sin(inflow_angle) * np.cos(inflow_angle)
            )
            updated_tangential_induction_factor = 1.0 / (
                tangential_denominator / (solidity * tangential_force_coefficient) - 1.0
            )
            updated_tangential_induction_factor = max(updated_tangential_induction_factor, 0.0)

            if (
                abs(updated_axial_induction_factor - axial_induction_factor) < convergence_tolerance
                and abs(updated_tangential_induction_factor - tangential_induction_factor)
                < convergence_tolerance
            ):
                axial_induction_factor = updated_axial_induction_factor
                tangential_induction_factor = updated_tangential_induction_factor
                break
            axial_induction_factor = updated_axial_induction_factor
            tangential_induction_factor = updated_tangential_induction_factor

        # Converged — store results
        axial_velocity = freestream_speed * (1.0 - axial_induction_factor)
        tangential_velocity = (
            angular_velocity * station_radius * (1.0 + tangential_induction_factor)
        )
        relative_speed = np.sqrt(axial_velocity**2 + tangential_velocity**2)
        inflow_angle = np.arctan2(axial_velocity, tangential_velocity)
        angle_of_attack = inflow_angle - twist_angle_radians[station_index]
        lift_coefficient = lift_curve_slope * np.sin(angle_of_attack)
        circulation = 0.5 * chord[station_index] * lift_coefficient * relative_speed

        density = 1.0  # caller multiplies by density if needed; BEM force is per unit ρ here
        # Use standard BEM rotor-plane element forces
        # differential_thrust = ½ ρ relative_speed² c normal_force_coefficient B dr  →  stored as per-unit-ρ
        normal_force_coefficient = lift_coefficient * np.cos(
            inflow_angle
        ) + constant_drag_coefficient * np.sin(inflow_angle)
        tangential_force_coefficient = lift_coefficient * np.sin(
            inflow_angle
        ) - constant_drag_coefficient * np.cos(inflow_angle)
        differential_thrust = (
            0.5 * relative_speed**2 * chord[station_index] * normal_force_coefficient * n_blades
        )
        differential_torque = (
            0.5
            * relative_speed**2
            * chord[station_index]
            * tangential_force_coefficient
            * n_blades
            * station_radius
        )

        axial_induction_factor_array[station_index] = axial_induction_factor
        tangential_induction_factor_array[station_index] = tangential_induction_factor
        inflow_angle_array[station_index] = inflow_angle
        angle_of_attack_array[station_index] = angle_of_attack
        lift_coefficient_array[station_index] = lift_coefficient
        thrust_per_radius_array[station_index] = differential_thrust
        torque_per_radius_array[station_index] = differential_torque
        circulation_array[station_index] = circulation

    # Integrate thrust and torque
    thrust = float(np.trapezoid(thrust_per_radius_array, radial_position))
    torque = float(np.trapezoid(torque_per_radius_array, radial_position))
    power = torque * angular_velocity
    rotor_disk_area = np.pi * rotor_radius**2
    dynamic_pressure_per_density = 0.5 * freestream_speed**2
    thrust_coefficient = thrust / (dynamic_pressure_per_density * rotor_disk_area)
    power_coefficient = power / (dynamic_pressure_per_density * rotor_disk_area * freestream_speed)

    df = pd.DataFrame(
        {
            "radial_position": radial_position,
            "normalized_radial_position": radial_position / rotor_radius,
            "chord": chord,
            "twist_angle_degrees": np.degrees(twist_angle_radians),
            "inflow_angle_degrees": np.degrees(inflow_angle_array),
            "angle_of_attack_degrees": np.degrees(angle_of_attack_array),
            "lift_coefficient": lift_coefficient_array,
            "drag_coefficient": drag_coefficient_array,
            "axial_induction_factor": axial_induction_factor_array,
            "tangential_induction_factor": tangential_induction_factor_array,
            "thrust_per_radius": thrust_per_radius_array,
            "torque_per_radius": torque_per_radius_array,
            "circulation": circulation_array,
        }
    )
    df.attrs["thrust_coefficient"] = thrust_coefficient
    df.attrs["power_coefficient"] = power_coefficient
    df.attrs["thrust"] = thrust
    df.attrs["torque"] = torque
    df.attrs["power"] = power

    return df


def actuator_disk_velocity_ratio(
    axial_induction_factor: float,
    normalized_radial_position: np.ndarray,
) -> np.ndarray:
    """Far-wake axial velocity deficit from actuator-disk momentum theory.

    In the far wake (x → ∞), u/U∞ = 1 − 2a uniformly within the wake disc.
    Outside the disc (|r/R| > 1) the deficit vanishes.

    Parameters
    ----------
    axial_induction_factor: scalar induction factor inferred from thrust coefficient
    normalized_radial_position: radial position normalised by rotor radius
    Returns
    -------
    axial_velocity_ratio: array with the same shape as normalized_radial_position
    """
    normalized_radial_position = np.asarray(normalized_radial_position, dtype=float)
    axial_velocity_ratio = np.ones_like(normalized_radial_position)
    inside = np.abs(normalized_radial_position) <= 1.0
    axial_velocity_ratio[inside] = 1.0 - 2.0 * axial_induction_factor
    return axial_velocity_ratio


def axial_induction_factor_from_thrust_coefficient(thrust_coefficient: float) -> float:
    """Axial induction factor from thrust_coefficient via momentum theory: 1 − √(1−thrust_coefficient) / 2."""
    if thrust_coefficient >= 1.0:
        return 0.5  # turbulent wake state (momentum theory invalid)
    return 0.5 * (1.0 - np.sqrt(max(0.0, 1.0 - thrust_coefficient)))


__all__ = [
    "solve_blade_element_momentum",
    "actuator_disk_velocity_ratio",
    "axial_induction_factor_from_thrust_coefficient",
]
