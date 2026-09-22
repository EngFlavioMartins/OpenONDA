"""Attached-flow blade-element/momentum references for axial rotors.

The inflow-angle residual follows Ning (2014); turbine high induction uses
Buhl (2005). See https://wisdem.readthedocs.io/en/master/wisdem/ccblade/theory.html.
This reference uses an inviscid thin-plate polar unless a profile drag is given.
It does not model stall, transition, or rotor/rotor interference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import brentq


def solve_blade_element_momentum(
    radial_position,
    chord,
    twist_angle_radians,
    n_blades,
    rotor_radius,
    freestream_speed,
    angular_velocity,
    lift_curve_slope=2 * np.pi,
    constant_drag_coefficient=0.0,
    max_iterations=200,
    convergence_tolerance=1e-10,
    *,
    hub_radius=0.0,
    radial_widths=None,
    density=1.0,
    mode="turbine",
):
    """Solve each annulus and integrate dimensional blade forces and shaft power.

    ``mode='turbine'`` uses axial velocity U(1-a), tangential velocity Ωr(1+a'),
    and incidence φ-β. ``mode='propeller'`` uses U(1+a), Ωr(1-a'), and β-φ.
    Power is positive extraction for a turbine and positive input for a propeller.
    Coefficients use ½ρU²A and ½ρU³A; dimensional outputs use ``density``.
    Stations must lie strictly between the hub and tip. Supply physical strip
    widths for cell-centred data; otherwise the sampled span is trapezoid-integrated.
    """
    r, c, beta = (
        np.asarray(value, dtype=float) for value in (radial_position, chord, twist_angle_radians)
    )
    if mode not in ("turbine", "propeller"):
        raise ValueError("BEM mode must be turbine or propeller")
    if freestream_speed <= 0 or angular_velocity <= 0:
        raise ValueError("This axial BEM reference requires positive inflow and angular speed")
    if (
        r.shape != c.shape
        or r.shape != beta.shape
        or np.any((r <= hub_radius) | (r >= rotor_radius))
    ):
        raise ValueError("BEM stations must have matching shapes and lie inside the hub/tip")
    propeller = mode == "propeller"
    records = []
    for radius, chord_i, pitch in zip(r, c, beta, strict=True):
        solidity = n_blades * chord_i / (2 * np.pi * radius)
        speed_ratio = angular_velocity * radius / freestream_speed

        def state(phi, radius=radius, pitch=pitch, solidity=solidity, speed_ratio=speed_ratio):
            sine, cosine = np.sin(phi), np.cos(phi)
            tip = (
                2
                / np.pi
                * np.arccos(np.exp(-n_blades / 2 * (rotor_radius - radius) / (radius * sine)))
            )
            hub = (
                1.0
                if hub_radius == 0
                else 2
                / np.pi
                * np.arccos(np.exp(-n_blades / 2 * (radius - hub_radius) / (hub_radius * sine)))
            )
            loss = max(tip * hub, 1e-12)
            alpha = pitch - phi if propeller else phi - pitch
            cl = lift_curve_slope * np.sin(alpha)
            cd = constant_drag_coefficient
            cn = cl * cosine + (-1 if propeller else 1) * cd * sine
            ct = cl * sine + (1 if propeller else -1) * cd * cosine
            k = solidity * cn / (4 * loss * sine * sine)
            kp = solidity * ct / (4 * loss * sine * cosine)
            if k < 0 or kp < 0 or (propeller and k >= 1) or (not propeller and kp >= 1):
                return None
            if propeller:
                a, ap = k / (1 - k), kp / (1 + kp)
                axial_factor, tangent_factor = 1 + a, 1 - ap
            else:
                a = k / (1 + k)
                if k > 2 / 3:
                    # Buhl's continuous high-thrust relation, solved on its physical branch.
                    a = brentq(
                        lambda value: (
                            8 / 9
                            + (4 * loss - 40 / 9) * value
                            + (50 / 9 - 4 * loss) * value**2
                            - 4 * loss * k * (1 - value) ** 2
                        ),
                        0.4,
                        1.0,
                    )
                ap = kp / (1 - kp)
                axial_factor, tangent_factor = 1 - a, 1 + ap
            residual = sine / axial_factor - cosine / (speed_ratio * tangent_factor)
            return residual, a, ap, cl, cn, ct, loss, axial_factor, tangent_factor

        upper = min(pitch, np.pi / 2 - 1e-6) if propeller else np.pi / 2 - 1e-6
        lower = 1e-6 if propeller else max(pitch + 1e-6, 1e-6)
        bracket = None
        previous = None
        for phi in np.linspace(lower, upper, 600):
            evaluated = state(phi)
            if evaluated is None:
                previous = None
                continue
            if previous is not None and previous[1] * evaluated[0] <= 0:
                bracket = previous[0], phi
                break
            previous = phi, evaluated[0]
        if bracket is None:
            raise ValueError(f"No positive-loading {mode} BEM root at radius {radius:g} m")
        phi = brentq(
            lambda angle: state(angle)[0],
            *bracket,
            xtol=convergence_tolerance,
            maxiter=max_iterations,
        )
        residual, a, ap, cl, cn, ct, loss, axial_factor, tangent_factor = state(phi)
        speed = np.hypot(
            freestream_speed * axial_factor, angular_velocity * radius * tangent_factor
        )
        loading = 0.5 * density * speed**2 * chord_i * n_blades
        records.append(
            {
                "radial_position": radius,
                "normalized_radial_position": radius / rotor_radius,
                "chord": chord_i,
                "twist_angle_degrees": np.degrees(pitch),
                "inflow_angle_degrees": np.degrees(phi),
                "angle_of_attack_degrees": np.degrees(pitch - phi if propeller else phi - pitch),
                "lift_coefficient": cl,
                "drag_coefficient": constant_drag_coefficient,
                "axial_induction_factor": a,
                "tangential_induction_factor": ap,
                "loss_factor": loss,
                "thrust_per_radius": loading * cn,
                "torque_per_radius": loading * ct * radius,
                "circulation": 0.5 * chord_i * cl * speed,
                "residual": residual,
            }
        )
    data = pd.DataFrame(records)

    def integrate(values):
        return float(
            trapezoid(values, r)
            if radial_widths is None
            else np.sum(values * np.asarray(radial_widths))
        )

    thrust = integrate(data.thrust_per_radius)
    torque = integrate(data.torque_per_radius)
    power = torque * angular_velocity
    reference = 0.5 * density * freestream_speed**2 * np.pi * rotor_radius**2
    data.attrs.update(
        thrust=thrust,
        torque=torque,
        power=power,
        thrust_coefficient=thrust / reference,
        power_coefficient=power / (reference * freestream_speed),
    )
    return data


def axial_induction_factor_from_thrust_coefficient(thrust_coefficient):
    """Low-induction actuator-disk branch; invalid operating points stay visible."""
    ct = np.asarray(thrust_coefficient, dtype=float)
    return np.where((ct >= 0) & (ct <= 1), 0.5 * (1 - np.sqrt(np.maximum(1 - ct, 0))), np.nan)


def actuator_disk_velocity_ratio(axial_induction_factor, normalized_radial_position):
    """Uniform far-wake deficit inside the expanded actuator-disk streamtube."""
    a = axial_induction_factor
    wake_radius_ratio = np.sqrt((1 - a) / (1 - 2 * a))
    return np.where(np.abs(normalized_radial_position) <= wake_radius_ratio, 1 - 2 * a, 1.0)
