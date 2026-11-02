"""Loads of the meshed-blade rotor and of the VLM rotor it is compared with.

Both cases write the force on each blade with the same column names:
``force_x`` (thrust, N), ``moment_x`` (torque about the rotor axis, N m) and
``power`` (W, positive when the fluid drives the rotor). The FVM blade forces
are summed over the three blade patches for the rotor totals.
"""

from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
VLM_SAMPLES = CASE_DIR.parents[1] / "vpm/06_rotor_flow/samples/rotor"
FIGURES = CASE_DIR / "figures"

FREESTREAM_SPEED = 7.0  # m/s
ROTOR_RADIUS = 6.0  # m
AIR_DENSITY = 1.225  # kg/m^3
ANGULAR_VELOCITY = 7.0 * FREESTREAM_SPEED / ROTOR_RADIUS  # rad/s
ROTATION_PERIOD = 2.0 * np.pi / ANGULAR_VELOCITY  # s
DISK_AREA = np.pi * ROTOR_RADIUS**2  # m^2
THRUST_REFERENCE = 0.5 * AIR_DENSITY * FREESTREAM_SPEED**2 * DISK_AREA  # N
POWER_REFERENCE = THRUST_REFERENCE * FREESTREAM_SPEED  # W
BLADE_COLUMNS = ("force_x", "force_y", "force_z", "moment_x", "power")


def read_table(path):
    """Named columns of a sampler CSV."""
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def fvm_blades():
    """Per-blade loads of the meshed-blade rotor, keyed by blade name, with the VLM column names."""
    table = read_table(CASE_DIR / "samples/blade_forces.csv")
    blades = {}
    for name in np.unique(table["patch"]):
        rows = table[table["patch"] == name]
        blades[str(name)] = {
            "time": rows["time"],
            "force_x": rows["total_force_x"],
            "force_y": rows["total_force_y"],
            "force_z": rows["total_force_z"],
            "moment_x": rows["moment_x"],
            "power": rows["power"],
        }
    return blades


def vlm_blades():
    """Per-blade loads of the VLM rotor when its run exists, else an empty dictionary."""
    files = list(VLM_SAMPLES.glob("vlm_surface_forces.csv"))
    if not files:
        return {}
    table = read_table(files[0])
    return {
        str(name): {
            column: table[table["surface"] == name][column] for column in ("time", *BLADE_COLUMNS)
        }
        for name in np.unique(table["surface"])
    }


def rotor_totals(blades):
    """Thrust and power coefficients of a rotor from its blade loads on a common time axis."""
    time = next(iter(blades.values()))["time"]
    thrust = sum(blade["force_x"] for blade in blades.values())
    power = sum(blade["power"] for blade in blades.values())
    return time, thrust / THRUST_REFERENCE, power / POWER_REFERENCE
