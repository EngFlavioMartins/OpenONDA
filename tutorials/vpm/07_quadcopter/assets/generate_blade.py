"""Compatibility exports for installed tutorial support helpers."""

from openonda.tutorial_support.vpm_quadcopter.generate_blade import (
    save_blade as save_blade,
    create_rotor_blade as create_rotor_blade,
)


if __name__ == "__main__":
    blade = create_rotor_blade()
    save_blade(blade, "blade.json")
