"""Compatibility exports for installed tutorial support helpers."""

from openonda.tutorial_support.vpm_delta_wing.generate_surface import (
    save_surface as save_surface,
    create_delta_wing_vertices as create_delta_wing_vertices,
    create_delta_wing as create_delta_wing,
)


if __name__ == "__main__":
    # Example: Generate and save a delta wing surface
    surface = create_delta_wing(
        root_chord=0.5,
        tip_chord=0.1,
        half_span=0.5,
        sweep_angle_degrees=45.0,
        angle_of_attack_degrees=0.0,
        n_chordwise_panels=8,
        n_spanwise_panels=20,
    )
    save_surface(surface, "delta_wing_surface")
