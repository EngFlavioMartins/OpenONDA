"""Compatibility exports for installed tutorial support helpers."""

from openonda.tutorial_support.vpm_flat_plate.generate_surface import (
    save_surface as save_surface,
    create_flat_plate_vertices as create_flat_plate_vertices,
    create_flat_plate as create_flat_plate,
)


if __name__ == "__main__":
    # Example: Generate and save a flat plate surface
    surface = create_flat_plate(
        chord=0.5,
        span=1.0,
        angle_of_attack_degrees=0.0,
        n_chordwise_panels=8,
        n_spanwise_panels=20,
    )
    save_surface(surface, "flat_plate_surface")
