# Flat-plate comparison

The [case guide](../README.md) defines Reynolds number, wall spacing and the Blasius comparison. See [FVM mesh setup](../../../../docs/fvm.md#mesh-setup) for wall resolution.

`mesh_plate.py` builds the stretched mesh. `profiles.py` samples wall-normal velocity and wall shear; `plot_blasius.py` and `plot_cf.py` compare them with Blasius theory.
