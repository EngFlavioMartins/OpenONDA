# Flat-plate comparison

The [case guide](../README.md) defines Reynolds number, wall spacing and the Blasius comparison. See [FVM mesh setup](../../../../docs/fvm.md#mesh-setup) for wall resolution.

The installed `fvm_boundary_layer.mesh_plate` helper builds the stretched mesh. the installed `fvm_boundary_layer.profiles` helper samples wall-normal velocity and wall shear; `plot_blasius.py` and `plot_cf.py` compare them with Blasius theory.
