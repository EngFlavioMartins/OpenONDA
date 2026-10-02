# Step-flow diagnostics

The [case guide](../README.md) defines the expansion geometry, inlet profile, mesh controls and run commands. See [FVM boundary conditions](../../../../docs/fvm.md#boundary-conditions) for the parabolic inlet and no-slip walls.

The installed `fvm_step_profile.mesh_step` helper builds the body-fitted grid. the installed `fvm_step_profile.reattachment` helper estimates the downstream sign change in near-wall velocity; `plot_profile.py` and `plot_comparison.py` plot the history and final flow.
