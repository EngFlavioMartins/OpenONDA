# Inspect and compare the flow

Each tutorial plots its own results:

```bash
./allplot.sh          # PNG and PDF
./allplot.sh png      # PNG only
./allplot.sh pdf      # PDF only
```

The plot scripts in `assets/` read `samples/` and `solution/` directly; edit them to change a comparison. For fields, open `solution/fvm.pvd`, `solution/vpm.pvd` or `solution/vlm.pvd` in ParaView; several cases provide `assets/paraview_state.py` with a ready-made view. Perceptually uniform colour maps for ParaView are in `docs/assets/scientific_colour_maps/`.

## Reading the fields

| Quantity | Interpretation |
| --- | --- |
| Velocity $\mathbf{u}$, m/s | Flow direction, wake deficit, near-wall profiles |
| Kinematic pressure $q=p/\rho$, m²/s² | Multiply by $\rho$ for pressure in Pa |
| Vorticity $\boldsymbol{\omega}$, 1/s | Local rotation; signed components show its sense |
| Particle strength $\boldsymbol{\Gamma}$, m³/s | Vorticity integrated over a particle volume, not a pointwise value |
| $C_D$, $C_L$ | Force divided by $\tfrac12\rho U^2A$ with the case's reference area $A$ |

When comparing runs, keep the colour limits, physical times, normalization and averaging windows the same. Use a sequential colour map for magnitudes and a diverging map centred on zero for signed fields. A smooth-looking wake does not prove accuracy: compare forces, profiles or vortex trajectories while refining the resolution and time step.

Your own plots can share the tutorial style through `openonda.plotting` (`set_style()`, `export_figure()`).
