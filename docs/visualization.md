# Inspect and compare the flow

From a tutorial directory with an `allplot.sh` launcher, plot saved results:

```bash
./allplot.sh          # PNG and PDF.
./allplot.sh png      # PNG only.
./allplot.sh pdf      # PDF only.
```

Plotters use local results or retrieve a published result archive when available. Check each case guide for incomplete runs and the data used in its comparisons.

Open `solution/fvm.pvd`, `solution/vpm.pvd`, or `solution/vlm.pvd` in ParaView. Select the field and time of interest; see [saved fields](solution_layout.md).

## Physical quantities

| Quantity | Interpretation |
| --- | --- |
| Velocity $\mathbf{u}$, m/s | Flow direction, wake deficit, and near-wall profiles. |
| Kinematic pressure $q=p/\rho$, m²/s² | FVM pressure divided by density; multiply $q$ by $\rho$ for Pa. |
| Vorticity $\boldsymbol{\omega}$, s⁻¹ | Vortex strength per unit volume; signed components show rotation direction. |
| Particle strength $\boldsymbol{\Gamma}$, m³/s | Volume-integrated vorticity; it is not a pointwise vorticity value. |
| $C_L$, $C_D$ | Forces divided by $\tfrac12\rho U^2 A$; use the case's reference area $A$. |

Keep common colour limits, physical times, normalization, and averaging windows when comparing runs. Use a sequential map for magnitudes and a zero-centred diverging map for signed fields. Particle count or a visually smooth wake alone does not establish accuracy; compare forces, profiles, or vortex motion while refining resolution and timestep.

The [tutorial index](tutorials.md) links each case to its physical model and setup.
