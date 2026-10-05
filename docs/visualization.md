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


## Figure layout for the thesis

Use `openonda.plotting.set_thesis_style()` and export PNG/PDF together with
`export_figure()`. Author the layout in each generator at a native width of
125 mm; do not use tight cropping or automatic layout to resize the canvas.
The same rules apply to diagnostic figures that are not included in the thesis.

- Measure the final left y-label at the production font size and choose an
  explicit left margin `x` that leaves about 2–3 pt of clear space. Set the
  right boundary to `1 - x`. Keep these paired margins for every plotted figure.
  Time-series families use one fixed margin sized for their longest labels.
- Put titles close to the top with 2–3 pt clearance. Use short, meaningful
  titles such as “Aspect ratio” and “Co-rotating”. Keep a small clear gap
  between the bottom x label and legend frame; include legend handles in visual checks.
- Hard-code subplot, colour-bar and legend positions for each figure. Compact
  unused panel gaps while preserving room for labels and ticks.
- Use at least 0.6 pt (0.21 mm) for visible curves, contours, axes, ticks,
  marker edges and annotation lines at native print size. Ordinary curves use
  1 pt. Convert this floor to pixels for raster and ParaView strokes. Keep
  deliberately absent edges absent.
- Display at most two significant digits in ticks, colour bars, annotations,
  legends and times. Retain full precision in calculations, sampled data,
  geometry and file identifiers. Use shared exponents or additive offsets
  when rounded tick values would conceal small changes.
- Use snake_case filenames. Rendered three-dimensional views should fill the
  horizontal space on a shorter canvas. Keep all geometry visible and the
  colour limits, glyph scaling and camera consistent between compared states.
  Update scale bars when changing the camera or viewport.

Validate exports with `validate_thesis_figure()` and inspect them at their final
print size. Scientific renders use saved solver states; do not launch simulations
or restore result archives solely to change their presentation. The ring renderer
retains its verified source information and camera in `figures/vortex_ring_scene.json`.
