# Inspect and compare the flow

Each tutorial plots its own results:

```bash
./allplot.sh          # PNG figures in figures/
```

The tutorials write PNG only; the thesis and paper draw their own PDF figures from the sampled data.

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

Your own plots can share the tutorial style: call `openonda.plotting.set_thesis_style()` before the figure and `export_figure()` to save it, as the tutorial scripts do.

## Presentation videos

`docs/assets/presentation/` holds full-HD (1920 × 1080) H.264 versions of the tutorial animations for slides. They show the scenes of the README GIFs, framed for 16:9, with text sized for a 1080p slide. The vorticity is drawn as unlit discs with a soft edge, blended from the farthest to the nearest: transparent below a cut, opaque from a second value and linear in between, so that weak vorticity does not veil the vortex tubes (`openonda.plotting.vorticity_discs`).

Each case's animation script writes its video with the `presentation` argument, from the case folder:

```bash
python -m assets.render_cube_flow_animation presentation
```

The script keeps one PNG per saved state in `solution/presentation_frames/`. With [RIFE](https://github.com/nihui/rife-ncnn-vulkan) installed (`rife-ncnn-vulkan`, MIT licence; its executable is `$OPENONDA_RIFE`, else the release unpacked in `~/opt`), the video plays at 50 frames per second for at least 10 s: RIFE interpolates as many frames between the saved states as that needs, for example 5 between each pair of the 99 saved states of the cube. Without it, the saved states play at 10 per second. The last state is held for one second.

| Video | Case | Script |
| --- | --- | --- |
| `cylinder_shedding.mp4` | [Coupled cylinder](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) | `assets/render_cylinder_animation.py` |
| `cube_flow.mp4` | [Coupled cube](../tutorials/coupled_fvm_vpm/02_cube_flow/README.md) | `assets/render_cube_flow_animation.py` |
| `wing_flow.mp4` | [Coupled NACA 4412 wing](../tutorials/coupled_fvm_vpm/03_naca4412_flow/README.md) | `assets/render_wing_flow_animation.py` |
| `vortex_ring.mp4` | [Vortex ring](../tutorials/vpm/02_vortex_ring/README.md) | `assets/render_vortex_ring_animation.py` |
| `flat_plate_wake.mp4` | [Flat plate](../tutorials/vpm/04_flat_plate/README.md) | `assets/render_flat_plate_animation.py` |
| `delta_wing.mp4` | [Two heaving wings](../tutorials/vpm/05_delta_wing/README.md) | `assets/render_delta_wing_animation.py` |
| `rotor_flow.mp4` | [Wind turbine](../tutorials/vpm/06_rotor_flow/README.md) | `assets/render_rotor_wake_animation.py` |
| `quadcopter_flow.mp4` | [Quadcopter in climb](../tutorials/vpm/07_quadcopter/README.md) | `assets/render_quadcopter_animation.py` |

The coupled cases write their FVM and particle fields often enough for about 500 saved states over a run (`FIELD_INTERVAL` in `setup.py`), which play at 50 frames per second without interpolation.
