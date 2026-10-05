# Two heaving wings: wake interaction

Two swept wings heave out of phase and pitch with the changing incident flow. The downstream wing crosses the upstream wake. See [VLM surfaces, motion and wake cores](../../../docs/vpm.md#vlm-surfaces-and-wakes).

## Run

From this directory:

```bash
./allrun.sh
./allplot.sh
```

`allrun.sh` removes previous generated output and starts a fresh run. `./allcontinue.sh` resumes saved native checkpoints. `./allplot.sh pdf` exports PDF.

## Physical inputs

Edit [setup.py](setup.py):

| Quantity | Value |
| --- | --- |
| Root/tip chord; span | 0.5/0.1 m; 1 m. |
| Wing separation; initial incidence | 2.5 m; 15°. |
| Freestream | 5 m/s toward negative $x$. |
| Density; molecular viscosity | 1.225 kg/m³; 0.001 m²/s. |
| Heave amplitude $A$; frequency $f$ | 0.2 m; 1 Hz, with opposite phases. |
| Step; duration | 0.0025 s; 20 s (8000 steps). |

Prescribed vertical velocity is $\dot z=A(2\pi f)\sin(2\pi ft+\phi)$. Pitch rate follows the changing flow angle, with a pivot one-third of the root chord behind the leading edge. Each half-wing has 8 chordwise and 18 spanwise panels, with geometric spacing ratio 3. Initial surface rotation precedes translation; moving pivots use world coordinates.

The Gaussian wake uses CPU/FMM induction, transposed stretching and core spreading, with `wake_core_overlap=2.5`. There is no LES closure or strength relaxation. Particles outside the configured bounds are removed; extend those bounds if a changed motion or duration requires a larger wake.

## Results and scope

Force/loading/motion tables are in `samples/delta_wing/`, with wake planes and integrals every 0.025 s. Coupled backups use the same interval; open `solution/vlm.pvd` or `solution/vpm.pvd` in ParaView.

Figures show vertical forces, motion power, the last three heave cycles, particle strength and three downstream wake planes. The strength sum is $\sum_p|\boldsymbol{\Gamma}_p|$ in m³/s, not conserved scalar circulation. Wake fields average one full measured heave cycle.

Compare successive heave cycles and refine the time step, wing panels and wake cores before interpreting interaction loads. `python -m openonda.tutorial_runner . assets.postprocess render-gif` exports an animation from the saved native frames.

This attached-flow VLM does not model separated leading-edge vortices, stall or viscous particle–wall interaction. The initial 15° incidence is a prescribed demonstration, not validation of separated delta-wing aerodynamics. The animation reads the saved native wing panels and their bound circulation.
