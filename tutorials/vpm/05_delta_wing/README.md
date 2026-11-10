# Two heaving wings: wake interaction

Two swept wings heave out of phase and pitch through part of the changing flow angle, so their incidence oscillates. Each wing sheds a chain of linked vortex loops, and the downstream wing crosses the upstream wake. See [VLM surfaces, motion and wake cores](../../../docs/vpm.md#vlm-surfaces-and-wakes).

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
| Wing separation; initial incidence | 2.5 m; 8°. |
| Freestream | 5 m/s toward negative $x$. |
| Density; molecular viscosity | 1.225 kg/m³; 0.001 m²/s. |
| Heave amplitude $A$; frequency $f$ | 0.25 m; 2 Hz, with opposite phases. |
| Step; duration | 0.005 s; 6 s (1200 steps, 12 heave cycles). |

Prescribed vertical velocity is $\dot z=A(2\pi f)\sin(2\pi ft+\phi)$, a Strouhal number $2fA/U_\infty=0.2$ and a reduced frequency $\pi fc_\mathrm{root}/U_\infty=0.63$. The pitch angle follows 60% of the heave flow angle $\arctan(\dot z/U_\infty)$, which peaks at 32°, so the incidence swings by about ±13° around 8°. The pivot lies one-third of the root chord behind the leading edge. Each half-wing has 8 chordwise and 18 spanwise panels, with geometric spacing ratio 3. Initial surface rotation precedes translation; moving pivots use world coordinates.

Loads include the unsteady Kutta–Joukowski term. The Gaussian wake uses direct GPU induction (it stays below about 45,000 particles), transposed stretching and core spreading, with `wake_core_overlap=2.5`. $C_s=0.17$ Smagorinsky LES, with the selective eddy viscosity that comes with it, damps the breakdown of the far wake; without it, the clumping far wake stopped an earlier run at the Lagrangian CFL limit. An outflow fringe fades the particle strengths between 5.5 and 7 m downstream of the rear wing, where the particles are removed; deleting them abruptly at the face let the clumping far wake reach the Lagrangian CFL limit. Extend those bounds if a changed motion or duration requires a larger wake.

## Results and scope

Force/loading/motion tables are in `samples/delta_wing/`, with wake planes every 0.025 s and flow integrals every 0.25 s. Particle frames are written every 0.05 s and backups every second; open `solution/delta_wing/vlm.pvd` or `solution/delta_wing/vpm.pvd` in ParaView.

Figures show the lift and thrust coefficients, on each wing's planform area of 0.3 m², and the power each wing's motion puts into the flow; their phase averages over the final four heave cycles, with the cycle-to-cycle range shaded; the particle strength; and three downstream wake planes. The impulsive start loads lie off the history axes. Over heave cycles 9–12, which repeat to within the line width, the mean thrust coefficient is 0.27 on the front wing and 0.38 on the rear wing, which flies through the front wing's wake; the mean lift coefficients are 0.53 and 0.37 (printed by `plot_delta_wing_force_cycles`). The strength sum is $\sum_p|\boldsymbol{\Gamma}_p|$ in m³/s, not conserved scalar circulation. Wake fields average one full measured heave cycle.

Compare successive heave cycles and refine the time step, wing panels and wake cores before interpreting interaction loads. `figures/delta_wing.gif` animates the particle wakes in real time with the lift coefficient of each wing; `assets/delta_wing.gif`, at the top of the main README, is a copy.

This attached-flow VLM does not model separated leading-edge vortices, stall or viscous particle–wall interaction. Incidences up to about 21° are a prescribed demonstration, not validation of separated delta-wing aerodynamics.
