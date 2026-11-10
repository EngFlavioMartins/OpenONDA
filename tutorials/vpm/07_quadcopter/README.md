# Quadcopter in climb

Four counter-rotating, two-bladed rotors generate interacting wakes in axial climb. See [VLM blade/wake setup](../../../docs/vpm.md#vlm-surfaces-and-wakes) and [LES/core spreading](../../../docs/vpm.md#diffusion-and-les).

## Run

From this directory:

```bash
./allrun.sh
./allplot.sh
```

`allrun.sh` removes previous generated output. `./allcontinue.sh` resumes saved native backups; `python setup.py` also continues automatically. `./allplot.sh pdf` exports PDF.

## Geometry and resolution

Edit [setup.py](setup.py). Each rotor has radius $R=0.15$ m, hub radius 0.03 m and two blades with root/tip chords 0.025/0.015 m and pitch 28°/12°. Each blade uses 4 chordwise and 12 spanwise panels. Rotor centres are at $(\pm0.18,\pm0.18,0)$ m, leaving 6 cm between neighbouring tips, with alternating rotation directions.

The speed is 4000 rpm and axial climb speed is 5 m/s. Density is 1.225 kg/m³ and viscosity is $1.5\times10^{-5}$ m²/s. Blade chord Reynolds numbers are approximately 22,000–63,000. Rotation starts at operating speed, with the leading edges facing the rotational relative wind.

The wake uses Gaussian particles, transposed fast-multipole (FMM) induction, core spreading and $C_s=0.3$ Smagorinsky LES with a selective eddy viscosity of coefficient 2. These damp the stretched, collapsing tubes of the merged wake; with the defaults ($C_s=0.17$, coefficient 0.5) the wake column clumped after about eight revolutions. Pedrizzetti relaxation, a factor of 0.1 per step with moment restoration, keeps the particle strengths aligned with the vorticity they induce; without it the strengths drifted out of alignment within three revolutions. Particles are not split: adaptive filament splitting multiplied the particles in the stretched starting vortex until they outgrew the particle capacity. Particles outside $x,y\in[-1,1]$ m and $z\in[-1,0.3]$ m are removed, about $6.7R$ below the rotors; enlarge this region before studying a longer retained wake.

There are 72 steps per revolution (5° per step), with $\Delta t=2.083\times10^{-4}$ s. The run spans 20 revolutions, or 0.3 s. At a 3 m/s climb the four starting vortices lifted vorticity back into the rotor plane between the slipstreams, and the blades struck it; the 5 m/s climb carries it away. Wake planes and particle frames are written six times per revolution, flow integrals once per revolution and backups every two revolutions.

## Loads, references and outputs

For each rotor, $A=\pi R^2$ and the rotorcraft coefficients are

$$
C_T=\frac{T}{\rho A(\Omega R)^2},\qquad
C_P=\frac{P}{\rho A(\Omega R)^3}.
$$

Positive $P$ is shaft input, the negative of fluid-on-blade rotational power. The reference is isolated-rotor BEM with a thin-plate lift polar and hub/tip losses. It omits rotor interference, transition, profile drag and stall; the power excludes motor electrical losses. Over revolutions 10–20 the mean thrust coefficient of a rotor is 0.0101, against 0.0102 from BEM, and stays within ±5% of that mean; the Lagrangian CFL number stays below 0.05 (`samples/quadcopter/vlm_surface_forces.csv`, `flow_integrals.csv`).

Force/loading records and planes at $z=-0.35$ and −0.70 m are in `samples/quadcopter/`. Open `solution/quadcopter/vpm.pvd` and `vlm.pvd` for the coupled geometry. Wake figures average the final four revolutions, after the starting vortex has passed both planes. `figures/quadcopter_flow.gif` animates the four particle wakes with the thrust and power coefficients averaged over the rotors.

The impulse check requires the wake to remain inside the retained domain. A complete run still needs panel, time-step and particle-core convergence. This case demonstrates interacting attached-flow rotor wakes; VLM does not resolve low-Reynolds-number blade boundary layers or stall.
