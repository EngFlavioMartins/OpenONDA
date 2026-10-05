# Quadcopter in climb

Four counter-rotating, two-bladed rotors generate interacting wakes in axial climb. See [VLM blade/wake setup](../../../docs/vpm.md#vlm-surfaces-and-wakes) and [LES/core spreading](../../../docs/vpm.md#diffusion-and-les).

## Run

From this directory:

```bash
./allrun.sh
./allplot.sh
```

`allrun.sh` removes previous generated output. `./allcontinue.sh` resumes saved native backups; `python -m openonda.tutorial_runner . setup` also continues automatically. `./allplot.sh pdf` exports PDF.

## Geometry and resolution

Edit [setup.py](setup.py). Each rotor has radius $R=0.15$ m, hub radius 0.03 m and two blades with root/tip chords 0.025/0.015 m and pitch 12°/6°. Each blade uses 4 chordwise and 12 spanwise panels. Rotor centres are at $(\pm0.16,\pm0.16,0)$ m with alternating rotation directions.

The speed is 4000 rpm and axial climb speed is 0.8 m/s. Density is 1.225 kg/m³ and viscosity is $1.5\times10^{-5}$ m²/s. Blade chord Reynolds numbers are approximately 22,000–63,000. Rotation starts at operating speed, with the leading edges facing the rotational relative wind.

The wake uses Winckelmans particles, transposed treecode induction, core spreading and $C_s=0.20$ Smagorinsky LES. Core spreading with this algebraic kernel is a second-moment diffusion model. Adaptive splitting checks each step when strength doubles. Particles outside $x,y\in[-1.5,1.5]$ m and $z\in[-3,1]$ m are removed; enlarge this region before studying a longer retained wake.

There are 96 steps per revolution (3.75° per step), with $\Delta t=0.00015625$ s. The run spans 24 revolutions, or 0.36 s. Fields are sampled eight times per revolution; coupled backups occur every three steps.

## Loads, references and outputs

For each rotor, $A=\pi R^2$ and the rotorcraft coefficients are

$$
C_T=\frac{T}{\rho A(\Omega R)^2},\qquad
C_P=\frac{P}{\rho A(\Omega R)^3}.
$$

Positive $P$ is shaft input, the negative of fluid-on-blade rotational power. The reference is isolated-rotor BEM with a thin-plate lift polar and hub/tip losses. It omits rotor interference, transition, profile drag and stall; the power excludes motor electrical losses.

Force/loading records and planes at $z=-0.35$ and −0.70 m are in `samples/quadcopter/`. Open `solution/vpm.pvd` and `vlm.pvd` for the coupled geometry. Wake figures average the final six revolutions.

The impulse check requires the wake to remain inside the retained domain. A complete run still needs panel, time-step and particle-core convergence. This case demonstrates interacting attached-flow rotor wakes; VLM does not resolve low-Reynolds-number blade boundary layers or stall.
