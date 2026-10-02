# Wind turbine: blade loading and a viscous wake

A three-bladed turbine extracts power from axial wind. VLM supplies attached blade circulation and loads; a particle wake supplies induced velocity, molecular diffusion and LES. See [VLM](../../../docs/vpm.md#vlm-surfaces-and-wakes) and [diffusion/LES](../../../docs/vpm.md#diffusion-and-les).

## Run

From this directory:

```bash
./allrun.sh
./allplot.sh
```

`allrun.sh` removes generated output and starts a fresh calculation. `./allcontinue.sh` resumes compatible backups; `python setup.py` also continues automatically. `./allplot.sh pdf` exports PDF.

## Geometry and operating point

Edit [setup.py](setup.py) and [blade.json](assets/blade.json). The JSON contains chord/twist geometry and chordwise/spanwise panel counts. Inputs are rotor radius $R=6$ m, hub radius 1 m, wind speed $U=7$ m/s, density $\rho=1.225$ kg/m³ and viscosity $\nu=1.5\times10^{-5}$ m²/s. Tip-speed ratio $\lambda=\Omega R/U=7$ gives $\Omega=8.167$ rad/s; rotation ramps over one nominal revolution. Blade chord Reynolds numbers are approximately 0.47–1.15 million.

The authored step is 0.006 s and duration 15 s (2500 steps). The Gaussian wake uses treecode induction, core spreading, Smagorinsky LES with $C_s=0.20$, and wake-core overlap 2.5. Selective viscosity uses coefficient 0.5; filament splitting checks every five steps when strength doubles. These numerical models require resolution checks together with the blade panels and time step.

## Loads and wake interpretation

For disk area $A=\pi R^2$, figures use

$$
C_T=\frac{T}{\tfrac12\rho U^2A},\qquad
C_P=\frac{P}{\tfrac12\rho U^3A},
$$

where positive $P$ is extracted shaft power. The BEM reference uses the same blade geometry, a thin-plate lift polar and hub/tip losses. The final five nominal revolutions define the mean loading and wake profiles.

Wake planes lie at the disk, 1 diameter and 2 diameters downstream. Streamwise lines span $x/D=-1$ to 3 at $r/R=0.25$ and 0.65. Ideal actuator-disk velocities are $u_d/U=1-a$ at the disk and $u_\infty/U=1-2a$ in the far wake; 1D or 2D need not be far enough for that limit. Finite-distance vortex-cylinder induction is a separate approximate comparison.

Loading CSVs are in `samples/rotor/`, with fields every 0.06 s. Coupled backups every 0.024 s are in `solution/`; open `vlm.pvd` and `vpm.pvd` in ParaView. Run `python assets/validate_results.py --pre-plot` to check completion, full averaging windows, load/wake stationarity, BEM comparison and impulse balance.

Converged rotor loads and induction have not been established. An earlier configuration stopped at 7.68 s; the current stabilization settings and 15 s horizon need a complete run and spatial/time/core refinement. Short startup or continuation checks cannot establish the developed wake.

The blade model omits boundary layers, transition, stall and profile drag. Wake LES does not add these blade-section physics. See the [BEM formulation](https://wisdem.readthedocs.io/en/master/wisdem/ccblade/theory.html) for the reference model.
