# Wind turbine with meshed blades

The three-bladed turbine of the [VLM rotor tutorial](../../vpm/06_rotor_flow/README.md) with its blades fully meshed: the same planform, twist, radius, wind, tip-speed ratio, spin-up, time step, end time and output cadence, so that power and blade forces compare blade for blade; only the blade model differs. The FVM resolves the blades inside a flat box, a disk normal to the rotor axis, which turns with the rotor; the vortex particles carry the wake in the inertial frame.

| Quantity | Value |
| --- | --- |
| Rotor | $R=6$ m, hub radius 1 m, three blades 120° apart, no hub body |
| Blades | Chord lines of `assets/blade.json`, a copy of the VLM blade file `../../vpm/06_rotor_flow/assets/blade.json` (0.6 m at the root, 0.35 m at the tip, twist of the VLM design), with a NACA 0018 thickness cut to a flat base at 95 % of the chord; closed at root and tip |
| Wind; air | $U_\infty=7$ m/s along $+x$; $\rho=1.225$ kg/m³, $\nu=1.5\times10^{-5}$ m²/s |
| Rotation | $\lambda=\Omega R/U_\infty=7$, $\Omega=8.167$ rad/s about $-x$, sin² spin-up over one revolution |
| FVM disk | $[-2.4,2.4]\times[-9,9]\times[-9,9]$ m in the frame of the rotor |
| Cell and particle spacing | 0.2 m at the interface, 0.025 m at the blades |
| FVM step; coupling interval | 0.0005 s; 0.006 s (the VLM particle step) |
| End time; output | 15 s; fields, wake planes and lines every 0.06 s, backups every 0.024 s |

## Physics and numerical model

The blade surfaces are generated at setup from the copied VLM blade file (`assets/blade_surface.py`): each of its 23 radial stations gives a chord line, the NACA 0018 thickness is laid off normal to it in the plane of the station, the section ends in a flat base at 95 % of the chord, and the sections are lofted and capped. The 18 % section keeps the blade tip more than two cells thick; with 12 % the 0.04 m tip collapsed cells in the mesher's wall projection. Three STL files, one per blade, become the wall patches `blade_0`, `blade_1` and `blade_2`.

The disk is a coupling box with all six faces on the particle interface, meshed as exact Cartesian planes, with the blades three octree levels finer; its sides keep the blade tips eight background cells inside, as the mesher requires. The whole mesh turns with `fvm.RotatingMotion`, the rigid ALE form of [moving meshes](../../../docs/fvm.md#moving-meshes); the coupling geometry is kept in the turning body coordinates and the particles stay in the inertial frame ([moving bodies](../../../docs/coupling.md#moving-bodies)). The relative flow at the blade tips, about 50 m/s, sets the FVM step: twelve substeps per coupling interval keep the tip Courant number near one with three outer PIMPLE passes. Both solvers use the equilibrium Smagorinsky closure, as the [cube](../02_cube_flow/README.md); the VLM rotor uses the Smagorinsky closure in its wake alone. The particle domain holds the disk at every angle, with an outflow fringe at its downstream end.

## Status

The case is configured but has not run: on 2 November 2026 the mesher built the octree mesh of about 350 000 cells for every configuration tried (12 % and 18 % sections, 0.05 m and 0.025 m blade cells, sharp and blunt trailing edges) and then failed its surface-constrained wall projection on the lofted blades, with inverted cells or cell centres inside the blade. The blade surfaces were then wound inward; wound outward, as every other tutorial body, the 18 % blades with 0.025 m cells still fail the same projection with non-positive cell volumes. The wall projection of thin, twisted surfaces is a mesher limitation to fix before this case runs; the case files are the intended configuration.

## Run

Run commands from this directory.

```bash
./allrun.sh
./allplot.sh
```

`./allrun.sh` starts from $t=0$; `./allcontinue.sh` resumes the latest accepted coupled backup. `python setup.py --end-time 0.03 --compute-device cpu` runs a short check on the CPU; the particle solver otherwise takes the GPU. The mesh of about half a million cells takes some minutes and a few GB of memory to generate and is cached in `assets/`.

## Results

| File | Content |
| --- | --- |
| `samples/blade_forces.csv` | Pressure, viscous and total force, moment about the rotor centre and `rotational_power`, `translational_power`, `power` of each blade patch every 0.006 s, as the VLM `vlm_surface_forces.csv` |
| `samples/flow_integrals.csv`, `samples/wake_*D.pvd`, `samples/streamwise_r*.csv` | Particle invariants, wake planes at 0, 1 and 2 diameters and streamwise lines at $r/R=0.25$ and 0.65, as the VLM case |
| `figures/rotor_performance` | $C_T$ and $C_P$ on the disk area against nominal revolutions, with the VLM rotor when its run exists; `allplot.sh` prints the means of the last five revolutions and writes `figures/auxiliary/rotor_performance.csv` |
| `figures/blade_forces` | Thrust, torque and power of each blade, meshed against VLM |
| `figures/rotor_wake_t*` | Blades, FVM vorticity on the plane through the axis and the particle wake at the last saved time |
| `solution/fvm.pvd`, `solution/vpm.pvd` | Turning FVM fields and particle states |

The coefficients use $C_T=T/(\tfrac12\rho U_\infty^2\pi R^2)$ and $C_P=P/(\tfrac12\rho U_\infty^3\pi R^2)$ with $P=\mathbf{M}\cdot\boldsymbol{\Omega}$ the shaft power the fluid delivers, the definitions of the VLM case.
