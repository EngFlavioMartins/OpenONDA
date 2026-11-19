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
| End time; output | 15 s; fields every 0.042 s over the last two revolutions, wake planes and lines every 0.018 s from 7.5 s, backups every 0.024 s |

## Physics and numerical model

The blade surfaces are generated at setup from the copied VLM blade file (`assets/blade_surface.py`): each of its 23 radial stations gives a chord line, the NACA 0018 thickness is laid off normal to it in the plane of the station, the section ends in a flat base at 95 % of the chord, and the sections are lofted and capped. The 18 % section keeps the blade tip more than two cells thick; with 12 % the 0.04 m tip collapsed cells in the mesher's wall projection. Three STL files, one per blade, become the wall patches `blade_0`, `blade_1` and `blade_2`.

The disk is a coupling box with all six faces on the particle interface, meshed as exact Cartesian planes, with the blades three octree levels finer; its sides keep the blade tips eight background cells inside, as the mesher requires. The whole mesh turns with `fvm.RotatingMotion`, the rigid ALE form of [moving meshes](../../../docs/fvm.md#moving-meshes); the coupling geometry is kept in the turning body coordinates and the particles stay in the inertial frame ([moving bodies](../../../docs/coupling.md#moving-bodies)). The relative flow at the blade tips, about 50 m/s, sets the FVM step: twelve substeps per coupling interval keep the tip Courant number near one with three outer PIMPLE passes. Both solvers use the equilibrium Smagorinsky closure, as the [cube](../02_cube_flow/README.md); the VLM rotor uses the Smagorinsky closure in its wake alone. The particle domain holds the disk at every angle, with an outflow fringe at its downstream end.

## Status

The case is configured but has not run: the mesher stops in its wall projection on the blades. Since 19 November 2026 the mesher puts the wall on the sharp edges of a blade, the rims of its root and tip caps and the two edges of its flat base ([mesh setup](../../../docs/fvm.md#mesh-setup)); before, wall faces cut across those edges and left 27 cell centres inside the blades. A straight slab of the tip section, pitched 17° and tilted 7° against the grid, now meshes with the cells of this case, also with a 0.0125 m refinement box around its trailing edge. On the three lofted blades the octree template of 351 105 cells, 412 139 with the wall layer, takes about four hours on two cores to the end of the mesh optimisation, which cannot untangle about 80 faces at the flat bases, 7.4 mm wide at the tip and 12.7 mm at the root under the 0.025 m wall cells. The wall projection then gives one or two of the thin wall-layer cells there, near 4.7 m radius on blades 1 and 2, a non-positive volume at every relaxation, and the mesher stops with "Mesh contains non-positive cell volumes". The case files are the intended configuration.

## Run

Run commands from this directory.

```bash
./allrun.sh
```

`./allrun.sh` starts from $t=0$; `./allcontinue.sh` resumes the latest accepted coupled backup. `python setup.py --end-time 0.03 --compute-device cpu` runs a short check on the CPU; the particle solver otherwise takes the GPU. Meshing takes about four hours on two cores and a few GB of memory before it stops (see Status); a completed mesh is cached in `assets/`.

## Results

| File | Content |
| --- | --- |
| `samples/blade_forces.csv` | Pressure, viscous and total force, moment about the rotor centre and `rotational_power`, `translational_power`, `power` of each blade patch every 0.006 s, as the VLM `vlm_surface_forces.csv` |
| `samples/flow_integrals.csv`, `samples/wake_*D.pvd`, `samples/streamwise_r*.csv` | Particle invariants, wake planes at 0, 1 and 2 diameters and streamwise lines at $r/R=0.25$ and 0.65, as the VLM case |
| `solution/fvm.pvd`, `solution/vpm.pvd` | Turning FVM fields and particle states |

The coefficients use $C_T=T/(\tfrac12\rho U_\infty^2\pi R^2)$ and $C_P=P/(\tfrac12\rho U_\infty^3\pi R^2)$ with $P=\mathbf{M}\cdot\boldsymbol{\Omega}$ the shaft power the fluid delivers, the definitions of the VLM case.
