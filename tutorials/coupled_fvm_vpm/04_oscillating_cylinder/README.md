# Elastically mounted cylinder with two degrees of freedom

A no-slip cylinder on equal springs and dampers in the streamwise and cross-flow directions vibrates in a uniform stream. The structure and the operating points are those of the fully meshed study of Martins and Avila (2019), *Int. J. Mech. Sci.* 156, 235–249 (`../two_cylinders.pdf`): mass ratio $m^*=2.4$, ideal added mass $C_A=1$, damping ratio $\zeta=0.5\,\%$ and the low-Reynolds set $\mathrm{Re}=250\,V^*$ of its table 4. The fluid force on the wall drives the cylinder (`fvm.ElasticMount`); the FVM box and mesh ride with it and one particle layer carries the outer wake. The [reference flow](reference_flow/README.md) solves the same problem with the FVM alone.

| Quantity | Value |
| --- | --- |
| Diameter; freestream | 1 m; 1 m/s in $+x$ |
| Density; viscosity | 1 kg/m³; $1/\mathrm{Re}$ m²/s with $\mathrm{Re}=250\,V^*$ |
| Reduced velocities | $V^*=U_\infty/(f_{n,w}D)=4$, 6 and 8, hence $\mathrm{Re}=1000$, 1500 and 2000 |
| Structure | $m^*=4m/(\rho\pi D^2L)=2.4$; $f_{n,w}=\frac{1}{2\pi}\sqrt{k/(m+\rho\pi D^2C_A/4)}=U_\infty/(V^*D)$ with $C_A=1$; $\zeta=b/(2\sqrt{km})=0.005$; equal in $x$ and $y$ |
| FVM domain | $[-2.4,4.4]\times[-2.4,2.4]$ m, relative to the cylinder |
| Cell and particle spacing | 0.04 m at the interface, 0.02 m at the wall |
| FVM step; coupling interval | 0.0025 s; 0.04 s |
| End time; statistics window | 25 natural periods ($25V^*$ s); the last 10 periods |

## Physics and numerical model

The mesh, transfer region and particle model follow the [fixed cylinder](../01_cylinder_shedding_flow/README.md); the wall cells are one octree level finer for the boundary layers at $\mathrm{Re}$ up to 2000. The FVM mesh and particle frame translate with the cylinder, as described in [moving bodies](../../../docs/coupling.md#moving-bodies). Along each direction the cylinder obeys $m\ddot{x}+b\dot{x}+kx=F$ with the pressure and viscous force $F$ on its wall, integrated exactly over every FVM step with the force of the last accepted step ([moving meshes](../../../docs/fvm.md#moving-meshes)); the paper integrates the same equation with an Adams–Moulton scheme at 200 steps per period, this case takes 1600. A startup vortex behind the cylinder breaks the symmetry, as in the fixed-cylinder case.

The target of the three runs is the paper's *Low Re, $\zeta=0.5\,\%$* response contour (figures 6 and 12, cases 1-1 of table 4): the upper branch with its peak $A_y^*=0.794$ at $\mathrm{Re}=1500$ (point A1, $V^*=6$) and the lower branch through $A_y^*=0.594$ at $\mathrm{Re}=2100$ (point A2). The Reynolds number of each run follows from the paper's fixed natural frequency: its low-Reynolds set spans $\mathrm{Re}=750$–$3300$ for $V^*=3$–15 and point A1 is its $V^*=6$ peak, so $\mathrm{Re}=250\,V^*$. The paper solves the three-dimensional unsteady RANS equations (SST $k$–$\omega$) over a span of $\pi D$; this case is planar and laminar, so the two-dimensional shear-layer instabilities that the RANS model damps are resolved here, and the amplitudes need not coincide.

## Run

Run commands from this directory. Each line of `allrun.sh` runs one reduced velocity; the results go to `solution/vstar_<V*>` and `samples/vstar_<V*>`.

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
./allplot.sh
```

`./allrun.sh` starts from $t=0$; `./allcontinue.sh` resumes the latest accepted coupled backup of each run. `python setup.py --reduced-velocity 6 --end-time 2 --compute-device cpu` runs one short check on the CPU; the particle solver otherwise takes the GPU.

## Results

| File | Content |
| --- | --- |
| `figures/response` | $A_y^*$, $A_x^*$ and $f_y/f_{n,w}$ against $V^*$; `allplot.sh` prints them with the statistics window and writes `figures/auxiliary/response.csv` |
| `figures/histories_vstar_*` | Displacements, lift coefficient and the $x$–$y$ trajectory of each run |
| `figures/vorticity_vstar_*` | FVM and particle vorticity at the last upper turning point and the following downward crossing of each run |
| `figures/coupling_cost_vstar_*` | Wall time by solver part and particle count |
| `solution/vstar_*/fvm.pvd`, `solution/vstar_*/vpm.pvd` | Moving FVM fields and particle states, the particles relative to the box |

The reduced amplitude $A^*$ is the mean of the highest 20 % of the oscillation peaks in the window, the paper's definition. The paper's response in `figures/response` comes from `assets/references/martins_avila_2019_low_re_zeta_0p5.csv`: the two values of its table 5 (points A1 and A2), and nine values digitized from the raster of its figure 12 by locating the filled blue circle markers of the *Low Re, $\zeta=0.5\,\%$* series on the axes of the figure (about ±0.005 in $A_y^*$ and ±1 % in $\mathrm{Re}$; the digitized values reproduce A1 and A2 to 0.001). Markers hidden by annotations or other series (two of the thirteen, and the three lowest points near $\mathrm{Re}=750$) are omitted. Every point of that file is labelled by its source in the CSV and in the figure legend.
