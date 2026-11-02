# Elastically mounted cylinder with two degrees of freedom

A no-slip cylinder on equal springs and dampers in the streamwise and cross-flow directions vibrates in a uniform stream. The structure and the operating points are those of the fully meshed study of Martins and Avila (2019), *Int. J. Mech. Sci.* 156, 235–249 (`../two_cylinders.pdf`): mass ratio $m^*=2.4$, ideal added mass $C_A=1$, damping ratio $\zeta=0.5\,\%$, and its points B1 and B2 of table 5 in the medium-Reynolds set of table 4. The fluid force on the wall drives the cylinder (`fvm.ElasticMount`); the FVM box and mesh ride with it and one particle layer carries the outer wake. The [reference flow](reference_flow/README.md) solves the same problem with the FVM alone.

| Quantity | Value |
| --- | --- |
| Diameter; freestream | 1 m; 1 m/s in $+x$ |
| Density; viscosity | 1 kg/m³; $1/\mathrm{Re}$ m²/s |
| Runs | B1: $\mathrm{Re}=6400$, $V^*=U_\infty/(f_{n,w}D)=7.68$; B2: $\mathrm{Re}=8300$, $V^*=9.96$ |
| Structure | $m^*=4m/(\rho\pi D^2L)=2.4$; $f_{n,w}=\frac{1}{2\pi}\sqrt{k/(m+\rho\pi D^2C_A/4)}=U_\infty/(V^*D)$ with $C_A=1$; $\zeta=b/(2\sqrt{km})=0.005$; equal in $x$ and $y$ |
| FVM domain | $[-2.4,4.4]\times[-2.4,2.4]$ m, relative to the cylinder |
| Cell and particle spacing | 0.04 m at the interface, 0.005 m at the wall |
| FVM step; coupling interval | 0.0025 s; 0.04 s |
| End time; statistics window | 25 natural periods ($25V^*$ s); the last 10 periods |

## Physics and numerical model

The mesh, transfer region and particle model follow the [fixed cylinder](../01_cylinder_shedding_flow/README.md); the wall cells are three octree levels finer, 0.005 m, for the laminar boundary layer, about $3D/\sqrt{\mathrm{Re}}=0.033$ m thick at $\mathrm{Re}=8300$. The FVM mesh and particle frame translate with the cylinder, as described in [moving bodies](../../../docs/coupling.md#moving-bodies). Along each direction the cylinder obeys $m\ddot{x}+b\dot{x}+kx=F$ with the pressure and viscous force $F$ on its wall, integrated exactly over every FVM step with the force of the last accepted step ([moving meshes](../../../docs/fvm.md#moving-meshes)); the paper integrates the same equation with an Adams–Moulton scheme at 200 steps per period, this case takes 3072 (B1) and 3984 (B2). A startup vortex behind the cylinder breaks the symmetry, as in the fixed-cylinder case.

The targets are the paper's points B1 ($A_y^*=1.137$ at $\mathrm{Re}=6400$, upper branch) and B2 ($A_y^*=0.677$ at $\mathrm{Re}=8300$, lower branch) of table 5, both in the *Medium Re, $\zeta=0.5\,\%$* set (case 1-2 of table 4, $m^*\zeta=1.2\,\%$). The paper gives these points by Reynolds number only. Its sets hold the natural frequency fixed, $f_n=0.1$, 0.3, 1.0 and 3.3 Hz, and their Reynolds ranges over $V^*=3$–15 follow $\mathrm{Re}=2500\,f_n V^*$ with the medium frequency $f_n=1/3$ Hz (the low set gives $\mathrm{Re}=250\,V^*$, the medium set $2500$–$12000$, whose reduced-velocity axis in figure 6 ends near 14). Hence $V^*=3\,\mathrm{Re}/2500$: 7.68 for B1 and 9.96 for B2. This mapping is a reading of the paper, not a value it states. The paper solves the three-dimensional unsteady RANS equations (SST $k$–$\omega$) over a span of $\pi D$; this case is planar and laminar, so the two-dimensional shear-layer instabilities that the RANS model damps are resolved here, and the amplitudes need not coincide.

## Run

Run commands from this directory. Each line of `allrun.sh` runs one point, `python setup.py --reynolds-number 6400 --reduced-velocity 7.68` for B1 and `--reynolds-number 8300 --reduced-velocity 9.96` for B2; the results go to `solution/re_<Re>_vstar_<V*>` and `samples/re_<Re>_vstar_<V*>`.

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
./allplot.sh
```

`./allrun.sh` starts from $t=0$; `./allcontinue.sh` resumes the latest accepted coupled backup of each run. `python setup.py --reynolds-number 8300 --reduced-velocity 9.96 --end-time 2 --compute-device cpu` runs one short check on the CPU; the particle solver otherwise takes the GPU.

## Results

| File | Content |
| --- | --- |
| `figures/response` | $A_y^*$, $A_x^*$ and $f_y/f_{n,w}$ against $V^*$; `allplot.sh` prints them with the statistics window and writes `figures/auxiliary/response.csv` |
| `figures/histories_re_*` | Displacements, lift coefficient and the $x$–$y$ trajectory of each run |
| `figures/vorticity_re_*` | FVM and particle vorticity at the last upper turning point and the following downward crossing of each run |
| `figures/coupling_cost_re_*` | Wall time by solver part and particle count |
| `solution/re_*/fvm.pvd`, `solution/re_*/vpm.pvd` | Moving FVM fields and particle states, the particles relative to the box |

The reduced amplitude $A^*$ is the mean of the highest 20 % of the oscillation peaks in the window, the paper's definition. The paper's response in `figures/response` is the two values of its table 5 (points B1 and B2), from `assets/references/martins_avila_2019_medium_re_zeta_0p5.csv`, plotted at $V^*=3\,\mathrm{Re}/2500$.
