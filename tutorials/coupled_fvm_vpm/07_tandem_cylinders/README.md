# Two cylinders in tandem at $Re=150$

Two circular cylinders of diameter $D$ have their centres $8D$ apart along a uniform stream. Each cylinder carries the FVM domain, mesh, time steps and transfer settings of the baseline run of the [single cylinder](../01_cylinder_shedding_flow/README.md); the two FVM domains do not touch, so the wake of the upstream cylinder crosses the gap as particles only and reaches the outer boundary of the downstream domain through the same particle set. The [reference flow](reference_flow/README.md) solves both cylinders on one FVM mesh.

| Quantity | Value |
| --- | --- |
| Diameter; centre spacing | 1 m; 8 m along $x$, upstream centre at the origin |
| Freestream; Reynolds number | 1 m/s in $+x$; $UD/\nu=150$ |
| FVM domain of each cylinder | $[-2.4,4.4]\times[-2.4,2.4]$ m about its centre, the box of case 01 |
| Cell and particle spacing | 0.04 m |
| FVM step; coupling interval | 0.008 s; 0.04 s, five FVM steps |
| Particle domain | $[-5,38]\times[-5,5]$ m, outflow fringe over the last 15 m |
| End time | 200 s |

## Physics and numerical model

The coupled solver holds one FVM domain per cylinder and one particle set. In each coupling interval the particles advance once; the boundary data of both outer boundaries are evaluated from the whole particle field; both FVM domains advance with their own substeps; the vorticity of each domain replaces the particles of its transfer region; and the boundary data of both domains are evaluated again from the particles after both transfers. The coupling iterations converge when both domains meet the tolerances. Before the first step the solver checks that the remeshing regions of the two domains stay at least four particle spacings apart over the run, so that the two transfers change distinct particles.

With `--heave-frequency f` the downstream cylinder heaves across the stream, $y=0.2D\sin(2\pi f t)$, on its own moving mesh. The particles then stay in the inertial frame, and the coupling geometry of the moving domain is kept in its body coordinates.

## Run

Run commands from this directory.

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
```

`./allrun.sh` runs the fixed cylinders from $t=0$ into `solution/fixed/` and `samples/fixed/`; `python setup.py --heave-frequency f` runs the heaving downstream cylinder into `solution/heaving/`. `./allcontinue.sh` resumes the latest accepted coupled backup. `python setup.py --end-time 0.12 --compute-device cpu` runs a short check on the CPU; the particle solver otherwise takes the GPU.

## Results

A run writes, per cylinder, `samples/<run>/<cylinder>/forces_history.csv` (pressure and viscous forces and moments every 0.04 s) and `fvm_transverse_x2D.csv`; the heaving cylinder also writes `motion_history.csv`. The particle samples are `samples/<run>/flow_integrals.csv` every coupling interval and `samples/<run>/<cylinder>_transverse_x2D.csv` and `_x4D.csv`, velocity and vorticity on transverse lines 2 and 4 D behind each cylinder every 0.12 s.
