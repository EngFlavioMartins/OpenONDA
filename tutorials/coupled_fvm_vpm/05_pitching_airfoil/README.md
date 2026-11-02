# Pitching NACA 0012 airfoil at $Re=1000$

A no-slip NACA 0012 airfoil pitches about its quarter chord in a uniform stream, $\alpha(t)=\alpha_0\sin(2\pi f t)$. The FVM box and mesh turn with the airfoil; one particle layer carries the outer wake. The [reference flow](reference_flow/README.md) solves the same moving-body problem with the FVM alone.

| Quantity | Value |
| --- | --- |
| Chord; freestream | 1 m; 1 m/s in $+x$ |
| Density; viscosity | 1 kg/m³; $10^{-3}$ m²/s |
| Pitch motion | $\alpha_0=10°$ about the quarter chord; $f=0.2$ Hz, $k=\pi f c/U_\infty=0.63$ |
| FVM domain | $[-1.6,2.6]\times[-1.2,1.2]$ m, in the frame of the airfoil, mid-chord at the origin |
| Cell and particle spacing | 0.025 m |
| FVM step; coupling interval | 0.005 s; 0.025 s |
| Statistics window | $15\le t<30$ s, three pitch periods |

## Physics and numerical model

The mesh, transfer region and particle model follow the [elastically mounted cylinder](../04_oscillating_cylinder/README.md). Here the mesh turns instead of translating, so the coupled FVM box sweeps a different part of the particle domain at each angle. The transfer region, transfer lattice, interpolation stencils and walls are kept in the coordinates of the turning box; the particles stay in the inertial frame, which the pitch axis does not leave. Particle output and profiles are therefore at inertial positions, as in the reference.

Both runs use three outer PIMPLE passes per FVM step, which converge the steps of the accelerating mesh ([FVM moving meshes](../../../docs/fvm.md)).

## Run

Run commands from this directory.

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
./allplot.sh
```

`./allrun.sh` starts from $t=0$; `./allcontinue.sh` resumes the latest accepted coupled backup. `python setup.py --end-time 0.5 --compute-device cpu` runs a short check on the CPU; the particle solver otherwise takes the GPU.

## Results

| File | Content |
| --- | --- |
| `figures/forces` | Drag, lift and quarter-chord moment against the FVM only; `allplot.sh` prints the mean drag and the first-harmonic lift and moment amplitudes and phases |
| `solution/fvm.pvd`, `solution/vpm.pvd` | Turning FVM fields and particle states |
