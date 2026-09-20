# Cube reference flow

Body-fitted flow around a unit cube at Re = 1000, with U = 1 m/s and
kinematic viscosity 0.001 m²/s. The setup declares the domain, geometric
refinement, equilibrium Smagorinsky model, boundary conditions and samplers.

Run a standalone case with the installed OpenONDA API:

```bash
python setup.py --name fine --dx 0.06
```

The grid-independence assessment uses four exact wall spacings in a geometric
progression, followed by a timestep control on the identical fine mesh:

```bash
./allrun.sh
```

This explicit `--campaign` assessment requires an OpenONDA source checkout with
its `studies` package available. The study owns resource admission and mesh
qualification; the tutorial remains the single physical configuration.
The fine reference level runs to 30 s with visualization and rolling checkpoints
every 0.25 s; the other levels run to 120 s. Fresh outputs use
`campaigns/geometric_r15_30s_fine`. Already-running simulations retain their
submitted settings.
See [GRID_CAMPAIGN.md](GRID_CAMPAIGN.md) for domains, spacings, statistics,
restart and reporting commands. Each case writes force and velocity-profile
samples plus native solver metadata. Campaign outputs are separate from the
standalone `solution/` and `samples/` directories.

The launcher contains only run commands. Postprocessing is a separate action;
run completion alone does not establish grid independence.
