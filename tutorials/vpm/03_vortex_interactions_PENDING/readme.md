# Vortex interactions — LBM Fig. 5

Run from this directory with the OpenONDA environment active:

```sh
bash allrun.sh
bash allplot.sh
```

`allrun.sh` first calls `allclean.sh`, which deletes this tutorial's previous
`solution/`, `samples/`, `figures/`, and `study_results/` data. The scripts use
plain `python` from the installed OpenONDA environment, with no `PYTHONPATH`
override.

Because OpenONDA is installed in that environment, a single case also works
from another directory with `python /absolute/path/to/setup.py baseline`.

[`setup.py`](setup.py) is the single active case definition. It builds the
unperturbed Re=3000 CS/LES baseline, then adds either stretching viscosity or
moment-preserving Pedrizzetti relaxation. Earlier ring studies motivated these
two simple candidates; their performance with this Fig. 5 baseline is untested.
The conservative particle transfer is common to every case, including the
baseline; it maintains resolution without adding damping or projection.

The physical and numerical baseline is R=1, circulation=π per ring, initial
separation=1, Gaussian physical core=0.1, h=initial σ=0.05, dt=0.00375,
SSPRK3, transposed tree induction, Cs=0.20 with a fixed 0.053 LES filter, and
zero imposed disturbance. Every case requests 2400 steps (t=9), subject to
native particle, memory, and numerical-health limits. A stopped method does
not prevent later commands in `allrun.sh` from running.

Each run starts at t=0 and writes to matching
`solution/<method>/` and `samples/<method>/`
directories. `allclean.sh` removes them before a full rerun. For one method
run directly, clear that method's old output directories first; the solver
appends samples and rejects duplicate initial times.

`allplot.sh` reads these three cases and writes meridional vorticity sections,
diagnostic histories, and leapfrogging kinematics to
`figures/leapfrogging_study/`. All figures use the thesis Matplotlib template
and are at most 12.5 cm wide. Each simulation has one color and marker in
every plot; solid and dashed lines distinguish the two rings. The supplied
LBM trajectory is the unperturbed Fig. 5 case; it does not provide a time or a
three-dimensional breakdown field. See [reference provenance](assets/references/README.md).

`assets/` contains the figure scripts, the shared `postprocess.py` plotting
helper, and the digitized LBM reference. `allplot.sh` uses whatever sampler
output has been saved so far; unfinished cases are skipped until they produce
data.
