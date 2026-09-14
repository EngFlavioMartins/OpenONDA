# Writing code and experiments in OpenONDA

Start with the physical question, the reference result, and the outputs needed
to judge it. Write the smallest set of cases that answers that question. The
[Lamb–Oseen setup](../../tutorials/vpm/01_lamb_oseen_vortex/setup.py) and
[vortex-ring setup](../../tutorials/vpm/02_vortex_ring/setup.py) show the intended
style for VPM tutorials and studies.

## Write the physics first

- Put physical inputs at the top of `setup.py`: geometry, circulation, core
  radius, Reynolds number, disturbance, and viscosity. Follow with particle
  resolution, time step, run length, sample cadence, and numerical limits.
- Write derived quantities where readers can see the physical relationship:

  ```python
  RING_STRENGTH = np.pi                 # circulation [m²/s]
  REYNOLDS_NUMBER = 3000.0             # Γ/ν
  KINEMATIC_VISCOSITY = RING_STRENGTH / REYNOLDS_NUMBER
  ```

- Use short comments for units, formulas, or a non-obvious physical choice.
  Do not narrate what the next line of Python already says. Name values after
  CFD quantities, such as `CORE_RADIUS`, `PARTICLE_SPACING`, and
  `MAX_LAGRANGIAN_CFL`; do not bury them in generic configuration tables.
- Keep `setup.py` in reading order: inputs, derived quantities, initial vortex
  geometry, `Numerics`, samplers and backups, `RunPlan`, then the solver call.
  A `build_case` function may share common physics; a short `run_case` may run it.
  Put plotting, analytical solutions, and diagnostics in `assets/`.
- Use `if` only when a physical or numerical choice truly changes: for example,
  Lamb–Oseen's merging sample cadence or DVH heat-transfer cadence. Keep that
  choice beside the affected value. Avoid extra switches, fallback chains,
  restart selection, and wrappers around a single solver call.
- Use CFD language in names, comments, and documentation. Prefer `circulation`,
  `core_radius`, `stretching`, `diffusion`, `remeshing`, and `sample_time` to
  generic software terms such as “contract,” “handshake,” “orchestration,” or
  “gate” when describing a flow calculation.

## Keep the runnable experiment small

- Keep one active `setup.py` per tutorial. Use `argparse` only for meaningful
  physical variants. For a stabilization comparison, use one reference baseline
  and two or three practical methods, all with the same initial conditions,
  resolution, time step, and output cadence. Change one method per case.
- Check that the baseline reproduces the stated reference before judging a
  stabilization method. Set a practical run budget; add cases only when the
  existing result calls for them. Numerical survival alone is not physical
  agreement.
- Use the installed OpenONDA package and plain `python`. Fix a broken
  installation instead of adding `PYTHONPATH`, `sys.path`, interpreter selection,
  or working-directory bootstrapping to run and plot scripts. A case-local path
  for its own samples or assets is fine.
- Make `allrun.sh` a short, explicit list of `python setup.py ...` commands.
  Make `allplot.sh` read exactly their outputs. Let `allclean.sh` remove only
  generated data in its tutorial directory; call it explicitly when fresh
  outputs are required. Do not hide restart, skip, cleanup, or output rewriting
  logic in `setup.py`.

## Check what the run means

Keep literature reference inputs separate from `solution/`, `samples/`, and
`figures/`. Test imports, case construction, shell commands, and the plot input
list before a costly run. For VPM, inspect particle count, CFL, divergence,
vorticity alignment, and energy/enstrophy drift. Check that coherent vortex
cores stay resolved. Report early stops and disagreement with the reference;
particle clipping, stronger damping, or a higher capacity cannot replace a
resolution check.

The [tutorial style guide](tutorial_style.md) covers file layout and figures;
the [test guide](../../tests/README.md) covers numerical verification.
