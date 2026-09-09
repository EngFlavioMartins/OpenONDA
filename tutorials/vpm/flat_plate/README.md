# Flat plate: VLM with a free particle wake

After installing OpenONDA, run a case directly:

```sh
python setup.py --mode static --angle 5
python setup.py --mode moving --angle 5
```

`./allrun.sh` runs the ten angles in both frames. `./allplot.sh` plots their saved samples; `./allplot.sh pdf` exports PDF. `allclean.sh` removes this case's generated results.

`python assets/validate_results.py` checks completion, native force sampling,
settled lift, full-vector bound/wake closure, moving/static agreement, reflection
symmetry and the generated PNG/PDF figures. Add `--pre-plot` to check the samples
before rendering. These checks supplement the physical reference comparisons
in the figures; they do not claim exact agreement with lifting-line theory.

The plate has chord 1 m, span 10 m and reference speed 10 m/s. The moving plate accelerates through still fluid; the stationary plate encounters an inclined freestream. The case selects bound-leg Kutta–Joukowski loads plus the unsteady pressure contribution from changing surface potential jump. Forces and moments are integrated by the solver. Separation, stall and viscous skin friction are outside this VLM model.

The particle wake uses direct Biot–Savart summation. At this case's particle
count, direct induction provides a practical circulation reference without
tree approximation error. Both the velocity and transposed stretching rate
use the same regularized pair interactions.

Each case writes to `samples/exp_<mode>_aoa<angle>/`:

- `vlm_forces.csv`: total forces, coefficients, moments, separate unsteady-pressure forces and bound/wake vector-strength budget, recorded every step.
- `vlm_spanwise_flat_plate.csv`: sectional loading, circulation and physical strip widths.
- `vlm_chordwise_flat_plate.csv`: panel circulation, pressure jump, forces and relative velocities at bound midpoints.
- `vlm_*.vtp` and `vlm.pvd`: lattice geometry, velocities and loads for ParaView, selected by `VLMSampler`.
- `flow_integrals.csv` and any configured field probes: VPM diagnostics and sampled flow. Velocity probes include the attached VLM field.

Numerical checkpoints, logs and the default `vpm_metadata.json` remain under `solution/<case>/`. Geometry/flow samples are recorded every five steps and backups every forty steps. Each backup also writes `vlm_<step>.vtp` and updates `vlm.pvd` beside the VPM files, including the final step. Open `solution/<case>/vlm.pvd` in ParaView to animate the surface and color its circulation, pressure jump, velocity or forces. Checkpoints include the VLM geometry, circulation and load history, and motion state required to continue the coupled calculation. Plotting reads the solver's metadata and geometry to recover reference values; no tutorial metadata file is written.

The figures compare the force polar, moving/static histories, spanwise loading, induced-velocity profiles, bound/wake strength closure and the force/impulse budget. The steady polar averages the final five chord lengths. The velocity profile uses circulation-weighted bound-midpoint velocities: these are the velocities entering the sectional Kutta–Joukowski force. The impulse figure integrates the sampled surface forces and compares them with the change in native bound-plus-wake fluid impulse; it reads CSV samples only.

The reference is rectangular-wing lifting-line theory, solved with independent half-span collocation points. Lift, induced drag, spanwise circulation and downwash share this one solution. The elliptic loading is shown only as a shape comparison. Lifting-line theory is a small-incidence, high-aspect-ratio approximation; finite chord and the viscous particle wake can produce larger differences near the tips. The transient figure shows the steady limit rather than presenting a two-dimensional Wagner approximation as an exact three-dimensional ramp response.

The startup figure shows the first two chord lengths, including the static
plate's impulsive-start load. A separate figure shows the subsequent wake
development so that this peak does not obscure convergence. The finite first
pressure-time load is an average over the first time step, not a resolved
infinite-frequency impulsive force.

Qualification is in progress. Current CPU runs include native bound/free strength
exchange and unsteady pressure. Completed results and budget figures are stored
here. The angle/frame cases are being checked sequentially; their preceding
results remain in qualification
subfolders when replaced. The full sweep must pass before all these results are
treated as a qualified reference. See the [current audit checklist](../../../docs/reviews/2026-09-vlm-todos.md).
