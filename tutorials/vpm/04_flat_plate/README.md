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

In ParaView, `exp_static_*` has a fixed plate (leading edge at x=0, trailing edge
at x=1) and air flowing toward positive x. Its wake travels toward positive x.
In `exp_moving_*`, the plate travels toward negative x through still air; the
wake is left behind on its trailing-edge side. All surface and particle files
use their case's fixed spatial coordinates, without a camera or coordinate
transform that follows the plate. A negative angle such as `aoan05` changes the
incidence, not the streamwise direction. For example, the static -5° inflow is
(9.962, 0, -0.872) m/s, while the moving -5° plate reaches (-10, 0, 0) m/s.

The particle wake uses direct Biot–Savart summation. At this case's particle
count, direct induction provides a practical circulation reference without
tree approximation error. Both the velocity and transposed stretching rate
use the same regularized pair interactions.

Each case writes native outputs to two destinations. Attached VLM force, loading
and surface state are owned by the VPM accepted-step/checkpoint lifecycle; this
setup has no independent `VLMSampler` clock.

- Under `samples/exp_<mode>_aoa<angle>/`: `vlm_forces.csv` contains total forces, coefficients, moments, separate unsteady-pressure forces and bound/wake vector-strength budget, recorded every step; `vlm_spanwise_flat_plate.csv` contains sectional loading, circulation and physical strip widths; `vlm_chordwise_flat_plate.csv` contains panel circulation, pressure jump, forces and relative velocities at bound midpoints; `flow_integrals.csv` and any configured field probes contain VPM diagnostics and sampled flow. Velocity probes include the attached VLM field.
- Under `solution/<case>/`, open `vpm.pvd` or `vlm.pvd` in ParaView. Particle HDF5/XDMF frames are below `vpm/`; VLM surface frames are below `vlm/`.

Numerical checkpoints, logs and the default `vpm_metadata.json` remain under
`solution/<case>`. Backups are every forty accepted steps (0.5 s); the backup
clock is therefore the only native VLM surface cadence. Open the VTP companion
files from `solution/<case>/` in ParaView to inspect circulation, pressure jump,
velocity or forces. Checkpoints include the VLM geometry, circulation and load
history, and motion state required to continue the coupled calculation.
Plotting reads the solver's metadata and geometry to recover reference values;
no tutorial metadata file is written.

The figures compare the force polar, moving/static histories, spanwise loading, induced-velocity profiles, bound/wake strength closure and the force/impulse budget. The steady polar averages the final five chord lengths. The velocity profile uses circulation-weighted bound-midpoint velocities: these are the velocities entering the sectional Kutta–Joukowski force. The impulse figure integrates the sampled surface forces and compares them with the change in native bound-plus-wake fluid impulse; it reads CSV samples only.

The reference is rectangular-wing lifting-line theory, solved with independent half-span collocation points. Lift, induced drag, spanwise circulation and downwash share this one solution. The elliptic loading is shown only as a shape comparison. Lifting-line theory is a small-incidence, high-aspect-ratio approximation; finite chord and the viscous particle wake can produce larger differences near the tips. The transient figure shows the steady limit rather than presenting a two-dimensional Wagner approximation as an exact three-dimensional ramp response.

The startup figure shows the first two chord lengths, including the static
plate's impulsive-start load. A separate figure shows the subsequent wake
development so that this peak does not obscure convergence. The finite first
pressure-time load is an average over the first time step, not a resolved
infinite-frequency impulsive force.

The fresh 2026-09-10 CPU sweep completed all 20 authored cases and passed
`python assets/validate_results.py`; the native strength-closure residuals are
below `4e-6`. The preceding current outputs remain in
`solution/historical_2026-09-10_pre-rerun/` and
`samples/historical_2026-09-10_pre-rerun/`. The sweep is a reproducible
attached-flow reference, not a stall, separation or skin-friction model.
Historical native states and sampled scientific evidence remain available;
obsolete VLM visualization exports have been deleted. Every retained VLM
surface now has a matching native VPM backup at the same physical time.
See the [VPM guide](../../../docs/vpm.md#vlm-coupling) for VLM assumptions
and the remaining limitations of rotor calculations.
