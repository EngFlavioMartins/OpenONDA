# Historical quadcopter-rotor diagnostics

These earlier diagnostic runs are retained for reproducibility. They are not
part of the tutorial validation sequence, and no further isolated quadcopter
rotor studies are scheduled. The single rotor validation case is
`vpm/rotor_flow`, the wind turbine.

Reproduce a historical case directly with `python setup.py --case coarse`. `./allrun.sh` runs
the seven completed cases in dependency order. `./allplot.sh` plots the saved native data;
`./allplot.sh pdf` writes vector figures. Plotting does not run a simulation.

Each case writes native metadata, logs and sparse checkpoints to
`solution/<case>/`, and native force, sectional-loading and flow-integral samples
to `samples/<case>/`. Figures are in `figures/`. The stored solutions were
transferred from the qualification runs without repeating them.

| Case | Degrees per step | Blade panels | Revolutions | Relaxation |
|---|---:|---:|---:|---:|
| `coarse` | 3.75 | 4 × 12 | 1–6 | 0 |
| `time_refined` | 1.875 | 4 × 12 | 1–3 | 0 |
| `mesh_refined` | 3.75 | 8 × 24 | 1–3 | 0 |
| `relaxed` | 3.75 | 4 × 12 | 1–6 | 0.3 |
| `relaxed_time_refined` | 1.875 | 4 × 12 | 1–3 | 0.3 |
| `continue_8` | 3.75 | 4 × 12 | 7–8 | 0 |
| `continue_12` | 3.75 | 4 × 12 | 9–12 | 0 |
| `relaxed_moments` | 3.75 | 4 × 12 | 1–12 | 0.3, preserve moments |

All cases use Gaussian particles, common core overlap 2.5 and CPU/FMM induction.
`continue_8` loads `solution/coarse/vpm_000576.h5`; `continue_12` loads
`solution/continue_8/vpm_000768.h5`. Each continuation writes its own output.
All seven original cases, including the twelve-revolution continuation, have
completed. `relaxed_moments` failed at step 3: its moment correction exceeded
the native strength-growth limit (0.1206% versus 0.1%). Its partial native
outputs are retained; it has no flow-integral samples yet.
A completed run does not by itself establish stationarity.

Forces are recorded every two steps and flow integrals every twelve steps;
checkpoints are written every two revolutions and at completion. The mesh-refined
case also samples VLM geometry once per revolution. Postprocessing uses the
solver's native metadata, embedded geometry and CSV samples. No metadata writer
or checkpoint extractor is needed.

The resolution figures compare matching complete revolutions. At the third
revolution, halving the time step changes mean thrust/power by about
0.0071%/0.00065%; doubling the blade mesh changes them by 3.37%/2.19%.
Pointwise loading remains more sensitive near the blade root. The relaxation
figure also shows the longer unrelaxed continuation: its last six revolutions show thrust/power
half-window drift of 6.01%/12.22%, despite mean errors against BEM of only
−0.96%/−4.69%. Final misalignment is 75.08 degrees, and the twelfth-cycle wake
impulse predicts about 1.97 times the blade thrust. This wake is not qualified. Keeping the same relaxation factor per
step while halving the time step also changes its physical relaxation frequency.

The wake-health figure compares native strength/vorticity misalignment, normalized
divergence and wake-impulse change divided by integrated blade thrust over each
complete revolution. A ratio near one supports momentum consistency. It is a
wake-only diagnostic; bound-impulse changes and other unsteady terms are not
included. It must be considered alongside the load and resolution comparisons.

The `relaxed_moments` comparison retains the same grid, time step and relaxation
factor while enabling the solver's native preservation of vector strength and
impulse moments during relaxation. This tests the large impulse/load mismatch
in the ordinary relaxed case. The current solver rejects this candidate. Its
inputs remain available with `python setup.py --case relaxed_moments`, and its
failed outputs are retained. It is excluded from `allrun.sh` because the native
solver rejected it. Its acceptance limit remains unchanged. This diagnostic
branch is closed; it does not gate the quadcopter demonstration.

The coarse relaxed result predates the native `wake_core_overlap` option and
used the equivalent experimental overlap formula. The reproducible case now
expresses that setting through the solver's public configuration. Its original
native metadata is retained unchanged; it does not contain that later field.

`allclean.sh` removes these studies' generated output directories.
