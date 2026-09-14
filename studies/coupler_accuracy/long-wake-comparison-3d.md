# Longer matched 3D wake comparison

The selected precision-corrected, iterated coupling is now advancing through
400 exchanges, from physical time `0.5` to `20.5`. This comparison is running;
its completion, wake development and force/profile agreement are unproven.
The shorter [time-resolution experiment](interface-time-resolution-3d.md)
did not justify replacing the `0.05` exchange interval with `0.01`.

## Fixed problem and recorded observations

The run retains the unit cube and the original matched native cells: 16,936
cells in the small FVM domain and 53,752 in the independent full FVM. The
nominal near-body spacing is `0.0625`; the small box remains approximately
`[-1.5,1.5]^3`. Both FVMs use `dt=0.01`, implicit Euler, Gauss gradients,
linear-upwind convection, three outer correctors, two pressure correctors,
unit relaxation and absolute linear tolerances `1e-11`. Both models are
laminar with `ν=0.001`.

VPM retains fully three-dimensional FMM induction, RK2, GBD and single-precision
particle fields. Only the qualified auxiliary body query is evaluated in
double precision. The existing direct 108-panel cube and buffered-M4 renewal
are retained. The fixed-predictor interface iteration uses at most 12 sweeps
with unchanged normal and derivative thresholds `1e-6`. Every interval's
first endpoint map must replay fields and clocks bitwise. A capped interval
is explicitly reported, rather than being called converged.

The full reference evolves independently; it never supplies an evolving
boundary value or particle correction to the hybrid. Drag and volume-weighted
velocity errors are observed every exchange (`0.05`). The
[qualified profile observer](profile-validation-3d.md) records centreline and
off-axis velocities every `0.25`, retaining both the full-reference stencil
and the matched small-FVM stencil. Each profile frame also retains particle
and panel source fields. No two-dimensional approximation, span replication,
force rescaling or phase shift is introduced.

The initial state is the same saved physical state at `t=0.5` used by the short
comparisons. These are newly seeded matching initial-value problems; they are
not a restart of the tutorial's currently saved LES reference. The existing
reset/startup transient remains visible. Reaching a later time alone will not
be treated as proof of a statistically developed wake.

## Accepted FVM checkpoints and their qualification

The [checkpoint runner](cube_profile_checkpoint_trial_3d.py) saves canonical
full and hybrid FVM states at every profile time, including the initial and
final states. Each save must leave both solvers' fields, histories, panel
strengths and clocks bitwise unchanged. It records velocity, pressure, flux
histories and time-integration state, allowing independent wall-force and
field reconstruction. It does not claim a complete coupled restart at every
profile time; the VPM restart schedule is unchanged.

Before this longer run, the checkpoint writer was tested over three advancing
intervals against the existing original-coupling control. The
[observer verification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/profile-checkpoints-observer-verification.json)
finds bitwise equal comparison histories, all six saved comparison arrays,
17 FVM checkpoint entries, 11 boundary-history entries and 11 numeric VPM
datasets. Its profile figure is byte-identical to the previously inspected
figure. The [force verifier](verify_profile_checkpoints_3d.py) independently
recomputes both wall forces at each of four saved times: all
[eight force checks](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/profile-checkpoints-force-verification.json)
have zero difference from the observer's records.

## Running case and remaining evidence

The [run directory](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/long-wake-twenty-time-iterated)
contains live comparison histories, iteration residuals, profiles and accepted
FVM checkpoints. It uses the same frozen source workspace as the qualified
short cases. All 799 original source/input files were verified unchanged
before launch. The new observer and checkpoint scripts were added under new
filenames. The first twenty completed intervals reproduce the recorded short
comparison history exactly; this is a prefix check, not a completed long-run
qualification.

The [accepted 70-exchange prefix](long-wake-prefix-through-four-3d.md), through
physical time `4.0`, is now independently verified. All 70 intervals converge
and replay their first endpoint map bitwise. Thirty wall-force reconstructions
and 480 scalar checks pass. Relative drag-history RMS error is `0.7589%`.
At `t=4.0`, whole-small-domain and near-body velocity RMS errors are
`0.0102895 U∞` and `0.00817650 U∞`; centreline and off-axis near-wake line
errors reach `0.0284607 U∞` and `0.0244384 U∞`. Direct evaluation of the saved
particle/panel sources leaves the wake discrepancy essentially unchanged.
The instantaneous drag difference of `-0.000623526` is near a force-curve
crossing, not a measure of history agreement. Verified forces, profiles and
figures are linked in the prefix report. The parent process remains live;
later observations and completion still require qualification.

Completion requires checking terminal process success, every interval's
convergence and replay flags, archived sources, and the independently
recomputed full/hybrid forces and profiles. The complete force history and
individual profiles must remain available even if a late-time summary is also
reported. Matching mesh size or converged interface residuals will not be
substituted for measured agreement with the full FVM.

The requested near-roundoff force/profile agreement and a qualified developed
three-dimensional wake remain outstanding. This long run will supply evidence
about how the measured short-transient error evolves; it is not itself a
promise that the target will be met.
