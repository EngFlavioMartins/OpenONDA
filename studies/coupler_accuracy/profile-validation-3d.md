# Matched profiles of the fully 3D cube solution

The [profile observer](cube_interface_profile_observer_3d.py) now records the
tutorial's centreline and `y/D=0.75` off-axis line, including points outside
the small FVM domain. The points span `-3 <= x/D <= 10` at spacing `0.0625`,
with `z=0`; points inside or on the cube are excluded. These lines observe the
fully three-dimensional solvers. They introduce no two-dimensional flow or
spanwise approximation.

The current tutorial inputs use different FVM sampling operators: the hybrid
`LineSampler` defaults to five-neighbour IDW, while the reference explicitly
uses 12-neighbour affine interpolation. The study therefore explicitly uses
the native 12-neighbour affine sampler for both FVMs. Near the artificial
boundary, truncating the available cells can still change the interpolation
stencil even when all shared cell velocities are identical. Each frame stores
both the full-reference line and a reference line evaluated with exactly the
hybrid's small-FVM stencil. This keeps sampling error distinct from solver
field error.

## Observation neutrality

The observer runs after the existing accepted-state comparison. Before and
after each new line query, it requires bitwise equality of FVM primary fields
and histories, VPM particle fields, body-panel strengths and solver clocks.
No reference sample enters the hybrid equations or particle renewal.

The first advancing qualification compares the new observer with the existing
promoted-query original-coupling control through three intervals, from physical
time `0.5` to `0.65`. Profiles are recorded at the initial state and each accepted
endpoint. The [verification](verify_profile_observer_3d.py) establishes:

- Bitwise identical complete comparison histories and all six saved comparison arrays.
- Bitwise identical 17 FVM checkpoint entries, 11 boundary-history entries and
  11 numeric VPM datasets.
- Independent reconstruction of each FVM profile from saved cell velocities
  and interpolation weights, allowing only the arithmetic roundoff bound for
  summing the 12 weighted terms in a different order.
- Exact initial equality of the hybrid and reference profiles on the shared
  small-FVM stencil.

The inherited cell centres differ by at most `8.88e-16` due to summation order
when cropping the native mesh. The observer retains the original trial's
`1e-13` geometric matching gate. Both sampled fields then use the same actual
small-mesh coordinates, indices and weights.

## Qualified short transient

At physical time `0.65`, the following are vector line RMS errors divided by
the unit freestream speed. Exterior means the sampled points with `|x|>1.5`,
not a volume-weighted exterior norm.

| Measurement | Centreline | Off-axis y/D=0.75 |
| --- | ---: | ---: |
| VPM, whole fluid line | 0.00803256 | 0.01682888 |
| VPM, exterior line points | 0.00154280 | 0.00091352 |
| Hybrid FVM versus reference on the same small stencil | 0.00592443 | 0.00264295 |
| Reference full-stencil versus small-stencil difference | 0.00431600 | 0.00126095 |

The reference-stencil difference is measurable even though it uses the same
reference velocity field. It is not a coupler error. Conversely, matching a
sampling operator does not remove the measured hybrid field error. These
are original-coupling observations from an early transient; they neither
qualify the iterated solution's developed wake nor establish long-time accuracy.

![Qualified centreline, off-axis and force observations](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/profile-observer-verification-3d.png)

The figure was visually inspected. The
[verification record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/profile-observer-verification-3d.json)
contains the metrics, source hashes and advancing equality checks. The
[observation record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/profile-observer-three-step-control/profile-observation-3d.json)
links each frame's cell velocities, query velocities, particles, panel field
and common geometry. The observation is now available for the longer coupled
comparison; neutrality has so far been qualified over three intervals.

The [longer wake comparison](long-wake-comparison-3d.md) additionally saves
canonical checkpoints of both FVMs at every profile time. Its checkpoint
writer preserves the same short control bitwise, and all eight full/hybrid
forces reconstructed from its four qualification snapshots match exactly.
The [first 70 exchanges of the longer comparison](long-wake-prefix-through-four-3d.md)
now have independently verified forces and profiles through `t=4.0`, including
direct particle/panel evaluation at four times. The centreline and off-axis
near-wake vector RMS errors reach `2.846% U∞` and `2.444% U∞`. The parent run
remains in progress; these qualified observations do not demonstrate the
requested wake agreement.

## Reference files currently present

A fresh [inventory](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-reference-inventory-20260913/reference-inventory.json)
finds the cube's reference samples and native meshes present for `very_coarse`,
`coarse` and `medium`. At the captured inspection, each latest force epoch and
metadata reaches time `20`. The coarse history contains a reset; only its
latest epoch belongs to the latest run. The inventory archives the inspected
metadata, force tables and meshes. It does not establish who recovered or
regenerated these files, or qualify their restart or physical accuracy.

Their global cell counts are 5,216, 6,404 and 11,804, and their minimum stored
nominal cell sizes are approximately `0.15`, `0.1275` and `0.105`. These are
nominal mesh levels, not cell-volume lengths or exact projected surface-cell
dimensions. They differ from the matched `h=0.0625` study and cannot be
substituted for its independent reference merely because a directory is
named `medium`. The user's equal-resolution constraint remains in force.
