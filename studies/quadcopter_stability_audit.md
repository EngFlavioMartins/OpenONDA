# Quadcopter strain-stop audit

The saved unrefined quadcopter wake contains a strongly stretched, concentrated
filament. Its particle-induced strain already exceeds the existing integration
limit. Changing the limit or merely suppressing the health check is unsupported.
The default case now enables the existing physical filament-refinement algorithm
at every accepted step, using its unchanged factor-of-two strength criterion.
This is an implementation change requiring a clean numerical qualification;
it is not evidence that the full refined case has completed.

## Reproduce the bounded diagnosis

```bash
python studies/audit_quadcopter_stability.py tutorials/vpm/07_quadcopter
```

The argument is the case directory containing the original stopped solution.
The script reads native HDF5 arrays and the recorded failure target, evaluates
one direct Winckelmans particle-strain sum, and writes
`studies/quadcopter_stability_audit.json`. It does not construct a solver,
initialize a compute device, advance a timestep, or mutate the case. Numeric
work arrays are well below 100 MB. The quantitative evidence is retained in
[the JSON](quadcopter_stability_audit.json).

## Saved state and concentration

The latest local checkpoint is `vpm_000296.h5`: step 296, time 0.04625 s,
59,200 particles. The log records target particle 48974 at
(0.155600876, 0.159615174, 0.017785233) m. Its strain norm is
7,309.406 s⁻¹; the configured timestep 0.00015625 s produces increment
1.142095 against the unchanged maximum of 1. The frozen-state timestep ceiling
is approximately 0.000136810 s.

A direct sum of the saved particle-only Winckelmans strain gives
6,910.779 s⁻¹ at that target, or increment 1.079809. This independent sum excludes
VLM/bound-vortex contributions and bypasses tree approximation. It does not
reproduce the complete combined gradient, but establishes that excessive strain
is already present in the particle field.

The dominant inducing source is storage index 155, group 1. Its strength is
0.11035678, core radius 0.01107128 m, and distance from the target 0.00868168 m.
Its individual strain norm is 5,910.685 s⁻¹. Recorded history at the same storage
index is:

| Step | Strength | Core radius [m] |
| ---: | ---: | ---: |
| 3 | 0.000660848 | 0.00647554 |
| 30 | 0.000759957 | 0.00670228 |
| 90 | 0.001250331 | 0.00758814 |
| 180 | 0.002855468 | 0.00894393 |
| 270 | 0.018439714 | 0.01005663 |
| 295 | 0.096626982 | 0.01100656 |
| 296 | 0.110356780 | 0.01107128 |

From step 3 to 296, strength increases about 167 times while radius increases
about 1.71 times; strength/core³ therefore increases about 33 times. This
storage-index trace is supporting evidence, not a reconstructed birth identifier.
The original configuration has no topology-changing filament refinement.

All loaded fields are finite. Core-radius minimum/median are approximately
0.003340/0.012129 m. The target core is 0.010825 m and its nearest nonself
neighbor distance is 0.003557 m. Thus neither zero cores nor absent local overlap
explains this stop. Finite overlap alone does not certify that the stretched
filament remains spatially resolved.

## Lineage and compatibility

Neither `filament_reference_vortex_strength` nor `filament_reference_length`
exists in this original checkpoint. `StabilizationManager.capture_reference_state`
only captures these arrays when filament refinement or divergence relaxation is
enabled; backup writing saves them only when initialized and matching the cloud.
Disabled refinement therefore did not retain its birth/reference state.

Enabling refinement after restoration cannot reconstruct the historical strength
ratio from this backup. New reference magnitudes at the current state would miss
that accumulated stretching. No restart compatibility checks were weakened.
The new refined setup and the archived unrefined stop are different numerical
configurations; the old data remain untouched and a clean isolated run is required.
The new policy checks each step because the strongest source increases roughly
14% in the final recorded interval. It uses the existing factor 2, not a new
absolute-strength threshold, artificial soft limit, or memory budget.

## Why automatic RK subdivision is a separate change

Temporal strain-limited RK subdivision is mathematically reasonable as a temporal
resolution mechanism, using the existing strain limit. It cannot independently
establish spatial resolution or prevent continuing filament concentration.
Naively looping `integrator.advance()` with smaller intervals is unsafe here:
`VLMSolver.particle_transport_step` initializes the transported-bound ledger for
each RK context. Multiple independent contexts before one macro wake emission
would overwrite earlier subinterval exchanges.

Any future implementation requires a macro-scoped cumulative transactional VLM
ledger, stagewise strain checks, full rejected-trial rollback, and consistent
macro clocks, wake emission, split diffusion, sampling and restart state. No
subcycling was implemented during this audit.

Qualification must retain the strain limit and hard particle capacity, validate
lineage across wake births/splits/restarts, verify moment/strength conservation,
and run the refined case through the original 0.04625 s stop and onward. A short
startup pilot alone does not complete that qualification or the 24-revolution run.
