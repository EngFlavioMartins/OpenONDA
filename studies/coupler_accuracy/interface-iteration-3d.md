# Fixed-predictor interface iteration in the fully 3D cube

The latest 20-interval matched-medium comparison, with
[promoted auxiliary panel queries](panel-derivative-precision-3d.md), converges
all 20 intervals in three sweeps each. RMS relative drag error falls from
2.082% to 0.555%, and final drag error changes from −2.094% to +0.756%.
Final near-body FVM velocity error falls 2.50%; the whole-domain error falls
only 0.29%. This is a verified short transient, with remaining reference error
despite interface convergence. The requested developed-wake force/profile
match is still unachieved.

The original native-query comparison documented below converged twelve
intervals, while eight reached the fixed 12-sweep cap. Its force improvement
motivated isolating the numerical derivative floor and repeating the complete
comparison with corrected query precision.

The first interval alone gave the opposite force conclusion: nine sweeps
reduced both boundary residuals below the original `1e-6` thresholds and
improved velocity, but worsened drag error. The complete short trajectory below
therefore matters when judging this coupling change.

The [preceding endpoint audit](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/velocity-projection-and-mass-flux-3d.md)
identified different FVM and VPM boundary traces at the same accepted time.
The normal driver advances FVM using a VPM prediction, replaces particles
from the FVM endpoint, and then recomputes VPM boundary history. This experiment
isolates that change by solving a fixed-point problem within each interval.

## What is iterated

The [scoped implementation](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/experimental_interface_iteration.py)
wraps the existing solver in the experiment's Python process. Production
defaults are unchanged. The accepted FVM start, its time histories, the
advected VPM predictor and the start-time boundary history remain fixed.

Each sweep restores those states, advances the same five FVM substeps using
the candidate endpoint normal velocity and tangential derivative, then renews
particles from that FVM endpoint. Renewal always starts from the same advected
predictor. The newly induced boundary trace becomes the next candidate.
VPM is not advanced again. Transfer and FVM counters must describe exactly one
accepted coupling interval, regardless of the number of sweeps.

The iteration uses full relaxation and area-weighted RMS residuals for normal
velocity and tangential derivative. It stops only when both residuals meet
their fixed `1e-6` thresholds, or when the prescribed maximum is reached.
Every interval records whether it actually converged. Reference fields and
reference forces are observations after acceptance; they do not drive a sweep.

The experiment retains the unit cube, small approximately `[-1.5, 1.5]^3`
domain, 16,936 small-FVM cells, 53,752 full-FVM cells, matching `0.0625`
near-body spacing, `ν=0.001`, laminar equations and the existing buffered-M4
renewal. It uses three-dimensional induction, stretching and body response.
The FVM step is `0.01`, the coupling interval is `0.05`, and the initial
physical state is the reference at `t=0.5`.

## Exact control and rollback qualification

Before any additional sweep, the first endpoint map in every interval is
repeated from the saved states. All 24 numerical-array and clock fingerprints
must match bitwise. These cover FVM velocity, pressure and flux histories,
particle fields, the post-renewal boundary trace, acceptance counters and
viscosity. A failed replay terminates the experiment.

Independent processes use `TI_CPU_MAX_NUM_THREADS=1`, in addition to the
OpenMP and BLAS limits. The callback verifies the actual Taichi thread count.
An initial comparison with the default Taichi thread pool did not reproduce
all fields bitwise across processes, so it is not used as a qualified pair.
No equality threshold was relaxed to admit it.

With the single-thread setting, zero sweeps (the original algorithm) and one
sweep (the wrapped algorithm with its replay check) give identical comparison
histories and complete numeric checkpoints: 17 decoded FVM entries, 11 decoded
boundary-history entries and 11 numeric VPM datasets. This establishes that
the wrapper alone does not change this tested interval.

The [independent verifier](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/verify_interface_iteration_3d.py)
recomputes boundary residuals from saved traces and native face areas. It
reconstructs the final wall force from pressure and viscous stress independently
of the force sampler, and recomputes all three final velocity-error metrics.
The [one-interval verification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-iteration-one-step-verification-3d.json)
covers 170 source/artifact records and 48 independent metrics, with zero
reported metric difference. Source archives preserve the verifier used for
that result.

## First interval: physical time 0.50 to 0.55

The full-FVM drag coefficient at the endpoint is `1.5090799958339747`.
Velocity errors below are volume-weighted RMS divided by `U∞`. The near-body
region uses `max(abs(cell centre)) < 1`, as in the advancing comparison; it is
different from the near-body sample set in frozen reconstruction studies.

| Case | Drag coefficient | Relative drag difference | Whole small-FVM velocity error | Near-body FVM error | Sampled VPM error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original / one-sweep replay | 1.4945063781 | −0.965729% | 0.00265429854 | 0.00081795809 | 0.03677996824 |
| Six sweeps, not converged | 1.4867253137 | −1.481345% | 0.00241392487 | 0.00072569576 | 0.03677604400 |
| Nine sweeps, converged | 1.4867253216 | −1.481345% | 0.00241392491 | 0.00072569579 | 0.03677604374 |

Normal residual falls from `1.15637e-3 U∞` to `1.46677e-10 U∞`.
The derivative residual falls from `2.77650e-3 U∞/D` to
`6.12196e-10 U∞/D`. Six sweeps were insufficient because the derivative
residual remained `2.47405e-6`; a separate run with a maximum of 12 converged
at sweep nine without changing either threshold. The six shared sweeps replay
identically between those two runs.

![Convergence and force response within one physical interval](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-iteration-one-step-3d.png)

The first converged interface has a more accurate small-FVM velocity field but
a less accurate drag coefficient. Convergence of the coupled endpoint map is
therefore separate from convergence to the full-FVM solution. This single
interval does not predict the force response over the complete short run.

## Twenty intervals: physical time 0.50 to 1.50

Both runs use one Taichi CPU thread and the same source/configuration apart
from the fixed-predictor iteration. Their full-reference drag/time histories
and final shared-cell reference velocities are bitwise identical. The reference
is used only by the observer. The reference endpoint is
`Cd=1.0803953399803075`.

| Final measurement | Original coupling | Maximum 12 sweeps |
| --- | ---: | ---: |
| Drag coefficient | 1.0577548037 | 1.0885606756 |
| Relative drag difference | −2.095579% | +0.755773% |
| Whole small-FVM velocity RMS / U∞ | 0.00768620894 | 0.00766336859 |
| Near-body FVM velocity RMS / U∞ | 0.00200495464 | 0.00195480212 |
| Sampled VPM velocity RMS / U∞ | 0.03124526768 | 0.03124271135 |

Across the 20 equally weighted coupling endpoints, excluding the shared initial
state, RMS relative drag error decreases by 73.35% (`2.08246% → 0.554974%`).
The largest absolute relative drag error decreases from 2.49154% to 1.48134%.
The corresponding RMS-over-time FVM errors decrease by 0.768% for the whole
small domain and 2.191% near the body. The sampled VPM error is essentially
unchanged and rises by 0.0013% under this history metric. Velocity accuracy
does not improve uniformly at every time; near-body error is worse in several
early intervals.

![Matched 20-interval force, velocity and convergence histories](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-iteration-trajectory-3d.png)

The run accepts 186 logical sweeps plus 20 first-map replay evaluations.
Intervals 3, 4, 5, 7, 11, 17, 18 and 19 reach the 12-sweep cap without satisfying
the derivative threshold. Their largest final derivative residual is
`3.10822e-6 U∞/D`. All convergence thresholds are unchanged. The final interval
does converge, at sweep seven. The run took 2,257 seconds including its
independent reference advancement and replay checks; this is not a production
performance measurement.

The [trajectory verifier](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-iteration-trajectory-verification-3d.json)
checks 341 source/artifact records and 390 independent metrics with zero
reported difference. It rechecks the one-interval zero/one-sweep identity,
matches the longer control's shared prefix, verifies every interval's bitwise
map replay, and independently recomputes final forces and all final velocity
metrics. It does not claim a separate 20-interval zero/one-sweep trajectory
identity. The plot was visually inspected.

Independent force decomposition attributes almost all of the change in drag
to pressure. Pressure drag changes from `1.00466034` to `1.03537633`; viscous
drag changes from `0.05309446` to `0.05318435`. This describes the response to
interface iteration and does not establish a new pressure boundary policy.

The [saved-endpoint audit](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-interface-iteration-endpoint-audit-medium/coupled-boundary-stage-audit-3d.json)
recovers the FVM trace from its actual conservative flux and ghost velocities,
then compares it with post-replacement VPM history and the replayed reference:

| Endpoint measurement | Original coupling | Iterated interface |
| --- | ---: | ---: |
| Normal replacement jump / U∞ | 0.000714324 | 1.80396e-9 |
| Derivative replacement jump [U∞/D] | 0.00435081 | 8.17341e-7 |
| FVM normal trace error versus reference / U∞ | 0.00486556 | 0.00492641 |
| FVM derivative trace error versus reference [U∞/D] | 0.01323168 | 0.01217799 |

Both cut fluxes remain conservative to roundoff. Closing the interface removes
most of its replacement jump but leaves substantial reference boundary errors.
The [auxiliary panel precision experiment](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/panel-derivative-precision-3d.md)
now connects the small residual floor to single-precision query differences.
In a matched frozen-source test, promoted queries converge the first three
intervals in three sweeps each, while retaining the corrected drag to `6.28e-8`
at the third endpoint. The complete promoted-query comparison now converges
all 20 intervals in three sweeps each, with final drag `1.0885607282705918`
against reference `1.0803953399803075`. Its
[independent verification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/precision-twenty-step-verification.json)
checks 243 source/artifact records and 138 independently recomputed metrics,
with zero metric difference. Production defaults remain unchanged, and
developed-wake force/profile agreement remains unvalidated.

![Fully converged precision-corrected 3D interface trajectory](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/precision-twenty-step-trajectory.png)

## Reproduction

Use the OpenONDA Python environment from the repository root and new output
directories. The explicit Taichi limit is necessary for this reproducibility
check; `OMP_NUM_THREADS` alone does not set the Taichi pool size.

```sh
env PYTHONPATH=. TI_CPU_MAX_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python studies/coupler_accuracy/cube_interface_iteration_3d.py \
  --oracle studies/coupler_accuracy/results/cube-3d-medium-laminar-oracle \
  --steps 1 --iterations 0 --output /private/tmp/cube-interface-control
env PYTHONPATH=. TI_CPU_MAX_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python studies/coupler_accuracy/cube_interface_iteration_3d.py \
  --oracle studies/coupler_accuracy/results/cube-3d-medium-laminar-oracle \
  --steps 1 --iterations 1 --output /private/tmp/cube-interface-replay
env PYTHONPATH=. TI_CPU_MAX_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python studies/coupler_accuracy/cube_interface_iteration_3d.py \
  --oracle studies/coupler_accuracy/results/cube-3d-medium-laminar-oracle \
  --steps 1 --iterations 12 --output /private/tmp/cube-interface-twelve
```

The result records retain each sweep's boundary fields, source hashes, clock
checks, convergence flags and the child trial's saved solution and comparison.
