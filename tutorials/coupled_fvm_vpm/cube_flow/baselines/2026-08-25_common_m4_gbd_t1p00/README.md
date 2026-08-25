# Cube-flow pre-change benchmark

This record freezes the short run generated on 2026-08-25 before the next
coupler implementation. It is the benchmark that the replacement must beat in
both physics and elapsed time.

The run was interrupted during coupling step 106. Steps 1--105 are coherent;
step 106 is excluded. The two complete backup boundaries are at `t=0.50` and
`t=1.00`. This is a short-run diagnostic benchmark, not evidence of long-time
stability.

## Headline targets

| Quantity | Baseline |
|---|---:|
| Warm backup interval, `t=0.50 -> 1.00` | **1276.500 s (21:16.500)** |
| Warm interval, summed timed stages | 1258.906 s (20:58.906) |
| Cold/ramp backup interval, `t=0.00 -> 0.50` | 861.191 s (14:21.191) |
| Mean drag error, `t=0.05 -> 0.50` | **5.494%** |
| RMS drag error, `t=0.05 -> 0.50` | 5.801% |
| Worst drag error | **7.717% at `t=0.30`** |
| Drag error at `t=0.50` | -4.319% |

The exact scheduled-output timestamp difference is the primary timing number.
The coupler log independently gives about 21:17 from the start of step 51 to
the start of step 101. Timed stages exclude the checkpoint save. The current
134.93 MiB atomic checkpoint completed in roughly one second at log resolution;
checkpoint I/O is not the dominant cost.

The 68.896 s step 100 is not a backup-I/O cost. Similar VPM diagnostic spikes
occur every 20 steps. Step 50 is also a checkpoint step and took a normal
17.967 s.

## Valid physics errors

The coherent drag and line-profile reference ends at `t=0.50`. No error after
that time is valid. At `t=0.50`, the streamwise-velocity errors below are
normalized by `U_inf`; the VPM rows include only its downstream authority,
`1.25 < x <= 5`.

| Region | Mean | RMS | p95 | Maximum |
|---|---:|---:|---:|---:|
| FVM centreline, solid excluded | 1.449% | 1.647% | 2.674% | 3.569% |
| FVM line at `y=0.75` | 0.664% | 0.732% | 1.091% | 1.349% |
| VPM downstream centreline | 0.276% | 0.426% | 0.909% | 1.409% |
| VPM downstream line at `y=0.75` | 0.263% | 0.440% | 1.061% | 1.480% |

The downstream velocity error is small, but that does not establish correct
vorticity transport.

## Wake-vorticity reference conflict

The reference directory contains two `t=0.50` wake fields from different
reference runs. They agree closely in velocity but disagree by a factor of
4.55 in integrated vorticity magnitude. Wake-vorticity error is therefore
diagnostic, not a certified acceptance number yet.

Using the older dedicated wake reference gives a velocity-vector RMS error of
0.588%, vorticity relative RMS error of 1.343, and an `|omega|` content ratio of
0.622 (a 37.8% deficit). Using the newer full-slice reference gives 0.456%,
4.006, and 2.831 (a 183.1% excess), respectively. Both comparisons show a
material vorticity-field disagreement, but they do not establish its direction.
A single same-configuration reference must be regenerated before this becomes
a pass/fail metric. These values use linear interpolation onto the dedicated
reference's 126 by 76 wake grid; the content ratio is a two-dimensional slice
proxy, not three-dimensional circulation.

## Numerical and scaling guardrails

- Particles after transfer: 216,432 at `t=0.50`; 305,347 at `t=1.00`.
- Particle growth over the warm interval: 88,915.
- At the `t=1.00` checkpoint, downstream `x > 1.25` contains 7,169 particles
  and `sum(|Gamma_p|) = 0.0500298 m^3/s`.
- Peak FVM Courant number: 1.07493.
- Peak continuity error: `1.1195e-7`.
- Non-finite values and unconverged FVM solves: zero.
- Maximum corrected boundary-flux residual: `6.75e-14 m^3/s`.
- Maximum post-correction blend cross-divergence: `2.78e-10`.

## Competition rule

Run the candidate with the same mesh, time steps, four FVM ranks, sampling and
backup schedule, hardware, and no competing workload.

1. Compare valid physics through `t=0.50`. Stop immediately if the candidate is
   already clearly worse.
2. Otherwise continue only through the second checkpoint at `t=1.00` and
   compare the warm `0.50 -> 1.00` interval.
3. "Faster" means less than 1276.500 s. Because this baseline has only one warm
   interval, require at least a 5% reduction (at most 1212.675 s) before calling
   the speedup robust rather than measurement noise.
4. "Better" means no regression in drag or authority-only velocity profiles,
   clean numerical guardrails, and improved wake transport once the coherent
   vorticity reference exists. Particle growth and topology churn must also be
   reported; a faster step obtained by deleting the wake is not acceptable.

## Provenance and limitations

The exact clean source commit cannot be recovered because run metadata did not
record Git state. Reflog and log-format evidence support commit
`b31cb0c738fd7ad1962fb8bfb0c839f8e415cc8f` plus the uncommitted change of
`FVM_BOX` from `[-3,3]^3` to `[-1.5,1.5]^3`, later captured in commit
`f7173ac11ac3ed53de4d044f28631e4f830e7136`. Treat that provenance as inferred,
not certified.

The run used the absolute common-M4' lattice blend, GBD diffusion, `h=0.03125`,
`dt_FVM=0.005`, `dt_VPM=0.010`, a three-spacing blend width, four partitioned
FVM ranks, 549,344 FVM cells, Python 3.11.15, Taichi 1.7.4, and Apple arm64.
Machine-readable values and evidence hashes are in `metrics.json`.
