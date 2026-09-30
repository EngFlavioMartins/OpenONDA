# Delta-wing runtime investigation, 29 September 2026

The late-run slowdown is strongly associated with competing workloads on the
laptop. There are also substantial implementation costs, including work that
is absent from the displayed step timer. The available evidence does not
indicate failed hardware. It does not establish an isolated-machine speedup.

## Historical evidence

The delta-wing case explicitly requests CPU execution, f32 FMM, SSPRK3, core
spreading, 576 VLM panels, and backups/flow integrals/three wake planes every
10 steps. Its wake population stabilizes around 104,000–106,000 particles.

The September 29 session resumes from step 2230. Its elapsed clock restarts:
`11:44:43.2` is the resumed session's duration, not the total cost of all 2781
accepted steps. The log ends with KeyboardInterrupt at accepted step 2781;
the latest scheduled native checkpoint is step 2780, t=6.95 s.

| Observation on September 29 (CEST) | Evidence |
| --- | --- |
| 15:12:30.9: delta step 2580 | 23.200 s evolution, 103,706 particles |
| 15:13:38: rotor process starts | PID 2466189; cwd `tutorials/vpm/06_rotor_flow` |
| 15:13:47: rotor log starts | Backend CUDA; confirmed as a current GPU compute process |
| 15:16:11.7: delta step 2581 | 61.814 s evolution, 103,715 particles |
| Steps 2582–2592 | Most steps 58–75 s, one 23.668 s; population remains almost unchanged |
| Steps 2700–2781 | Median evolution time 84.26 s |
| Steps 2770–2780 | 147.66 s/step end to end, including scheduled work |

Before the jump, steps 2500–2599 have a 21.14 s median (this range also
contains the first slow steps). The count, particle bounds, and core-radius
quantiles show no comparable jump. For example, maximum core radius changes
from 0.14817 m at step 2580 to 0.14800 m at 2590.

System activity records bracket the same transition: five-minute load rises
from 19.31 at 15:10 to 25.70 at 15:20 and 26.59 at 15:30 on a 20-logical-CPU
machine. CPU idle falls from 25.12% to 14.93% and 10.00%. CPU pressure rises;
I/O wait stays approximately 0.02–0.04%. This supports compute contention,
not a disk-write bottleneck, as the main explanation for the sudden jump.

At inspection, the machine was an i7-12700H laptop with roughly 15 GiB RAM,
about 9 GiB of swap occupied, six cylinder MPI workers, additional FVM work,
a CUDA rotor run, and a CUDA quadcopter qualification. The RTX 3060 Laptop
GPU was at 100% utilization. Swap activity is present historically, but
occupied swap alone is not evidence that delta was thrashing. Shared laptop
power/thermal limits may contribute; the measurements do not isolate that
contribution from CPU scheduling and memory contention.

## Costs identified in the implementation

- `EvolutionStepper.advance` measures the physical evolution block, then
  reports its time. `VPMSolver.advance` subsequently refreshes accepted-state
  health and dispatches output. Thus the displayed `Step wall time` is not
  the complete per-step wall time.
- Accepted-state health calls the stage RHS again, including the FMM velocity
  and gradient and VLM field. SSPRK3 already evaluates three physical stages.
  This fourth evaluation checks the accepted state; simply dropping it would
  remove the strain-increment safety check.
- A backup refreshes the stage field again and then calls
  `compute_vorticities_kernel`. That kernel has nested loops over every pair
  of particles, independent of the FMM setting. At 106,297 particles it visits
  11,299,052,209 ordered pairs per call.
- Bound-wing induction loops over all 576 panels for every particle and RK
  stage. FMM accelerates the particle self-induction; it does not accelerate
  this VLM-to-particle loop.
- Normal log text is small. Expensive field reconstruction and diagnostics,
  rather than printing the log, need to be separated from actual file I/O.

## Measurement method and limits

A frozen copy of `source/`, `openonda/`, and the tutorial setup was created
under `/tmp/openonda-delta-profile/frozen`. The step-2780 checkpoint was read
into a case rooted in `/tmp`; original solution and sample directories were
not modified. The numerical configuration is unchanged. Detailed timers and
Python profiling were enabled, with warm-up and steady advance separated.
Other users' simulations were not stopped or altered, so these timings
represent the contended machine and are not clean hardware benchmarks.
The snapshot includes commit `02c47b5e` (active-source scratch allocation,
22:19), which postdates the interrupted run. The profile therefore diagnoses
the current implementation with the saved physical state; it is not a
bit-for-bit reconstruction of the earlier process's executable code.

`source_hashes.json` records the frozen source and input hashes. The log
segment JSON files preserve parsed step/count/evolution/elapsed observations;
`host_*.txt` preserve historical sysstat output collected before this profile.
`profile_driver.py` is the exact local measurement driver; its paths refer to
the frozen `/tmp` workspace described above, not a production run launcher.

## Checkpoint profile

After a separate initialization/restart and a warm-up step, advancing from
2781 to 2782 took 137.46 s wall time and 688.41 s aggregate process CPU time.
Taichi had 20 worker threads configured. The latter CPU/wall ratio is about
five CPU equivalents; it is not evidence that all 20 CPUs were available to
the simulation. The first advance took 278.21 s and included JIT compilation;
it is excluded from the steady-state breakdown below.

| Work in the warmed advance | Seconds | Share |
| --- | ---: | ---: |
| FMM, four evaluations | 112.82 | 82.1% |
| Bound VLM field, four evaluations | 18.42 | 13.4% |
| All remaining work | 6.23 | 4.5% |

Within FMM, exact near-field work costs 57.18 s and multipole-to-local
translations cost 35.41 s across the four evaluations. The tree builds cost
0.91 s. Each evaluation performs about 0.665–0.670 billion exact near-field
particle interactions; the three-dimensional overlapping wake still has a
large near-field workload. The `direct_strength_rate_fallbacks` diagnostic is
zero. Thus FMM is active and saves interactions, but does not make this cloud
cheap on a contended CPU.

The physical-evolution call took 107.12 s. Accepted-state health then took
29.66 s, including its fourth FMM/VLM evaluation. The health figure overlaps
the FMM/VLM rows above; it must not be added to their sum. Output dispatch on
this non-output step took 0.00056 s. Most warmed time is inside native Taichi
kernels, not Python bookkeeping or console logging.

The first standalone backup took 82.24 s: 76.90 s to refresh derived fields,
including 39.49 s for all-pairs vorticity, and 5.20 s in the output writer.
This first backup includes approximately 12.4 s of kernel creation/compilation
across its operations, so it is not a warmed backup benchmark. It still
clearly separates the computational preparation from serialization and I/O.

The separate flow-integral/quality diagnostic took 181.28 s. Of this,
177.03 s was the Fourier integral path: its large-array quadratic reductions
cost 115.52 s, twelve FFTs cost 18.86 s, and four particle-to-grid scatters
cost 15.56 s. This was the first diagnostic on the restored state, with a
newly fitted diagnostic grid, rather than the historical run's persistent
grid. It measures a real expensive path but does not exactly reproduce the
old run's output interval.

During that calculation the process's resident high-water mark reached
4.47 GiB, versus about 0.75 GiB during the ordinary step. The sampled maximum
resident-plus-swapped footprint was 4.85 GiB. Across all concurrent jobs,
available RAM briefly fell to 0.32 GiB and swap was almost exhausted.
`memory-monitor.jsonl` records these observations. The calculation completed
successfully, and the temporary profiling process exited normally. This
shows why output-related allocation and memory traffic matter on this
machine, even though disk writes themselves are small.

The three wake-plane samplers were not separately benchmarked. The
historical end-to-end interval includes them; the individual measurements
above must not be presented as a complete replay of every scheduled output.

## Priorities

1. Give the heavy cases separate execution windows or explicit shared CPU/GPU
   budgets. The historical rotor-start correlation is strong; isolating
   shared power limits from scheduling would require a controlled quiet-run
   comparison. No other simulation was paused to manufacture such a result.
2. Report complete `advance` and scheduled-output wall times. Preserve the
   evolution breakdown but name it accurately.
3. Optimize the dominant FMM near-field and multipole-translation kernels.
   The current case explicitly selects CPU. A supported GPU FMM backend
   requires its own timing and numerical parity check on an available device;
   current FMM advertises Vulkan/Metal/CPU, not CUDA.
4. Reuse valid accepted-state derived fields for backups, with explicit state
   and provider invalidation, and replace the quadratic vorticity traversal
   with a validated accelerated implementation. Preserve health checks and
   the requested backup cadence.
5. Reduce Fourier-diagnostic peak storage and temporary-array traffic,
   especially in `quadratic_integrals`, while preserving its measurement
   definition. Simply making samples less frequent would change the requested
   output, not fix the implementation cost.

Only this investigation's evidence files were added. Production solvers,
tutorial configurations, existing solutions and other simulations were not
changed. The original delta checkpoint and log hashes were verified unchanged
after the bounded profile completed.
