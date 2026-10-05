The cylinder tutorial passed an early-time CUDA execution and restart check on
30 September 2026. It reached **t = 0.36 s**, with **8,753 particles**, after six
profiled intervals and three intervals resumed from the native checkpoint.
This does not establish developed vortex shedding, long-time stability, or
mesh/time-step convergence; the configured case ends at 100 s.

The tutorial's numerical settings were retained: the generated 143,600-cell
mesh, four FVM ranks, h = 0.04 m, particle/GBD spacing 0.048 m, FVM dt = 0.004 s,
coupling dt = 0.04 s, and up to three ordinary interface sweeps. CUDA was
requested explicitly and checked inside the process using both the solver's
backend and Taichi's `Arch.cuda`. Particle fields and device kernels use CUDA;
FVM, wall geometry, renewal algebra, and host/device staging still involve CPU
work. The machine has an RTX 3060 Laptop GPU with 6 GiB memory. Another
simulation was active on the GPU, so these are measurements on a shared machine.

The t = 0.24 s and t = 0.36 s end-state checks found finite FVM velocity and
pressure, finite particle positions and strengths, positive finite particle
cores and volumes, and zero particles in strict solid interiors. Scheduled
force/probe sampling at t = 0.20 s and field/slice/checkpoint output at
t = 0.24 s succeeded. The native restart passed configuration and checkpoint-file
validation, continued the journal without duplicate steps, and wrote a final
checkpoint with coupling/VPM step 9, FVM step 90, and time 0.36 s. The final
checkpoint-file hashes and stored particle arrays also passed verification. Renewal
strength and linear-impulse residuals remained within their recorded tolerances
at all nine accepted intervals.

**Interface convergence needs qualification.** The first two intervals hit the
three-sweep cap without reaching the 1e-5 tolerance. Their final normal/gradient
residuals were 4.11e-4/4.73e-4 and 1.09e-5/1.25e-5, respectively. Intervals 3–9
reached tolerance. The final residuals were 4.92e-6 and 5.68e-6. No tolerance,
iteration cap, geometry correction, or physical boundary model was changed.

The baseline used cProfile on every MPI rank and synchronized VPM section
timers. Its six intervals took 685.07 s; including initialization from the
saved mesh, the run took 919.35 s. Creating and validating the mesh beforehand
took 652.16 s under cProf

ile. Loading that exact native mesh took 1.18 s.
These startup costs are separate from advancement.

| Interval | Physical time (s) | Particles after renewal | Profiled wall time (s) | Interface converged |
| --- | ---: | ---: | ---: | --- |
| 1

| 0.04 | 5,841 | 173.04 | No |
| 2 | 0.08 | 6,462 | 91.66 | No |
| 3 | 0.12 | 6,795 | 92.71 | Yes |
| 4 | 0.16 | 7,254 | 131.45 | Yes |
| 5 | 0.20 | 7,552 | 100.86 | Yes |
| 6 | 0.24 | 7,959 | 95.34 | Yes |

For a representative warmed interval, step 2 divides approximately as follows.
Boundary refresh is separated from the transfer total because it evaluates
particle-induced fields. Particle state checks also evaluates fields after renewal.

| Sequential phase | Seconds |
| --- | ---: |
| FVM advancement, including its communication | 71.306 |
| Particle advancement, including wall/GBD handling | 12.610 |
| Boundary induction and post-renewal refresh | 5.545 |
| Transfer calculations and gathers | 1.569 |
| Particle state checks/field refresh | 0.489 |
| Interface state capture and restore | 0.132 |
| Accepted FVM output | 0.003 |

The largest supporting costs are:

- **MPI communication and waiting:** step 2 spent 7.75 s in halo exchange and
  6.58 s in global sum/max reductions on rank zero. These are nested within FVM
  advancement and include load imbalance. Rank 3 spent less time in these calls
  and more time computing gradients; the measurements are not pure network time.
- **CPU wall queries:** signed distances took 5.08 s in step 2 and 5.44–7.50 s
  in steps 3–6. These are nested within wall motion, mask preparation, and
  remeshing. In step 2, only 1.80 s belongs to the repeated-start intersection
  prefilter targeted by the optimization below.
- **GPU kernel preparation:** compilation is prominent on first use and can
  recur when scratch storage grows. At step 4, 8,318 post-GBD particles exceeded
  the FMM workspace's 8,272-source capacity. Replacing it with capacity 16,544
  recreated fields and triggered kernel materialization. Boundary evaluation
  rose from 0.67 to 25.22 s, and accepted-state field refresh from 0.78 to
  15.50 s. These are overlapping phase/compilation measurements, not additive
  overhead estimates. A Taichi compiler-entry call count is not a count of
  unique compilations.
- **Checkpoint field preparation:** the step-6 checkpoint took 8.64 s, including
  7.93 s preparing particle fields and 0.23 s serializing the VPM checkpoint.
  File serialization and particle state copying were not dominant here.

Startup additionally spent about 31.30 s constructing global VTK connectivity,
18.87 s validating topology, 14.59 s hashing the mesh, and 21.00 s preparing
fixed donor stencils. These did not recur in the measured advancement steps.
The new connected-component bookkeeping took about 0.015 s in step 1.

One small shared-code optimization was applied in
`source/coupler/geometry.py::TriangulatedWall.first_intersections`: consecutive
segments with exactly identical starts reuse that start's signed distance.
Distinct starts retain the existing path; query order, triangle intersection
tests, cache bounds, and numerical tolerances are preserved. Temporary storage
is linear in the query batch. There is no geometry recognition or case-specific
branch. The FMM memory allocation settings was retained because changing its memory/JIT
tradeoff needs broader GPU qualification.

A paired host benchmark compared frozen, hash-verified before/after source
snapshots. Each fixture used 512 sources with 64 target segments each. Five
alternating timing pairs followed warmup, with the distance cache cleared
before each invocation. Every measured intersection fraction and normal was
bitwise identical.

| Wall fixture | Distance-query rows before → after | Median whole-query time before → after (s) |
| --- | ---: | ---: |
| Curved | 16,384 → 512 | 0.277 → 0.223 |
| Rotated | 32,768 → 512 | 0.397 → 0.339 |
| Concave | 32,768 → 512 | 0.491 → 0.395 |
| Thin | 8,192 → 512 | 0.289 → 0.185 |

The deterministic reduction in signed-distance query rows is 16–64 times; timing
samples vary substantially under shared load. This is a wall-query
microbenchmark, not an end-to-end solver speedup claim. The targeted slice was
only about 2% of baseline step 2.

The optimized code then resumed the CUDA checkpoint with cProfile and detailed
timers disabled. Interval 7 took 206.41 s with fresh-process kernel preparation;
warm intervals 8 and 9 took 92.02 and 82.82 s. Interval 9's final forced stop
checkpoint lies outside its interval timer. These different states and timing
modes cannot establish a before/after solver speedup. They verify continuation
with the optimization and provide ordinary-run timing observations.

All 60 focused tests passed across native solid geometry, visible interpolation,
wall retries, arbitrary-wall GBD, and wall-aware renewal. Coverage includes
rotated, concave, curved, thin and multiple bodies, exact repeated-start result
parity, one-ULP-distinct starts, and outside-domain queries. Ruff and
`git diff --check` passed. No full second tutorial GPU run was performed.

Machine-readable results are in
[cylinder_cuda_profile_2026-09-30.json](cylinder_cuda_profile_2026-09-30.json).
The checkpoint, profile, log and source-snapshot paths in the JSON record
identify the original measurement environment. The one-off profiling drivers
have since been retired; these observations remain historical numerical
evidence. Initial instrumentation errors were corrected before the measured
advancement run, and the separate cold-start log records the aborted wrapper.
