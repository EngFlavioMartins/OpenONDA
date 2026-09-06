# ADR: laptop-sized cylinder convergence experiment

Status: provisional sizing implemented; production admission blocked by LSQ quality.
Date: 2026-09-06. Authorization: user explicitly requested a physically meaningful
change of dimensions and mesh family to fit this machine.

## Constraints

Measured using psutil: macOS arm64, 10 logical CPUs, 16 GiB physical RAM,
4.13 GiB initially available, 20.72 GiB free disk. Availability changes as other
applications run. The old D/160 family was estimated at 33.5 million cells,
incompatible with this laptop. Raising a budget was not a solution.

## Options and decision

- Keep the old 3D family: best continuity with earlier inputs, but infeasible.
- Shrink the upstream/lateral distances drastically: cheaper but changes blockage
  and outlet interaction in poorly controlled ways; rejected.
- Use a lattice-aligned 1D span and affordable dyadic three-grid family: chosen. This preserves a
  meaningful laminar shedding problem and distant in-plane boundaries, at the
  cost of restricting the physical claim to quasi-two-dimensional flow.

The base domain is [-8,24] x [-10,10] x [-0.5,0.5] in D units. Cylinder D=1,
U=1, Re=150, nu=1/150. The original STL is clipped at span planes without caps.
No-slip cylinder, uniform inlet, pressure outlet and slip lateral/span boundaries
define the problem. Lateral extent is 20D, so finite-boundary influence must be
measured rather than assumed zero. The enlarged control uses [-12,28] x [-12,12].
A second control reduces the span by 25% to 0.75D. Both use the fine wall spacing.
This tests sensitivity within a span-limited model, not unrestricted 3D physics.

The classical infinite-cylinder wake first becomes linearly unstable to 3D
perturbations near Re=188.5. This supports considering a quasi-2D Re=150 model,
but does not establish exact two-dimensionality of our 3D Cartesian discretization.
[Barkley and Henderson, JFM 322](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/threedimensional-floquet-stability-analysis-of-the-wake-of-a-circular-cylinder/61575FBF0BC45054592D46382DEF30BB).

D/8, D/16, D/32 use r=2 between grids, while keeping native-like dyadic
transitions within each mesh. The scheme, STL, box extents, and patch refinement
policy are identical across the spatial triplet. This is a new experiment;
earlier meshes and force histories cannot be relabeled and reused.

## Resource and numerical acceptance

The working budget is 3 GiB and 300,000 cells; each launch also checks current
available RAM. Disk stays above an 8 GiB reserve. Only two published checkpoints
are retained per run; old user outputs are not removed. Fields are output every
50 units rather than every 25. Base timestep is 0.001 with separate half/quarter
timestep comparisons; CFL and all other numerical quality gates remain active.

There are eleven runs: three spatial baselines, four timestep refinements,
two tighter-iteration controls, one enlarged-domain and one reduced-span control.
Each still needs at least 20 settled shedding cycles. Domain/span differences
must fit within 20% of each metric's error tolerance and enter the combined
budget. A fine mesh that fits RAM but fails the accuracy budget does not pass.

## Capacity evidence and next steps

An initial 0.2D-span/D36 trial was stopped in volume untangling after exceeding
ten CPU minutes. Its observed RSS was only 0.514 GiB at one sample; memory was
not the limiting issue. It is not accepted or retained as the default. The
subsequent D/27, 1.8D-span trial failed wrapper construction with a non-manifold
edge after 261 seconds and 1.79 GiB peak RSS. These are rejected trials, not defaults.

The final lattice-aligned `fine_domain` geometry completed in 470 seconds:

| Measurement | Result |
|---|---:|
| Cells | 266,429 |
| Peak process RSS, construction and checking | 1.74 GiB |
| Full probe elapsed time | 489 seconds |
| Independent OpenFOAM all-topology/all-geometry check | Mesh OK |
| Minimum volume | 2.26e-7 |
| Maximum non-orthogonality | 41.41 degrees |
| Maximum skewness | 0.9195 |
| Inverted face pyramids / rank-deficient LSQ cells | 0 / 0 |
| Maximum LSQ condition | 11.6711; production limit 9, rejected |

Raw evidence: `artifacts/reference-flow-laptop-20260906/binary_domain/capacity.json`,
`mesh.npz`, and `checkMesh.log`. The artifact name records the exploratory run;
its domain and spacing exactly match the final `fine_domain` configuration.
No diagnostic mesh was installed in the production campaign.

Dry-run estimates (not measurements) are 4,392 / 33,473 / 261,151 cells for the
spatial triplet, 294,329 for the enlarged domain and 197,522 for the reduced span.
The largest estimate is 2.74 GiB using 10,000 bytes/cell. Actual FVM memory and
step cost still require measurement after production mesh admission. The other
four final-profile meshes have not been built. D/8 is intentionally a coarse
convergence anchor; a failed observed-order/error test must remain inconclusive.

Guarded probe outputs are in `artifacts/reference-flow-laptop-20260906/` at the
repository root. They are diagnostic meshes, not accepted production artifacts.
The probes cap process RSS at 3 GiB and preserve an 8 GiB disk reserve.
Measured counts, peak memory, native quality checks, and any solver startup
results must be inspected before claiming the heaviest case is qualified.

Mesh quality issues are not waived by this physical resizing. The immediate
next gate is LSQ operator qualification or local geometry improvement, as defined
in `REFERENCE_FLOW_QUALIFICATION_PLAN.md`; do not simply increase limit 9.
Then build/check all five meshes and benchmark production FVM initialization,
matrix/preconditioner memory, timesteps, and checkpoint I/O on the largest one.
Only after those gates pass should the full statistical campaign be started.
The full campaign has not been executed; this turn establishes a memory-sized
candidate, not a guarantee that the heaviest mesh runs the production solver.
