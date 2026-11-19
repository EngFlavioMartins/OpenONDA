# Part II run report

Status: in progress. This file is filled in as the runs finish; every number in it is copied from a named field of a delivered file.

## Acceptance tests of Task A

`tests/coupler/test_multi_domain_acceptance.py` (local `tests/` repository), each check followed by its planted fault. Runs: short coupled cylinder flows of 0.4 s (ten coupling intervals) with the particle solver on CUDA, from exports of the frozen commits named below.

Round-off, measured before any acceptance run: two identical one-domain case 01 runs on CUDA differ by 5.65e-7 of the largest force and 9.98e-8 of the largest strength, with equal positions and particle counts (939). The tolerances are ten times that floor, rounded up to a power of ten: force 1e-5, strength 1e-6, position 1e-6 particle spacings. Distant bodies: the force histories of each of two cylinders 200 D apart and of the single cylinder differ by at most 2e-3 of the largest single-cylinder force (stated on 18 Nov, before any run).

| Check | Measured difference | Outcome | Planted fault | Fault's difference | Fault caught |
| --- | --- | --- | --- | --- | --- |
| One domain through the new code path = single-domain coupler (commit 1cdea5cf) | force 5.576e-7, position 0, strength 9.977e-8 | passed | boundary data +1 % | force 9.513e-3, particle count differs | yes |
| Two domains in both orders | force 1.237e-6, position 0, strength 1.003e-7 | passed | transfers one after the other with boundary data refreshed in between | force 7.039e-2, particle count differs | yes |
| Two cylinders 200 D apart = single cylinder | pending | pending | cylinders 3 D apart | pending | pending |
| Continued from a two-domain backup = uninterrupted | force 1.998e-6, position 0, strength 1.244e-7 | passed | backup without the boundary-data correction of one domain | force 5.545e-4, strength 4.201e-6 | yes |

The first check ran from export 336f854f, the others from 1c627a9a (identical code; it adds the case 07 cylinder surface files). The first attempt of the two-domain checks failed before any flow because the case 07 surface was not in the repository; a later attempt of the distant-bodies checks failed because the one-cylinder run used the multi-domain output layout, a fault of the test driver.

## Acceptance tests of Task B

Pending (see the case 06 section).

## Runs

Pending.

## Cost estimates

| Run | Commit | First 2 % | Wall time of the 2 % | Extrapolated wall time | Particles at 2 % |
| --- | --- | --- | --- | --- | --- |
| S1 reference (case 05, FVM only) | 336f854f | 1 s of 50 s | 70 min meshing (69,085 cells), then 200 steps in 31.5 min (9.45 s per step) | 26 h for 10,000 steps | none |

## Deviations

- The particle solver runs on CUDA, not Vulkan: only CUDA has the fast slip-slab image queries (0.026 s per boundary evaluation against about 60 s on Vulkan or the CPU).
- Case 06: the author agreed to finer trailing-edge cells. Boxes around the trailing edge alone leave the octree surface non-manifold, so the whole blade is refined from 0.025 m to 0.0125 m, with 0.00625 m boxes along the flat base from 1.5 m radius.
- Output-only changes to cases 04, 05, 06 and the VLM rotor to record what the analysis needs (motion, wake lines with vorticity, particle invariants every interval); case 05 runs 50 s as this file asks.
- Machine sharing: another session's coupled wing run holds CPUs 0–3 and 12–15 and up to 7 GB of the GPU until about 22–23 Nov. Part II runs use the other 16 cores, one particle solver at a time, at most about 4 GB of GPU memory, and stop themselves when the card has less than 1,000 MiB free; a stopped run resumes from its last backup.
- The acceptance tests live in the local `tests/` repository, which has no remote.

## Stopped runs

None so far.

## Analysis checks

`part2_analysis/checks.py`, run before any analysis: all three passed (synthetic signal recovered within 0.17 %, a 2 % change reported at 17.7 combined standard errors; a run compared with itself gives zero differences; a force equal to the body velocity is 90.0° ahead of the displacement and does positive work). To be rerun at the frozen commit.
