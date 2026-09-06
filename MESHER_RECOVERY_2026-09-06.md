# Mesher recovery: build recovered, delivery not yet accepted

**Follow-up qualification update:** a subsequent isolated built-in-mesh test
completed 20 actual FVM steps with finite fields and converged linear solves,
using diagnostic LSQ limit 15 instead of production limit 9. This supersedes
the zero-step diagnostic below as the latest runtime evidence, but does not
establish force accuracy or statistical convergence. The user's acceptance of
the overall mesh appearance also changes the next-work priority: exact native
topology/cell-count parity is no longer a delivery gate. Follow
[REFERENCE_FLOW_QUALIFICATION_PLAN.md](REFERENCE_FLOW_QUALIFICATION_PLAN.md)
for the current localized mesh qualification and proper grid-study plan.

The interrupted worker's implementation now completes the full built-in D/40
cylinder pipeline. A 523,534-cell candidate, actual mesh-section images, native
references, and resumable checkpoints are saved. **The task is not complete:**
16 cells fail native concavity checking, the first template differs from the
native reference, and the configured solver conditioning limit rejects both
the Python candidate and real cfMesh meshes. No grid-independence result is claimed.

## Inspect the result

- [Built-in versus native mesh sections](/Users/flaviomartins/OpenONDA/artifacts/mesher-recovery-20260906/builtin_vs_native.png).
- [Diagnostic candidate VTU](/Users/flaviomartins/OpenONDA/artifacts/mesher-recovery-20260906/builtin-coarse/mesh.vtu) and [NPZ](/Users/flaviomartins/OpenONDA/artifacts/mesher-recovery-20260906/builtin-coarse/mesh.npz).
- [Candidate report](/Users/flaviomartins/OpenONDA/artifacts/mesher-recovery-20260906/builtin-coarse/mesh_report.json), [manifest](/Users/flaviomartins/OpenONDA/artifacts/mesher-recovery-20260906/builtin-coarse/mesh_manifest.json), and [explicit rejection notice](/Users/flaviomartins/OpenONDA/artifacts/mesher-recovery-20260906/builtin-coarse/NOT_ACCEPTED.md).
- [Independent check after export-order correction](/Users/flaviomartins/OpenONDA/artifacts/mesher-recovery-20260906/builtin-check-ordered/checkMesh.log).
- [Solver startup rejection](/Users/flaviomartins/OpenONDA/artifacts/mesher-recovery-20260906/flow-smoke/summary.json).

All existing reference-flow solutions were preserved. The candidate has **not**
been installed into `solution/coarse`. It was saved before the mesh-only entry
point gained its solver-conditioning gate and is retained as diagnostic evidence.

## What happened to the worker

The task **Mesher (Worker)** was idle after an interrupted turn, not continuously
making progress. Its last successful command stopped at wrapper generation;
there was no successful final D/40 volume-optimisation/export/solver qualification.
Its last uncommitted acceleration also failed an existing native surface test:
198 coordinate components differed, with a maximum discrepancy around 0.01015.

The separate scalar fast path used the unnormalized span for a normalized Newton
step limit and did not carry updated stabilization into the next gradient
iteration. It also changed branch-sensitive arithmetic. Replacing that duplicate
implementation with a compiled version of the reference calculation fixed the
regression. A floating exponent is important: integer-power lowering changed a
nearly singular triangle fan's Newton branch even with fastmath disabled.

Two additional workflow defects hid incomplete delivery:

1. Native `checkMesh` was invoked through a nonexistent user-bin executable;
   the runner did not require an explicit successful quality verdict. The case
   also lacked the fvSchemes/fvSolution dictionaries that checkMesh needs.
2. Mesh-report serialization failed on Path objects and multi-element arrays.
   Mesh-only validation omitted the solver's LSQ conditioning gate.

## Changes made in this recovery

- Compiled the existing surface optimizer and retained its native five-pass
  iteration counts, arithmetic ordering, stabilization, and Cramer-rule update.
- Compiled volume cell-centre accumulation and bad/low-quality face scans.
  All criteria remain active; no large-mesh skips or quality exclusions were added.
- Repaired native oracle invocation, required an explicit passing checkMesh
  verdict, and supplied the minimal mesh-check dictionaries.
- Repaired JSON serialization, atomic directory publication, and code-file
  provenance coverage. Mesh-only publication now computes the configured LSQ
  geometry and enforces the same geometric quality thresholds as solver startup.
- Corrected OpenFOAM export ordering, including owner/neighbour reversal with
  matching face reversal. Corrected import of cfMesh's full-length neighbour
  lists with trailing -1 boundary entries and cells appearing only as neighbours.
- Added regression tests for the compiled surface path, serialization,
  no-overwrite behavior, quality-gated publication, and OpenFOAM interchange.

No meshing sizes, refinement boxes, physical parameters, or acceptance limits
were loosened. No code was committed or pushed by this recovery. Unrelated VPM
worktree changes were left alone.

## Measured results

| Measurement | Built-in candidate | Native STL reference | Native double-precision experiment |
|---|---:|---:|---:|
| Template cells | 443,616 | 492,048 | 432,430 |
| Final cells | 523,534 | 567,378 | 511,044 |
| Maximum non-orthogonality | 42.274° | 62.740° | 38.390° |
| Native concavity failures | 16 | 0 | 0 |
| Maximum OpenONDA LSQ condition | 11.7791 | 13.3840 | 11.6135 |
| Cells with LSQ condition > 9 | 67 | 32 | 96 |
| Full native checkMesh verdict | Fail: concavity | Pass | Pass |

The Python candidate has one connected fluid region, zero unclosed cells,
positive cell volumes, zero nonpositive owner/neighbour pyramids, and passing
native face-tetrahedron checks. Wall vertices are within 4.81e-13 of the clipped
STL. The post-optimization wall constraint moved vertices by up to 1.436e-4.

The 16 concave cells are **already present before that final wall constraint**:
their largest face-plane cosine is 0.005990604, unchanged by projection. See
[cell IDs and before/after measurements](/Users/flaviomartins/OpenONDA/artifacts/mesher-recovery-20260906/concavity.json).
They are core cells near the cylinder/span-end region, not inverted volumes.
They must not be dismissed merely because the other geometry checks pass.

The original exported candidate additionally failed upper-triangular ordering
on 320,527 faces. After correcting the exporter, that check passes and only
the 16-cell concavity failure remains. Sorting did not move any vertices or
change geometric connectivity.

The reconstructed build used about 365 seconds through the saved pre-volume
checkpoint plus 297 seconds for resumed optimization/finalization before the
entry-point validation/export tail. This is not a clean cold/warm benchmark.
The resumed process recorded 3.48 GiB peak RSS. A full 1,379,481-face template
bad-face scan took 6.25 seconds after compilation; previously a 1% scan alone
took 1.87 seconds, excluding cell-centre construction. Full-volume traces end
with zero bad faces and zero faces failing cfMesh's default low-quality criteria.

## Two distinct remaining decisions/problems

### 1. Input and topology parity are not established

Native STL loading rounds the requested span coordinate 0.6 to
0.600000023841858. Python's declarative box keeps 0.6. This affects intersection
classification on a Cartesian lattice and invalidates a naive same-dictionary
parity claim.

The FMS experiment preserves double precision and welds clipping/annulus seams
at 12 decimal places (maximum coordinate adjustment 4.21e-13). It changes the
native result substantially and passes checkMesh, but is **not** a completed
proof of identical canonical inputs: it is a diagnostic experiment, not the
new production reference. The remaining template count difference must be
localized before attempting another final-coordinate parity comparison.

Also, `_verify_cfmesh_surface_topology` in the mesher is a rejection guard with
simple incidence/degeneracy checks. It is not a port of cfMesh's full iterative
topological repair routines. Its zero counters do not establish native repair
equivalence for arbitrary STLs. Do not present this implementation as a completed
arbitrary-geometry translation on the strength of cylinder screenshots.

### 2. The existing LSQ gate conflicts with the native target

The reference setup's limit of 9 was justified in its comment using an older
D/12 mesh with condition 8.402. It is not a demonstrated limit for this D/40
cfMesh case. Both real native meshes exceed it when evaluated using OpenONDA's
LSQ operator. The FVM smoke test therefore completed **zero** steps: startup
rejected the Python candidate at condition 11.7791.

[Conditioning evidence and worst-cell locations](/Users/flaviomartins/OpenONDA/artifacts/mesher-recovery-20260906/conditioning.json)
distinguish this acceptance-policy conflict from a claim that LSQ=11.78 proves
an unusable discretization. The limit was not increased here. A follow-up must
either justify/calibrate a limit against native meshes and solver convergence,
or make improving conditioning below 9 an explicit additional product requirement
that goes beyond matching native output. Do not quietly relax it to get green tests.

## Focused next work — do not restart from scratch

1. Establish one double-precision canonical input contract, including patch
   labeling, span seams, wall clipping, triangle coordinates and native decoded
   coordinates. Freeze both the input hashes and linked native-library hashes.
2. Compare octree leaf/classification records at the first divergent stage.
   Reconcile 443,616 versus the genuinely identical-input native template;
   do not tune final point smoothing to compensate for different topology.
3. Use the saved candidate to localize the 16 concave cells and the 67 LSQ>9
   cells. Preserve native operations/iteration counts while repairing the first
   divergence. The worst Python LSQ cell is 443832 near
   (-0.04207, 0.51731, -0.57621).
4. Resolve the LSQ acceptance policy explicitly, then require a passing
   independent checkMesh and 20 actual solver steps before promoting coarse.
5. Only then qualify D/80 and D/160 and run the statistical grid-independence
   study. Neither those mesh builds nor force-history convergence were completed.

## Reproduction and checkpoints

All paths below are under
`/Users/flaviomartins/OpenONDA/artifacts/mesher-recovery-20260906`.

- `run_builtin.py`: measured original build, with stage checkpoints.
- `resume_builtin.py`: trusted pre-volume checkpoint continuation through the
  same optimizer, wall constraint, finalizer, and publication routine.
- `build_cfmesh_template.pickle`, `add_cfmesh_wrapper_layer.pickle`,
  `before-volume.pickle`, `optimise_cfmesh_mesh.pickle`: saved stages.
- `build-events.jsonl`, `resume-events.jsonl`: timing traces. The stack logs'
  periodic “Timeout” labels are faulthandler sampling intervals, not failures.
- `native-entry-verified/repeat-1`: rerun of the repaired native command with
  a passing manifest and checkMesh log.
- `native-fms-check`: diagnostic double-precision native experiment.
- `builtin-check-ordered`: independently checked Python export after ordering fix.

The pickle files were generated locally by this recovery and preserve internal
addressing plus mapper callbacks. **Never load pickle files from an untrusted
source.** NPZ/VTU are the portable inspection formats. The resume script refuses
to overwrite its existing output; choose a fresh destination before rerunning.
With the newly added conditioning gate, this unchanged candidate is now expected
to be rejected before publication.

Fresh production entry points (use a new output directory):

```bash
cd /Users/flaviomartins/OpenONDA
/opt/anaconda3/envs/OpenONDA/bin/python tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/mesh.py --case coarse --output-dir /private/tmp/openonda-coarse-new-attempt
/opt/anaconda3/envs/OpenONDA/bin/python tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/native.py --output /private/tmp/openonda-native-new-attempt --repeats 1
```

## Verification status

- 38 focused tests passed: native optimizer contracts, octree controls,
  checkMesh parsing, reference-study tools, delivery gates and interchange.
- Focused Pyrefly: zero errors; stub warnings remain.
- `git diff --check`: clean.
- The broader `test_cylinder_reference_builds_smooth_conformal_wall_cells`
  test still fails its one-to-one wall/core-interface assertion. It assumes
  a cylinder-only column topology, whereas the current workflow wraps all
  patches and constructs intersection cells. Its compatibility with the native
  wrapper must be established; neither that assertion nor its later geometric
  thresholds were weakened here. This is not an all-tests-passing signoff.

**Bottom line:** the performance/serialization/oracle failures have been
repaired and the full coarse mesh is now reproducible. Geometric/native parity,
solver acceptance, and grid-study completion remain open, with concrete evidence
instead of another unbounded meshing attempt.
