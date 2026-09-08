# VPM leapfrogging study: twelve-hour continuation

The user clarified that further simulations are authorized and that the
intended total budget is approximately half a day (12 hours). The previous
50-minute run caps were too short to answer the late-time question.

Started 2026-09-08 13:42 UTC (15:42 Amsterdam). Finish calculations and
assessment by approximately 2026-09-09 01:42 UTC (03:42 Amsterdam).

## Scientific decision sequence

1. Run the unstabilized LES baseline farther, targeting t=8.0025 with a
   four-hour solver cap. Use SSPRK3, transposed stretching, Smagorinsky
   Cs=.20, GBD/Lagrange6, h=.04, sigma=.04, dt=.0075, Re_Gamma=3000 and
   zero imposed disturbance. GBD/Lagrange6 is a provisional affordable
   choice supported by earlier transport controls; it is not certified best.
2. Inspect the baseline's native sampled trajectories, core deformation,
   energy/enstrophy, divergence, misalignment, strain/CFL and particle count.
   Score the predeclared common axial intervals against LBM without fitting
   or extrapolation. Separate numerical failure, physical deformation,
   sampler loss of core identity, and a computational budget stop.
3. Choose at most one stabilization intervention from the observed defect.
   Divergence/misalignment can motivate constrained realignment; loss of
   resolution can motivate refinement. Excess damping or a timestep error
   first requires a numerical sensitivity comparison. Do not automatically
   repeat weak realignment, whose previous short result did not improve
   the measured trajectory. Do not force an otherwise healthy baseline to fail.
4. Reserve a focused timestep or spacing comparison to check whether any
   apparent improvement survives numerical refinement. Compare at common
   physical times/distances; never interpret unequal wall-time survival as
   improved physics. Finish with the supported conclusion even if negative.

Provisional allocation: baseline 4 hours, one motivated comparison up to
4 hours, sensitivity up to 2.5 hours, assessment and overhead 1.5 hours.
Reallocate unused time based on evidence; do not exceed the overall deadline.
The t=8 target is not a guarantee that the run or physical breakdown will be
reached within four hours. Check measured progress rather than extrapolating
from startup speed. The old baseline grew from 68,952 to 195,662 particles.

## Reproducibility and execution

The source snapshot is
`.study-runtimes/vpm-halfday-20260908T134205Z/` under the repository root.
It contains the current VPM dependencies and tutorial source, including
uncommitted work, with per-file SHA-256 values in `source_manifest.json`.
Source files are read-only. All comparison runs must execute from this same
snapshot and verify the recorded source fingerprint. Do not change its
source while Taichi is compiling or running. This avoids the differing
source fingerprints that invalidated the earlier controlled comparison.

`campaign.json` records the exact baseline command, supervisor PID, start,
deadline and log. Its initial sandboxed launch failed to initialize Metal
before step zero; the unchanged command was relaunched with GPU access.
The baseline tag is `les_halfday_20260908_baseline`. Output is written to
the normal tutorial `study_results/` directory through a directory link.
The supervisor uses caffeinate to prevent idle sleep during its calculation.

All flow diagnostics and fields remain VPM FlowIntegralsSampler,
RingDiagnosticsSampler and SurfaceSampler outputs. Existing tutorial
postprocessing reads those outputs. There are no reconstructed particle
snapshots, custom continuation files or replacement field evaluators.
Normal VPM final backups are retained. A source snapshot is retained for
reproducibility, not as an alternative simulation or diagnostic workflow.

Thirty configuration and lifecycle checks passed before launch. The launch
must additionally be verified from a progressing solver log and saved
sampler outputs; successful process creation alone is not simulation success.

The GPU relaunch is advancing and has written native scalar and field
samples. Its source fingerprint is
`13cd1b20d3dec09120bbae8848e821e812d16f908c5baa0ca7c999c073e3f550`.
The app follow-up `complete-twelve-hour-vpm-study` checks this task every
15 minutes, chooses the next bounded comparison from evidence, and is to
pause after final assessment or at the deadline. It reports meaningful
changes rather than unchanged progress. No subsequent run is preselected.
