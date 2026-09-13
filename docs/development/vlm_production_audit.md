# VLM production audit — 13 September 2026

This is the original discovery report. Subsequent implementation and test results
are in [VLM production repair results](vlm_production_repair_results.md). Several
software blockers below have since been repaired; the rotor qualification gate
remains open.

**Decision: do not release the current VLM–VPM rotor configuration as scientifically
validated production software.** The audit found and fixed real output defects,
verified cross-surface coupling, and identified reproducible remaining blockers.
Passing software regressions does not establish converged wind-turbine aerodynamics.

Scope: working checkout based on `98479c375eb773f0683df5a41cac5bea301671ec`,
including the pre-existing VLM/VPM changes. Unrelated FVM/coupler changes were
preserved. Validation here uses the CPU backend; GPU qualification is not implied.

## Findings repaired

| Finding | Correction and evidence |
|---|---|
| Backup refresh omitted bound VLM induction and external stage providers | Refresh the complete transport field at the accepted clock through the same stage contract as evolution, without publishing exchange rates. A moving two-wing regression compares saved and live velocity against the complete field. |
| Same-revision CPU caches could override newly computed backup fields | Serialize fresh device snapshots. Regression poisons velocity/vorticity caches before writing and verifies the HDF5 values and unchanged positions, strengths and circulation. |
| Empty-wake plane/line samples were all zero | Remove the early return; preserve freestream and any solved bound/body field. Tested with an empty cloud both before and after a VLM solve. |
| Scheduled lines overwrote preceding samples | Honor `csv_time_series`, append native time/step rows atomically, validate the existing schema, and reject duplicate times. Restarted output-manager tests preserve all frames and reject legacy snapshots without overwriting them. |
| Crossing records could be published before step acceptance | Buffer records until the physical phases and clock commit succeed. Observe before diffusion changes particle identities. Strict rejected trials publish no accepted event files. |
| Leakage diagnostics repeatedly allocated Taichi fields and potentially enormous host pair arrays | Reuse probe workspaces; evaluate finite-core targets with a Taichi direct pair sum. Bound both dimensions of the retained independent host reference. Test all four radial kernels in f32/f64 against host sums. |
| Stage adapters allocated a new device field per query and used recyclable Python object identities as cache revisions | Borrow background metadata and use monotonically increasing revisions; test reuse and unique stage revisions. |
| Checkpoint auditor rejected valid binary VTP | Read encoded/raw-appended VTK through the VTK reader. A native binary-file regression and all nine retained rotor checkpoint prefixes now pass. |
| Leakage failures disappeared silently | Log diagnostic failures explicitly; missing evidence must not be interpreted as zero leakage. |

The collision broad phase now culls particles in bounded vectorized batches,
preserving the previous box margins. Exhaustive randomized comparisons and the
existing moving/warped/edge collision cases check classification preservation.

## Backup ownership and cross-surface coupling

VPM's output manager owns numerical backups. The current rotor setup requests
one every **four accepted steps**, or **0.024 s** at its authored timestep.
Each HDF5 write creates its VLM VTP companion at that same step/time. Coupled
cases reject a separate VLM geometry sampler. Force/loading CSVs are written
every accepted step; wake field samples use their separate 0.06 s schedule.
Those are different output products, not competing numerical backup clocks.

The new moving two-wing test checks periodic steps 2 and 3's explicit final
backup, exact paired geometry/force/circulation/velocity arrays, and unchanged
primary state. The retained nine rotor backups pass the publication-clock audit
through step 1152, t=6.912 s. This does **not** retroactively repair old velocities
or establish physical accuracy from timestamps alone.

**Yes: wake from one VLM surface induces velocity at the other surfaces.**
The accepted solve evaluates the entire free-particle cloud at every collocation
point and solves one global influence system. Tests check the default lagged
solve against an independent host Biot–Savart sum, prove the cross-body
contribution is nonzero, and show that changing a particle's group label leaves
the responsive circulation unchanged. Removing its strength changes the
receiving surface's circulation. Off-diagonal influence blocks are nonzero.
Group labels describe provenance; they do not isolate aerodynamic interaction.

## Rotor evidence and remaining release blockers

1. **The retained rotor is a failed run.** Metadata ends at step 1280 of 1667,
   t=7.68 s, with strain-based Lagrangian CFL 1.07 exceeding its limit of 1.
   The final sampled particle-vorticity maximum rises from about 122,000 to
   176,000 s⁻¹ between 7.56 and 7.62 s. Selecting a 7.5 s endpoint does not
   demonstrate numerical stability or convergence.

2. **The authored timestep already warns of under-resolved wake convection.**
   On the real 396-panel rotor, dt=0.006 s gives the setup diagnostic 5.67;
   its suggested bound is approximately 0.00106 s. This is a resolution warning,
   not proof that a smaller timestep alone cures the late growth. Refining dt
   also increases emitted particle count: 135 particles/step at dt=0.001 s over
   7.5 s would exceed one million particles, beyond the current 600,000 capacity.

3. **Detailed crossing classification is prohibitively expensive on this host.**
   The ten-step startup smoke reached only step 3, t=0.018 s, N=405, before its
   wall-time guard stopped it (235 s including startup/output). A separate
   synthetic replay of 16 recorded particles took 15.26 s, with 15,147 detailed
   classifications; 14.70 s was inside the moving finite-surface classifier.
   Profile results include instrumentation overhead and are not a throughput
   guarantee. The detailed geometric calculation needs a compiled implementation
   checked against the existing geometric oracle. Broad-phase culling alone
   does not solve the near-blade cost.

4. **The optional responsive temporal formulation is incomplete.** A static
   identical-geometry counterexample gives exactly the same new-row matrix
   for accepted and virtual rows, but the virtual row omits an old-circulation
   normal-velocity contribution of 0.004326 m/s. It also does not establish a
   consistent moving-row/transport closure. Do not certify this option from its
   algebraic residual. The default lagged policy avoids this particular virtual
   row but still requires coupled timestep-convergence evidence.

5. **Induction validation remains incomplete.** Retained fields exist at 1D,
   3D and 5D, with their last frame at 7.62 s. There is no retained 2D field or
   streamwise history, and the requested exact terminal averaging window is
   unbracketed. The finite-distance postprocessor consequently returns no finite
   terminal-window comparison scores. The new setup records four streamwise
   lines from −1D to 3D at r/R=0, 0.25, 0.65, 1.1, and plots signed ux, uy, uz
   information. Its plotting code rejects incomplete histories. The provided
   startup plot is explicitly unqualified; it is not a steady induction result.

The retained five-revolution load means are:

| Quantity | VLM/VPM | Matched BEM | Relative difference | Half-window drift |
|---|---:|---:|---:|---:|
| CT | 0.722836 | 0.683817 | +5.71% | 0.96% |
| CP | 0.530535 | 0.486252 | +9.11% | 1.60% |

Normalization uses actual blade radius 6.005739 m, disk area πR², density
1.225 kg/m³, and U=7 m/s. Shaft power uses each blade's fluid-on-body moment
and actual angular velocity, including the startup ramp. These definitions
and pressure-load moments are covered by numerical regressions. The comparison
uses a matched thin-plate lift model with no profile drag; it does not validate
real turbine airfoil polars, stall, nacelle or tower physics.

The finite-distance reference is an engineering comparison using
[Li et al. (2025)](https://wes.copernicus.org/articles/10/2515/2025/) and
[Li et al. (2022)](https://wes.copernicus.org/articles/7/75/2022/).
Those models' idealizations do not provide an uncertainty budget for this
finite-blade LES wake. The existing 25% axial/35% azimuthal screens remain
diagnostic flags, not production acceptance criteria.

## Documentation, precision and acceleration

Added numerical docstrings for the bound-exchange initializer, wake append
operation, triangle pressure loads, dense solver cache and iterative algebra
kernels. The inventory records 324 VLM function definitions including nested
helpers, with 41 still lacking docstrings, mostly private OpenVSP parsers,
configuration validators and local helpers. The inventory names each one;
documentation completeness is not claimed.

The ordinary Gaussian and Winckelmans finite-target reference checks use strict
double-precision tolerances in f64. HIGH_ORDER_GAUSSIAN and SUPER_GAUSSIAN still
use an erf approximation with an approximately 1e-7 error floor in device code,
including f64; their tests explicitly allow that existing approximation.
Selecting f64 does not remove it. Rotor uses ordinary Gaussian.

Existing influence assembly, bound induction/gradients, shedding and load kernels
already use Taichi. Further priorities are the detailed moving-surface classifier
and reuse/batching of geometry-only work. Dense SciPy LU is reasonable for the
396-panel case; moving a small solve onto a GPU is not automatically an improvement.
The direct finite-target diagnostic is memory-bounded now but remains O(MN);
its production cost still needs measurement at a mature rotor wake.

## Evidence and decision rule

Reproduction scripts, raw logs, JUnit results, documentation inventory,
retained-load statistics, counterexample and the labeled startup figure are in
`studies/vlm_production_audit/`. **358 distinct tests passed**: 287 broad VLM,
backup, stage/kernel and rotor regressions, 49 output-contract tests and 22
checkpoint-auditor tests. The strengthened cross-surface oracle passed an
additional focused rerun. Final counts and file hashes are recorded in
`validation_summary.json` and `source_sha256.json`. Earlier failing logs are retained as discovery
evidence; they are not final validation results.

The short rotor pilot exposed the line-overwrite bug before its final repair;
its original native snapshot is preserved. The final time-series behavior is
verified by the output tests, not represented as a completed new rotor run.

Production approval requires closure of the measured runtime/formulation issues,
a healthy resolved rotor run with complete native induction histories, and a
predeclared convergence/reference comparison for CT, CP and all three velocity
components. No repeated full rotor campaign was launched to work around failed
evidence, and no thresholds or failed native outputs were relabeled as passes.
