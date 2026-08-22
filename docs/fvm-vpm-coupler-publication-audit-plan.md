# FVM-VPM coupler simplification and physics-audit plan

**Status:** minimal operator implemented; exact/manufactured and short production-resolution time gates pass; long-time phase and spatial-convergence gates remain  
**Scope:** `tutorials/coupled_FVM_VPM/cube_flow/` only  
**Baseline snapshot:** local `development` at `a39eb34c64d5a03985998b6f13523f1d464a4942`  
**Baseline data policy:** treat `cube_flow/samples/` and `cube_flow/referenceFlow/samples_backup/` as read-only reference artifacts

## Constraints that govern the work

- Keep the cube case's `vorticity_mixed` outer-boundary formulation. It is the reference boundary condition for this audit.
- Keep VPM-owned GBD viscous diffusion, its global/absolute regeneration threshold, its population limit, and its conservative treatment. Do not introduce a local-maximum vorticity criterion.
- Remove duplicate population management from the coupler. The coupler exchanges fields and coordinates authority/time; the VPM owns particle diffusion, stabilization, pruning, and capacity policy.
- Do not modify the fully meshed FVM reference or tune empirical constants to improve agreement.
- Do not overwrite the current sample data. Every executable comparison will run in an isolated case directory and write separate audit artifacts.
- Preserve `create_fvm_solver -> create_vpm_solver -> CouplerSetup -> create_coupler(...)` construction unless a confirmed correctness issue requires a narrowly scoped API change.
- No damping, clipping, smoothing, relaxation, deconvolution gain, conservation repair, or threshold may enter the minimal path unless it follows from the discrete operator and passes the identity, fixed-point, solenoidality, and convergence gates below.

## Actual current coupling timestep

The current call graph is not the advertised minimal three-operation model. One interval is:

1. `FVMVPMCoupler._advance_vpm()` advances the VPM from `t_n` to `t_(n+1)`. In the cube case this includes VPM advection, stretching, LES state, and GBD diffusion/regeneration. These are VPM-owned operations and remain in scope only as external solver evolution.
2. `evaluate_vpm_boundary()` evaluates the body-complete VPM trace at `t_(n+1)`:
   - evaluates VPM velocity at every active blending cell and every coupling-patch face;
   - updates volumetric `Utarget` for the FVM blending source;
   - evaluates the retained mixed trace: normal velocity and tangential normal-velocity gradient;
   - unconditionally projects the full velocity trace to zero net flux;
   - separately projects the scalar normal trace again;
   - on the first interval, initializes the missing `t_n` history by copying this already-advanced `t_(n+1)` trace.
3. `advance_fvm_substeps()` advances the FVM from `t_n` to `t_(n+1)`:
   - linearly interpolates boundary and blending data at each FVM substep;
   - projects the interpolated full velocity trace again;
   - projects the interpolated normal trace again;
   - solves PIMPLE with the mixed vorticity boundary condition;
   - also applies the always-enabled scale-selective volumetric source based on `lambdaRelax` and `Utarget`.
4. `VorticityTransfer.transfer()` samples FVM cell velocity and velocity gradient and hands the result to `continuous_transfer()`:
   - uses a four-neighbour inverse-distance-squared Taylor reconstruction;
   - multiplies sampled FVM velocity by a synthetic wall-distance taper before curl;
   - computes lattice circulation from the velocity trace;
   - tapers current particle circulation around/interior to the solid;
   - remeshes every particle in the box plus transfer buffer, including particles where FVM authority is zero;
   - forms a scalar authority blend of VPM/FVM circulation-related fields;
   - applies Gaussian residual “inverse mollification” with a gain derived from `transfer_amplification_cap`;
   - soft-prunes, locally redistributes removed strength, deletes interior-solid nodes, then globally repairs invariants;
   - may cap the complete output cloud and globally repair invariants again;
   - appends exterior particles with default volume and core radius instead of their original values.
5. `vpm.replace_vortex_particles()` replaces the whole cloud:
   - uploads new position, circulation, radius, and volume;
   - resets velocity and eddy viscosity to zero;
   - resets zone IDs and, by omission, group IDs and gradients;
   - resets stabilization replacement/lineage state;
   - uses the default `report_removal=True`, so a representation rebuild is reported as physical removal of the old cloud.
6. `resynchronize_vpm_boundary()` re-evaluates the corrected VPM endpoint for the next interval and performs further zero-flux projection.

### Current field-mutation inventory

| Stage | VPM particle fields affected | FVM fields affected | History/diagnostics affected |
|---|---|---|---|
| VPM advance | Position, velocity, strength, radius/volume through GBD, viscosities, gradients, IDs/lineage through VPM-owned algorithms | None | VPM time and step |
| VPM boundary evaluation | None | `Utarget` is replaced | Previous/next velocity, normal velocity, tangential gradient, optional pressure history |
| FVM subcycling | None | Boundary ghost data, velocity, pressure, face flux, turbulence state; `lambdaRelax` source acts volumetrically | FVM time and step |
| FVM-to-VPM transfer | Entire particle cloud is rebuilt | Read-only sampling of velocity and gradient | Prune/cap/conservation/“flux” diagnostics |
| Boundary resync | None | Blending endpoint cache | Previous boundary endpoint is overwritten at fixed physical time |

## Confirmed audit findings

These findings were verified against the current source rather than inferred from old comments:

1. **`eta=0` is not an identity.** `continuous_transfer()` remeshes all particles in the box plus buffer. The existing test explicitly accepts remesh diffusion in a no-FVM-authority zone, which conflicts with the required identity.
2. **Complete free-wake state is destroyed.** The pure transfer API carries only position and circulation. Exterior volume/radius are reconstructed from global defaults, and whole-cloud upload resets velocity, viscosities, IDs, gradients, strain, vorticity cache, and stabilization lineage.
3. **Whole-cloud replacement uses physical-removal semantics.** The coupler omits `report_removal=False`.
4. **Coupler-side pruning and population capping are active.** `soft_prune`, boundary-dependent pruning, local redistribution, `transfer_max_particles`, and two invariant-recovery paths are in the production transfer.
5. **Invariant repair can redistribute solid-deleted circulation.** Target invariants are captured before exact interior-node deletion and recovered afterward.
6. **The wall taper creates coupler vorticity.** The sampled FVM velocity is multiplied by a smooth wall-distance weight before curl, introducing `grad(w) x u`.
7. **The Gaussian correction is not an enforced amplification cap.** The parameter sets `correction_gain = min(cap - 1, 1)` and only measures the resulting amplification afterward.
8. **The current transfer is not solenoidal by construction.** It blends circulation/vorticity-related quantities with spatially varying scalar authority. There is no compatible `div_h(curl_h(.))` construction or transfer divergence gate.
9. **Volumetric blending is always enabled.** Coupler initialization always registers nonzero `lambdaRelax`; the FVM applies a scale-selective filtered relaxation by default. There is no minimal no-blending reference mode.
10. **Boundary-flux correction is unconditional and repeated.** Full-vector and scalar-normal projections occur during evaluation, every substep, and resynchronization, with no dimensionless acceptance threshold or physically significant residual failure.
11. **`flux_ratio` is not a transport flux.** It is an outflow-band ratio of sums of vorticity/circulation magnitudes.
12. **The first boundary interval has incorrect endpoint history.** The VPM is advanced to `t_1` before any `t_0` boundary trace is captured, then the `t_1` trace is copied into both endpoints for the FVM interval `[t_0,t_1]`.
13. **Current tests preserve patch behavior.** Tests presently certify pruning, population deletion plus global repair, remeshing-induced diffusion at zero authority, repeated projection, and approximate rather than fixed-point transfer. These tests must be replaced where their expected behavior is nonphysical.

## Target minimal coupling timestep

The target split step is:

### Initialization at `t_0`

1. Evaluate and store the body-complete `vorticity_mixed` VPM boundary trace at `t_0`.
2. Measure its dimensionless net flux residual once. Correct only a residual consistent with numerical quadrature/discretization; fail clearly if the required correction is physically significant.
3. Do not register or evaluate volumetric FVM blending fields in the minimal mode.

### Interval `t_n -> t_(n+1)`

1. Let the VPM advance its own state to `t_(n+1)`, retaining its global GBD diffusion/pruning policy.
2. Evaluate the retained VPM mixed boundary trace at `t_(n+1)`.
3. Advance FVM substeps using only time interpolation between the valid `t_n` and `t_(n+1)` boundary endpoints. Perform at most one accepted flux correction on each endpoint, not on every interpolation of the same data.
4. Transfer FVM-resolved wake information through one compatible operator:

   ```text
   delta_u = eta * (I_h[u_FVM] - E_h[u_VPM])
   delta_Gamma = V_h * curl_h(delta_u)
   ```

   This is the correction form of `u* = u_VPM + eta(u_FVM-u_VPM)`. It is the preferred baseline because `eta=0` gives an exact zero correction, identical donor fields give an exact fixed point, and `div_h(delta_Gamma)=0` follows from one compatible discrete curl/divergence pair. Existing particles are not remeshed merely to support interpolation.
5. Apply only the mathematically nonzero correction in the actual authority/correction support. Preserve every pre-existing particle and all of its fields outside that support. New correction particles, if needed, are initialized once through the VPM API and are subsequently owned by the VPM. If the required capacity exceeds the VPM limit, fail before mutating the cloud.
6. Re-evaluate the corrected `t_(n+1)` boundary endpoint only if the transfer materially changes the represented boundary field; store it as the next interval's `t_n` endpoint without another FVM solve.

The additive correction application is a design hypothesis, not an exemption from testing. If its compatible discrete realization cannot pass the fixed-point, repeated-transfer, locality, and convergence gates, it will be rejected rather than patched. The only fallback considered will be replacement of the strict `eta>0` authority subset through a state-complete, non-removal API; the buffer and `eta=0` particles remain byte-for-byte untouched.

## Implementation phases and gates

### Phase 0 — concurrency check and immutable baseline

- After approval, re-read `git status`, inspect every change made by the other agent, and rebase this plan onto the then-current `development` state before touching overlapping files.
- Record the commit, package versions, hardware, mesh counts, time steps, and case constants.
- Hash the existing cube `samples/` and fully meshed `samples_backup/`; never run `allclean.sh` or the tutorial in those source directories.
- Run the current operator and cube case only in an isolated copy. Archive logs, timings, particle counts, force histories, profiles, fields, and diagnostics as the “before” dataset.

### Phase 1 — write fundamental tests before deleting patches

Add tests that expose the current failures:

- `eta=0` exact identity for particles inside the numerical stencil buffer.
- Complete-state identity covering position, velocity, strength, radius, volume, molecular/eddy/effective viscosity, group/zone IDs, velocity gradient, strain, cached vorticity, removal counters, and stabilization lineage.
- One-shot manufactured fixed point where the synthetic FVM donor is evaluated from a known VPM field.
- 50 repeated transfers with no physical evolution; track field norms, circulation, enstrophy, strength spectrum, population, radius/volume distributions, and discrete divergence.
- Compatible `div_h(curl_h(u))` identity on the transfer lattice, including a normalized interior divergence metric and explicit boundary treatment.
- Manufactured time-varying boundary data over the first and later coupling intervals.

The initial versions of these tests may be marked as expected failures only long enough to prove that they detect the current defects. They become mandatory passing tests in the same implementation series.

### Phase 2 — isolate and qualify interpolation/differential operators

- Separate FVM cell-to-lattice interpolation from curl and particle mutation so each can be tested independently.
- Test constant velocity, affine velocity, solid-body rotation, quadratic velocity, a smooth analytic vortex, and a closed vortex ring.
- Test uniform Cartesian, smoothly graded Cartesian, and mixed/nonuniform FVM meshes.
- Use two test tracks:
  - exact analytic gradients, to measure only the four-neighbour Taylor interpolator;
  - gradients produced by the FVM, to measure the complete production path.
- Measure `L1`, `L2`, and `Linf` velocity/curl error under refinement and document the theoretical and observed consistency order. Constant/affine exactness and convergent quadratic/vortex results are release gates.
- Keep direct FVM-vorticity interpolation only as an offline A/B diagnostic. It must not be applied simultaneously with velocity-to-curl transfer.
- Verify gradient tensor conventions explicitly; do not rely on current comments.

### Phase 3 — implement one solenoidal correction operator

- Apply authority to the velocity difference and take one compatible discrete curl.
- Use an interpolation guard solely to support stencils. It must never enlarge physical mutation support.
- Remove synthetic wall-velocity tapering. Keep exact exclusion of particle centers inside the solid as a placement rule only.
- Do not scalar-blend two vorticity fields.
- Do not apply an empirical Gaussian residual gain. A kernel-moment consistency inverse may act on the velocity defect only if its coefficient is derived, the exact fixed point is preserved, and error converges with `h`.
- Do not prune, cap, redistribute, or repair global invariants in the coupler.
- Preflight VPM capacity and raise an error that reports required/current/maximum counts before any state mutation.
- Add or narrow a VPM subset-mutation API only if necessary. It must preserve untouched fields and stabilization lineage and distinguish representation updates from physical removal.

### Phase 4 — simplify time and boundary coordination

- Capture the actual `t_0` VPM boundary state before the first VPM advance.
- Interpolate exactly between endpoint data for each FVM substep; test a linear-in-time manufactured trace at every substep time.
- Preserve the cube's `vorticity_mixed` normal-velocity/tangential-gradient boundary condition.
- Define a dimensionless flux residual `|integral(u.n dA)| / (U_ref A)` with a tolerance derived from quadrature/solver precision. Correct only below that limit; warn or fail above it.
- Project one canonical endpoint representation at most once. Derive scalar normal data from that corrected endpoint and remove redundant projections.
- Audit whether post-transfer boundary resynchronization is required by the explicit split. Retain it only as endpoint consistency, never as an extra relaxation/Picard correction.

### Phase 5 — remove obsolete machinery and diagnostics

Expected coupler removals, subject to the tests above:

- `transfer_vorticity_cutoff`
- `transfer_boundary_prune_multiplier`
- `transfer_max_particles`
- `transfer_amplification_cap`
- `soft_prune()`
- `redistribute_locally()`
- coupler uses of `recover_invariants()` and `source/coupler/conservation.py` if no independent diagnostic-only user remains
- Gaussian mollification/inverse-mollification transfer functions
- coupler-side population-cap logic
- synthetic wall-distance velocity taper
- volumetric `BlendingZone` from the minimal path, plus scale-selective compensation code if A/B evidence does not justify retaining an explicitly experimental mode
- tests whose only purpose is to certify those patches

Diagnostics changes:

- Preserve raw, non-forcing circulation/impulse/enstrophy budgets as diagnostics.
- Add normalized `div_h(omega)` before/after transfer and local transfer error split by `eta=0`, ramp, and FVM-authority regions.
- Rename `flux_ratio` to an honest field-comparison name such as `outflow_band_vorticity_magnitude_ratio`, or remove it. Do not call a magnitude ratio a flux.
- If a transport diagnostic is retained, define the surface, normal, velocity, vorticity/circulation quantity, quadrature, units, and sign.
- Record particle additions separately from physical removals and VPM-owned GBD regeneration.

### Phase 6 — interface transport tests

- Advect a smooth closed vortex/ring across the authority ramp.
- Track centroid, peak and integrated strength, core/radius distribution, enstrophy, spectrum, population, and normalized divergence before, during, and after crossing.
- Require no discontinuous strength jump and no source/sink tied to `grad(eta)`.
- Repeat with refined particle spacing and with coupling/FVM time steps refined together while keeping integer subcycling.
- Require decreasing field and phase error and demonstrate that the result is insensitive to coupling-step refinement within the measured order.

### Phase 7 — cube-only validation and optimization

Run in this order:

1. Unit/manufactured suite.
2. Isolated one-step cube smoke test.
3. Short canonical cube run with the minimal operator and no volumetric FVM blending.
4. Time-step and transfer-resolution refinement runs.
5. Production-length cube validation if the earlier gates pass.

Compare against both the immutable current hybrid samples and the immutable fully meshed FVM reference:

- velocity and vorticity profiles/fields, reported separately in the transfer ramp and global domain;
- force histories and mean drag;
- circulation and enstrophy budgets;
- vorticity/force spectra where the record length is sufficient;
- particle count, transfer cost, total wall time, and memory;
- normalized divergence;
- coupling-time-step sensitivity.

The reported drag “delay” will be measured rather than judged visually:

- interpolate both force records onto a common time grid without time warping;
- remove only a documented startup interval and, for correlation, the mean/trend;
- report the lag that maximizes normalized cross-correlation, the zero-lag correlation, and matched physical-event peak times;
- repeat under time-step refinement to distinguish a one-coupling-step history error from a physical/numerical phase error;
- report amplitude error separately so a phase shift cannot hide force bias.

Volumetric blending will be considered only as an A/B experiment after the no-blending baseline passes. It may be retained outside the baseline only if it fixes a reproducible defect while improving or preserving field error, force phase/amplitude, spectra, convergence, conservation diagnostics, and divergence. No gain tuning is allowed.

## Test hierarchy and release gates

### Rung 0 — exact identities

- Uniform velocity gives zero transferred vorticity and no particles.
- Solid-body rotation gives the exact constant compatible curl.
- `eta=0` gives exact complete-state identity, including metadata/lineage and population.
- Consistent FVM/VPM data is a fixed point.
- 50 repeated transfers without evolution show no statistically or monotonically accumulating drift; exact-state quantities remain exact where the operator promises identity.
- The first interval uses distinct, correct `t_0` and `t_1` manufactured endpoints.

### Rung 1 — manufactured transfer

- Constant, affine, quadratic, solid-body rotation, smooth analytic vortex, and closed-ring fields.
- Uniform and graded/nonuniform meshes.
- Documented convergence order for velocity and curl.
- Local field, circulation, enstrophy, spectrum, and `div_h(omega)` errors.

### Rung 2 — interface transport

- Vortex crossing without strength jump, artificial damping/amplification, or vortex-line source/sink.
- Coupling-step and spacing refinement.
- Exact identity outside authority throughout the crossing.

### Rung 3 — coupled cube

- Fully meshed FVM versus hybrid velocity/vorticity profiles and fields.
- Force amplitude and measured phase lag.
- Spectra, circulation/enstrophy budgets, divergence, and transfer-zone-local errors.
- Runtime and particle population compared with the archived current implementation.

No later rung can waive a failure in an earlier rung merely because the cube simulation runs or its drag curve looks better.

## Verification commands after implementation

- Focused coupler tests during development: `pytest tests/coupler -m "not mpi"`
- Relevant FVM and VPM tests for changed APIs/operators.
- Formatting and lint for touched files with Ruff.
- Pyrefly before and after Python API/signature changes under `source/coupler`, `source/solvers/FVM`, or `source/utilities`, without increasing the existing error baseline.
- Serial and, where supported, MPI smoke tests.
- Isolated cube audit runs and post-processing; never run destructive tutorial cleanup in the versioned sample directories.

## Final report contents

The implementation report will include:

1. Exact before/after coupling timestep call graphs.
2. Every removed operator and parameter with the failing property or lack of necessity that justified removal.
3. Every retained nontrivial operator with its mathematical/physical justification.
4. Results for identity, complete-state preservation, fixed point, repeated-transfer drift, divergence, interpolation convergence, and interface transport.
5. Quantitative before/after cube results, including runtime, particle count, force amplitude, and measured force phase lag.
6. A list of remaining questionable operations and the evidence still needed.
7. Issues discovered beyond the original audit list.
8. The exact commit, commands, environment, and artifact paths needed to reproduce the results.

## Known questions to resolve with evidence, not tuning

- Whether the additive curl correction provides adequate one-pass local fidelity with Gaussian blobs at the cube's current `sigma/h`, or whether the resolution itself must be changed.
- The actual consistency order of the four-neighbour Taylor reconstruction on the cube's graded mesh.
- Whether endpoint resynchronization is required after an exact correction operator or is redundant.
- Whether the residual force delay is entirely the missing `t_0` history, an explicit partitioned-splitting error, or a spatial transfer/outer-boundary propagation effect.
- Whether any volumetric FVM relaxation has a defensible role once the boundary history and transfer operator are corrected.
