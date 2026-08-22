# FVM–VPM coupler simplification and physics-audit report

**Status:** the simplified operator passes the exact and manufactured gates and
the short production-resolution cube time-refinement gate. The cube case is
not yet publication-ready: a production-length phase comparison and spatial
representation-divergence convergence remain release gates.

## Scope and immutable baseline

The audit started from `development` commit
`a39eb34c64d5a03985998b6f13523f1d464a4942` and was limited to
`tutorials/coupled_FVM_VPM/cube_flow/`. The cube's `vorticity_mixed` boundary
condition remains unchanged. VPM-owned GBD diffusion/regeneration retains its
absolute global threshold (`0.05 h^3` in the cube case), global population
policy, and moment treatment. No local-maximum vorticity criterion was added.

The versioned hybrid `samples/` and fully meshed
`reference_flow/samples/` were not modified. Every executable audit ran
in an isolated directory under `/private/tmp`.

## Coupling timestep before the changes

For one interval `t_n -> t_(n+1)`, the old driver performed:

1. Advance the VPM, including VPM advection, stretching, LES state, and GBD.
2. Evaluate the advanced VPM at coupling faces and volumetric blending cells.
   Project the vector trace and scalar normal trace to zero flux, update
   `Utarget`, and, on the first interval, copy the already advanced `t_1` trace
   into both the `t_0` and `t_1` history slots.
3. Interpolate those traces during FVM subcycling, project them again, impose
   the mixed vorticity boundary condition, and solve with the scale-selective
   volumetric source `lambdaRelax (Utarget - G*U)`.
4. Interpolate FVM velocity with a four-neighbour Taylor reconstruction,
   multiply it by a synthetic wall-distance taper, and compute lattice curl.
5. Remesh the complete overlap plus numerical buffer, scalar-blend VPM and FVM
   circulation, apply bounded local Gaussian inverse mollification, delete
   solid nodes, soft-prune, redistribute locally, optionally cap population,
   and force global circulation/impulse invariants.
6. Replace the complete VPM cloud with physical-removal semantics. Position
   and strength were reconstructed; velocity, viscosities, IDs, gradients,
   radii/volumes, and stabilization lineage could be reset or replaced even
   where FVM authority was zero.
7. Re-evaluate and project the corrected VPM boundary endpoint.

The old coupler therefore modified all particle fields in a stencil/remesh
buffer, not only the region of actual FVM authority.

## Minimal coupling timestep after the changes

Initialization evaluates and stores the physical VPM mixed boundary trace at
`t_0` before any VPM advance.

For each interval:

1. If the preceding handoff changed particles and GBD is active, the VPM first
   applies its existing global regeneration at zero elapsed time. This removes
   the dense transfer representation using the VPM's global criterion; it is
   not coupler pruning. The VPM then advances its own physics and applies its
   normal viscous GBD step.
2. Evaluate the body-complete `vorticity_mixed` trace at `t_(n+1)`. Measure
   `|integral(u.n dA)|/(U_ref A)`. Apply one minimum-L2 normal correction only
   below a spacing-based acceptance limit; fail above it.
3. Advance the FVM using linear interpolation between valid endpoint traces.
   There is no volumetric FVM relaxation and no repeated substep projection.
4. Evaluate FVM and VPM velocity on one wall-aligned correction lattice and
   form the velocity defect

   ```text
   d = eta (I_h[u_FVM] - E_h[u_VPM]).
   ```

   For an isotropic blob with second moment `m2 sigma^2`, apply the leading
   consistency inverse to the defect only,

   ```text
   d_c = (I - m2 sigma^2/6 Laplacian_h) d,
   delta_Gamma = h^3 curl_h(d_c).
   ```

   The Laplacian term is used only where its complete active stencil exists.
   It has no empirical gain or cap. Because it acts on the defect, both
   `eta=0` and a consistent FVM/VPM state give exactly zero correction.
   For a configured nonzero VPM-only dead zone, the velocity profile is inset
   by one curl stencil while the declared correction authority is unchanged;
   this prevents the centred curl from reaching an `eta=0` particle node.
5. Add `delta_Gamma` to a coincident, same-radius lattice particle when one
   exists; otherwise append a correction particle. No existing particle is
   rebuilt. Capacity is checked before mutation and exhaustion raises an
   error instead of deleting wake particles.
6. Notify the VPM that its externally supplied representation must pass
   through VPM-owned GBD before the next evolution. Re-evaluate the corrected
   boundary endpoint for the next interval.

The discrete correction has `div_h(curl_h(d_c))` at roundoff. In production
runs its normalized correction-divergence L2 remained approximately
`4e-17`.

## Removed operators and parameters

| Removed | Reason |
|---|---|
| Coupler overlap/buffer remesh | Violated exact identity at `eta=0` and changed free-wake state. |
| `soft_prune` and boundary prune multiplier | Population management belongs to VPM/GBD. |
| Local redistribution after pruning | Repaired an error introduced by coupler pruning. |
| Coupler population cap | Silently deleted wake content; replaced by pre-mutation capacity failure. |
| Global `recover_invariants` forcing | Could move solid-deleted or pruned circulation to unrelated fluid locations. |
| Whole-cloud `replace_vortex_particles()` transfer | Reset persistent fields and reported a representation rebuild as physical removal. |
| Synthetic wall-distance velocity taper | Added the unphysical term `grad(w) x u`. |
| Scalar vorticity/circulation authority blend | Does not preserve solenoidality across `grad(eta)`. |
| Bounded local Gaussian residual correction | Failed to enforce its advertised amplification cap and did not make the old operator a fixed point. |
| Volumetric FVM blending and its scale-selective filter | Added damping and compensation outside the minimal exchange model. |
| Repeated vector and scalar zero-flux projections | Replaced by one guarded endpoint correction. |
| `flux_ratio` and patch-validating spectral/pruning scalars | They were not transport fluxes and could validate a locally wrong field. |
| `source/coupler/remesh.py` | No independent use remained. |
| `source/coupler/conservation.py` | No independent use remained. |
| `source/coupler/blending.py` | No independent use remained. |
| `transfer_vorticity_cutoff` | Obsolete coupler-pruning parameter. |
| `transfer_boundary_prune_multiplier` | Obsolete boundary-pruning parameter. |
| `transfer_max_particles` | Obsolete coupler-cap parameter. |
| `transfer_amplification_cap` | Misleading empirical residual-gain parameter. |

## Retained nontrivial operators

| Retained | Justification |
|---|---|
| Cube `vorticity_mixed` outer boundary | Supplies VPM normal velocity and tangential `du/dn`; it remains the qualified cube boundary. |
| Linear endpoint interpolation | Required for FVM subcycling between coupling levels. |
| Guarded minimum-L2 flux projection | Corrects only closed-surface quadrature/discretization residual; significant imbalance fails. |
| Four-neighbour inverse-distance Taylor interpolation | Affine exact for `G[i,j]=du_j/dx_i`; manufactured smooth fields converge at approximately second order on graded meshes. |
| Cosine authority partition | Provides a bounded C1 transition applied to velocity before curl. |
| One-cell curl guard for nonzero dead zones | Keeps the correction support strictly inside the declared `eta>0` region without masking a completed curl. |
| Compatible centred curl | Gives exact fixed point and a roundoff-solenoidal correction. |
| Gaussian second-moment defect correction | Leading-order inverse of the known blob convolution, with coefficient fixed by the kernel moment; no tunable gain. It preserves the exact fixed point. |
| Exact cube solid exclusion | Prevents open-solid correction particles without a synthetic fluid-side wall taper. |
| Same-kernel coalescing/additive particles | Applies one correction without rebuilding existing state. |
| Post-transfer endpoint synchronization | Stores boundary data consistent with the corrected representation at the same physical time. |
| VPM-owned global GBD | Performs diffusion, global thresholding, population management, and its existing moment treatment in the owning solver. |
| Interface and vortex-line diagnostics | Read-only quantities with literal definitions; they do not force conservation. |

For the production Gaussian kernel, the second-moment term reduced a
manufactured represented-velocity error at `h=0.03125` from `0.2024` to
`0.0946` at scale `2h`, and from `0.1438` to `0.0872` at scale `4h`.

## Fundamental and manufactured tests

- `eta=0`: no donor evaluation, update, addition, removal, or state change. A
  localized dead-zone test next to an active curl stencil also verifies no
  correction particle and no in-place update at the zero-authority node.
- Complete state preservation: position, velocity, vortex strength, radius,
  volume, molecular/eddy/effective viscosity, group/zone IDs, gradients,
  strain, cached vorticity, and a lineage sentinel remain exact.
- Fixed point: a real CPU VPM cloud supplies its physical velocity as the
  synthetic FVM donor. Twenty transfers add no particles and preserve all
  tested fields bit-for-bit.
- Repeated transfer: 50 pure-operator applications and the real 20-transfer
  fixed point show no population, strength, radius, volume, metadata, or
  divergence drift.
- Solenoidality: compatible-curl corrections, including the wall-aligned cube
  solid, have normalized `div_h(omega) < 5e-14`.
- Interpolation: constant and affine fields are exact; solid-body rotation has
  exact constant curl. Graded-mesh velocity errors `8.29e-3`, `1.47e-3`, and
  `4.34e-4` give observed orders `2.49` and `1.76`. Analytic-vortex curl errors
  `3.23e-2`, `8.13e-3`, and `2.04e-3` give orders `1.99` and `1.99`.
- Interface transport: a smooth closed vortex was moved through 14 authority
  positions. A consistent donor produced zero correction, no strength jump,
  and no population change at every position.
- Time history: a manufactured time-varying trace verifies distinct `t_0` and
  `t_1` endpoints and exact values at every FVM substep.
- Flux acceptance: a discretization-scale residual is corrected; a physically
  significant residual fails.
- Capacity: insufficient VPM capacity fails before any mutation.
- VPM lifecycle: insertion of new correction particles schedules one pending
  global GBD regeneration; strength-only updates and fixed-point transfers
  schedule none; checkpoint/restart preserves the pending state.

## Production-resolution cube results

All hybrid results below use the original spatial resolution:

- FVM mesh: 571,576 cells.
- Surface cell size: `0.015625`.
- VPM spacing: `h=0.03125`.
- Transfer lattice: 857,375 nodes, of which 827,584 are open correction nodes.
- VPM capacity: 1,500,000 particles.
- Production VPM/coupling interval: `0.03`.

No coarse smoke result is used as force-accuracy evidence.

### Matched old versus simplified transfer

The old executable was run from an untouched archive of the baseline commit.
Its first two production handoffs reported pruning roughly 859,000–915,000
lattice nodes and discarded 6.57%–6.85% of `sum|Gamma|`; the first took
5.60 s. The simplified first handoff
took 2.62 s, pruned nothing, and produced a roundoff-solenoidal correction.
Total first-interval times were 188.23 s old and 185.09 s simplified; these
single-run wall times include compilation/cache effects.

At the original `dt_FVM=0.01`, the simplified result is worse during one part
of the startup transient: at `t=0.06`, the current exact full-FVM value is
`2.7764`, the old hybrid is `2.8695` (+3.35%), and the simplified hybrid is
`1.2025` (-56.69%). Both hybrids over-predict the earlier oscillatory
`t=0.03` value, so the initial impulse is not a new transfer regression.

After startup, the production-resolution simplified curve is smoother and
more accurate than the archived hybrid. Against the immutable sparse full-FVM
reference at `t=0.10, 0.15, 0.20, 0.25, 0.30`, its MAPE is 3.33% and RMSE is
0.0694, versus 10.90% and 0.2573 for the old hybrid.

| Time | Archived full FVM | Old hybrid error | Simplified error |
|---:|---:|---:|---:|
| 0.10 | 2.2766 | +17.00% | -5.14% |
| 0.15 | 1.9528 | +1.36% | -2.19% |
| 0.20 | 1.7947 | -14.67% | +1.98% |
| 0.25 | 1.6866 | +19.70% | +3.36% |
| 0.30 | 1.6105 | +1.78% | +3.98% |

### Exact startup sampling and time refinement

The old fully meshed force archive sampled every 0.05 s and hid a severe
impulsive-start oscillation. A new isolated run of the unchanged full-FVM
physics at every native `0.01` step gave:

| Time | Exact full-FVM Cd |
|---:|---:|
| 0.01 | 129.7546 |
| 0.02 | -84.5699 |
| 0.03 | 14.9942 |
| 0.04 | 2.1509 |
| 0.05 | 3.2573 |
| 0.06 | 2.7764 |
| 0.09 | 2.3045 |
| 0.12 | 2.0615 |

Linear interpolation of the old 0.05-spaced reference across this interval is
invalid and created a misleading apparent time shift.

At production spatial resolution, coupled time controls gave the following
errors against the per-step `dt=0.01` full-FVM reference above:

| `dt_coupling` | `dt_FVM` | Cd(0.06) | Cd(0.09) | Cd(0.12) | MAPE 0.06–0.12 |
|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.01 | 1.2025 | 2.2057 | 2.0675 | 20.42% |
| 0.02 | 0.01 | 1.4375 | 1.8363 | 2.3579 | 27.64% |
| 0.01 | 0.01 | 1.1059 | 2.0409 | 1.8081 | 27.97% |
| 0.015 | 0.005 | 2.7472 | 2.2834 | 2.0723 | 0.83% |

Refining coupling cadence alone at `dt_FVM=0.01` is non-monotone and does not
remove the startup error. A same-resolution full-FVM reference at
`dt_FVM=0.005` gave `Cd(0.03)=3.6194` and `Cd(0.06)=2.6692`. The systematically
refined hybrid (`dt_coupling=0.015`, `dt_FVM=0.005`) gave `3.4865` (-3.67%) and
`2.7472` (+2.92%) at those checkpoints. A `dt_coupling=0.020` control gave
`3.5107` and `2.8278`; retaining `dt_coupling=0.030` gave `-2.2273` and
`2.7385`. Thus the FVM refinement controls the settled error, while a coupling
interval below `0.03` is required to resolve the first impulse.

The refined `dt_FVM=0.005`, `dt_coupling=0.015` result remains a useful startup
control, but it more than doubles the production cost and has not been shown to
improve the long-time phase quantities targeted by this case. Production uses
the previous `dt_FVM=0.01`, `dt_coupling=0.03` pair. Its explicit nearest
complete endpoint is `20.01`, exactly 667 coupling intervals.

### Production runtime correction

A rejected run coupled `SURFACE_CELL_SIZE=0.015` directly to `h=0.03` and used
`dt_coupling=0.015`. At `t=0.24`, it carried 465,299 particles before transfer
and 995,144 afterwards; the VPM step took 100.5 s. The non-commensurate spacing
also produced normalized correction-divergence values `L2=0.0674` and
`Linf=0.292`, instead of the roundoff values obtained at `h=0.03125`.

The production VPM spacing is now explicit rather than derived from the FVM
surface size, and transfer setup rejects an axis-aligned body whose dimensions
are not integer multiples of `h`. GBD regeneration is requested only when the
handoff inserts particles; coincident strength updates do not trigger a second
population pass. Particle logs report physical population changes rather than
host storage types.

## Remaining questionable or open items

1. A production-length run at `h=0.03125` through several shedding cycles is
   still required for force cross-correlation, spectra, and event-time phase.
   The current `t<=0.30` record is too short for a defensible lag claim.
2. The compatible lattice correction is roundoff-solenoidal, but the evolved
   Gaussian cloud is not guaranteed to preserve continuous `div(omega)=0`.
   Representation-divergence convergence under GBD remains a release gate.
3. The second-moment defect inverse is mathematically specified and passes the
   fixed point, but its full production convergence with FVM gradient error
   still needs a spatial refinement series.
4. Pending zero-time GBD followed by the physical GBD step performs two
   VPM-owned regenerations after each handoff. It is correct and keeps the
   transfer cloud at the expected population, but its cost should be reduced
   without changing operator order or moving pruning into the coupler.
5. Exact solid handling is qualified only for the wall-aligned cube. Generic
   curved immersed bodies remain outside this audit.
6. Endpoint resynchronization is physically consistent with the explicit
   split, but should be retested under spatial refinement.
7. The hybrid FVM uses one outer PIMPLE corrector and no non-orthogonal
   corrector, while the full reference uses two and one respectively. The
   Cartesian non-orthogonal correction may be null, but the nonlinear
   convergence difference needs an isolated A/B test on the hybrid; the
   fully meshed reference must remain unchanged.

## Additional issues discovered

- Partitioned transfer initialization returned worker ranks before the
  collective cube-wall face gather. Rank zero then waited in `MPI_Gather`
  while workers entered a later `MPI_Bcast`. Wall geometry is now gathered
  collectively before workers return; a real two-rank body-wall smoke test
  completes two coupled steps and writes a partitioned checkpoint.
- SciPy `cKDTree` construction on a large regular GBD lattice recursively
  overflowed the C stack (about 80,577 repeated build frames) and caused a
  `SIGSEGV`. Both GBD lattice-tree constructions now use
  `compact_nodes=False`; the exact production run passes the former crash
  point and a regression test enforces the option.
- The old force-output cadence concealed the full-FVM startup oscillation and
  made linear interpolation unsuitable for diagnosing phase delay.
- The cube runner and geometry audit referenced `cube_flow_setup.py`, while
  the real file is `cubeFlow_setup.py`; those paths were corrected.
- Post-processing searched `referenceFlow/samples/`, while the immutable
  archive is `reference_flow/samples/`; both locations are now handled.
- The archived `couplingFace_xmax` full-FVM sample is at `x=3.5`, not at the
  hybrid boundary `x=1.5`; it is not used as a matched interface plane.

## Verification status

- Complete coupler suite excluding parallel/GPU/slow markers: 83 passed.
- Two-rank partitioned body-wall coupled smoke test: passed through two
  coupling intervals and checkpoint output.
- Complete serial FVM suite excluding parallel/GPU/slow markers: 427 passed.
- Focused VPM diffusion, sampler, checkpoint, active-box, and GBD audit tests:
  52 passed.
- Ruff formatting/lint and `git diff --check`: passed on touched files.
- Pyrefly: the focused changed-coupler check passes with zero diagnostics.
- The production-resolution refined cube checker passes through `t=0.12`
  with peak FVM CFL `1.19`, peak continuity residual `7.44e-5`, accepted
  boundary flux, and solenoidal local transfer.
- Production and reference runs use isolated output directories; source-tree
  sample and backup artifacts remain untouched.
