# Changes to the FVM–VPM coupling algorithm

State of `development` on 12 November 2026. The baseline is commit `10ea19e2` (the code before this
work); `git diff 10ea19e2..HEAD -- openonda/coupler openonda/vpm docs` shows every change. Commit
hashes below name where each change entered.

Status: items 1–8 are tested (coupler and VPM suites) and are the code of the final runs of the
cylinder, the cube and the wing, which started on commit `37d7d7e3` on 12 November 2026. The two
commits after it (`ba514539`, `b50e86fd`) remove analysis scripts and the unused per-transfer
threshold; the transfer they compute is bit-identical. Two settings are still being chosen by
runs: the tolerance of the coupling iterations (item 5) and the outer-band weight (item 4).

## 1. Vorticity transfer: direct, conservative assignment (`92674846` and the commit after it)

Files: `openonda/coupler/transfer_lattice.py`, `openonda/coupler/vorticity_transfer.py`,
`openonda/coupler/config/types.py`, `docs/coupling.md`, `docs/terminology.md`.

Before: on the transfer lattice the particle strengths were corrected toward the FVM vorticity by
one Van Cittert (deconvolution) step,
`Γ_new = Γ^VPM + Γ_mis + γ (Γ_mis − ℳ_σ[Γ_mis])`, `Γ_mis = η (Γ^FVM − ℳ_σ[Γ^VPM])`,
with the Gaussian representation `ℳ_σ`, gain `γ = min(κ_max − 1, 1)` and a limiter at `κ_max`
times the peak target strength. Next to a wall the strengths of a band of two spacings were reset
to zero before the step.

Why it was replaced: `ℳ_σ` loses the part of each Gaussian that falls inside the solid, so the
step creates circulation at a wall. With `σ = h` and `γ = 0.8` a wall sheet on one lattice node
returns 1.17 times the FVM circulation, a sheet on two nodes 1.10, an exponential layer of
thickness `4h` 1.09. The amount depends on the shape of the layer. The wing run stopped at
t = 0.44 s with 1.12 times the FVM circulation next to the wall.

Now (`blend_lattice_strength`): each lattice node holds
`Γ_new = Γ^VPM + η (Γ^FVM − Γ^VPM)`,
where `Γ^FVM` is the FVM circulation of the lattice cube of the node (Stokes integral of the
reconstructed FVM velocity), `Γ^VPM` the M4′-remeshed particle strength and `η` the blending
weight. In the replacement region (`η = 1`) the node takes `Γ^FVM` itself, so the transfer holds
the FVM circulation of every cube exactly, for any layer shape. In the blending ramp the strengths
are blended linearly. No Gaussian representation enters the transfer. The particle-represented
vorticity is therefore the FVM vorticity smoothed once by the core radius `σ = h`; the smoothing
does not accumulate, because the replacement region is rebuilt from the FVM at every transfer.

Unchanged after the assignment: the part of a wall sheet that falls on lattice nodes just inside
the wall is moved to the nearest fluid node; strengths below the removal threshold are removed and
redistributed; the moment correction restores the total strength and the linear impulse, now only
of what the removal changed.

Changed with it: a fluid-connected component with no strength above the removal threshold is
removed as a whole, and its strength counts as removed strength in the budget. Before, the moment
correction could not act on such a component and kept all its strengths. The Gaussian step had
joined all strengths into one component; without it, the roundoff of the Stokes integral in the
uniform stream (about 1e-19 m³/s per node) formed 11,211 isolated components at the first wing
transfer and kept 62,000 particles that carried 1e-14 m³/s in all.

The near-wall circulation-ratio bounds are now (0.97, 1.03), from (0.9, 1.1): only the removal
moves the ratio from one.

Tests (`tests/coupler/test_conservative_transfer.py`):
- the ratio is one for five near-wall layer shapes (sheet on one node, on two nodes, exponential
  layer of 4h, thin layer under an opposite layer, layer peaked at 2h);
- each replaced node holds the FVM circulation of its cube;
- the replaced strengths do not depend on the previous particles;
- the ramp blend;
- roundoff and a weak isolated pocket emit no particle, and the pocket appears in the removed
  strength.

The old transfer gives ratios of 1.013–1.088 and region errors of 0.9–3 % on the same five
layers.

Wing start-up, the saved inputs of the transfers of the run that stopped, old and new transfer on
the same input (the old one reproduces the saved output exactly):

| t (s) | Near-wall emitted − FVM (m³/s), old / new | Region error, old / new | Moment correction, old / new | Particles, old / new |
|---|---|---|---|---|
| 0 | +1.8e-2 / +4e-16 | 1.6e-3 / 4e-17 | 1.9e-5 / 6.1e-8 | 21,928 / 11,446 |
| 0.04 | +3.2e-2 / +4.0e-3 | 9.5e-4 / 1.8e-6 | 1.4e-4 / 2.6e-4 | 33,774 / 27,531 |
| 0.08 | +1.3e-1 / +4.8e-3 | 8.5e-4 / 1.8e-6 | 8.2e-5 / 5.1e-4 | 37,213 / 30,420 |
| 0.12 | +2.7e-1 / +3.8e-3 | 7.7e-4 / 1.8e-6 | 1.6e-4 / 4.4e-4 | 39,750 / 33,496 |

The new near-wall difference is what the removal takes there: 9e-4 of the near-wall L1 norm of 13.1
m³/s. The old one grew to the 12 % that stopped the run at t = 0.44 s. The moment correction is a
few 1e-4 of the strength because it now repairs only the removal.

Removed: `correct_lattice_strength`, the limiter, the wall-band reset, the settings
`max_strength_ratio` (`κ_max`) and `transfer_correction_steps`, the strength-ratio and
representation-residual diagnostics, and the `κ` study of the cylinder tutorial: its runs
(`setup.py`, `allrun.sh`, `run_sensitivity_study.sh`), its analysis (`strength_ratio_history.py` and
the `κ` parts of the other study scripts) and its text (`README.md`, `coupling_findings.md`).

## 2. Wall treatment of the transfer (`1e02b236`, `36bb75ad`, `1fc5ecb0`, `3f3aa6eb`)

- The FVM target of a lattice cube cut by a wall is the Stokes integral over its fluid part, kept
  whole; strength on nodes inside the wall within one spacing is relocated to a fluid node.
- A circulation budget of every transfer compares the emitted strengths with the FVM target, over
  the replacement region and over the nodes within three spacings of a wall
  (`FVMBudget`, `check_fvm_budget`); the run stops when the near-wall ratio leaves its bounds.
- Wall distances are queried only near closed walls (`near_surface`, `signed_distance_near`);
  `TriangulatedWall.encloses_solid` separates a wall around a body from one around the fluid.

## 3. Removal threshold of the transfer (`e07860d6`, `3f3aa6eb`, `a6454f11`, `8965343f`)

- The threshold scales with a reference speed shared by the whole coupler, taken in the mesh frame
  (free stream minus frame velocity; free stream alone at the first transfer).
- The threshold of a coupling interval Δt is `τ_transfer h³ √(U Δt / h)`, so the removal per
  convective time does not depend on the interval. The F0 cylinders at one FVM step per interval
  decided it against a fixed threshold per transfer, which is deleted (with its setting
  `transfer_removal_scaling`).
- The threshold falls linearly with the blending weight from the transfer value to the VPM value
  at the transfer surface.

## 4. Outer band (`cc353b28`)

`band_blending_weight` blends the particles of a band outside the transfer surface toward the FVM
vorticity, to keep the particle wake and the FVM wake from decorrelating there (the cause of the
cube failure at t ≈ 38 s). Not yet run on the cube.

## 5. Coupling iterations and boundary data (`1da0175d`, `8ce69f50`, `3f3aa6eb`)

- The tolerances of the coupling iterations scale with the flow (reference speed).
- The first estimate of an interval extrapolates the boundary-data correction of the previous
  intervals (`boundary_data_extrapolation`), also over unequal intervals. On the cylinder this
  gives 1.01 iterations per interval.
- A restart restores both stored corrections and their intervals (`BACKUP_FORMAT_VERSION = 19`,
  `BOUNDARY_SCHEMA_VERSION = 6`).

## 6. Stops and warnings (`62712c69`, `62bf3c93`, `47e33056`, `384592cd`)

- The run stops or warns when a transfer loses strength.
- The transfer is checked once before the solvers start.
- The velocity mismatch at the outer FVM boundary is measured as an RMS per face side
  (`face_side_rms`); warning at 0.03 and stop at 0.1 of the reference speed
  (`boundary_mismatch_rms_warning`, `boundary_mismatch_rms_limit`).

## 7. Timing (`554e8ff3`, `e99e1ed8`)

`VPMSolver.phase_seconds` times the phases of the particle step when
`LoggingConfig.detailed_timing` is on; the coupler writes them to `coupler_diagnostics.jsonl`
(`timing_seconds.vpm_phases`).

## 8. Cases and post-processing (`716dcc71`, `045615c2`, `f231039f`)

- Cases 04 and 05 no longer override coupler defaults.
- The cube reference forces are corrected for the blockage of its slip walls (Maskell) in
  `02_cube_flow/assets/plot_forces.py`; the READMEs of the cube and the wing state the force basis.

## Results so far

All errors are against the FVM-only reference of the same case (cylinder and cube corrected for
blockage). The final runs are in progress, so the windows are short and the numbers preliminary.

Offline, same saved state through the old (Van Cittert) and the new (direct) transfer:

- Wing start-up, t = 0.12 s: near-wall excess of circulation +2.7e-1 old, +3.8e-3 new; circulation
  error of the replacement region 7.7e-4 old, 1.8e-6 new; particles 39,750 old, 33,496 new.
- Cylinder, t = 22 s: near-wall ratio 1.024 old, 1.008 new; velocity mismatch at the outflow face
  after the transfer 0.724 % old, 0.712 % new (rms, in free-stream speeds). The smoothing of
  item 1 does not raise the boundary mismatch there.

Final runs on `37d7d7e3`, state of 12 November 2026, 20:47:

| Case | Flow time | Error against the reference (window) | Near-wall ratio |
|---|---|---|---|
| Cylinder, N = 5 | 43.4 of 100 s | C_D +0.36 %, St −0.33 %, C_L rms −0.64 % (28.4–43.4 s) | 0.998 |
| Cube | 8.2 of 60 s | drag −0.09 % (3–6 s) | 0.996 |
| Wing | 2.72 of 24 s | lift −0.46 %, drag −0.23 % (1–2 s) | 0.999 |

The wing run with the old transfer stopped at t = 0.44 s (near-wall ratio 1.12); with the new
transfer the ratio is 0.997 there. Before this work the particles next to the wing held about 30 %
of the FVM bound circulation.

Cost: with every core of the machine in use the wing takes about 9,200 wall seconds per flow second
with two coupling iterations per interval (baseline 9,701), so the target of 30 % less wall time is
not met yet; it depends on the tolerance of the coupling iterations.
