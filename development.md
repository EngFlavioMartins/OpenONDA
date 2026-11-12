# Changes to the FVM–VPM coupling algorithm

State of `development` on 12 November 2026. The baseline is commit `10ea19e2` (the code before this
work); `git diff 10ea19e2..HEAD -- openonda/coupler openonda/vpm docs` shows every change. Commit
hashes below name where each change entered.

Status: items 2–8 are tested (coupler and VPM suites) and were used in the cylinder runs. Item 1 is
committed as work in progress: its tests and its comparison against saved flow states are not
finished, and no case has run with it yet.

## 1. Vorticity transfer: direct, conservative assignment (in progress)

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

Removed: `correct_lattice_strength`, the limiter, the wall-band reset, the settings
`max_strength_ratio` (`κ_max`) and `transfer_correction_steps`, the strength-ratio and
representation-residual diagnostics, and the `κ` study of the cylinder tutorial (`setup.py`,
`allrun.sh`, `README.md`). Analysis scripts of that tutorial that read the removed fields are still
to be cleaned.

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
- `transfer_removal_scaling` selects `convective` (threshold tied to the convective time) or
  `per_transfer`. The cylinder runs so far favour `convective`; the other is to be deleted.
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

- Cylinder, revised code without item 1, `convective` threshold, one FVM step per transfer, window
  t = 10–21.8 s (short): C_D +0.49 %, St −0.46 %, C_L rms −0.22 % against the blockage-corrected
  FVM-only reference.
- Wing: stopped at t = 0.44 s for the reason given in item 1.
- Cube: not yet run on the revised code.
