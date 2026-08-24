# FVM–VPM Coupler Investigation Ledger

Last updated: 2026-08-24

This file is the durable record of coupling experiments. A setting is retained
only when a controlled comparison improved the result or fixed a demonstrated
contract violation. Rejected settings are restored to the stated baseline.

## Accepted algorithm

The only FVM-to-VPM transfer algorithm in the code is absolute overlap-state
replacement:

1. For every fluid FVM cell in the transfer region, create one particle at the
   cell centre with `Gamma = cell_volume * FVM_vorticity` and with the actual
   cell volume.
2. For hard replacement (`eta_blend_width = 0`), delete all existing particles
   in the transfer region before injecting the FVM particles. Particles outside
   the region are not modified.
3. For a positive `eta_blend_width`, use a C1 cosine `eta` ramp at the transfer
   box boundary. Existing and FVM states are weighted by `1-eta` and `eta`.
   Thus the knob enables/disables blending without an additive corrector.
4. Advance the VPM normally. Its existing GBD diffusion/regeneration remains
   responsible for producing the next regular particle cloud. No extra
   remeshing/regeneration pass is performed by the coupler.
5. Resolve the boundary-only panel potential against the current particle
   state before every FVM boundary trace. This is a fixed-time harmonic-state
   refresh; it does not modify particle circulation, advance panel history,
   shed wake, or evolve geometry.

Deleted production algorithms include velocity-defect correction,
conservative-vorticity-defect correction, hybrid-defect correction,
interface-flux injection, scatter/lattice defect transfer, tail-budget logic,
and their configuration modes. They are retained below only as experiment
history so they are not repeated.

## Retained baseline values

| Quantity | Retained value | Evidence |
|---|---:|---|
| FVM domain | `(-3, 3)^3` | Standalone box-convergence error was 0.78%/1.09% in Cd at t=0.05/0.10; the original `(-1.5,1.5)^3` box gave 11.94%/13.42% |
| Refined and transfer box | `(-1.25,1.25)^3` | Original refined region retained; no unsupported xmax/refinement drift |
| FVM outer cell size | `0.25` m | Box-convergence change only; near-body and transfer resolution unchanged |
| FVM time step | `0.005` s | Matches the fresh validation reference |
| VPM/coupling step | `0.010` s | Two exact FVM substeps |
| Boundary mode | `vorticity_mixed` | Dirichlet replacement control changed Cd error 6.145% to 6.150%; no benefit, reverted |
| `eta_blend_width` | `0.0` m | Hard replacement/blending off is the validated baseline; on/off and partition behavior are unit tested |
| Core-radius ratio | `1.0` | Literal cell-state replacement baseline |
| GBD threshold | `0.02*h^3`, absolute | Existing physical VPM diffusion; not retuned during replacement validation |
| Pressure anchoring | off | Pressure datum cannot alter a closed-body pressure force |
| Post-transfer boundary resync | on | Required so the next FVM interval starts from the replaced current state |

## Root causes established by evidence

| ID | Root cause | Evidence | Fix |
|---|---|---|---|
| R1 | Additive defect algorithms retained/added circulation in the overlap instead of imposing one state | Earlier runs showed monotonic VPM enstrophy and `sum(abs(Gamma))`, growing correction clouds, and eventual blow-up; direct conservative variants were contractive but produced 11–59% Cd errors | Delete all defect modes; absolute delete/reinject state replacement |
| R2 | The velocity-defect loop was blind to grid-scale vorticity | Measured transfer gain reaches zero at Nyquist; the old VPM field carried 66.8% of enstrophy above half-Nyquist and reached `max(abs(omega))=1280` versus about 90 in FVM | Removed rather than patched |
| R3 | The original FVM box was too small | Uncoupled `+/-1.5` box differed from the full reference Cd by +11.94%/+13.42% at t=0.05/0.10 | Retain only the evidence-backed `+/-3` coarse outer box |
| R4 | VPM samplers ran before same-time FVM replacement | Before the fix, VPM centreline max error at t=0.05 was 10.10%; sampling the replaced state reduced it to 4.87% with no transfer change | Coupled VPM advance defers scheduled output; coupler samples after replacement and resync |
| R5 | Boundary-only panel strengths were stale relative to the evaluated particle state | Code solved the panel before VPM evolution, then evaluated boundary velocity after evolution/GBD; after FVM replacement it combined new particles with the same old panel strengths. Fixed-time refresh reduced Cd error at t=0.05 from 6.145% to 1.153%, with viscous force and transfer budgets unchanged | Refresh the panel solve from current particles before each boundary trace |
| R6 | The checked-in reference was stale relative to current FVM time step/code | Fresh `dt=0.005` reference Cd is 2.907174/2.172972 at t=0.05/0.10; old checked-in reference was 3.38734/2.27656 | All acceptance numbers below use the fresh reference |

## Controlled experiments and disposition

Drag errors use the fresh `dt_FVM=0.005` fully meshed reference unless noted.

| Experiment | One controlled change | Result | Disposition |
|---|---|---|---|
| Original velocity-defect baseline | None | Good early results with visible time lag; additive high-frequency error accumulated and blew up near t=1 | Rejected/deleted |
| Pedrizzetti magnitude removal, factor 0.30 | Add grid-scale sink | Stable to t=1.65, worst Cd error 37.5% | Reverted/deleted |
| Pedrizzetti magnitude removal, factor 0.15 | Weaker sink | Worst Cd error 18.8% | Reverted/deleted |
| Pedrizzetti rotation only | Preserve strength magnitude | Cd error reached 68.2% at t=0.5 | Reverted/deleted |
| Direct conservative M4-prime defect, `+/-1.5` | Replace velocity correction by `V*omega` defect | Stable/contractive through t=0.2, Cd errors about +59%/+58% at 0.05/0.10 | Rejected/deleted |
| Velocity defect, current code, `+/-1.5` | Control | Cd +6.24%/+16.74%; correction grew to 1.616 by t=0.10 | Rejected/deleted |
| Dirichlet boundary, old defect algorithm | Boundary mode only | Cd 3.0901 versus 3.0885 at t=0.05 | Reverted |
| Downstream bound `xmax=3.5` only | Extend outlet | Cd 3.072 at t=0.05, worse than control | Reverted |
| FVM step 0.010 | Time step only | Cd error -6.55% at t=0.10; correction still grew | Reverted to 0.005 |
| Hybrid defect, `+/-1.5` | Low-band velocity plus high-band conservative defect | Cd +18%/+28.8% at 0.05/0.10 | Rejected/deleted |
| Uncoupled FVM box `+/-2.5` | Domain convergence | Cd +1.97%/+2.34% | Improved but not retained |
| Conservative defect, box `+/-2.5` | Transfer control | Correction contracted 0.568 to 0.432; Cd +11.07%/+11.25% | Rejected/deleted |
| Hybrid defect, box `+/-2.5` | Restore resolved response | Cd +2.87%/+5.37% fresh | Outside gate; rejected/deleted |
| Uncoupled FVM box `+/-3` | Domain convergence | Cd +0.78%/+1.09% | Retained geometry |
| Hybrid defect, box `+/-3`, 1% tail | Old best candidate | Cd +1.39%/+2.98%; faster but still additive algorithm | Deleted at user request |
| Absolute replacement, stale sampler/panel | Requested method, first control | Stable to t=0.10; Cd 6.145%/6.406%; VPM centreline max 10.10%/6.41% | Exposed R4 and R5 |
| Post-replacement sampling | Output order only | At t=0.05, VPM centreline max 10.10% to 4.871%; FVM max 4.882%; Cd unchanged at 6.145% | Retained |
| Dirichlet boundary with absolute replacement | Boundary operator only | Cd 2.728369, 6.150% error versus 6.145% baseline; profiles unchanged | Reverted to `vorticity_mixed` |
| Skip initial t=0 replacement | Startup order only | Cd error 6.145% to 6.132%; profiles unchanged | Insufficient; initial replacement restored |
| Current-particle panel refresh | Harmonic state timing only | Cd error 6.145% to 1.153% at t=0.05; pressure force moved 0.8261 to 0.8985 versus 0.9145 reference; profiles improved slightly | Retained |
| Repeat corrected run to t=0.10 | Stability/reproducibility | t=0.05 Cd repeated within `1.2e-7`; all ten steps stable; metrics below | Accepted |

## Accepted accuracy results

Trial: `/private/tmp/openonda_cube_replacement_panel_refresh_t010_trial`

Reference: `/private/tmp/openonda_reference_dt0005_t1_20260824`

| Time | Metric | Error |
|---:|---|---:|
| 0.05 | Cd | 1.153% |
| 0.05 | FVM centreline max / mean | 4.882% / 0.852% |
| 0.05 | VPM centreline max / mean | 4.805% / 0.276% |
| 0.05 | FVM off-axis max / mean | 1.520% / 0.349% |
| 0.05 | VPM off-axis max / mean | 0.910% / 0.178% |
| 0.10 | Cd | 0.844% |
| 0.10 | FVM centreline max / mean | 3.505% / 0.715% |
| 0.10 | VPM centreline max / mean | 2.310% / 0.225% |
| 0.10 | FVM off-axis max / mean | 1.656% / 0.307% |
| 0.10 | VPM off-axis max / mean | 1.066% / 0.160% |

All measured Cd and velocity-profile metrics at both validated sample times are
below the 5% acceptance limit.

## Verification commands

```text
pytest -q tests/coupler tests/vpm/test_global_regeneration_threshold.py
ruff check source/coupler tests/coupler tutorials/coupled_fvm_vpm
pyrefly check --python-version 3.11 --search-path /opt/anaconda3/envs/OpenONDA/lib/python3.11/site-packages source/coupler
python tutorials/coupled_fvm_vpm/cube_flow/assets/measure_trial_errors.py <trial> <reference>
```

Taichi emitted cache-lock warnings because concurrent processes could not write
its user cache. Tests and simulations completed successfully; the warnings did
not affect solver state or metrics.
