# FVM–VPM coupler instability — investigation report

> **Resolved / historical report.** The defect/corrector algorithms discussed
> below were deleted. The production method now performs absolute FVM overlap
> replacement, `Gamma = V_cell * omega_FVM`, followed only by normal VPM
> evolution and its existing GBD diffusion. The final missing accuracy defect
> was a stale boundary-panel solution: new particles were evaluated with panel
> strengths solved from the previous particle state. Fixed-time panel refresh
> reduced fresh-reference Cd error from 6.145% to 1.153% at t=0.05 and 0.844%
> at t=0.10; every measured velocity-profile error is below 5%. The complete
> accepted/rejected experiment ledger is at the repository root in
> `COUPLER_INVESTIGATION_LOG.md`.

Case: `tutorials/coupled_fvm_vpm/cube_flow` (cube at Re = 1000, hybrid LES).
Baseline commit: `50186c9` (working-tree snapshot taken at the start of this work).
Date: 2026-08-24.

Everything below is measured from solver output. Claims that were tested and
**failed** are kept in, marked as such, so they are not re-investigated.

---

## 1. Symptom

The coupled run diverges: VPM velocity and vorticity grow without bound while
the FVM half stays clean. The problem appeared during the solver reorganisation
of the preceding week.

Original diverging run: `solution/` in this directory (started 10:32, killed at
coupling step ~106, t ≈ 1.06 s).

| quantity | value |
|---|---|
| FVM residuals | 1e-8 (velocity), 8e-7 (pressure) |
| FVM max Courant | 0.76, constant |
| FVM total enstrophy | 80.7, flat for the whole run |
| FVM max abs omega | 90.4 |
| VPM total enstrophy | 71.9 → 109 → 144 → 392 → 570 (steps 20, 40, 60, 80, 100) |
| VPM eddy/molecular viscosity ratio | 1.0 → 5.05 |
| VPM max abs omega at t = 1.0 | **1280.1**, at (−0.375, 0.1875, 0.5625) |
| FVM abs omega at that same point | **15** |

There is no single failure step. Enstrophy rises monotonically from the first
coupling step. The FVM is never affected, because the VPM reaches it only
through the outer boundary condition, far from the body.

## 2. Where the error lives

Binned by lattice layer off the cube wall (h = 0.03125), rms abs omega,
VPM particles vs the nearest FVM cell, at t = 1.0:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12 | ≥13 |
|---|---|---|---|---|---|---|---|---|---|---|
| VPM/FVM | 2.32 | 2.77 | **5.82** | **6.02** | 3.49 | 2.52 | 1.99 | 1.39 | 1.09 | 1.07 |

A sharp shell 2–3 cells off the wall, decaying to correct by layer 13.

Spectral content on an 18³ sub-cube where the particle cloud is 100 % occupied
(no zero-fill artefact):

| band abs k / k_nyq | [0,.25) | [.25,.5) | [.5,.75) | [.75,1) | ≥1 |
|---|---|---|---|---|---|
| FVM | 45.3 % | 28.8 % | 12.5 % | 7.8 % | 5.7 % |
| VPM | 7.8 % | 25.4 % | 28.7 % | 25.2 % | 12.9 % |

rms abs omega: FVM 7.96, VPM 50.8. **66.8 % of VPM enstrophy sits above half
Nyquist, versus 26 % for the FVM** (and the FVM figure is inflated by
nearest-cell sampling, so the true contrast is larger).

Per-particle consistency — the vorticity a particle *carries* (Gamma/V) against
the vorticity its own cloud *induces* (curl u from the VPM's `velocity_gradient`):

| | median angle | strength-weighted angle | rms magnitude ratio |
|---|---|---|---|
| t = 1.0 | **68.1°** | 61.6° | **3.99** |

A consistent VPM discretisation requires 0° and 1.0.

## 3. Mechanism (this is the core result)

The transfer corrects the VPM only through a velocity defect:

```
Gamma += h^3 * curl_h[ (I - m2 sigma^2 / 6 * lap_h) * eta * (u_F - u_V) ]
```

Its per-mode loop gain, measured by driving the production routine
`solenoidal_velocity_correction` directly and matching the closed form exactly:

```
G(k) = exp(-sigma^2 k^2 / 2) * [1 + (m2 sigma^2 / 6)(4/h^2) sin^2(kh/2)] * sin(kh)/(kh)
```

Error dynamics are `e_{n+1} = (1 - G) e_n + s`, so the steady-state amplification
of any per-step source `s` is `1/G`:

| lambda/h | 16 | 8 | 6 | 4 | 3 | 2.67 | 2.29 | 2 |
|---|---|---|---|---|---|---|---|---|
| G | 0.937 | 0.758 | 0.597 | 0.278 | 0.081 | 0.035 | 0.006 | 0.000 |
| **1/G** | 1.1 | 1.3 | 1.7 | 3.6 | 12.4 | 28.8 | 159.8 | **inf** |

Both the Gaussian blob factor `exp(-sigma^2 k^2/2)` and the central-difference
curl `sin(kh)/(kh)` drive the gain to zero at grid scale. The coupler is
structurally blind there.

The correction itself is **not** the noise source — its discrete divergence is
4e-17 to 5.6e-17 at every diagnostic step. It is the amplifier.

Nothing supplies a sink. The case runs `StabilizationConfig.bounded_domain`,
which enables particle retention only; the solver's own
`regularization_divergence_trigger` (0.04) and
`regularization_misalignment_trigger` (20°) never fire, while the measured
misalignment is 68°.

Vortex-strength budget per coupling step, from `coupler.log`:
transfer adds ~+0.8, VPM evolution removes ~−0.6, sum(abs Gamma) climbs
monotonically 1.26 → 31.2 over 105 steps. The loop never closes.

## 4. Tested and ruled out — do not re-investigate

| hypothesis | test | result |
|---|---|---|
| Deconvolution/eta step at the body injects the shell | production routine fed a smooth analytic defect with eta = 0 inside a body | no spurious shell; `m2 = 1.5` vs `m2 = 0` differ by 2 % |
| `physics/engine.py` rewrite in `9bc77be` | full diff read | cosmetic renames only (`p` → `parent`) |
| Transfer correction is non-solenoidal | `divergence_correction_l2` every 10 steps | 4e-17 — solenoidal to machine precision |
| GBD threshold pruning injects divergence | FVM field on the same lattice, thresholded at 0.02 / 0.05, divergence on fully occupied stencils only | 0.0012 before, 0.0012 / 0.0004 after — no injection |
| Missing vorticity×volume injection route causes it | head-to-head vs velocity route, error against FVM omega by layer | velocity route only 1.2–1.4× noisier, **uniformly at all layers**, no near-wall concentration; peak injected abs omega 47.5 vs 48.2 (FVM 54.2) |

Note on the last row: an earlier figure of "2.4× L2, max 106 vs 31" was computed
on a lattice that wrongly included nodes inside the solid. Excluding them, the
two routes are nearly equal. The injection route is a genuine regression worth
restoring (§6) but it is **not** the cause.

## 5. Experiments

All variants differ from this case by a single `StabilizationConfig` block and
nothing else. Runs live in the session scratchpad, not in the repo.

| run | change | reached |
|---|---|---|
| expA | none (baseline reproduction) | t = 0.60 |
| expB | pedrizzetti f = 0.3, `preserve_vortex_strength=False` | t = 1.65 |
| expE | pedrizzetti f = 0.3, `preserve_vortex_strength=True` | t = 0.50 (killed) |
| expD | pedrizzetti f = 0.15, `preserve_vortex_strength=False` | t = 1.20 |

### 5.1 Baseline reproduces the divergence

expA enstrophy 12.7 → 35.6 → 75.3 → 100.4 → 113.2 → 123.2 → **145.6**
(steps 5…60), with per-5-step increments *rising* at the end
(4.7, 5.3, 8.2, 14.2). Net sum(Gamma), which must stay ~0, rises 3.6e-3 → 1.6e-2.

### 5.2 A grid-scale sink removes the blow-up

expB internal consistency, versus the original run at the same time:

| metric | original t=1.0 | expB t=0.5 | expB t=1.0 | expB t=1.5 |
|---|---|---|---|---|
| max abs omega (FVM ≈ 90) | 1280.1 | 77.9 | 166.1 | 152.0 |
| median angle carried vs induced | 68.1° | 16.4° | 16.6° | **9.8°** |
| rms ratio carried/induced | 3.99 | 1.75 | 1.63 | **1.61** |
| enstrophy above half-Nyquist | 66.8 % | 25.0 % | 25.4 % | **20.6 %** |

Flat or improving. The numerical blow-up is gone.

### 5.3 …but at an unacceptable accuracy cost

Cd error against the fully meshed reference (`reference_flow/samples`):

| t | reference Cd | original | expA | expB (f=0.3) | expD (f=0.15) |
|---|---|---|---|---|---|
| 0.20 | 1.795 | +10.0 | +14.4 | −15.8 | **−2.6** |
| 0.40 | 1.502 | +8.0 | +15.0 | −12.1 | +10.5 |
| 0.60 | 1.382 | +7.5 | +17.3 | −32.8 | **+3.3** |
| 0.80 | 1.317 | +12.1 | — | −30.1 | −7.8 |
| 1.00 | 1.268 | +8.8 | — | −15.2 | −18.8 |
| 1.20 | 1.203 | — | — | +4.8 | −17.9 |
| **worst** | | **12.1 %** | 17.3 % | **37.5 %** | **18.8 %** |

VPM centreline u_x RMS error vs the same reference (U_inf = 1; the FVM
centreline is ~0.02 in every run, so this is entirely the VPM half):

| t | original | expA | expB | expD |
|---|---|---|---|---|
| 0.40 | 0.3213 | 0.3333 | 0.2623 | 0.3156 |
| 0.60 | 0.2720 | 0.2847 | 0.1328 | 0.2216 |
| 0.80 | 0.2232 | — | **0.0304** | 0.1434 |
| 1.00 | 0.1650 | — | 0.0487 | 0.1085 |
| 1.20 | — | — | 0.0957 | 0.1063 |

### 5.4 Rotation without magnitude removal is actively harmful

expE (`preserve_vortex_strength=True`, zero drain — sum(abs Gamma) change
+5.7e-9 per step) gave Cd +17.1 % → **+48.6 %** → **+68.2 %** at t = 0.3, 0.4, 0.5
and was killed.

Interpretation: the particles carry ~4× the vorticity their own field contains.
Rotating that oversized Gamma onto the induced direction does not remove it, it
makes it **coherent**. Incoherent grid-scale noise largely cancels in the
induced velocity; coherent, physically oriented, 4×-oversized vorticity does
not. **Magnitude removal is the mechanism, not a side effect.**

### 5.5 Conclusion from the sweep

The four runs are one scalar: how much of the Gamma component the induced field
does not support is removed per step. 0 → blow-up; 0.15 → best; 0.30 →
over-diffused; rotation-only → worst. Every setting trades force accuracy
against velocity accuracy along a single line. **No point on that line meets a
"both errors small" requirement.** A different lever is needed.

## 6. Source changes made

Working tree relative to `50186c9`. No solver behaviour changes by default.

| file | change |
|---|---|
| `source/coupler/config/types.py` | `vorticity_transfer_mode: Literal["velocity_defect", "vorticity_defect"] = "velocity_defect"`, validation, `to_dict` entry |
| `source/coupler/interpolation.py` | `FVMVelocityInterpolator.sample_cell_field` — plain IDW on the cached stencil, no Taylor term |
| `source/coupler/vorticity_transfer.py` | `vorticity_defect_correction`, `_sample_vpm_vorticity`, branch in `transfer()`, mode in the `TransferGrid` log, `__all__`, module docstring |
| `tests/coupler/test_physical_coupling_contracts.py` | one attribute (`transfer_mode`) on the hand-built fixture |

Background: commit `bba39d1` ("Improve solver coupling, performance, and
stability") deleted the `handoff_transfer_mode` selector and its
`omega = fvm.get_vorticity_field(); cell_circ = omega * cell_volumes` route,
hard-wiring the velocity route. `vortex_strength_from_velocity_trace` survived
as an unused function exercised only by a test.

The restored route computes, per transfer-lattice node:

```
Gamma_node = h^3 * eta * (omega_F - omega_V)
```

with `omega_F` from the FVM's own `velocity_gradient` via
`_vorticity_from_gradient`, and `omega_V` from `compute_vorticity_at_points`.

Verified: reproduces an analytic donor exactly (max abs error 0.0), reports zero
divergence for a solenoidal donor, 15/15 coupler tests pass, ruff check and
format clean, nomenclature scan passes.

### 6.1 Known defect in the current implementation

It is **not** literally `cell_volume * cell_vorticity` per FVM cell. Three
departures, the third of which is a real weakness:

1. Volume is the lattice cell `h^3`, not the FVM cell volume (the particle at
   that node has `particle_volume = h^3`; near the wall the FVM cells are h/2,
   eight per lattice cell).
2. It is a **defect** `(omega_F - omega_V)`, not an absolute injection, because
   `transfer()` is incremental — an absolute `Gamma = V omega` would re-add the
   whole field every coupling step.
3. It is a **gather** (inverse-distance from 4 donor cells), not a
   volume-weighted **scatter**. A gather does not conserve circulation.

Measured cost of (3) on the step-200 field, transfer box excluding the solid:

| | sum V omega (FVM cells) | sum h^3 omega (lattice gather) |
|---|---|---|
| net x, y, z | −9.6e−06, −6.5e−04, 1.30e−03 | 1.7e−03, −1.3e−02, **6.8e−02** |
| sum abs | 11.06246 | 11.01837 (ratio 0.9960) |

Total magnitude is 0.4 % low, acceptable. The **net circulation is ~50× too
large**. Given §3, that is exactly the kind of input the 1/G → inf amplification
converts into the blow-up, so it must be fixed before the route is judged.

Proposed fix (not yet applied): replace the gather with
`Gamma_node = sum_cells W(x_cell − x_node) * V_cell * omega_cell` using the M4'
kernel already present in `source/solvers/vpm/physics/diffusion/grid.py`. This
makes `sum Gamma` exactly equal `sum V omega`, and matches the deleted code.

## 7. Open questions for the next investigator

1. **Does the vorticity-defect route (with a conservative scatter) break the
   accuracy/stability trade-off of §5.5?** It changes *what* is injected rather
   than *how much* is removed, so it is the only lever identified that moves off
   that line. Untested.
2. **Wake amputation, independent defect.** No particles survive beyond
   x ≈ 1.5 in any run; 0 % of enstrophy lies beyond x = 1.25, while the VPM
   domain extends to x = 12. The GBD absolute threshold
   (`threshold = GBD_VORTICITY_FLOOR * h^3`) deletes the far wake. This caps
   far-field accuracy in every run above and is unrelated to the blow-up.
3. **Should the coupler own the sink?** `1/G → inf` is a property of the
   *transfer operator*, not of the VPM, yet the only cure lives in
   `StabilizationConfig`, which knows nothing about the coupler. At minimum
   `VorticityTransfer.setup()` already knows `h`, `core_radius_ratio` and `m2`
   and could log the amplification table and warn when no VPM sink is active —
   that alone would have surfaced this in the first run.
4. **Parameter regression.** The current settings (`sigma/h = 1.0`,
   `GBD_VORTICITY_FLOOR = 0.02`) give worse Cd than the previous ones
   (`1.1`, `0.05`): 17.3 % vs ~8 % over the same window. Not investigated.

## 8. Reproduction

```sh
cd tutorials/coupled_fvm_vpm/cube_flow
./allrun.sh        # ~33 s per coupling step, 4 FVM ranks + Metal VPM
./allplot.sh png   # figures/ ; needs reference_flow/samples for comparisons
```

Diagnostics used throughout: `solution/vpm.log` (`FLOW DIAGNOSTICS` blocks),
`solution/coupler_diagnostics.jsonl`, `solution/diagnostics.jsonl` (FVM),
`solution/coupler.log` (`[Coupler][VPMState]` for the Gamma budget),
`samples/forces_history.csv`, `samples/vpm_centreline.csv`, and
`reference_flow/samples/{forces_history,centreline}.csv` for the reference.
