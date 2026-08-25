# Native FVM–VPM cylinder shedding (Re = 150) — controlled validation experiment

This case is a controlled validation experiment, not a benchmark.  It tests the
**seed-amplitude hypothesis** of the coupled FVM–VPM solver: a coupled hybrid
calculation and an FVM-only reference share the same linear growth rate
σ and saturated shedding frequency St of the laminar Von-Karman instability,
but the coupling injects a larger initial antisymmetric disturbance A₀.  The
consequence predicted by linear theory is that the hybrid reaches the same
nonlinear onset amplitude earlier, by exactly

```
Δt = (1/σ) · ln(A₀,hyb / A₀,ref).
```

Everything is generated as solver-native data by OpenONDA's internal adaptive
Cartesian mesher and immersed-boundary (IBM) FVM solver — no external mesher,
no STL, no repository path on `PYTHONPATH`.  The single validation-only control
is the deterministic, divergence-free initial perturbation in
`seed_perturbation.py` (switched on with `OPENONDA_SEED_AMPLITUDE`).

## Cases

| directory        | contents |
|------------------|----------|
| `.`              | coupled hybrid FVM–VPM (`cylinder_shedding_flow_setup.py`) |
| `reference_flow/` | FVM-only reference, same IBM cylinder and numerics on a six-patch domain |
| `assets/`        | integrity gate, instability analysis, and figure scripts |
| `seed_perturbation.py` | the shared analytic streamfunction perturbation |

Physical problem: infinite cylinder (spanwise invariant, passes through the FVM
domain), D = 1, Re = 150, U∞ = 1, ρ = 1, ν = 1/150, laminar (SGS disabled in both
solvers).  Local resolution: body 0.0625 D (16 cells/D), wake 0.125 D, background
0.25 D — both cases share these spacings, so every comparative metric stays
consistent.  FVM dt = 0.02 (backward 2nd-order, CFL ≈ 0.32 in the body region),
VPM dt = 0.10, VPM particle spacing 0.125 D with a 1M particle cap, end time = 100
D/U∞ (a long horizon so the slow-growing unseeded reference, whose onset sits
near t ≈ 65, still has a robust ~30-unit saturated window for the frequency
estimate). Both cases run serial by default to keep memory use predictable;
`OPENONDA_FVM_CORES` selects the PETSc-replicated FVM path on larger machines.
The VPM backend remains independently selectable through
`OPENONDA_COMPUTE_DEVICE`.

## Run the full validation

```sh
./allvalidate.sh                    # unseeded
OPENONDA_SEED_AMPLITUDE=1e-4 ./allvalidate.sh
```

This cleans both cases, runs the reference and the hybrid to saturation, runs
the numerical-integrity gate (non-finite fields, unconverged solves, CFL,
continuity, transfer/VPM boundary-condition conservation, open-vortex-line leakage through the
slip z-faces, fine-lattice coincidence), computes the instability report
(σ, A₀, St, A*, t*, Δt_pred vs Δt_meas, verdict), makes all figures, and
archives the run under `runs/<seed>_<tag>_<timestamp>/`.

Exit code 0 means every objective acceptance criterion passed; the verdict is
printed in the report and stored in `solution/analysis_summary.json`.

## Simple interface

```sh
./allrun.sh                         # run the hybrid (clean + simulate)
OPENONDA_SEED_AMPLITUDE=1e-4 ./allrun.sh
./allplot.sh                        # png (or ./allplot.sh pdf)
./allclean.sh                       # remove all generated output + runs/
```

## Smoke test

```sh
OPENONDA_SMOKE=1 ./allvalidate.sh   # coarse, short; pipeline check only
```

## Objectives and acceptance criteria

The seed-amplitude hypothesis is **supported** only when, between the hybrid and
the reference:

- relative growth-rate difference |Δσ|/σ < 15 %
- relative saturated-frequency difference |ΔSt|/St < 5 %
- A₀,hyb > A₀,ref
- the measured onset shift (t*_ref − t*_hyb) matches (1/σ)·ln(A₀,hyb/A₀,ref)
  within ~0.25 shedding periods
- with equal nonzero seeds, the onset offset collapses (both cases carry the
  same initial disturbance)

It is **falsified** if σ or St differ materially, the hybrid does not carry a
larger A₀, equal seeds do not remove the onset offset, the hybrid envelope shows
amplitude jumps at coupling events, or the spatial mode differs from the
reference.

## Outputs

- `solution/` — run metadata, solver and coupler diagnostics, VTK volumes
- `samples/` — force histories, probe/lines/slices (reference names unprefixed,
  hybrid prefixed `fvm_*`, VPM prefixed `vpm_*`)
- `figures/` — `lift_history`, `midspan_probe_growth`, `linear_growth_fit`,
  `onset_alignment`, `shedding_spectrum`, `spanwise_coherence`,
  `vorticity_midspan_tXX` (identical levels across reference/hybrid FVM/VPM)
- `runs/` — one archived copy per validation run
