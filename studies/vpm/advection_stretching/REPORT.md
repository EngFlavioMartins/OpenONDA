# VPM advection–stretching qualification: engineering conclusion

## Decision

**Partial certification only.** Retain common-stage coupled SSPRK3 with pairwise `TRANSPOSED` stretching for isolated self-induced VPM evolution. Do not replace it with sequential advection/stretching or stored-gradient reuse. This is not a repository-wide production-default certification and does not establish that the rotor instability is explained or fixed.

Literature supports treating strength equation, time integration, regularization, and conservation as separate choices. Winckelmans distinguishes classical/direct, transpose, and mixed particle-strength equations and gives the pairwise transpose form a total-vorticity conservation property, while leaving convergence and overlap as separate questions ([thesis, §§3.1–3.4](https://thesis.caltech.edu/697/5/winckelmans-gs_1989.pdf); [JCP 109, 247–273](https://doi.org/10.1006/jcph.1993.1216)). Partitioned-RK theory explains why separately accurate substeps need not retain monolithic order under nonlinear coupling ([ETNA 63, 2025](https://doi.org/10.1553/etna_vol63s171)). The engineering choice below comes from OpenONDA’s executed tests, not from literature preference.

## What the experiments proved

- On manufactured nonlinear deformation and independent discrete-VPM oracles, sequential `x→Γ`/`Γ→x` evolution reduced strength accuracy to approximately first order; common-stage SSPRK3 recovered approximately third order. At the finest discrete oracle point, coupled RK3’s worst relative strength error was `1.68e-9`, versus `8.61e-5` for sequential `x→Γ`.
- On one exact steady Euler/Beltrami reference where `JΓ = JᵀΓ = SΓ`, coupled RK3 converged at order ≈3 for all formulations. At `dt=0.0125`, relative strength errors were `1.45e-7` (`DIRECT`), `1.35e-7` (`TRANSPOSED`), and `1.73e-7` (`MIXED`). This verifies integration of each equation on a common physical solution; the near-tie does **not** select the physically best general equation.
- The complete 14,080-particle leapfrog checkpoint advanced two steps in all three arms. Relative to pairwise RK3, tree-gradient RK3 ended with strength relative L2 error `4.41e-4` (per-particle p95 `2.47e-3`, max `4.74e-3`; worst original particle index `9251`, at `(1.444, 1.236, -0.648)`). Normalized total-strength drift was `4.97e-10` for pairwise RK3 and `1.13e-5` for tree-gradient RK3. Peak physical strength growth was similar, 17.6% versus 17.4%; “least growth” would therefore be the wrong selector.
- The code survey confirms the user’s concern: the direct pairwise stretching kernel regularizes each interaction with the mean target/source core radius, while the velocity-gradient field uses the source blob’s radius. With nonuniform cores they are different discrete operators, not merely algebraically equivalent contractions with different roundoff. On leapfrog’s 64 diagnostic targets, pairwise rates differed from the source-blob `JᵀΓ` reference by 2.14–2.46% over the six RK stages. The tree approximation added a 2.64–2.71% gradient error and 2.43–2.81% rate error against that source-blob reference.
- The complete 70,200-particle rotor checkpoint advanced two steps with tree-gradient RK3 and with current production numerics (RK2, tree gradient, CS, LES, bounds and freestream). Both remained finite. Stage-wise tree-gradient error was 3.09–3.12%, and transposed-rate error was 1.81–2.26% on 64 targets evaluated against all particles. These were **unforced replays**: historical VLM loading and wake insertion were unavailable.

## What the experiments suggest

| complete step, f32 Vulkan | N=4,000 | N=14,000 | N=35,000 | N=70,200 |
|---|---:|---:|---:|---:|
| pairwise TRANSPOSED, RK3 isolated | 3.01 s | 22.99 s | 113.23 s | skipped; 455.51 s projected |
| tree-gradient TRANSPOSED, RK3 isolated | 2.02 s | 11.85 s | 49.08 s | 125.31 s |
| current rotor numerics, unforced | 2.05 s | 12.27 s | 50.20 s | 96.40 s |

On this hardware and current complete-step path, tree-gradient stretching is already faster at 4,000 particles; no production-scale crossover in favour of pairwise evaluation was observed. This does not contradict the earlier isolated-evaluator microbenchmark: the current coupled step marks gradients as required unconditionally and performs fused tree velocity/gradient work before adding pairwise stretching. Removing that unused gradient work is now a specific optimization target, not a reason to change the equation or integrator.

The rotor’s short replay and stage values (`max χs≈0.0625`, `max χΓ≈0.152`) do not reveal a temporal threshold. They support monitoring stage envelopes, but do not validate `χs=0.2` as a stability boundary.

## What remains untested

- The 70,200-particle pairwise rotor replay was rejected by the predeclared 120 s/step feasibility gate after a measured 35,000-particle step projected 455.51 s/step. No pairwise rotor result is inferred.
- No replay includes historical VLM/body forcing or particle insertion, so the rotor instability is neither reproduced, explained, nor fixed.
- Two-step survival is not a long-horizon stability result. Projection ablation, full forced replay, wake-overlap convergence, and production-default qualification remain open.
- `TRANSPOSED` is provisionally selected for the isolated leapfrog use case because of temporal accuracy and pairwise conservation, not proven universally more physically accurate than `DIRECT` or `MIXED`.

Reproducible records are in `results/formulation_comparison.csv`, `results/full_checkpoint_replay.csv`, `results/full_replay_stage_summary.csv`, `results/full_replay_comparisons.csv`, and `results/scale_timing_*.csv`. Instrumented replay wall times include independent f64 stage diagnostics; only `scale_timing_*.csv` contains uninstrumented complete-step timings.
