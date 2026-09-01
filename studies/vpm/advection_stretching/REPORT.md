# VPM Advection–Stretching Numerical Qualification

Study commit: `48d72e997337b4f6aaeeceb325d1367876e29514` (the manifest records the pre-study dirty tree).

## Decision

**For self-induced vortex-interaction cases, retain common-stage coupled SSPRK3 with exact pairwise TRANSPOSED stretching; do not replace it with a stored-gradient proxy.** The current vortex-interactions setup already selects this path. Do not promote it to the repository-wide default until coupled FVM/VLM cases are qualified. Coupled RK4 is a research reference, not a production selection.

The reason is reference error, not suppressed growth: sequential x→Γ and Γ→x splitting is only first order on the nonlinear closed deformation, whereas common-stage RK3 recovers approximately third order. The stage-gradient reuse method matches coupled RK3 only for prescribed flows; self-induced tests expose its frozen-source approximation. Exact pairwise TRANSPOSED remains preferred when conservation is important; accumulated/tree gradients are a performance alternative whose tolerance and cancellation error must be accepted explicitly.

## Algorithms and implementation audit

- `fractional_x_gamma`: SSPRK3 advection with Γ frozen at Γn, followed by SSPRK3 strength evolution with x fixed at x(n+1).
- `fractional_gamma_x`: the reverse sequential split.
- `parallel_lagged`: both complete subsolves start from (xn, Γn).
- Strang candidates use the two explicit half/full/half orderings.
- Coupled RK2/RK3/RK4 evaluate u and the selected strength equation at identical stage states and stage times.
- `reuse_stage_gradients` freezes source Γ during the advection field stages, then reuses those J samples; it is not monolithic RK.
- The exponential candidate applies exp(Δt Javg) particle by particle with Simpson-like stage weights.

Executed stage states are in `results/stage_ledger.csv`; the implementation trace and unused-work finding are in `results/implementation_audit.json`. The nonsymmetric-shear orientation test and f64 pairwise-versus-gradient identity passed to roundoff.

## Literature context

Winckelmans’ primary derivation distinguishes classical, transpose and mixed particle-strength equations because a finite regularized particle vorticity field is not generally divergence-free; the formulations are therefore not interchangeable after discretization. It identifies exact total-vorticity conservation and a weak-solution property for the pairwise transpose scheme, while warning that convergence and particle overlap remain separate concerns ([Winckelmans thesis, §§3.1–3.4](https://thesis.caltech.edu/697/5/winckelmans-gs_1989.pdf)). The later Winckelmans–Leonard JCP paper likewise treats strength formulation, regularization, diffusion and conservation as distinct design questions ([JCP 109, 247–273](https://doi.org/10.1006/jcph.1993.1216)).

A documented three-dimensional vortex particle–panel implementation selected TRANSPOSED for total-vorticity conservation and advanced the particle equations with a stated second-order Adams–Bashforth treatment ([implementation report](https://citeseerx.ist.psu.edu/document?doi=4335baa8a42916e85156195681628cc646e1c12a&repid=rep1&type=pdf)). This does not validate OpenONDA’s sequential RK split. Modern partitioned-RK analysis explains why: nonlinear coupling creates additional mixed order conditions, so reusing stages from one partition does not automatically inherit monolithic order ([Tran, Southworth & Buvoli, ETNA 63, 2025](https://doi.org/10.1553/etna_vol63s171)). Literature therefore motivates TRANSPOSED and common-state integration, but the OpenONDA choice below comes from the executed evidence.

## Manufactured-flow convergence

| method | observed px | observed pΓ | finest EΓ | finest Ex |
|---|---:|---:|---:|---:|
| coupled_rk4_reference | 4.96 | 4.96 | 2.101e-08 | 2.211e-08 |
| strang_gamma_x_gamma | 2.95 | 2.97 | 3.593e-06 | 1.694e-05 |
| averaged_gradient_exponential | 2.95 | 2.95 | 6.343e-06 | 1.694e-05 |
| coupled_rk3 | 2.95 | 2.95 | 1.712e-05 | 1.694e-05 |
| reuse_stage_gradients | 2.95 | 2.95 | 1.712e-05 | 1.694e-05 |
| strang_x_gamma_x | 2.99 | 2.95 | 1.717e-05 | 2.125e-06 |
| coupled_rk2 | 2.92 | 2.91 | 3.713e-05 | 3.642e-05 |
| fractional_gamma_x | 2.95 | 1.02 | 5.184e-04 | 1.694e-05 |
| parallel_lagged | 2.95 | 1.02 | 5.184e-04 | 1.694e-05 |
| fractional_x_gamma | 2.95 | 1.05 | 5.202e-04 | 1.694e-05 |

The nonlinear map is an exact volume-preserving composition of shears and returns to identity after one cycle. Its inverse, determinant, divergence and gradient were checked numerically to roundoff. Closed cycles can superconverge for symmetric schemes, so the time-varying rotating-strain and random histories are retained as noncommuting controls.

At 16 steps across 128 deterministic random trace-free histories:

| method | median EΓ | worst EΓ |
|---|---:|---:|
| coupled_rk4_reference | 1.784e-05 | 4.907e-05 |
| strang_gamma_x_gamma | 1.222e-04 | 4.748e-04 |
| averaged_gradient_exponential | 5.073e-04 | 2.403e-03 |
| coupled_rk3 | 9.330e-04 | 3.612e-03 |
| parallel_lagged | 9.330e-04 | 3.612e-03 |
| fractional_x_gamma | 9.330e-04 | 3.612e-03 |
| reuse_stage_gradients | 9.330e-04 | 3.612e-03 |
| fractional_gamma_x | 9.330e-04 | 3.612e-03 |
| strang_x_gamma_x | 9.330e-04 | 3.612e-03 |
| coupled_rk2 | 5.950e-03 | 1.447e-02 |

## Production operating envelope

| checkpoint | N | max χs | max χr | max χx | max χΓ | median h/σ |
|---|---:|---:|---:|---:|---:|---:|
| leapfrog_healthy_050 | 14080 | 0.124 | 0.313 | 5.1 | 0.172 | 0.266 |
| leapfrog_late_150 | 14080 | 0.0893 | 0.243 | 12 | 0.162 | 0.182 |
| rotor_healthy_515 | 69525 | 0.0745 | 0.166 | 14.8 | 0.126 | 0.295 |
| rotor_prefailure_520 | 70200 | 0.0761 | 0.174 | 7.52 | 0.144 | 0.295 |
| rotor_rejected_520 | 70200 | 0.0761 | 0.174 | 7.52 | 0.144 | 0.295 |

| rotor checkpoint | stored-tree/exact-J relative L2 | stored/exact TRANSPOSED rate relative L2 |
|---|---:|---:|
| rotor_healthy_515 | 3.218e-02 | 2.718e-02 |
| rotor_prefailure_520 | 2.883e-02 | 1.936e-02 |
| rotor_rejected_520 | 2.883e-02 | 1.936e-02 |

The ring checkpoints lack stored gradients, so their J metrics use a deterministic 256-target independent f64 evaluation against every source. Rotor checkpoints use the stored production gradient for every particle; the comparison uses the 64 strongest plus 64 spatially spread targets against all sources. Exact O(N²) checkpoint energy and projection corrections were not present in the files; these are explicitly inconclusive rather than fabricated.

## Discrete VPM oracle and checkpoint replay

| method | median EΓ | worst EΓ |
|---|---:|---:|
| coupled_rk4_reference | 1.025e-14 | 2.381e-12 |
| coupled_rk3 | 1.930e-11 | 1.682e-09 |
| strang_x_gamma_x | 7.334e-09 | 2.454e-07 |
| strang_gamma_x_gamma | 7.029e-09 | 2.555e-07 |
| coupled_rk2 | 2.474e-08 | 7.300e-07 |
| averaged_gradient_exponential | 7.724e-06 | 5.899e-05 |
| reuse_stage_gradients | 7.724e-06 | 5.900e-05 |
| fractional_gamma_x | 5.668e-06 | 8.609e-05 |
| fractional_x_gamma | 5.669e-06 | 8.611e-05 |
| parallel_lagged | 5.930e-06 | 8.619e-05 |

Production-Δt isolated local replay (external forcing was unavailable):

| method | median EΓ | worst EΓ |
|---|---:|---:|
| coupled_rk4_reference | 2.075e-09 | 2.191e-09 |
| coupled_rk3 | 1.920e-07 | 2.645e-07 |
| strang_x_gamma_x | 5.075e-06 | 5.309e-06 |
| strang_gamma_x_gamma | 6.418e-06 | 7.334e-06 |
| coupled_rk2 | 1.542e-05 | 2.084e-05 |
| reuse_stage_gradients | 2.012e-04 | 2.349e-04 |
| averaged_gradient_exponential | 2.013e-04 | 2.351e-04 |
| fractional_x_gamma | 4.002e-04 | 4.360e-04 |
| fractional_gamma_x | 4.038e-04 | 4.393e-04 |
| parallel_lagged | 4.092e-04 | 4.452e-04 |

Each DIRECT/TRANSPOSED/MIXED candidate is compared with DOP853 integrating that same semi-discrete equation. Tightened DOP853 and a fixed 640-substep RK4 cross-check are in `oracle_verification.csv`. Replay results are classified as isolated local neighbourhoods: absent recorded body/coupling forcing prevents a scientifically valid forced replay.

## Conservation and evaluator cost

Raw, unprojected total strength, linear impulse, kernel-corrected angular impulse and kinetic-energy drift are in `conservation.csv`. For TRANSPOSED cases, exact-pair coupled RK3 retained total strength to 1.6e-16 relative drift, while frozen-source accumulated-gradient reuse reached 5.6e-5. This shows that algebraic equivalence in exact arithmetic does not preserve pairwise cancellation under a different accumulation/update path. No projection result is presented because no finalist projection implementation was introduced into the independent harness; production projection remains a separate ablation.

| N | tree/direct rate relative L2 | direct net-rate norm | tree net-rate norm |
|---:|---:|---:|---:|
| 256 | 2.441e-02 | 3.970e-08 | 4.685e-03 |
| 1024 | 3.375e-02 | 2.208e-07 | 1.563e-02 |
| 4096 | 5.216e-02 | 1.892e-06 | 1.057e-01 |

| N | evaluator | median [ms] | dispersion [ms] |
|---:|---|---:|---:|
| 256 | direct | 0.954 | 1.342 |
| 256 | tree | 30.524 | 6.025 |
| 1024 | direct | 3.862 | 2.222 |
| 1024 | tree | 43.643 | 6.925 |
| 4096 | direct | 11.717 | 4.558 |
| 4096 | tree | 89.808 | 15.220 |

Timings use warmed production Taichi f32 kernels, 10 repetitions, and explicit device synchronization. Candidate operation counts come from executed study code. Tree and direct results are not called mathematically identical: tree opening error and loss of exact pairwise accumulation are reported.

## Practical criterion

Treat `χs = Δt ||S||₂` and `χΓ` as accuracy controls, not universal stability theorems. The random prescribed histories kept coupled-RK3 strength error below 1% through the sampled χs=0.307, and exact-pair local checkpoint replay stayed below 4.1e-6 even at 2× production Δt. Yet the rejected rotor snapshot has χs=0.076 and χΓ=0.144. Therefore no scalar temporal threshold in this evidence separates the rotor event: the current 0.2 warning may remain as a warning, but it is not a safety guarantee. Keep a conservative stage-wise target χs≤0.2 and χΓ≤0.2 while separately tightening and qualifying the rotor tree-gradient evaluator. A future production change should record all common RK stages because a beginning-of-step value cannot bound the stage maximum.

## Failures and limitations

- Forced checkpoint replay was impossible because historical body/FVM/VLM forcing and insertions were not recorded.
- The replay is a 32-particle nearest-neighbour extraction, not the full 70,200-particle rotor state.
- Exact checkpoint energy was not computed because it is O(N²) and unavailable in the backup.
- Projection-enabled finalist ablation was not implemented; raw conservation remains the selection evidence.
- No production default was changed. These limits prevent claiming full architecture certification, but not the temporal-order conclusion.

## Files and commands

Changed only the study directory plus five compact permanent tests. The existing dirty tracked files were preserved untouched. Commands are encoded in `allrun.sh`: the manufactured, discrete-cloud, checkpoint-replay, production-envelope, performance, and plotting runners, plus `/home/flavio-martins/anaconda3/envs/OpenONDA/bin/python -m pytest -q tests/vpm/test_advection_stretching_qualification.py`.

Figures: error–step size, error–wall time, sampled χs boundary, excess growth relative to reference, raw invariant drift, work decomposition, Pareto view, production envelope, and direct/tree kernel cost are under `figures/`.
