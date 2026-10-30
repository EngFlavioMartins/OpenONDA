# Coherence audit of 28 October 2026: solver, thesis Chapter 4 and Paper I

Scope: the OpenONDA coupler (development tree of 28 October 2026), thesis `chapters/ch04_hybrid_solver.tex` (section 4.7 in particular), the Part I manuscript, and the records of the cylinder study in `tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow` (`README.md`, `assets/coupling_findings.md`, `docs/plans/2026-10-19-coupling-error-sources.md`, `docs/reports/2026-10-23-coupling-error-sources.md`, `figures/auxiliary/*.csv`, the stored `coupler_diagnostics.jsonl` and particle frames of the N = 5 and N = 2 runs). Terms follow `OpenONDA/docs/terminology.md`.

The audit answers three questions. Does the manuscript describe the solver as it is and the study as it was done? Does the cylinder study isolate the parameters that thesis section 4.7 says must be isolated, and are its rules flow-agnostic? What has to change in each document so that the three read as one work?

## 1. Summary

1. **The algorithm is described consistently** in the code, Chapter 4 and the paper (defaults `L_inset = 8.75h`, `δ_0 = 2h`, `δ_1 = 6h`, `τ_transfer = 0.05`, `κ_max = 1.8`, `γ = min(κ_max − 1, 1)`, moment-correction limit 8 %, `M = 6`, tolerances 1e-5, `L_remesh = 1.5 U∞ Δt_VPM + 2h`, lattice extended two nodes). I checked every method statement of the paper against `coupler/config/types.py`, `transfer_lattice.py`, `vorticity_transfer.py`, `coupling_iteration.py` and `boundary.py`. One inconsistency: `parts/declarations.tex` still cites the September commits (`dfa6b54f`, `66964084`, `45da734c`) while `parts/cylinder.tex` and the report give `3b4c0f23`–`97d65b09`. Fixed.
2. **Section 4.7 of the thesis and the executed study do not yet match.** Section 4.7 asks for four levels per refinement sequence, for the FVM step at a fixed coupling interval, the particle spacing and the coupling interval at a fixed FVM step as separate sequences, and for the transfer limits to be shown not to act. The study has three spatial levels (sign change, no order), two FVM-step levels, no particle-spacing series, no series over `δ_0`, `δ_1`, `κ_max`, `τ_transfer`, and, as found below, the strength limit acts in the runs with short coupling intervals. Section 3 lists the gaps and section 4 the runs that close them.
3. **New finding from the stored diagnostics (to be verified by the author).** In the N = 2 run the strength ratio `κ` sits at its limit `κ_max = 1.8` in about 80 % of the transfers of the developed flow, whereas the baseline never reaches it (`κ ≤ 1.66`). The coupling-interval series therefore confounds the number of transfers per convective time with the action of the strength limit at the wall. Section 3.2.
4. **New finding from the stored particle frames (to be verified).** The circulation and the peak vorticity of the shed vortices against their streamwise position are the same in the N = 2 and the N = 5 runs within about 1 % from 2D to 15D. The per-interval error is therefore not an accumulated smoothing of the wake vortices, and the GBD remeshing of the free wake is excluded as its cause. The error enters the FVM through the boundary data on `Σ_FVM`, from the particles near the body. Section 3.2.
5. **Flow-agnostic calibration is not established and the documents should not imply it.** Two Reynolds numbers in one regime, whose vorticity thicknesses differ by 23 %, do not decide between `δ_ω` and `D` as the lateral scale, and the downstream reach shows no common trend. The paper already says this; the OpenONDA `docs/coupling.md` states the rules more confidently than the evidence allows (section 7).
6. **Symbols.** The paper used `Π`, `Π_ref`, `ΔΠ_ref`, `e_VPM`, `e_interface` and `Λ = Δt_VPM max_p‖S(x_p)‖_∞`; the thesis and the terminology file use `Q`, `Q_ext`, `U_ext`, `e = e_FVM + e_coupling` and `Λ = Δt_VPM‖S‖_{1,∞}`. The paper now follows the terminology file (section 5).
7. **Paper text.** The manuscript was consistent with the evidence but read as an enumerated argument (three requirements, three questions, five contributions, three design decisions, seven steps, each announced before it is stated), with few sentences that a reader could not predict. The revision of 28 October rewrites the abstract, introduction, test programme, cylinder, discussion and conclusions in a plainer register, keeps every number, and states the two findings above where they change a conclusion (section 6).

## 2. What section 4.7 prescribes and what the study did

| Section 4.7 prescription | Case-01 study (19–25 Oct) | Status |
| --- | --- | --- |
| Error of a statistic measured against `Q_ext`, the FVM-only value extrapolated to zero cell size and step, with Eça–Hoekstra least squares; order only for monotone data | Done for `C̄_D`, `Ĉ_L`, `St` at Re = 150 (families C, X, T; space–time fit over 10 levels for `St`) | Done. At Re = 250 and 100 no extrapolation: errors normalised by `Q_FVM(h, Δt)` |
| Split `e = e_FVM + e_coupling` | Done for every run at Re = 150 | Done |
| FVM cell size sequence, ≥ 4 levels, other parameters fixed | Family XP: 3 levels at fixed `Δt_VPM = 0.04`; the 4th (h = 0.0283) inadmissible at that interval (Λ = 1.03); family X at fixed N = 4 (3 levels) | **Partial.** No order for the coupled solver. Needs the 4-level sequence at an interval admissible on the finest cells (section 4, run A) |
| FVM step at fixed `Δt_VPM`, ≥ 4 levels | Family P: 2 levels (N = 5, 10) | **Partial.** Shows `e_coupling` independent of `Δt_FVM` (0.07 points), no temporal order of the coupled solver |
| Particle spacing sequence (`h_VPM/δ_ω`, `σ/h_VPM`) | Family S not run | **Not done** |
| Coupling interval at fixed FVM step, ≥ 4 levels | Family E: N = 2, 4, 5 (3 levels; N = 8 inadmissible, N = 6 not run); plus T2 and P | **Partial**, and confounded with the strength limit (section 3.2) |
| Coupling-iteration count and tolerances compared; smallest count after which the state no longer changes | Family K: 1, 2, 4, tolerance | Done |
| Interface geometry in flow lengths (`ℓ_side/δ_ω`, `ℓ_wake/L_f`, `L_inset/h`, `δ_0/h`, `δ_1/h`) | B (lateral, downstream, Re 150 and 250), inset 4/8.75/14h | **Partial.** `δ_0`, `δ_1` never varied |
| Transfer limits (`κ_max`, `τ_transfer`, moment fraction) set so that they rarely act; a limit that acts at most intervals belongs in the parameter study | Baseline: `κ ≤ 1.66`, removal 0.3 %, moment correction 0.11 % per transfer. **N = 2: `κ = 1.80` in 80 % of the transfers** | **Open.** By the thesis's own rule, `κ_max` must enter the study (run B) |
| In-run indicators compared with `e_coupling` over all runs | Done (mismatch r = 0.82–0.84 for `Ĉ_L`; residuals no indication) | Done |
| Refining cells and steps together at fixed N is not sufficient | Family C shows the cancellation | Done |
| Induction and iteration tolerances tightened until the data no longer change | Iteration: K study. Induction tolerance (multipole/image sums): not varied | **Not done** for the induction tolerance; state it as assumed |

## 3. What the study isolates

### 3.1 Family by family

- **K (coupling iterations).** Clean. One parameter, everything else fixed. One iteration changes the drag by +3 %, two converge within 0.15 points.
- **P (FVM step at fixed coupling interval).** Clean as a null result on two levels (`e_coupling` changes by ≤ 0.07 points). It does not give a temporal order of the coupled solver and the paper should not say "the coupled solution then follows the temporal convergence of the FVM" as a measured fact (now stated as expected, with the two levels named).
- **E, T2 (coupling interval).** Varying `Δt_VPM` changes at once the particle time step, the number of transfers per convective time, the number of GBD steps, the remeshing extension `L_remesh`, the span of the linear boundary-data interpolation, and, as found below, whether the strength limit acts. The direction of the trend (error grows as the interval shrinks) excludes a time-truncation error of the particle advance as the dominant term. It does not identify which once-per-interval operation adds the error. The exponent 1.7 rests on one run at 125 intervals per convective time.
- **B (FVM domain).** Lateral clearance: clean within one regime; the two series do not separate `δ_ω` from `D` (23 % difference). Downstream reach: no trend. The paper says both.
- **Inset (4, 8.75, 14h).** Confounded with the lateral clearance (the replacement region moves from 5.7 to 4.2 `δ_ω` from the body), which by family B should raise the error, while the error fell. The sign argument makes the transfer's share a lower bound; the paper states it this way.
- **X, XP (space).** Three levels with a sign change of `e_coupling(Ĉ_L)`. The Strouhal-number coupling error at fixed interval falls as +3.07, +1.00, +0.35 % for h = 0.08, 0.0566, 0.04, which is a decay of order about 3 per level; the lift-amplitude coupling error does not converge monotonically. No order of the coupled solver follows.
- **R (Re 100, 150, 250).** Direction of the interval trend common, size not; lift amplitude unaffected at Re = 100.
- **F (factorial).** Shows that the domain and interval effects do not add. Two levels each; clean for that statement.

### 3.2 Two findings from the stored data (28 Oct, scripts in `notes/analysis_2026-10-28/`)

**Strength limit in the N = 2 run.** From `sensitivity/box2.4_sub2_inset8.75/solution/coupler_diagnostics.jsonl` and the baseline file, over 40 ≤ t ≤ 55:

| Run | κ median | κ max | transfers with κ ≥ 1.79 | mean boundary mismatch | max mismatch | removed fraction per transfer | moment correction per transfer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N = 5 (baseline) | 1.649 | 1.662 | 0 % | 2.9e-3 U∞ | 1.3e-2 U∞ | 0.30 % | 0.11 % |
| N = 2 | 1.800 | 1.800 | 81 % | 4.0e-3 U∞ | 2.8e-2 U∞ | 0.30 % | 0.12 % |

In the N = 2 run the limit acts from the first interval (κ max 1.97 before the limit was reached in the start-up, then pinned at 1.800). The mechanism is consistent with Appendix A of the paper: for a wall mode the particles cannot represent, the error factor `(1 − m̂)(1 − γ m̂)` is close to one, so each transfer adds about `(1 + γ)` times the same mismatch to the coefficient, and the coefficient grows with the number of transfers until GBD diffusion between transfers balances the growth or the limit stops it. More transfers per viscous time therefore raise the wall-layer strength ratio. Whether this amplified wall layer is what changes the boundary data on `Σ_FVM` is not shown by the diagnostics; the mean velocity mismatch is 40 % larger at N = 2, which is at least consistent.

Consequence for the documents. The paper's sentence "the strength ratio κ reached 1.68, 93 % of its limit, so the strength correction was never limited" is true of the baseline and is now stated for the baseline only, with the N = 2 numbers added to the coupling-interval section and the discussion. The thesis rule in section 4.7 ("a limit that acts at most coupling intervals belongs in the parameter study") applies, so `κ_max` must be varied (run B below) before the exponent of the interval trend is interpreted.

**Shed vortices in the wake.** From six particle frames of each run between t = 44 and 54, segmented by watershed on `|ω_z|` (threshold 0.1 U∞/D, peaks above 0.6 U∞/D):

| x range [D] | N = 5: Γ/(U∞D), ω_peak D/U∞ | N = 2: Γ/(U∞D), ω_peak D/U∞ |
| --- | --- | --- |
| 3.4–4.0 (inside Σ_transfer) | 3.49, 2.58 | 3.49, 2.60 |
| 5.2–6.0 (free wake) | 3.35, 2.32 | 3.33, 2.29 |
| 8–9 | 3.12, 1.96 | 3.10, 1.93 |
| 11–12 | 2.99, 1.67 | 2.97, 1.64 |
| 14–15 | 2.89, 1.42 | 2.86, 1.41 |

Circulation differs by at most 1.3 % and the peak vorticity by at most 2 %, in both regions, and the differences do not grow with the number of GBD steps a vortex has experienced (2.5 times more at N = 2). The lift amplitude, a wall-pressure integral of the FVM, differs by 1.4 points between the runs. Since the FVM receives the particles only through `Σ_FVM`, and the free wake is the same, the difference comes from the boundary data induced by the particles inside and just outside `Ω_FVM`, where the transfer acts. This narrows the per-interval error to the interface and excludes the VPM's remeshing of the wake. It also removes the paper's earlier hypothesis that "the circulation of a shed vortex should decrease with the number of transfers it experiences": it does not.

### 3.3 Is the calibration flow-agnostic?

No, and it should not be claimed. What the study shows within laminar periodic shedding from a smooth body: the lateral clearance of the replacement region sets the largest domain error, the downstream reach does not matter between 0.35 and 2.3 formation lengths, the domain and interval effects do not add, and the coupling interval should be the longest stable one. What it does not show: the scale of the lateral threshold (`δ_ω` or `D`), any rule for `δ_0`, `δ_1`, `κ_max`, `τ_transfer` or `σ/h`, any temporal or spatial order of the coupled solver, the share of each transfer operation in the per-interval error, or anything outside the regime. The cube and the wing test the lateral rule in another regime and are the right next step; they do not test the interval rule, because their coupling interval is set by stability alone.

## 4. Runs that close the gaps (prioritised; all on case 01, h = D/25 unless stated; about 3 h GPU each)

| Run | Purpose | Settings | Cost |
| --- | --- | --- | --- |
| **B1. Strength limit** | Decide whether the per-interval error is the wall-layer amplification | N = 2 and N = 5 with `κ_max = 1.0` (γ = 0, direct assignment of the mismatch); and N = 2 with `κ_max = 3` (limit inactive) | 3 runs |
| **A. Spatial order at an admissible interval** | Four levels, one order | `Δt_VPM = 0.032` (N = 8 at Δt_FVM = 0.004), h = 0.08, 0.0566, 0.04, 0.0283, interface widths in metres; FVM-only reference at Δt_FVM = 0.004 exists for all four | 4 runs (the finest is the expensive one) |
| **C. Temporal order at a fixed interval** | Four levels of the FVM step under `Δt_VPM = 0.04` | N = 5, 10, 20, 40 (`Δt_FVM` = 0.008 … 0.001); references `reference_dt0.002` exists, add 0.001 | 2 coupled + 1 reference |
| **D. Coupling interval, one more level** | Replace the single run at 125 intervals per convective time | N = 6 at Δt_FVM = 0.008 (`Δt_VPM = 0.048`, Λ = 0.93) and N = 3 | 2 runs |
| **E. Blending depths** | Rule for `δ_0`, `δ_1` | `δ_1 = 3h, 6h, 10h` at `δ_0 = 2h`; `δ_0 = 0, 2h, 4h` at `δ_1 = 6h` | 4 runs |
| **F. Particle spacing** | `h_VPM/h_FVM` and `σ/h` | `h_VPM = h_FVM/2` at N = 5 (family S of the plan) | 1 run |
| **G. Reference near wake** | Remove the resolution contribution from `e_coupling` | Reference with cells `h` out to x = 4.6D, Re = 150, Δt_FVM = 0.008 | 1 CPU run |
| **H. Statistics window** | Sampling error of the 2.7-period window | Continue the baseline and N = 2 to t = 100 (`allcontinue.sh`) and report 40–55 against 55–100 | 2 continuations |
| **I. Diagnostics on all runs** | Check κ, mismatch and removal per transfer for the 36 runs | Script `strength_ratio_history.py` over every `coupler_diagnostics.jsonl` | minutes |

B1 and I come first, because their outcome changes how the interval series is written up. A, C and D turn "no order" into an order or a documented absence of one, which is what section 4.7 promises. G is the one change to the yardstick that could move the headline −1.4 %.

## 5. Terms and symbols: discrepancies found and fixed

| Where | Was | Now (per `terminology.md`) |
| --- | --- | --- |
| Paper, test programme, tables, nomenclature | `Π`, `Π_coupled`, `Π_ref,h`, `Π_ref`, `ΔΠ_ref` | `Q`, `Q_coupled`, `Q_FVM(h, Δt_FVM)`, `Q_ext`, `U_ext` |
| Paper | `e_coupling = e_VPM + e_interface` | removed; the split into VPM and interface parts is stated in words (the two are not measured separately) |
| Paper, formulation | `Λ = Δt_VPM max_p ‖S(x_p)‖_∞` | `Λ = Δt_VPM ‖S‖_{1,∞}`, with the thesis's definition named |
| Paper, declarations | commits `dfa6b54f`, `66964084`, `45da734c` | `3b4c0f23`–`97d65b09`, as in the cylinder section |
| Paper, wing section | "Both solvers use the Smagorinsky closure of the cube" (closure is reserved for turbulence closure; fine) | unchanged |
| Thesis Ch. 4 | `h_FVM`, `h_VPM` in table 4.2, `h` elsewhere | unchanged (the table distinguishes the two spacings on purpose); the paper now says `h_FVM = h_VPM = h` once |
| OpenONDA `coupling_findings.md` §3.1 and `README` | replacement region starts "inset + zero-weight layer + blend end depth = 16.75 cells = 0.67D" | `L_inset + δ_1 = 14.75h = 0.59D` (the zero-weight layer lies inside `δ_1`); table values corrected |
| OpenONDA report 2026-10-23, §3.3 table | "8.75, 6 and 2 cells; 0.67D together" | 0.59D |
| Paper figure captions, "coupling boundary" (meeting booklet only) | retired term | not in the paper |

Words checked across the three documents and found consistent: vorticity transfer, transfer region, transfer surface, transfer lattice, transfer inset, zero-weight layer, blending ramp, replacement region, remeshing region, boundary data, coupling interval, coupling iteration, end-of-interval estimate, previous-interval correction, weak-strength removal, moment correction, strength correction, strength ratio, removal threshold, lateral clearance, wake reach, vortex-formation length, vorticity thickness. Retired words still present before this revision: none in the paper, none in Chapter 4; "coupling boundary" and `L_form` in `notes/meetings/2026-10-26/meeting_figures.tex` (left, dated record).

## 6. Paper: assessment and changes

**Does it do what it should?** The paper's job is to present the algorithm, measure its accuracy, its order, its sensitivity and its errors, and give a procedure and worked examples. Against that list:

- Algorithm: complete and checked against the code.
- Accuracy and error split: done for the cylinder at Re = 150 against `Q_ext`; at Re = 250 and 100 against the same-cell FVM only (said).
- Order: not obtained, for the reasons in section 3; the paper says so, and now says what sequence would give it.
- Sensitivity: coupling iterations, coupling interval, FVM domain, inset; not `δ_0`, `δ_1`, `κ_max`, `τ_transfer`, `σ/h`. The limitations now list these by name.
- Procedure: seven steps in the discussion; kept, reworded as instructions with their evidence.
- Examples: the cylinder setup table, the cube and wing configurations. The cube and wing results are pending; the paper cannot be submitted with placeholders (noted in the revision notes of 13 Oct).

**Style.** The text obeyed the writing rules but read as generated: every section opened by announcing how many things follow, items were enumerated before being developed, and the same hinge ("therefore") carried most sentences. The revision removes the announcements, lets the argument carry its own order, varies the hinges, and states a result before its qualification. Numbers, citations and labels are unchanged except where section 3.2 adds new ones; every added number is marked in a LaTeX comment with its source.

**Scientific statements changed.**
1. "Strength correction never limited" → stated for the baseline; N = 2 numbers added (cylinder, coupling-interval subsection; discussion).
2. "The circulation of a shed vortex should decrease with the number of transfers, which is a direct test" → replaced by the measured result that it does not, and the error is located in the boundary data from the near-body particles.
3. "The coupled solution converges in time only within a fixed interval, and it then follows the temporal convergence of the FVM" → the second half is now an expectation from two levels.
4. Limitations now name the parameters that were not varied and the induction tolerance.

## 7. Thesis Chapter 4

Objective errors corrected (typos and wrong words only): "It was also that the inverse step also enlarges" → "It was also found that the inverse step enlarges"; "an linearly-varying velocity field that also takes into account the gradient term" → "a linearly varying velocity field"; "ridigly-moving" → "rigidly moving"; "we atempt" → "we attempt"; caption of fig. 4.5 "trimmed due interaction with walls" → "due to interaction with walls"; "a Dirichlet condition ... in validated hybrid solvers" unchanged.

Not changed, for the author to consider:
- Section 4.7, "Transfer limits": the rule is right and now has a case. Consider adding one sentence that the cylinder study found the strength limit active at short coupling intervals, so that section 5 can report run B1.
- Section 4.7, "Refinement sequences": the sequences are stated; section 5 will have to report which were completed (A, C, D above).
- Section 4.6 "we attempt to find the optimum value of N" is the only first-person plural in the chapter.
- Table 4.2 lists `h_FVM/δ_ω` and `h_VPM/δ_ω`; the paper uses one `h`. Consistent, since the paper states `h_FVM = h_VPM`.
- `OpenONDA/docs/coupling.md` says "A deeper inset, 15–20 spacings for the cylinder, is more accurate" and the report says "the default of 8.75 stays; 14 gains 0.13 points". The thesis and paper say the default stays. `coupling.md` should be softened (done).

## 8. OpenONDA records

- `tutorials/.../assets/coupling_findings.md`: replacement-region start corrected from 16.75 to 14.75 cells (0.59D) in §3.1 and its table, with a dated note; the earlier numbers were derived with the old `L_T + δ_0 + δ_buf` formula.
- `docs/reports/2026-10-23-coupling-error-sources.md`: the "0.67D together" entry of the normalisation table corrected to 0.59D.
- `docs/coupling.md`: the inset recommendation aligned with the report.
- `docs/plans/2026-10-28-coupling-error-followup.md`: the runs of section 4, with the two findings and the scripts that produced them.
