# Cylinder coupling sensitivity at $Re=150$

This note records what the sensitivity study of the [coupled cylinder tutorial](../README.md) measured, and the rules drawn from it. It is the evidence behind Paper I (section 8) and thesis Chapter 5. Terms follow [terminology](../../../../docs/terminology.md).

All runs of 19–25 October 2026 used one solver version (development `3b4c0f23` to `97d65b09`). Statistics are taken over $40\le t\le55$ s (2.7 shedding periods) in every run and reference: the mean drag $\bar C_D$, the lift amplitude $\hat C_L$ and the Strouhal number $St$.

## Error measure

Each statistic $Q$ is compared with $Q_\mathrm{ext}$, the FVM-only value extrapolated to zero cell size and step (Eça and Hoekstra least squares, `assets/error_analysis.py`). The error of a coupled run splits into two measured parts:

- FVM error $e_\mathrm{FVM}=Q_\mathrm{FVM}(h,\Delta t_\mathrm{FVM})/Q_\mathrm{ext}-1$, the FVM-only run on the same cells and step;
- coupling error $e_\mathrm{coupling}=(Q_\mathrm{coupled}-Q_\mathrm{FVM})/Q_\mathrm{ext}$, what the particle wake and the interface add.

| $Re=150$ | $\bar C_D$ | $\hat C_L$ | $St$ |
| --- | ---: | ---: | ---: |
| $Q_\mathrm{ext}\pm U_\mathrm{ext}$ | $1.311\pm0.003$ | $0.518\pm0.011$ | $0.1838\pm0.0002$ |
| $e_\mathrm{FVM}$ at $h=D/25$, $\Delta t_\mathrm{FVM}=0.008$ s | −0.14% | −3.11% | −1.22% |
| $e_\mathrm{coupling}$ of the baseline | −0.09% | −1.42% | +0.33% |

On the baseline cells the FVM error is larger than the coupling error in all three statistics. A repeat of the baseline agreed within 0.01 points.

## What each parameter does

| Parameter (flow-scaled) | Runs | Effect on $e_\mathrm{coupling}$ |
| --- | --- | --- |
| Coupling iterations | 1, 2, 4, tolerance $10^{-5}$ | One iteration: +2.96% drag, +2.09% lift amplitude. From two on, within 0.15 points; the tolerance needs 1.8 iterations on average |
| Coupling intervals per convective time $D/(U\Delta t_\mathrm{VPM})$ | $N=2,4,5$ at $\Delta t_\mathrm{FVM}=0.008$ s; $N=2$ at three FVM steps; $N=10$ | Lift amplitude −1.42% at 25, −1.55% at 31, −2.82% at 62.5, −7.08% at 125. Halving $\Delta t_\mathrm{FVM}$ at a fixed interval changes it by at most 0.07 points |
| Lateral clearance $\ell_\mathrm{side}/\delta_\omega$ (body surface to the side faces of the replacement region) | four margins at $Re=150$ and $250$ | Excess lift-amplitude error over the widest domain: about 4 points at 1.2, 2 at 2, 0.6–1.3 at 2.7, none at 5 |
| Wake reach $\ell_\mathrm{wake}/L_f$ | 0.35 to 2.33, both $Re$ | Up to 2 points, no common trend. Short domains looked better only because their positive coupling error cancelled the negative FVM error |
| Transfer inset $L_\mathrm{inset}$ | 4, 8.75, 14 cells | Lift amplitude −1.67%, −1.42%, −1.29% |
| Cell size at a fixed coupling interval | $h=0.08$, 0.0566, 0.04 m | $e_\mathrm{coupling}(\hat C_L)$ +2.17%, −0.90%, −1.35%: sign change, no order. $St$: +3.07%, +1.00%, +0.35%. $h=0.0283$ m inadmissible at $\Delta t_\mathrm{VPM}=0.04$ s ($\Lambda=1.03$) |

The FVM domain and the coupling interval do not add their errors (2×2 factorial, defect 2.75 points in lift amplitude): more coupling intervals amplify the error the domain sets.

## Where the per-interval error comes from (28 October 2026)

Two checks on stored output, `notes/analysis_2026-10-28/` of Paper I and `assets/strength_ratio_history.py`:

1. **The shed vortices are the same.** Between the $N=5$ and $N=2$ runs, the circulation and peak vorticity of the shed vortices agree within 1.3% and 2% from the transfer region to $x=15D$, although a vortex of the $N=2$ run passes 2.5 times more GBD steps. Grid-based diffusion of the free wake is not the source; the error enters through the boundary data from the particles near the body.
2. **The strength limit acts at short intervals.** The strength ratio $\kappa$ of the transfer sits at $\kappa_\mathrm{max}=1.8$ in 82% of the transfers of the $N=2$ run and never above 1.66 in the baseline; the mean boundary velocity mismatch is $4.0\times10^{-3}U$ against $2.9\times10^{-3}U$. The coupling-interval series therefore changes the transfer count and the action of the strength correction at the wall together. Section `b1` separates them.

## Rules for a user

For laminar periodic shedding from a smooth body. Whether the lateral rule scales with $\delta_\omega$ or with $D$ is not decided (the two Reynolds numbers differ by 23% in $\delta_\omega$); the cube and the wing test it.

1. Measure $\delta_\omega$ and $L_f$ on a coarse FVM-only solution.
2. End the replacement region at least $5\delta_\omega$ (here $1.3D$) from the body on the sides.
3. Place the downstream transfer surface behind the formation point; never choose it by the size of the total error.
4. Use the longest coupling interval that the particles admit on the finest planned cells, $\Lambda\approx0.8$.
5. Refine in time by refining $\Delta t_\mathrm{FVM}$ inside a fixed coupling interval; refining both at a fixed $N$ does not converge.
6. Converge the coupling iterations (tolerance $10^{-5}$).
7. During a run, keep the mean boundary velocity mismatch below $3.5\times10^{-3}U$, and check that $\kappa$ does not sit at $\kappa_\mathrm{max}$ in most transfers.
8. Compare with an FVM-only solution of the same far field (walls and inlet far away), with converged linear solves (momentum tolerance $10^{-9}$).

## Not yet measured

Sections of `assets/run_sensitivity_study.sh` that close the gaps against thesis section 4.7, in priority order:

| Section | Measures |
| --- | --- |
| `b1` | $\kappa_\mathrm{max}=1$ and 3 at $N=2$ and 5: is the per-interval error the strength correction at the wall? |
| `d` | $N=3$ and 6: the interval trend without resting on the single run at 125 intervals per convective time |
| `a` | Four cell sizes at $\Delta t_\mathrm{VPM}=0.032$ s: the spatial order of the coupled solver |
| `c` | $N=20$, 40 at $\Delta t_\mathrm{VPM}=0.04$ s: the temporal order inside a fixed interval |
| `g` | FVM-only reference with $h$ cells over the coupled FVM domain: removes the near-wake resolution difference from $e_\mathrm{coupling}$ |
| `e` | Blending depths $\delta_0$, $\delta_1$ |
| `h` | Baseline and $N=2$ to 100 s: sampling error of the 40–55 s window |
| `f` | $h_\mathrm{VPM}=h_\mathrm{FVM}/2$ (needs a 12 GB GPU) |

`./allplot.sh` writes their results to `figures/auxiliary/error_transfer_limits.csv` (κ, limit fraction, mismatch and $e_\mathrm{coupling}$ against $\kappa_\mathrm{max}$, $N$ and the blending depths), `error_window_reference.csv` (sections `g`, `h`), and the order tables for families `A` and `P`.
