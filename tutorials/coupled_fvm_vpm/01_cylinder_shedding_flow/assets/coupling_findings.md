# Calibration of the coupling interface on the cylinder at $Re=150$

This note records what the calibration study of the [coupled cylinder tutorial](../README.md) measures and, once its runs are done, what it found. It is the evidence behind Paper I (section 8) and thesis Chapter 5. Terms follow [terminology](../../../../docs/terminology.md).

The study is the matrix of `allrun.sh`: a baseline and five groups, 18 coupled runs, each changing one setting of the baseline. Every run uses the current transfer, in which each lattice node of the replacement region takes the FVM circulation of its lattice cube. Results of the earlier transfer are not used.

## Error measure

Each statistic $Q$ (mean drag $\bar C_D$, Strouhal number $St$, rms lift $C_{L,\mathrm{rms}}$, over whole shedding periods) gives the coupling error

$$e = Q_\mathrm{coupled}/Q_\mathrm{FVM}(h,\Delta t_\mathrm{FVM}) - 1,$$

against the FVM-only run on the same cells and FVM step. The FVM-only domain has slip walls 32 $D$ apart; its values are corrected for blockage before the comparison: the drag with the complete Allen–Vincenti expression $C_D\,(1 - \tfrac12 C_D\,D/H - 2.467\,(D/H)^2)$ (Sadeh and Saharon 1982, eq. 3.5), which includes the drag of the axial pressure gradient, the lift by $(U/U')^2$ and $St$ by $U/U'$. The FVM's own discretization error is reported apart, from the reference grids $h=0.08$, 0.06, 0.04 and 0.03 m.

Runs are compared over $40\le t\le55$ s. A difference is resolved when it exceeds $\pm2\sigma$, with $\sigma=\sqrt{2\sigma_w^2+s_d^2}$:
- $\sigma_w$, the window-to-window spread of the developed flow;
- $s_d$, the run-to-run spread of the drift of the 40–55 s window.

Both are measured by `window_uncertainty.py` on the runs that reach 95 s.

## FVM-only references

`reference_flow/allrun.sh` lists them. The coupler does not enter an FVM-only run, so the stored grids $h=0.08$, 0.06, 0.04 and 0.03 m remain valid. They ran to 100 s with commit `3f771630`.
- **Solver.** The FVM changes between that commit and the transfer of the study, `37d7d7e3`, are in the mesher, documentation, messages, backups, moving meshes and the force-sampler interface.
- **Mesh.** Compared with the stored reference mesh, the current coupled mesh at $h=0.04$ m has, for each of its 19,964 cells, a reference cell at the same centre within $10^{-4}$ m. The exceptions are 812 cells within 0.06 m of the outer coupled boundary, offset by at most $5\times10^{-4}$ m. Within $0.75D$ of the axis the 680 cells agree within $4\times10^{-6}$ m.

The boxes of margin 3.2 and 4.0 $D$ reach beyond the fine cells of those references. Two new references cover them, `h0.04_dt0.008_fb3.2` and `h0.04_dt0.008_fb4.0`, with cells of 0.04 m over the larger box.

## Findings

To be filled from the runs, group by group:

| Group | Question |
| --- | --- |
| T, coupling interval ($N=1$, 2, 3, 5, 6) | how the error per transfer accumulates with the number of transfers |
| M, coupling iterations (one pass, $10^{-3}$, $10^{-4}$, $10^{-5}$ linear, $10^{-5}$ constant) | the tolerance at which the error reaches the converged-coupling limit, and the iterations it costs |
| D, FVM box margin (1.6, 2.0, 2.4, 3.2, 4.0 $D$) | the smallest box whose error is resolved from the widest |
| I, transfer inset (4, 8.75, 14 cells) | whether the inset moves the error beyond the window uncertainty |
| S, spatial resolution ($h=0.08$, 0.06, 0.04 m at 0.048 s; particle spacing $2h$) | the order of the coupling error with the cell size, and the cost of coarser particles |

## Rules for a user

To be drawn from the findings.
