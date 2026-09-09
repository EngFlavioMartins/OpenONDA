# Vortex interactions

Run the official GBD/Lagrange6 baseline first, then compare stretching
viscosity, moment-preserving realignment and splitting:

```sh
./allrun.sh
./allplot.sh
```

Use the installed OpenONDA environment. All physical inputs are in
`setup_les.py`; each launcher line is one ordinary Python command. To run
just the baseline or the leading stabilized candidate:

```sh
python setup_les.py --variant baseline
python setup_les.py --variant stretching_viscosity
```

The baseline is an unperturbed pair of Gaussian rings with Re_Gamma=3000,
R0=1, a0=.1, separation=1, Smagorinsky Cs=.20, SSPRK3, transposed stretching,
GBD/Lagrange6, h=sigma=.04 and dt=.0075. Only the stabilization changes
between the four default cases. Stretching viscosity uses coefficient .5;
realignment preserves moments with frequency .384684814725/s; splitting
checks every five steps with twice the initial peak-strength threshold.

Each case requests 1200 steps (t=9) with the same 150-minute native runtime
cap. The four serial caps total ten hours, plus startup/final-output overhead.
This is a bounded study; the requested horizon is not guaranteed within the
cap. `allrun.sh` stops on an execution error; normal native health/budget
stops retain their actual status. Do not run overlapping copies of the suite.

The time horizon includes margin for x/R0=7. From saved tracks through t=3.3,
mean-speed extrapolation predicts both cores there near t=5.6. The slower
core's recent t=2.4–3.3 speed predicts t=8.33 for baseline and t=8.69 for
stretching viscosity. Hence t=9, rather than t=6. This is a rough extrapolation
through changing leapfrog speeds, not a guarantee. The report explicitly
records whether both tracked cores actually reached 7; a runtime cap can
still prevent that, and must not be described as successful full coverage.

## Official solution and samples

For each variant, the solver writes:

- `solution/les_<variant>/vpm_metadata.json`: native configuration and state.
- `solution/les_<variant>/vpm.log`: progress, health limits and failure details.
- `solution/les_<variant>/vpm_*.h5` and XDMF: native numerical backups every
  100 steps and at normal termination, including a native budget/health stop.
- `samples/les_<variant>/flow_integrals.csv`: native energy, enstrophy,
  impulse, LES, resolution and stabilization diagnostics every ten steps.
- `samples/les_<variant>/ring_diagnostics.csv`: native particle-group proxies.
- `samples/les_<variant>/core_section.pvd` and VTS: velocity and curl-derived
  vorticity on z=0, y>=0, at .02 spacing, initially/every .15 s/at termination.
- `samples/les_<variant>/cross_section.pvd` and VTS: the orthogonal y=0
  plane, including both signs of z, at .04 spacing, initially/every .30 s/at
  termination. Compare these planes for asymmetric deformation.

There are no tutorial-owned metadata writers or reconstructed diagnostic
fields. Plotting reads native samples. A forced process kill can leave the
latest metadata at its checkpoint state; it must not be relabeled completed.
The older `study_results/` data remain historical evidence, not the official
solution produced by this launcher. Existing seeded `solution/baseline` data
are separate from the new `solution/les_baseline`.

## Decide which stabilization helps

Use both numerical survival and LBM agreement. A method is a clear winner
only if it improves one without materially worsening the other; otherwise
report the tradeoff or an inconclusive result.

| Evidence | Figure/result | Decision |
|---|---|---|
| Native termination, last accepted time, health history | Report status table and `diagnostic_histories.png` | Identify numerical blow-up/health stop separately from completion, runtime cap or resource failure. A capped/completed run only gives a lower bound on survival. |
| Sampled-core radius versus axial travel | `core_trajectories.png` and fixed-interval RMS scores | Compare both rings on x/R0=.55–1.5, .55–2.5, .55–3.5 and .55–5.5, .55–7, without fitting or extrapolation. Longer coverage is useful only while the trajectory remains credible. |
| Core shape, vorticity peaks and bridge | Common-time core sections at t=1.5, 3.3, 4.5, 6, 7.5, 9, when saved | Detect excessive damping, deformation or loss of two resolved cores; inspect orthogonal VTS planes around any transition. |
| Energy/enstrophy, divergence, misalignment, CFL and particle count | `diagnostic_histories.png` | Check whether apparent stability simply comes from excessive damping or loss of resolution. |
| Native stabilization activity | Report JSON and flow CSV | Zero splitting events means splitting was not exercised; continuous stretching viscosity is identified by its coefficient rather than the discrete-event count. |

`allplot.sh` writes the report, trajectory and diagnostic figures to
`figures/les/`, and sampled core contours to `figures/core_sections/`.
Scores unavailable because of early termination remain unavailable. Compare
an apparent winner at the saved-time cadence and at every other output;
a close ranking needs a sensitivity check before certification.

The optional `python setup_les.py --variant halfdt` is a baseline time-step
check with the same physical horizon. It is outside the default four-run
budget; run it only if the final comparison needs it. Particle-resolution
convergence is not established by this check.

The existing Fig. 5 LBM data are an unperturbed kinematic reference and do
not establish an instability-breakdown time. VPM induction is unbounded;
the reference uses periodic boundaries. A meridional peak-tracking cutoff
is not proof of three-dimensional breakdown. Claiming matched physical
breakdown requires corresponding LBM field/time evidence; see
[reference provenance](assets/references/README.md).

## Prior result and other experiments

The prior bounded study selected GBD over CS on early-trajectory agreement
and cost; CS core spreading was confirmed active. Stretching viscosity .5
reduced radius RMS from 5.715% to 5.116% of R0 on x/R0=.55–3.5. Realignment
showed no resolved trajectory improvement and splitting did not activate.
Those results motivate this official comparison, but do not certify late
survival or breakdown physics.

The six seeded cases remain in `setup.py`; their imposed disturbance is a
separate physical experiment. Advanced controls remain in `assets/study.py`.
`allclean.sh` deletes solution, samples, figures and historical study results;
run and plot do not automatically clean existing data.

See [study record](../../../docs/reviews/2026-09-vpm-core-transport.md).
