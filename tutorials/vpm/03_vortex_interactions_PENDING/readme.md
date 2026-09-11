# Vortex interactions

Run the official CS/LES baseline first, then compare stretching
viscosity, moment-preserving realignment and splitting:

```sh
./allrun.sh
./allplot.sh
```

The default `allrun.sh`/`allplot.sh` battery is the unperturbed
Re_Gamma=3000 kinematic control; native health and wall-budget stops remain
reported as recorded rather than being relabeled completed. The separate seeded instability phase uses
the primary paper's Re=3415, epsilon/R0=.05, axial mode-eight displacement
and distinct native output names:

```sh
./allrun_breakdown.sh
./allplot_breakdown.sh
```

Do not score the seeded Re=3415 cases against the available Fig. 5 trajectory:
that reference is the unperturbed Re=3000 problem. Use the native particle
backups and both SurfaceSampler planes to establish mode growth and morphology
before calling an event physical breakdown.

For one bounded seeded case, the explicit host-level command is:

```sh
python setup_les.py --scenario seeded_breakdown --variant baseline \
  --compute-device METAL --steps 1200 --wall-minutes 24
```

Run cases serially on the exclusive Metal slot. The existing seeded
stretching-viscosity contrast uses the same command with
`--variant stretching_viscosity`; the moment-preserving and splitting cases
remain separate diagnostics and must not be silently substituted for the
unstabilized baseline.

Use the installed OpenONDA environment. All physical inputs are in
`setup_les.py`; each launcher line is one ordinary Python command. To run
just the baseline or the leading stabilized candidate:

```sh
python setup_les.py --variant baseline
python setup_les.py --variant stretching_viscosity
```

The baseline is an unperturbed pair of Gaussian rings with Re_Gamma=3000,
R0=1, physical Gaussian core a0=.1, separation=1, Smagorinsky Cs=.20,
SSPRK3, transposed stretching, CS core spreading with h/R0=.06 and
core-radius ratio 1, and dt=.0075. Only the stabilization changes between
the four default cases. Stretching viscosity uses coefficient .5;
moment-preserving Pedrizzetti relaxation uses frequency .384684814725/s;
splitting checks every five steps with twice the initial peak-strength
threshold. All four cases use the same native samplers and health limits.

Each case requests 1200 steps (t=9) with a 24-minute native runtime cap. The
official serial battery used 1:35:17.3 of GPU wall time; qualification added
about 2:11.6. This is a bounded study: the requested horizon is not
guaranteed within the cap. `allrun.sh` stops on an execution error; normal
native health/budget stops retain their actual status. Do not run overlapping
copies of the suite.

The RunPlan also declares a 600,000 active-particle ceiling, a 12 GiB process
RSS ceiling and a 2 GiB available-memory floor. These are lifecycle
`resource_limit` guards, distinct from physical `resolution_lost` health stops;
they are checked at accepted-step boundaries and do not prevent an
inside-step allocation peak.

The time horizon includes margin for x/R0=7. From saved tracks through t=3.3,
mean-speed extrapolation predicts both cores there near t=5.6. The slower
core's recent t=2.4–3.3 speed predicts t=8.33 for baseline and t=8.69 for
stretching viscosity. Hence t=9, rather than t=6. This is a rough extrapolation
through changing leapfrog speeds, not a guarantee. The report explicitly
records whether both tracked cores actually reached 7; a runtime cap can
still prevent that, and must not be described as successful full coverage.

The resolution qualification rejected h/R0=.08 after 80 steps because CS
diffusion was too strong and the pair did not reach the comparison interval.
The selected h/R0=.06 qualification ran 200 steps to t=1.5 and gave 2.305%
R0 radius RMS on the common x/R0=.55--1.5 interval with both rings covered.
Reproduce the qualification with:

```sh
python setup_les.py --variant baseline --qualification --particle-spacing 0.08
python setup_les.py --variant baseline --qualification \
  --particle-spacing 0.06 --qualification-name cs_qualification_h06 --steps 200
```

The seeded dynamic-coverage qualification is a separate Researcher 2 test.
It decreases the spacing to `.05 R0` while retaining the `.06 R0` numerical
core and matches the baseline's strength-weighted RMS `Cs*Delta`. It does not
change the physical core, circulation, seed, timestep, guards or stabilization:

```sh
./allrun_coverage_qualification.sh
```

The planned Metal command was capped at 400 accepted steps (`t=3.0`) and 15
minutes, with 27,200 fixed CS particles. On the recorded host it failed in
Taichi's Metal scalar-field probe before solver construction. The accepted
fallback used the same physics on six CPU threads: after strict two-versus-six
thread parity, a public restart reached step 369 (`t=2.7675`) under a 50-minute
wall cap. It crossed the old baseline health endpoint without reaching step
400; lifecycle is `wall_time_limit`, not `completed`.
Its static initial-field and coverage evidence is generated with:

```sh
python assets/check_initial_coverage_field.py
python assets/assess_particle_coverage.py
```

See the [Researcher 2 technical record](../../../docs/reviews/2026-09-vortex-interactions-researcher2-report.md)
for the predeclared predictions, native outcome and decision rule. The denser
fixed-core case improves coverage and numerical health but does not show
persistent mode growth or two-plane breakup; passing the old runtime horizon
alone is not physical breakdown. Regenerate the common-time mode, morphology,
field and coverage comparison with:

```sh
./allplot_coverage_qualification.sh
```

The bounded seeded p-moments qualification is also complete through step 300
(`t=2.25`) from onset. It reduces misalignment but materially changes the
group-0 radial mode and ringwise impulse/strength distribution, so it is an
onset diagnostic rather than a winning post-breakdown stabilizer. Reproduce
the read-only particle-geometry, weighting, two-plane core and frozen-event
attribution from the preserved native outputs with:

```sh
python assets/analyze_p_moments_attribution.py
```

The frozen event is explicitly a single-state operator decomposition, not a
continued solution. The technical record gives its source hashes, the old/new
installed-generation boundary and the proposed matched CS filter contrast. No
new simulation is launched by this postprocessor.

The next seeded CS filter contrast is prepared but held for compute
coordination. Its dry preflight verifies the current installed generation,
exactly matched primary particle arrays and mutual velocity/gradient, then
writes an inspectable JSON/Markdown receipt:

```sh
python assets/check_filter_pair_preflight.py
```

After explicit slot release, run the `.20` control and `0` contrast
sequentially using `allrun_filter_cs020_qualification.sh` and
`allrun_filter_cs000_qualification.sh`. Each targets 80 steps (`t=.6`) under a
five-minute native cap. Record actual control time before deciding whether the
second leg fits the remaining envelope; native caps are not hard process-wall
limits. Initial eddy/effective viscosity is expected to differ because `Cs` is
the isolated parameter.

## Official solution and samples

For each variant, the solver writes:

- `solution/cs_<variant>/vpm_metadata.json`: native configuration and state.
- `solution/cs_<variant>/vpm.log`: progress, health limits and failure details.
- `solution/cs_<variant>/vpm_*.h5` and XDMF: native numerical backups every
  100 steps and at normal termination, including a native budget/health stop.
- `samples/cs_<variant>/flow_integrals.csv`: native energy, enstrophy,
  impulse, LES, resolution and stabilization diagnostics every ten steps.
- `samples/cs_<variant>/ring_diagnostics.csv`: native particle-group proxies.
- `samples/cs_<variant>/core_section.pvd` and VTS: velocity and curl-derived
  vorticity on z=0, y>=0, at .02 spacing, initially/every .15 s/at termination.
- `samples/cs_<variant>/cross_section.pvd` and VTS: the orthogonal y=0
  plane, including both signs of z, at .04 spacing, initially/every .30 s/at
  termination. Compare these planes for asymmetric deformation.

There are no tutorial-owned metadata writers or reconstructed diagnostic
fields. Plotting reads native samples. A forced process kill can leave the
latest metadata at its checkpoint state; it must not be relabeled completed.
The older `les_*`, GBD and `study_results/` data remain historical evidence,
not the official CS solution produced by this launcher. Existing seeded
`solution/baseline` data are separate from the new `solution/cs_baseline`.

## Decide which stabilization helps

Use both numerical survival and LBM agreement. A method is a clear winner
only if it improves one without materially worsening the other; otherwise
report the tradeoff or an inconclusive result.

| Evidence | Figure/result | Decision |
|---|---|---|
| Native termination, last accepted time, health history | Report status table and [diagnostic histories](figures/cs/diagnostic_histories.png) | Identify numerical blow-up/health stop separately from completion, runtime cap or resource failure. A capped/completed run only gives a lower bound on survival. |
| Sampled-core radius versus axial travel | [core trajectories](figures/cs/core_trajectories.png) and fixed-interval RMS scores | Compare both rings on x/R0=.55–1.5, .55–2.5, .55–3.5 and .55–5.5, .55–7, without fitting or extrapolation. Longer coverage is useful only while the trajectory remains credible. |
| Core shape, vorticity peaks and bridge | Common-time core sections at t=1.5, 3.3, 4.5, 6, 7.5, 9, when saved | Detect excessive damping, deformation or loss of two resolved cores; inspect orthogonal VTS planes around any transition. |
| Energy/enstrophy, divergence, misalignment, CFL and particle count | [diagnostic histories](figures/cs/diagnostic_histories.png) | Check whether apparent stability simply comes from excessive damping or loss of resolution. |
| Native stabilization activity | Report JSON and flow CSV | Zero splitting events means splitting was not exercised; continuous stretching viscosity is identified by its coefficient rather than the discrete-event count. |

`allplot.sh` writes the report, trajectory and diagnostic figures to
`figures/cs/`, and sampled core contours to `figures/core_sections/`.
Scores unavailable because of early termination remain unavailable. Compare
an apparent winner at the saved-time cadence and at every other output;
a close ranking needs a sensitivity check before certification.

The seeded native-history postprocessor writes energy, enstrophy, health,
mode, morphology and impulse figures without reconstructing fields:

```sh
python assets/plot_seeded_history.py \
  --runs cs_breakdown_baseline cs_breakdown_stretching_viscosity \
  --output figures/cs_breakdown
python assets/plot_core_sections.py \
  --runs cs_breakdown_baseline cs_breakdown_stretching_viscosity \
  --format png --output figures/cs_breakdown_core_sections
```

`allplot_breakdown.sh` runs these native postprocessors in dependency order
for the two seeded variants currently present in this checkout. It writes
[diagnostic histories](figures/cs_breakdown/diagnostic_histories.png),
[mode histories](figures/cs_breakdown/mode_histories.png),
[morphology histories](figures/cs_breakdown/morphology_histories.png),
[impulse histories](figures/cs_breakdown/impulse_histories.png) and
[summary](figures/cs_breakdown/summary.md) to `figures/cs_breakdown/`, and
[seeded core sections](figures/cs_breakdown_core_sections/) to
`figures/cs_breakdown_core_sections/`. The bounded seeded p-moments CPU
qualification is stored under the explicit
`cs_breakdown_p_moments_cpu_t6_qualification` namespace. The default seeded
`p_moments` and `splitting` launcher namespaces remain unrun; the ordinary
launcher does not fabricate or score them.

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

## Historical controls and other experiments

The earlier GBD/Lagrange6 battery and `les_*` outputs are retained as
historical controls. They are not mixed into the official CS comparison and
do not certify late survival or breakdown physics.

The six seeded cases remain in `setup.py`; their imposed disturbance is a
separate physical experiment. Advanced controls remain in `assets/study.py`.
`allclean.sh` deletes solution, samples, figures and historical study results;
run and plot do not automatically clean existing data.

See the [official CS study record](../../../docs/reviews/2026-09-vpm-cs-leapfrog.md)
and the generated [comparison report](figures/cs/report.md).
