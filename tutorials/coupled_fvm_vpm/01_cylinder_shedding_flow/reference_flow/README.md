# Cylinder reference flow

This is a body-fitted reference study for flow around a unit cylinder at
Re = 150. The physical problem and numerical method are defined in `setup.py`.
Production runs use the top-level pipeline's conservative candidate family
`h = 0.10, 0.08, 0.064D`; each grid receives its own 12-hour wall budget.

The top-level [pipeline](../assets/run_pipeline.py) is the budgeted entry
point: it runs conservative reference candidates and matched coupled cases,
applies a 12-hour limit to each case, and stores independent logs and cost
records. Run a bounded screen first:

```bash
cd ..
python assets/run_pipeline.py --pilot --sensitivity none
```

`./allrun.sh` cleans and runs the single-grid `setup.py` case. Use
`./allcontinue.sh` to resume that grid's latest native FVM backup. Both launchers
accept the same `--name` and `-h` arguments as `setup.py`.

The separate pipeline supports `--reference-only`, `--reference-cores` and
`--root` through `python ../assets/run_pipeline.py`. Its default six-rank
reference configuration requires the optional MPI stack; select
`--reference-cores 1` for a serial campaign.

The explicit reference-family candidates used for a fine mesh check are:

| Case | Wall spacing h (m) | Span layers | Spanwise spacing (m) |
| --- | ---: | ---: | ---: |
| `grid_h008` | 0.08 | 12 | 0.08 |
| `grid_h00565685` | 0.0565685424949238 | 17 | 0.056471 |
| `grid_h004` | 0.04 | 24 | 0.04 |
| `grid_h00282843` | 0.0282842712474619 | 34 | 0.028235 |

Successive grids have a refinement ratio of sqrt(2). The realized near-body
spacing is passed directly as `h`:

```bash
python setup.py --name grid_h004 -h 0.04
```

Every grid uses the same 0.96 m resolved slip span, physical domain, Reynolds
number, timestep controls, force cadence and 100 s horizon. The span has enough
uniform layers to keep the near-body cells approximately isotropic. The wake
remains refined through 12 diameters downstream. Forces are sampled every
0.04 s, leaving more than one hundred samples per shedding cycle without
forcing the previous 0.004 s timestep cadence. Solutions and samples are
written below a unique campaign directory. A bounded mesh pilot is available
with `python ../assets/run_campaign.py --kind reference --pilot`; complete
campaigns publish `reference_selection.json`, which is the only reference
selection accepted by the comparison plots. The selection is accepted only
when its `force_grid_qualified` flag is true; that flag covers the force-grid
statistics/GCI gate and does not certify temporal, domain, span or profile
convergence. The four-grid table is an explicit candidate family, while the
default production pipeline remains the three-grid `.10, .08, .064D` family.

After all grids finish, post-process their force histories with:

```bash
python postprocess_grid_study.py
```

The script writes `grid_forces.json`, `grid_forces.csv` and `grid_forces.png`
under `figures/`. It reports mean drag, force RMS, lift-based Strouhal number
and Richardson/GCI estimates over the statistics window declared at the top of
the script.

`allclean.sh` removes generated solutions, samples and figures. It is never run
automatically. Full production completion and the finest-case 12-hour target
remain pending measured long-horizon runs. A per-case timeout is not evidence
that the case completed within the budget.
