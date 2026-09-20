# Cylinder reference flow

This is a four-grid body-fitted reference study for flow around a unit cylinder
at Re = 150. The physical problem and numerical method are defined in
`setup.py`; the grid names and wall spacings are defined in `allrun.sh`.

Run the grids with:

```bash
./allrun.sh
```

The launcher runs:

| Case | Wall spacing h (m) | Span layers | Spanwise spacing (m) |
| --- | ---: | ---: | ---: |
| `grid_h008` | 0.08 | 4 | 0.0625 |
| `grid_h00565685` | 0.0565685424949238 | 5 | 0.05 |
| `grid_h004` | 0.04 | 7 | 0.035714 |
| `grid_h00282843` | 0.0282842712474619 | 9 | 0.027778 |

Successive grids have a refinement ratio of sqrt(2). The realized near-body
spacing is passed directly as `h`:

```bash
python setup.py --name grid_h004 -h 0.04
```

Every grid uses the same quarter-diameter slip span, physical domain, Reynolds
number, timestep controls, force cadence and 80 s horizon. The span has enough
uniform layers to keep the near-body cells approximately isotropic. The wake
remains refined through 12 diameters downstream. Forces are sampled every
0.04 s, leaving more than one hundred samples per shedding cycle without
forcing the previous 0.004 s timestep cadence. Solutions and samples are
written below `solution/<name>` and `samples/<name>`.

After all grids finish, post-process their force histories with:

```bash
python postprocess_grid_study.py
```

The script writes `grid_forces.json`, `grid_forces.csv` and `grid_forces.png`
under `figures/`. It reports mean drag, force RMS, lift-based Strouhal number
and Richardson/GCI estimates over the statistics window declared at the top of
the script.

`allclean.sh` removes generated solutions, samples and figures. It is never run
automatically.
