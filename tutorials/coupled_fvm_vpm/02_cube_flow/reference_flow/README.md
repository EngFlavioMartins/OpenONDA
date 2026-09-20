# Cube reference flow

This is a four-grid body-fitted reference study for flow around a unit cube at
Re = 1000. The physical problem and numerical method are defined in `setup.py`;
the grid names and baseline spacings are defined in `allrun.sh`.

Run the grids with:

```bash
./allrun.sh
```

The launcher runs:

| Case | Wall spacing h (m) |
| --- | ---: |
| `grid_h010125` | 0.10125 |
| `grid_h00675` | 0.0675 |
| `grid_h0045` | 0.045 |
| `grid_h003` | 0.03 |

Successive grids have a refinement ratio of 1.5. Each command has only the
output name and baseline grid spacing:

```bash
python setup.py --name grid_h0045 -h 0.045
```

After all grids finish, post-process their force histories with:

```bash
python postprocess_grid_study.py
```

The script writes `grid_forces.json`, `grid_forces.csv` and `grid_forces.png`
under `figures/`. It reports mean drag, force RMS, Strouhal number and
Richardson/GCI estimates over the statistics window declared at the top of the
script.

`allclean.sh` removes generated solutions, samples and figures. It is never run
automatically.
