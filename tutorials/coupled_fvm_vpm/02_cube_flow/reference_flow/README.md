# Cube reference flow

This is a three-grid body-fitted reference study for flow around a unit cube at
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
Successive grids have a refinement ratio of 1.5. Each command has only the
output name and baseline grid spacing:

```bash
python setup.py --name grid_h0045 -h 0.045
```

After all grids finish, restore archived samples if needed and plot with:

```bash
./allplot.sh
```

The script writes `grid_forces.json`, `grid_forces.csv`, `grid_forces.png`, and
`grid_forces_fluctuations.png` under `figures/`. It reports mean drag, force RMS,
Strouhal number and guarded Richardson/GCI estimates over the declared
statistics window. The present 15–30 s archive contains fewer than ten force
cycles, so its force-grid statistics are **not yet qualified**. The local
0.03 m run remains excluded from the launcher until its runtime and sampling
are qualified; it is not silently counted as a fourth completed grid.

`allrun.sh` calls `allclean.sh` before starting from zero. Use `allcontinue.sh`
to resume the three existing grid outputs from native backups.
