# Cube reference flow

This is the body-fitted Re=1000 cube reference case. The STL, domain, boundary
types, refinement regions, LES model, samplers, and solver settings are declared
in `create_solver()` in `setup.py`.

Run the complete three-grid study with:

```bash
./allrun.sh
```

The script runs `coarse`, `medium`, and `fine` in sequence. Fields and meshes
are written below `solution/<name>/`; samples are written below
`samples/<name>/`. To run one case directly, use for example:

```bash
python -u setup.py --name coarse --dx 0.125
```

The background spacing is fixed at 0.5D. The near-body region
`[-1.5,2.5] x [-1.5,1.5] x [-1.5,1.5]` resolves to `2h`, and the downstream
wake `[0,8] x [-2,2] x [-2,2]` resolves to `4h`. The cube surface resolves to
the requested `h`. The audited fine mesh contains 280,632 cells.

After the fine resolution completes, OpenONDA writes `grid_study.json`,
`grid_study.csv`, `grid_study.md`, and `grid_study.png` under `solution/`. The
report includes force statistics, Richardson/GCI estimates, and centreline and
off-axis profile changes over the common final-half time window.

`allclean.sh` removes generated solutions, samples, and grid-study outputs.
