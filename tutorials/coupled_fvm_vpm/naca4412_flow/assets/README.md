# Native FVM–VPM NACA 4412 flow

This finite-span NACA 4412 case at 10 degrees and Re=1000 uses OpenONDA's
internal Cartesian mesher and immersed-boundary FVM solver. It does not require
an external mesher or a repository path on `PYTHONPATH`.

After installing OpenONDA, run:

```sh
./allrun.sh
./allplot.sh
```

The production horizon is 12 convective time units. The VPM advances in 0.04 s
coupling windows while the native FVM uses four 0.01 s substeps per window to
keep the immersed-boundary transient within its CFL limit. Edit the physical and numerical constants in `setup.py` for shorter
or differently resolved experiments. Run `python assets/check_run.py` explicitly
for result validation, and `./allclean.sh` to remove generated output.

The public `compute_device` field in `VPM_CASE.numerics` explicitly selects the
CPU when no supported GPU is available. Generated fields are written below
`solution/`, sampling histories below `samples/`, and plots below `figures/`.

The default 2.5-cell marker separation avoids an ill-conditioned direct-forcing
quadrature where the thin section meets the finite-span end caps. The FVM time
step must divide the VPM coupling window exactly; the core API validates it.
