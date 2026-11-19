# FVM reference for the tandem cylinders

The two cylinders of the [coupled case](../README.md) on one body-fitted FVM mesh, on the domain and boundaries of the [case 01 reference](../../01_cylinder_shedding_flow/reference_flow/README.md): $[-16,30]\times[-16,16]$ m, an inlet upstream, an outlet downstream and slip walls at the sides, one periodic cell across the span. Cells of 0.04 m fill both coupled FVM domains and the gap between them, $[-2.4,12.4]\times[-2.4,2.4]$ m; cells of 0.08 m follow the wake to the outlet. The step is the coupled FVM step, 0.008 s.

The mesh cannot hold two bodies in relative motion, so there is no reference for the heaving downstream cylinder.

## Run

```bash
./allrun.sh
```

`./allcontinue.sh` resumes the latest FVM backup. A run writes `samples/fixed/forces_upstream.csv` and `forces_downstream.csv` every 0.04 s and the transverse lines 2 and 4 D behind each cylinder every 0.12 s.
