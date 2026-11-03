# Cylinder FVM reference at $Re=150$

This FVM-only case provides the reference for the [coupled cylinder](../README.md). Geometry, freestream, viscosity, startup field and force normalization match the coupled case.

| Quantity | Value |
| --- | --- |
| Domain | $[-16,30]\times[-16,16]$ m |
| Mesh | 0.04 m over the coupled FVM domain $[-2.4,4.4]\times[-2.4,2.4]$ m, 0.08 m over $|y|\le3$ m from there to the outlet, 0.64 m elsewhere |
| Boundaries | Velocity inlet, zero-pressure outlet, slip sides and no-slip cylinder |
| Time step; end time | adaptive, cell Courant number at most 0.9; 100 s |

The domain follows the literature rather than a study of its own: slip walls 16 D from the axis, where Behr et al. (1995, Comput. Methods Appl. Mech. Engrg. 123) find the lift amplitude and mean drag at $Re=100$ within 0.2% of their widest domain, and an outlet 30 D downstream, twice the 14.5 D at which Tezduyar and Shih (1991, J. Eng. Mech. 117) find no change of the solution. `allrun.sh` lists every case: $Re=100$ and 150 at 0.04 m, and the grid sensitivity at the highest Reynolds number, $Re=250$, at 0.08, 0.0566, 0.04 and 0.0283 m. Forces, wake profiles and backups are written every 0.1 s.

## Run

Run commands from this directory.

```bash
./allrun.sh
./allplot.sh
```

`./allcontinue.sh` resumes the latest accepted backup. Each case writes force coefficients, wake profiles and centreline samples to `samples/<name>/`; `../allplot.sh` performs the coupled comparison.
