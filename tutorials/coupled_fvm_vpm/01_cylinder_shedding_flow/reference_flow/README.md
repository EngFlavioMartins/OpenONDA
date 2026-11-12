# Cylinder FVM reference at $Re=150$

This FVM-only case provides the reference for the [coupled cylinder](../README.md). Geometry, freestream, viscosity, startup field and force normalization match the coupled case.

| Quantity | Value |
| --- | --- |
| Domain | $[-16,30]\times[-16,16]$ m |
| Mesh | $h$ over the coupled FVM domain $[-2.4,4.4]\times[-2.4,2.4]$ m, $2h$ over $|y|\le3$ m from there to the outlet, $16h$ elsewhere; $h=0.04$ m for the coupled baseline |
| Boundaries | Velocity inlet, zero-pressure outlet, slip sides and no-slip cylinder |
| Time step; end time | fixed, $U_\infty\Delta t/h=0.2$ (0.008 s at $h=0.04$ m); 100 s |

The domain follows the literature rather than a study of its own: slip walls 16 D from the axis, where Behr et al. (1995, Comput. Methods Appl. Mech. Engrg. 123) find the lift amplitude and mean drag at $Re=100$ within 0.2% of their widest domain, and an outlet 30 D downstream, twice the 14.5 D at which Tezduyar and Shih (1991, J. Eng. Mech. 117) find no change of the solution. `allrun.sh` runs the references of the coupled study, the time step refined with the cells:
- the grid series at $Re=150$ with $h=0.08$, 0.06, 0.04 and 0.03 m, to 100 s, which gives the same-mesh reference of every coupled run on the 2.4 $D$ box or a smaller one, and the FVM's own grid error;
- `h0.04_dt0.008_fb3.2` and `h0.04_dt0.008_fb4.0`, to 55 s, with cells of $h$ over the coupled FVM domain of the box margins 3.2 and 4.0 $D$ (`--fine-box-margin`).

Forces are written every 0.04 s; wake profiles and fields every 0.2 s; backups every 1 s.

## Run

Run commands from this directory.

```bash
./allrun.sh
./allplot.sh
```

`./allcontinue.sh` resumes the latest accepted backup. Each case writes force coefficients, wake profiles and centreline samples to `samples/<name>/`; `../allplot.sh` performs the coupled comparison.
