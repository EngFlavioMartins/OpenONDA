# Pitching NACA 0012 airfoil: FVM reference

The FVM alone solves the [coupled pitching airfoil](../README.md) on a body-fitted mesh of $[-20,40]\times[-25,25]$ m that turns with the airfoil about its quarter chord. The cells match the coupled mesh over the box that the coupled FVM domain sweeps, then coarsen to two and four times the spacing through the wake. The inlet and side boundaries hold the undisturbed stream in the inertial frame and the outlet a fixed pressure; one periodic cell spans the airfoil.

```bash
./allrun.sh
./allplot.sh
```

`./allcontinue.sh` resumes the latest FVM backup; `python setup.py --end-time 0.5` runs a short check. `figures/forces` shows the drag, lift and quarter-chord moment.
