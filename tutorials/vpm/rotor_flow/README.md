# Wind-turbine wake

Run `python setup.py` (or `./allrun.sh`), then `./allplot.sh`; use `./allplot.sh pdf` for vector figures. The installed OpenONDA package supplies the solver and plotting dependencies.

The three-bladed turbine has a 6 m radius, 7 m/s wind and tip-speed ratio 7. Its prescribed rotation ramps smoothly over one nominal revolution. Native blade force/power/loading records and VLM/velocity samples are written under `samples/rotor/`. Native metadata and sparse numerical checkpoints are under `solution/`. Plotters read the run's own geometry, motion, density and sample times; they never regenerate design inputs or reconstruct a solver from a backup.

Blade chord Reynolds numbers are approximately 0.47–1.15 million, using the recorded chord, molecular viscosity and nominal relative speed `sqrt(U² + (Omega r)²)`. The wake LES resolves a different part of the flow from the inviscid blade-loading model.

The performance figure uses wind-turbine coefficient definitions:

- `CT = T / (0.5 rho U^2 A)`;
- `CP = P / (0.5 rho U^3 A)`, with positive `P` denoting extracted shaft power.

The native console also reports generic lift/drag coefficients using its displayed reference pressure and blade area. Use the disk-based `CT` and `CP` in these figures for the rotor theory comparison.

Power is the native fluid-on-blade rotational power, evaluated using actual angular speed, including the ramp. BEM uses the same recorded blade geometry, a thin-plate lift polar, Prandtl hub/tip losses and Buhl's high-induction relation. Loading profiles compare time-averaged sampled circulation and sectional lift with BEM. Wake profiles average native velocity samples in time and azimuth; the ideal far-wake reference includes streamtube expansion. Curves with more than 1% window drift are dotted.

The 14.4 s run samples forces every .012 s and fields every .06 s, with checkpoints about once per revolution. Wake planes at 1.5R, 3R and 4.5R are sampled during the last six revolutions. `python assets/validate_results.py --pre-plot` checks completion, force/power stationarity, BEM agreement, wake impulse and plane drift. Extend the run if the downstream wake has not settled; a plausible thrust coefficient alone does not establish a converged wake.

This is an inviscid blade-loading model coupled to a viscous/LES particle wake. It does not resolve blade boundary layers, transition, stall or airfoil profile drag. BEM agreement validates the corresponding attached-flow approximation, not every effect in a high-Reynolds-number turbine.

Reference: [CCBlade theory](https://wisdem.readthedocs.io/en/master/wisdem/ccblade/theory.html), citing Ning's bracketed inflow-angle method, Prandtl losses and Buhl's correction. The shared reference implementation lives in `openonda.rotor_theory`; independent momentum/scaling tests accompany it.

The current case uses Gaussian particles, common wake-core overlap 2.5 and
Pedrizzetti relaxation 0.3. The ongoing qualification run writes directly into
this tutorial's `solution/` and `samples/rotor/` folders. Its copied setup matches
these inputs. Performance/loading figures can be made before completion, but the
late wake-plane figures require those samples to have been written.
