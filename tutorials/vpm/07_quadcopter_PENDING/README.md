# Quadcopter in climb

Run `python setup.py` (or `./allrun.sh`), then `./allplot.sh`. Use `./allplot.sh pdf` for vector figures. The installed OpenONDA package supplies the solver and plotting dependencies.

Four counter-rotating, two-bladed flat-plate rotors operate at 4000 rpm in 0.8 m/s axial climb. The leading edges face the actual rotational relative wind; the quarter chords lie on each radial axis. Rotation starts at the operating speed. A spin-up in an already established axial inflow would initially expose almost stationary plates to nearly normal flow, which this attached-flow VLM does not describe.

At this operating point the blade chord Reynolds number is approximately 22,000–63,000, using `sqrt(U² + (Omega r)²) c / nu`. These small rotors are not in the wind-turbine tutorial's Reynolds-number range. The theoretical comparison assesses attached inviscid loading; transition and profile losses would require a blade-section model beyond this VLM.

The setup declares geometry, motion, fluid properties, resolution and native
samplers. Attached VLM loading is recorded on every accepted VPM step. The
VPM-owned backup clock is the sole coupled surface-output clock: `solution/`
holds restartable VPM+VLM HDF5 backups and their native VLM companion files,
while force/loading CSVs and sampled flow fields are in `samples/quadcopter/`.
Plotters use those native records and the actual blade geometry. They do not
import the setup, reconstruct a solver, or extract checkpoints.

`quadcopter_performance` shows thrust and input shaft power for each rotor. Positive shaft input is the negative of the native fluid-on-blade rotational power. The solver records each blade's torque about its own axis and contracts it with its actual angular velocity, so counter-rotation does not cancel power. Coefficients use the rotorcraft convention:

- `CT = T / (rho A (Omega R)^2)`;
- `CP = P / (rho A (Omega R)^3)`, with `A = pi R^2` for one rotor.

The native console also reports generic lift/drag coefficients using its displayed reference pressure and blade area. Use the disk-based `CT` and `CP` in these figures for the rotor theory comparison.

The reference uses the recorded chord/pitch distribution in an isolated-rotor blade-element/momentum calculation with tip/hub losses and an inviscid thin-plate lift polar. The axial ideal-momentum curve includes climb power. These references omit rotor interference, airfoil profile drag, stall and transition; they assess the inviscid loading model, not real motor electrical power. `quadcopter_wake` averages native velocity planes over the last six revolutions. Dashed circles mark the projected disks. Enstrophy is a separate wake diagnostic, not a convergence criterion.

The authored run spans 24 nominal revolutions at 3.75 degrees per step. VLM
forces and power are sampled on every accepted step, fields 8 times per
revolution, and full coupled restart backups 32 times per revolution
(every three steps, 11.25 degrees of rotation). The 24-revolution run therefore
provides 768 distinct VPM+VLM states for a slowed 30 fps animation, rather than
sampling the blades repeatedly at the same azimuth. There is no
independent VLM surface sampler. `python assets/validate_results.py` checks
completion and tail load drift against native metadata, compares BEM, checks
the ideal power requirement and rotor symmetry, and requires complete final
six-revolution velocity-plane histories. It compares the two three-revolution
mean vector fields relative to the induced velocity, with a 3% drift limit.
Subtracting the freestream prevents a large uniform inflow from hiding wake
changes. The native flow-integral sampler also records bound and coupled linear
impulse per density. The validator compares the final six-revolution coupled-
impulse change with integrated blade thrust, allowing 10% discrepancy; a
failure calls for checking wake health and any boundary losses. This comparison
requires retaining the wake inside the domain. Run it with `--pre-plot` to
assess numerical results before rendering figures. A completed solver run is
not automatically a converged solution.

Reference formulation: [CCBlade theory and its cited Ning/Buhl papers](https://wisdem.readthedocs.io/en/master/wisdem/ccblade/theory.html). The propeller branch is checked independently against annular axial and angular momentum in `tests/tutorials/test_rotor_theory.py`.

The quadcopter is a demonstration after validation with the flat plate, combined
motion of the delta wings, and the single wind-turbine rotor. It needs a stable,
complete run and useful native wake/load figures; a separate isolated-propeller
validation campaign is not a prerequisite. Earlier [diagnostic studies](studies/README.md)
and their results are retained for reproducibility, but are not separate
validation tutorials. No additional isolated quadcopter-rotor studies are scheduled.
