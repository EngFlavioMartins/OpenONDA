# Lamb–Oseen vortices: diffusion, translation and merger

Compare core spreading (CS), random walk (RWM), Diffused Vortex Hydrodynamics (DVH) and grid-based diffusion (GBD) for an isolated vortex, a counter-rotating dipole and a co-rotating pair. See [particle distributions](../../../docs/vpm.md#particle-distributions) and [diffusion models](../../../docs/vpm.md#diffusion-and-les).

## Run

From this directory with OpenONDA installed:

```bash
./allrun.sh
./allplot.sh
```

`allrun.sh` removes previous generated results. `./allcontinue.sh` resumes saved native backups. For one case, use `python -m openonda.tutorial_runner . setup vortex CS`; physical choices are `vortex`, `dipole`, `merging`, and methods are `CS`, `RWM`, `DVH`, `GBD`. A single RWM run is one realization; use `python -m openonda.tutorial_runner . assets.rwm_ensemble vortex --number-of-realizations 10` for the ensemble comparison. `./allplot.sh pdf` exports PDF.

## Physical and numerical inputs

Edit [setup.py](setup.py):

| Quantity | Value |
| --- | --- |
| Scalar circulation $\Gamma_0$ | ±1 m²/s; pair signs select translation or merger. |
| Circulation Reynolds number $\lvert\Gamma_0\rvert/\nu$ | 530; $\nu=1/530$ m²/s. |
| Initial velocity-peak radius $a_{c,0}$ | 0.125 m. |
| Gaussian vorticity radius $a_0=a_{c,0}/1.12$ | 0.1116 m. |
| Pair separation $b_0$; column length | 1 m; 5 m. |
| Particle spacing $h$; overlap $\sigma/h$ | 0.075 m; 1.2. |
| Step $\Delta t$; duration | 0.03233 s; 29.973 s (927 steps). |

The particles form a triangular transverse lattice extruded along $z$. Initial core compensation separates physical vortex width from particle smoothing. RK2 and transposed stretching are common to all methods. CS/RWM use direct induction; DVH/GBD use treecode. These are finite columns with unbounded induction, so end effects differ from an infinite two-dimensional vortex.

For the isolated Gaussian vortex, the radial reference is

$$
a^2(t)=a_0^2+4\nu t,\qquad
u_\theta(r,t)=\frac{\Gamma_0}{2\pi r}\left[1-e^{-r^2/a^2(t)}\right].
$$

The dipole tests mutual translation; the co-rotating pair tests separation, orientation and merger against [Cerretelli–Williamson data](assets/references/README.md). Figures use $\nu t/a_{c,0}^2$; this radius differs from the Gaussian radius above.

RWM uses ten independent seeds. Features are extracted from the ensemble-mean field; shaded intervals describe Monte Carlo uncertainty. See the short [RWM methodology](assets/references/rwm_statistical_methodology.md).

Fields and integrals are in `samples/<case>/`; checkpoints are in `solution/<case>/`. Figures compare profiles, kinetic-energy decay, dipole motion and merger. Diffusion domains include the physical spread; if you change duration or viscosity, enlarge them accordingly.
