# Vortex ring: stretching and LES

A perturbed Gaussian ring translates through otherwise still fluid. Compare discrete stretching formulations and the effect of Smagorinsky LES on mode growth. See [toroidal particles](../../../docs/vpm.md#particle-distributions), [stretching](../../../docs/vpm.md#induction-and-stretching) and [LES](../../../docs/vpm.md#diffusion-and-les).

## Run

From this directory:

```bash
./allrun.sh
./allplot.sh
```

`allrun.sh` removes previous generated output and runs all four variants. `./allcontinue.sh` resumes compatible checkpoints. To run one variant, use `python setup.py --variant dns_transposed`; alternatives are `dns_direct`, `dns_mixed` and `les_transposed`. `./allplot.sh pdf` exports PDF.

## Inputs and interpretation

Edit [setup.py](setup.py):

| Quantity | Value |
| --- | --- |
| Major radius $R_0$; Gaussian core $a_0$ | 1 m; 0.1 m. |
| Scalar circulation $\Gamma_0$ | $\pi$ m²/s. |
| $Re_\Gamma=\Gamma_0/\nu$ | 3000; $\nu=\pi/3000$ m²/s. |
| Particle spacing $h$; particle radius $\sigma$ | 0.035 m; 0.07 m. |
| Perturbation | 24 broadband Widnall modes, dimensionless amplitude 0.005 relative to $R_0$, seed 42. |
| Step; requested duration | 0.02 s; 60 s (3000 steps). |

Toroidal support is truncated at 5% of the represented Gaussian tail. Core compensation accounts for particle smoothing. All variants use treecode induction, SSPRK3 and core spreading. `les_transposed` adds $C_s=0.20$ Smagorinsky viscosity; the three DNS variants change only the discrete stretching choice.

Samples every 0.1 s track ring position, radius, circulation, mode amplitudes, energy and numerical health. Checkpoints every 0.5 s are in `solution/<variant>/`; diagnostics are in `samples/<variant>/`. Figures compare translation, energy, circulation, stretching stability and ring shape.

The calculation stops if configured strain, divergence or misalignment limits are exceeded. A stopped curve is a partial trajectory. Compare instability onset only while the physical core and particle overlap remain resolved; numerical growth is not sufficient evidence of a physical Widnall instability.
