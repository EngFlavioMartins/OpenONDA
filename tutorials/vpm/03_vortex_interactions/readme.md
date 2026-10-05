# Leapfrogging vortex rings

Two equal, coaxial Gaussian rings exchange radius and overtake each other. The DNS baseline is compared with selective eddy viscosity, strength alignment and particle splitting. See [toroidal initialization](../../../docs/vpm.md#particle-distributions) and [diffusion and stabilization](../../../docs/vpm.md#diffusion-and-les).

## Run

From this directory:

```bash
./allrun.sh
./allplot.sh
```

`allrun.sh` removes previous results and runs `baseline`, `selective_eddy_viscosity`, `pedrizzetti_relaxation` and `particle_splitting`. For one case, use `python -m openonda.tutorial_runner . setup baseline`. `./allcontinue.sh` resumes saved native checkpoints; `./allplot.sh pdf` exports PDF.

## Inputs and model comparisons

Edit [setup.py](setup.py): both rings have radius $R_0=1$ m, scalar circulation $\Gamma_0=\pi$ m²/s and Gaussian core $a_0=0.1$ m. Their centres start at $x=\pm0.5$ m. $Re_\Gamma=3000$ gives $\nu=\pi/3000$ m²/s. There is no imposed perturbation.

Toroidal particles have $h=\sigma=0.05$ m with physical-core compensation. All cases use transposed treecode induction, SSPRK3 and core spreading at $\Delta t=0.00375$ s, requesting 2400 steps (9 s). Conservative group-preserving redistribution is common to every case.

| Variant | Added numerical model |
| --- | --- |
| `baseline` | Molecular viscosity only. |
| `selective_eddy_viscosity` | Positive-stretching diffusion, coefficient $C=0.5=2C_w^2$. |
| `pedrizzetti_relaxation` | Aligns strength toward local vorticity, with global strength/impulse restoration. |
| `particle_splitting` | Every five steps, bisects particles whose strength doubles; children retain core radius. |

The added models can change leapfrogging timing and dissipation. Longer survival does not establish improved physical accuracy. A particle state limit stop leaves a partial trajectory.

## Read the results

`samples/<method>/` holds integrals, group histories and meridional fields. `solution/<method>/` holds checkpoints. Figures show field-core paths, core sections, energy, enstrophy and group centroids.

The [Cheng et al. reference](assets/references/README.md) is the unperturbed $Re_\Gamma=3000$ Fig. 5 case. Its periodic boundaries and much finer spacing differ from this unbounded VPM calculation; the overlay is a comparison, with convergence still required.

Core paths follow dominant positive vorticity maxima. Tracking ends when a bridge or competing peak prevents assigning two cores. `group_id` tracks each initial ring's vorticity contribution after redistribution, even after merger; group centroids need not remain physical vortex centres. Particle indices change during redistribution and splitting, so index trails cannot represent fluid trajectories.
