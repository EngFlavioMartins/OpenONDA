# Tutorial cleanup and launcher inventory

The subsequent [all-tutorial qualification](2026-09-tutorial-qualification.md)
records the complete September 8 test matrix, remaining numerical failures, and
verification that existing result files were preserved.

The tutorial inputs now use the installed Python package directly. Run, plot, and clean are separate actions; result validation is an explicit assets command. No tutorial setup writes its own run metadata.

## Configuration and verification

The seven FVM configurations and meshes match the saved pre-cleanup configurations. All 27 VPM variant configurations preserve physical inputs, numerical methods, time steps, horizons, and sampling schedules. The recorded differences are replacement of tutorial sampler subclasses by standard samplers with `initial=True`, and the portable CPU device for the FMM rotor and coupled cube (Vulkan is unavailable on macOS and FMM does not support Metal). Flat-plate variants now own separate solution directories, so metadata cannot be overwritten by the next variant. The three LES budget configurations match their previous research recipe exactly.

`python install.py` installs into the Python environment invoking it, runs `pip check`, and verifies the installed package from outside the source tree. The Conda installer delegates to the same Python installer. The installation check exercises direct script arguments, bundled tutorial assets, plotting, native meshing, Taichi, and a native FVM solve.

The installer passed in a new venv using the existing environment's numerical dependencies, then installed and verified the repaired package in the current `/opt/anaconda3/envs/OpenONDA` environment. OpenONDA is a normal package in `site-packages`, not an editable checkout. Both checks ran outside the repository with isolated Python imports. This verifies a fresh OpenONDA installation; it is not a fresh download of every dependency.

## Numerical and portability evidence

- **115 focused tests passed** across tutorial physics/schema contracts, all installed launchers, copied local edits, metadata consumers, sampler output, public induction controls, and FVM snapshot selection. Ruff and `git diff --check` passed for this change.
- **All 20 templates** were materialized from the installed distribution. Launchers were checked for direct command order, failure propagation, preservation of previous outputs, and case-local cleanup. Ten argument-bearing scripts were also invoked directly using isolated Python from an unrelated directory, including copied paths containing spaces.
- **Current machine environment:** the installed `openonda tutorial run`, `plot`, and `clean` commands also completed a full disposable Taylor–Green case, with the same history rows. The managed tool sandbox denies installed-package and home-cache writes, so this machine-environment check ran with approved cache access; no tutorial or shell configuration was added.
- **Full Taylor–Green run:** all 10 accepted steps, through `t=0.05`, produce exactly the same history CSV rows as the saved pre-cleanup run. Final velocity L2 error is `1.638436274500633e-4`, energy error `7.037134666406301e-5`, and maximum continuity error approximately `9.54e-15`.
- **Full coupled uniform-flow run:** both VPM steps and all six FVM substeps completed outside the repository.
- **Two-step integrations at the original FVM mesh:** cube, boundary layer, cylinder IBM, backward step, and Cartesian mesher completed. Two-step CPU VPM integrations completed for delta wing, quadcopter, static flat plate at 8 degrees, isolated Lamb–Oseen with CS, and the transposed vortex ring. These use reduced run horizons, particle allocation, and diagnostic field resolution in disposable test drivers; production inputs retain their original values.
- **Cleanup after actual solves:** all seven completed FVM/coupled copies were cleaned by their absolute script path from `/tmp`. Generated outputs disappeared and every setup/asset hash remained unchanged.
- **Flat-plate output path:** moving 5° and static 8° completed two CPU steps at their original time step. All four figures rendered from these short runs and the failed static-5° diagnostic output; raw CSV hashes stayed unchanged. The moving run's solver record contained the ramp duration, start time, and final velocity. These figures verify data compatibility, not a completed polar or Kelvin experiment.
- **Other VPM plots:** delta-wing and quadcopter launchers completed on short-run output. Ring motion, energy, circulation, and stability figures completed in PNG and PDF; the unavailable LES scene was explicitly skipped.
- **FVM plots:** Taylor–Green, cube, boundary layer, cylinder IBM, and backward-step figures rendered from copied-case results. Cube and IBM snapshot readers now select the last physical time from the solver's PVD series; `mesh.vtu` cannot accidentally replace the solution fields.

### Limits found by actual execution

The cleanup does **not** establish that every complete research campaign passes its numerical acceptance checks:

| Case | Observed limit |
| --- | --- |
| VPM rotor | CPU execution reaches the solver, then its first accepted state fails the Lagrangian CFL limit: `26.4 > 1` at the authored `dt=0.006`. |
| VPM flat plate, static 5° | First accepted CPU state fails the Lagrangian CFL limit: `1.02 > 1` at the authored `dt=0.0125`. A disposable half-step experiment also failed (`1.25 > 1`); reducing the time step alone has not established a remedy. Static 8° completed its two-step check. |
| Lamb–Oseen plotting | Isolated-vortex field, comparison, and energy figures rendered from the short CS run. The final merger scene requires the unrun `merging_gbd` experiment; a complete three-physics, four-method plot set was not qualified. |
| FVM airfoil | The default native mesh contains 634,864 cells; construction exceeded the 180-second check budget before a solver step. Coarsening is not a numerical-equivalence test and did not qualify this case. |
| VPM interactions | The 61,160-particle default completed one accepted step but exceeded the 180-second two-step/output budget. A separate 9,920-particle, one-step check completed with solver-owned metadata. |
| Long coupled/reference cases | Installation, direct launchers, and relevant existing tests were checked; their full flow-development and grid-convergence campaigns were not run. |

The authored time steps, run lengths, geometry, and solver health limits have not been relaxed to make these checks pass. The reported CFL stops require a separate numerical investigation; they are not import or launcher failures.

## What changed and why

- **All cases:** direct Python run/plot commands; no interpreter variables, environment setup, campaign shell loops, automatic cleaning, or validation from the run launcher. No setup contains `try`, `raise`, or assertions.
- **Lamb–Oseen and vortex ring:** removed custom metadata and restart infrastructure; standard samplers express initial output. Plots consume solver metadata. The physical diffusion and stretching comparisons remain visible.
- **Vortex interactions:** removed the shell campaign controller. Three explicit LES comparison commands use a real physical input deck; advanced research orchestration is an optional assets script.
- **Flat plate:** removed post-run CSV rewriting and aliases. Plots read raw solver forces and spanwise samples, derive travel in memory from recorded motion, and select the final spanwise step. Each variant has its own solver metadata directory.
- **Rotor and quadcopter:** removed tutorial checks and duplicate configuration recording. Rotor geometry uses the supplied blade asset; explicit FMM CPU selection enables the supported macOS backend.
- **FVM airfoil, boundary layer, and backward step:** geometry and analytical extraction are small scientific assets. Setups show construction and execution without status banners and duplicate validation.
- **FVM Taylor–Green, cube, cylinder IBM, and Cartesian mesher:** removed runner orchestration, sanity checks, redundant defaults, and status output. Analytical error and reattachment loops remain where they express the experiment.
- **Coupled cases:** removed tutorial sanity gates and duplicate output records, extracted the NACA geometry calculation, and used public solver context managers in reference runs.

Reusable API repairs are limited to functionality useful outside tutorials: standard sampler `initial` selection, public treecode opening-angle/order controls, serialization of standard VLM motion parameters, and locating the latest FVM snapshot from the solver record. Package verification now proves direct copied-script execution. No new tutorial manager or configuration framework was introduced.

## Simplification inventory

Counts below compare the start of this pedagogical cleanup pass with the resulting setup. Earlier metadata cleanup in the shared checkout is not included in the starting counts.

| Case | Setup lines before → after | Run commands | Clean lines |
| --- | ---: | ---: | ---: |
| `coupled_fvm_vpm/cube_flow/reference_flow` | 178 → 175 | 3 | 4 |
| `coupled_fvm_vpm/cube_flow` | 409 → 385 | 1 | 6 |
| `coupled_fvm_vpm/cylinder_shedding_flow/reference_flow` | 210 → 207 | 4 | 4 |
| `coupled_fvm_vpm/cylinder_shedding_flow` | 331 → 329 | 1 | 5 |
| `coupled_fvm_vpm/naca4412_flow` | 236 → 199 | 1 | 5 |
| `coupled_fvm_vpm/uniform_flow` | 82 → 63 | 1 | 4 |
| `fvm/airfoil_flow` | 258 → 137 | 1 | 5 |
| `fvm/boundary_layer` | 204 → 115 | 1 | 5 |
| `fvm/cartesian_mesher` | 92 → 86 | 1 | 4 |
| `fvm/cube_flow` | 139 → 116 | 1 | 5 |
| `fvm/cylinder_ibm` | 160 → 130 | 1 | 5 |
| `fvm/step_profile` | 250 → 154 | 1 | 5 |
| `fvm/taylor_green` | 185 → 145 | 1 | 4 |
| `vpm/delta_wing` | 197 → 185 | 1 | 4 |
| `vpm/flat_plate` | 280 → 238 | 20 | 4 |
| `vpm/lamb_oseen_vortex` | 300 → 289 | 12 | 5 |
| `vpm/quadcopter` | 175 → 164 | 1 | 4 |
| `vpm/rotor_flow` | 261 → 225 | 1 | 4 |
| `vpm/vortex_interactions` | 293 → 263 | 3 | 4 |
| `vpm/vortex_ring` | 210 → 174 | 4 | 5 |

The LES comparison has its own physical input deck, `vortex_interactions/setup_les.py`, with one variant argument. Its advanced research script is in `assets/study.py`. No Python campaign runner replaces the removed shell infrastructure.

Geometry generation, pressure/profile extraction, and analytical error calculations live in case assets. These are scientific utilities, not orchestration frameworks. The rotor uses its shipped blade geometry; OpenVSP regeneration is an explicit auxiliary operation.

## Deliberate exceptions

- `allclean.sh` keeps one `cd` anchored to its own location so deletion cannot affect an unrelated working directory. Run and plot launchers are invoked from the case directory.
- Lamb–Oseen retains physical distinctions between isolated and paired vortices, DVH heat-transfer sampling, and RWM ensemble sampling. Its adaptive statistical stopping rule remains part of the experiment.
- Flat-plate motion branches distinguish ramped body motion from a static wind-frame experiment.
- Coupled cube restart initialization remains available to its explicit research script. Solver-level restart checks stay in the library.
- Taylor–Green and backward-step cases retain accepted-step loops to measure analytical error and reattachment histories. Their public solver context manager closes resources and finalizes metadata.
- Default long research campaigns are preserved. Short integration checks establish execution and output compatibility; they do not replace convergence or full-horizon physical qualification.

## Every run and clean launcher

The 20 run files contain 102 lines in total; the 20 cleaners contain 91 lines. The 16 plot launchers contain 81 lines. All 21 setup files are free of `try`, `raise`, and assertions.

All run files use the active `python`; `#!/bin/bash -e` stops at the first failed command. None invokes cleaning, plotting, or validation.

### coupled_fvm_vpm/cube_flow

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution constant samples figures
rm -rf __pycache__ assets/__pycache__
rm -f ./*.log
```

### coupled_fvm_vpm/cube_flow/reference_flow

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py --name coarse --dx 0.125
python setup.py --name medium --dx 0.0625
python setup.py --name fine --dx 0.03125
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples __pycache__
```

### coupled_fvm_vpm/cylinder_shedding_flow

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
rm -f ./*.log
```

### coupled_fvm_vpm/cylinder_shedding_flow/reference_flow

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py --name coarse --dx 0.125
python setup.py --name medium --dx 0.0625
python setup.py --name fine --dx 0.03125
python setup.py --name very_fine --dx 0.01563
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples __pycache__
```

### coupled_fvm_vpm/naca4412_flow

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution constant samples figures .matplotlib __pycache__ assets/__pycache__
rm -f ./*.log
```

### coupled_fvm_vpm/uniform_flow

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples
```

### fvm/airfoil_flow

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
rm -f assets/*.msh assets/*.vtk
```

### fvm/boundary_layer

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
rm -f assets/*.msh assets/*.vtk
```

### fvm/cartesian_mesher

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
```

### fvm/cube_flow

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
rm -f assets/*.msh assets/*.vtk
```

### fvm/cylinder_ibm

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
rm -f assets/*.msh assets/*.vtk
```

### fvm/step_profile

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
rm -f assets/*.msh assets/*.vtk
```

### fvm/taylor_green

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
```

### vpm/delta_wing

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
```

### vpm/flat_plate

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py --mode moving --angle -10
python setup.py --mode moving --angle -5
python setup.py --mode moving --angle -2
python setup.py --mode moving --angle 0
python setup.py --mode moving --angle 2
python setup.py --mode moving --angle 5
python setup.py --mode moving --angle 8
python setup.py --mode moving --angle 10
python setup.py --mode moving --angle 12
python setup.py --mode moving --angle 15
python setup.py --mode static --angle -10
python setup.py --mode static --angle -5
python setup.py --mode static --angle -2
python setup.py --mode static --angle 0
python setup.py --mode static --angle 2
python setup.py --mode static --angle 5
python setup.py --mode static --angle 8
python setup.py --mode static --angle 10
python setup.py --mode static --angle 12
python setup.py --mode static --angle 15
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
```

### vpm/lamb_oseen_vortex

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py vortex CS
python assets/rwm_ensemble.py vortex --number-of-realizations 10 --converge
python setup.py vortex DVH
python setup.py vortex GBD

python setup.py dipole CS
python assets/rwm_ensemble.py dipole --number-of-realizations 10 --converge
python setup.py dipole DVH
python setup.py dipole GBD

python setup.py merging CS
python assets/rwm_ensemble.py merging --number-of-realizations 10 --converge
python setup.py merging DVH
python setup.py merging GBD
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
rm -f ./*.log
```

### vpm/quadcopter

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
```

### vpm/rotor_flow

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
```

### vpm/vortex_interactions

`allrun.sh`

```bash
#!/bin/bash -e

python setup_les.py --variant baseline
python setup_les.py --variant p_moments
python setup_les.py --variant halfdt
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures study_results
```

### vpm/vortex_ring

`allrun.sh`

```bash
#!/bin/bash -e

python setup.py --variant dns_direct
python setup.py --variant dns_transposed
python setup.py --variant dns_mixed
python setup.py --variant les_transposed
```

`allclean.sh`

```bash
#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
rm -f ./*.log
```
