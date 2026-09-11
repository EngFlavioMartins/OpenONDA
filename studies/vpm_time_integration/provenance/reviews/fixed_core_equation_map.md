# Equation and implementation map

All line numbers refer to the working tree inspected on 2026-09-10 unless a
historical object is named explicitly.

| ID | Mathematical object | OpenONDA / source anchor | Probe / evidence anchor |
|---|---|---|---|
| E1 | `zeta(rho)=pi^(-3/2) exp(-rho^2)` and `q(rho)=[erf(rho)-(2/sqrt(pi))rho exp(-rho^2)]/(4pi)`, with small-`rho` series and the audited unmatched splice | Host `source/solvers/vpm/kernels/base.py:387–404`; shared device factory `source/solvers/vpm/kernels/gaussian.py:15–86`; LBVH `source/solvers/vpm/physics/induction/treecode/lbvh.py:1140–1171`; shared constants `source/solvers/vpm/config/constants.py:95–107` | `probe.py:32–59,287–345`; `derivative_checks.csv`; `kernel_crossover.csv` |
| E2 | Pair core `sigma_ij=(sigma_i+sigma_j)/2`; `U_i=sum_j K_sigmaij(X_i-X_j) Gamma_j` | `source/solvers/vpm/numerics/kernels_common.py:568–601` | `probe.py:73–103` |
| E3 | Finite centre gradient and regularized off-centre gradient | `source/solvers/vpm/numerics/kernels_common.py:638–684` | `probe.py:73–103`; `derivative_checks.csv`; `kernel_crossover.csv` |
| E4 | Particle state `Gamma=omega V`, with units `m^3/s` | `source/solvers/vpm/physics/induction/base.py:45–69` | `probe.py:170–189` |
| E5 | Default/historical transposed stretching `S_i=(grad U_i)^T Gamma_i` | `source/solvers/vpm/physics/induction/stretching.py:7–13`; historical `bb1718c9:source/solvers/vpm/config/stretching.py:40–49` | `probe.py:112–122` |
| E6 | Coupled state `Y=(X,Gamma)` evaluated at common stages with fixed core over the RK step | `source/solvers/vpm/numerics/runge_kutta.py:184–339`; `source/solvers/vpm/physics/stage_rhs.py:301–335` | `probe.py:124–166` |
| E7 | SSPRK3 tableau `a21=1`, `a31=a32=1/4`, `b=(1/6,1/6,2/3)` | `source/solvers/vpm/numerics/rk_tableaux.py:87–103` | `probe.py:125–129` |
| E8 | Current accepted step: one coupled inviscid RK update; core spreading half steps around it; other diffusion after it | `source/solvers/vpm/core/evolution.py:274–280,469–502` | Explicitly excluded from the fixed-core probe; boundary recorded in `README.md` |
| E9 | Historical predecessor: full advection SSPRK3, then full stretching SSPRK3 at the advected positions | `bb1718c9:source/solvers/vpm/core/evolution.py:295–316`; advection stages `bb1718c9:source/solvers/vpm/physics/engine.py:334–363`; strength stages `:844–896` | `probe.py:139–145` (`historical_lie`) |
| E10 | Justified comparison split: `A/2`, `B`, `A/2`, each subproblem advanced by SSPRK3 | Not a production path | `probe.py:146–150` (`symmetric_strang`) |
| E11 | Volume spacing `ell_i=V_i^(1/3)`; sampled fill/mesh-ratio proxy | The same volume length is the LES filter width at `source/solvers/vpm/turbulence/smagorinsky.py:99–113`, but it is not the Gaussian core radius | `probe.py:205–257`; `geometry.csv` (`9x9x9` sampled box, not continuum supremum) |
| E12 | Scaled-Euclidean centered-difference tangent estimate and sampled logarithmic-norm/spectral diagnostics | Derived diagnostics, not production solver features or certified continuous bounds | `probe.py:348–453`; `tangent.csv` |

## Sign and tensor convention

For `r_ij=X_i-X_j`, production first accumulates
`q (r_ij x Gamma_j)/|r_ij|^3` and then negates the sum, so the mathematical
velocity used here is `q (Gamma_j x r_ij)/|r_ij|^3`.  The stored gradient has
row/column convention `J_ab=partial u_a/partial x_b`.  The probe uses
`J^T Gamma` because that is the historical/default `TRANSPOSED` mode being
compared.  The continuum material stretching contraction under this tensor
convention would be `J Gamma`; this report does not claim the two formulations
are identical.

## Historical reconstruction commands

```bash
git show bb1718c9:source/solvers/vpm/core/evolution.py
git show bb1718c9:source/solvers/vpm/physics/engine.py
git show bb1718c9:source/solvers/vpm/config/setup.py
git show bb1718c9:source/solvers/vpm/config/advection.py
git show bb1718c9:source/solvers/vpm/config/stretching.py
```
