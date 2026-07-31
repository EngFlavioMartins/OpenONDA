# FVM performance and code audit

## Result

The production speed limit was PETSc matrix transfer, not a missing JIT flag.
Uploading each owned CSR block in one call replaced one Python call per matrix
row. On the 448k-cell cube mesh, the median FVM step fell from 35.83 s to
8.52 s (4.2×) with identical residuals, continuity, y+, and force coefficients.

Warmed median time per serial FVM step:

| Assembly backend | 100k cells | 400k cells |
|---|---:|---:|
| NumPy | 2.963 s | 12.513 s |
| Numba | 1.732 s | 12.675 s |
| Taichi CPU | 1.386 s | 12.464 s |

The 400k-cell timings are effectively equal. The production FVM therefore
uses NumPy assembly and partitioned PETSc. The VPM remains on Metal in f32,
where Taichi accelerates the particle kernels. The FVM remains CPU/f64 because
pressure and force parity has not been established for reduced precision.

## Changes retained

- PETSc owned rows use bulk `Mat.setValuesCSR` insertion.
- Least-squares geometry and uniform-face topology checks use batched NumPy.
- The public setup is `FVMSetup` with explicit scheme, linear-solver, PIMPLE,
  force, time, output, and execution configurations.
- `phi` is consistently named volumetric face flux, with units m³/s.
- Dead loop assembly, one-off diffusion workflow, pass-through backend object,
  obsolete aliases, and non-functional FVM device/precision controls were
  deleted.

An attempted reuse of one PETSc matrix across the three momentum components
increased step time to 40–50 s and was removed. Reusing a stale GAMG
preconditioner also failed the production A/B (148.0 s versus 146.5 s per
coupling window) and was removed. Relaxing the pressure tolerance from 1e-8
to the OpenFOAM inner tolerance of 1e-6 preserved drag to 1e-9 but was slower
(159.6 s versus 146.5 s) and increased the continuity error, so 1e-8 remains.

## Code-quality findings

Generated-code style cannot identify whether a human, ChatGPT, or Claude wrote
a file. Objective warning signs are more useful: duplicated implementations,
undefined helpers, pass-through abstractions, verbose comments that restate
code, names with incorrect physical meaning, and configuration switches that
do not alter execution. Those instances were removed from the audited path.

The remaining large functions mostly encode coupled numerical branches rather
than generic software architecture. Refactor them only with term-by-term
physics tests; lowering a complexity score is not worth obscuring the discrete
equations.

## Reproducing the production run

From `tutorials/coupled_FVM_VPM/cubeFlow`:

```bash
./allrun.sh
```

`FVMSetup(cores=4)` relaunches the FVM under MPI/PETSc. Only rank zero creates
the Taichi/Metal VPM. The case requires no manual `mpiexec`, path bootstrap, or
backend environment variables.

## References

- Numba performance guidance: <https://numba.readthedocs.io/en/stable/user/performance-tips.html>
- Taichi performance tuning: <https://docs.taichi-lang.org/docs/performance>
- PETSc sparse matrix assembly: <https://petsc.org/release/manual/mat/>
- ChatGPT code-quality study: <https://arxiv.org/abs/2307.12596>
- Copilot code-quality study: <https://sanadlab.org/assets/pdf/NguyenMSR22.pdf>
