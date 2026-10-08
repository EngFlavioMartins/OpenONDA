# Known open issues

Limitations users may meet, and maintenance work still open. Report anything
else in the [issues](https://github.com/EngFlavioMartins/OpenONDA/issues); to
help fix one, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Limitations

- **Cartesian mesher on some closed bodies.** Wall projection leaves
  zero-area faces on a torus, and the finite NACA wing fails because the
  feature planes at its tips share no finite-facet intersection. The mesher
  rejects these meshes with an explicit error rather than producing invalid
  cells. Both cases are expected failures in
  `tests/fvm/test_mesh_consistency.py`; a fix will turn them into passes.
  Until then, import such geometries through Gmsh
  (`openonda.fvm.mesher.GmshImporter`).
- **Python 3.11 only.** Taichi 1.7 also supports Python 3.10 and 3.12, but
  OpenONDA is only tested on 3.11.
- **Linux and macOS only.** Windows is not supported.
- **Some tutorial figures predate the 2026-10-08 FVM fixes.** The PISO
  corrector update, freestream inflow flux and slip-wall stress changed the
  transient FVM solution. The Taylor–Green, boundary-layer, step and cube
  tutorials were re-run with the fixed solver. Still to refresh with
  `./allrun.sh` and `./allplot.sh` (or `./allcontinue.sh` to resume):
  `tutorials/fvm/airfoil_flow` (about 30 hours on a CPU) and the coupled
  `01_cylinder_shedding_flow` and `02_cube_flow` cases with their reference
  flows (GPU, several hours each).
- **Cube shedding frequency.** The FVM cube tutorial gives a mean drag
  coefficient of 1.536, inside the 1.45–1.58 reference band, and a Strouhal
  number of 0.135, just below the 0.140–0.150 band. The lift amplitude is
  still growing at the final time of 120 s, so a longer run is needed before
  the frequency is compared with the literature.
- **Restarts agree to solver tolerance, not bit for bit.** Preconditioners
  reused between steps are rebuilt after a restart, so a continued run
  matches an uninterrupted one to the linear-solver tolerance.
- **CUDA Gaussian image mesh needs CUDA headers.** Install
  `OpenONDA[gaussian-mesh-cuda12]`, or set `CUDA_PATH` to a CUDA 12 toolkit;
  otherwise the solver explains what is missing and CPU backends remain
  available.

## Maintenance

- **Public git history.** The history of `development`, `openfoam-journal`
  and `openfoam-wrapper` contains two copyrighted journal articles and about
  5.9 GB of retired outputs. Remove it when `main` is first published from
  `development`, as described in
  [docs/development/RELEASING.md](docs/development/RELEASING.md).
- **`main` is out of date.** It still holds release 0.1.1; publish the
  current tree, tag a release and mint its Zenodo DOI.
- **Large media in the tree.** About 100 MB of tutorial GIFs, videos and
  renders make clones heavy; move them to release assets.
- **cfMesh provenance.** Confirm the upstream commit `3ff855551482` cited in
  `openonda/fvm/mesh/CFMESH_ATTRIBUTION.md` and record it for every adapted
  module.
- **Static typing.** The package is fully annotated, but mypy and pyrefly
  report about 950 errors, mostly because mesh dictionaries are typed as
  `dict[str, object]`. A typed mesh structure would make type checking a CI
  gate.
- **GPU continuous integration.** The nightly GPU job needs a self-hosted
  runner with a CUDA device; GPU tests otherwise run only on maintainers'
  machines.
