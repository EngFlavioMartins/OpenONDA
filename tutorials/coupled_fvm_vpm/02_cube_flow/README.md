Run from your OpenONDA Python environment:

```bash
./allrun.sh
./allplot.sh
```

The cube occupies `[-0.5, 0.5]^3`; the FVM domain is `[-1.5, 1.5]^3`, giving one diameter of clearance on every side. Its mesh uses the same `12 × 0.06` background, `0.06` near-body box, and `0.06` cube-patch targets as `reference_flow/setup.py --name fine --dx 0.06`. The mesher resolves the nominal near-body and wall octree spacing to `0.045 m`; particle spacing is `0.06 m`. Four FVM processes, RK2/GBD VPM and coupling use `dt=0.01 s`, through `t=20 s`. Both solvers retain equilibrium Smagorinsky LES at Re=1000. Samples remain every `0.05 s` and backups every `0.5 s`.

The previous `0.05 s` VPM step reached a Lagrangian strain increment of `1.05` at `t=15.5 s`, exceeding the limit of `1`. The `0.01 s` step would give `0.21` at that same strain; the health limit remains enabled. This reduces the VPM/coupling timestep without changing the FVM mesh, timestep or schemes. It requires five times as many VPM advances and coupling exchanges per simulated second, while the nominal FVM step count is unchanged. Full-run stability, accuracy and runtime still need validation.

The mixed vorticity boundary and buffered M4 renewal are iterated from the same FVM start and VPM predictor until the normal-velocity and tangential-gradient RMS residuals reach `1e-6`, with a maximum of 12 sweeps. Residuals and convergence status are recorded in `solution/coupler_diagnostics.jsonl`. Provisional sweeps do not write force or field samples. The body potential also enters particle transport; its analytical gradient and f64 differentiated boundary queries avoid single-precision differencing noise. The consistency source is disabled.

`allrun.sh` runs `python setup.py` in the active OpenONDA environment. The coupled factory accepts `FVM_SETUP`, `VPM_CASE` and `COUPLER_SETUP`; it launches the requested MPI ranks, creates VPM only on rank zero, owns logging, and closes its solvers when the context exits. `FVM_SETUP.cores` is the CPU allocation. The library limits each MPI rank's BLAS/Numba pools to one thread and gives owner-only particle work the case CPU budget. PETSc retains separate momentum and pressure workspaces internally. The tutorial requires no runtime exports or rank checks.

For development, install the checkout once with `python -m pip install -e .` from the repository root; subsequent source edits then apply from any case directory without `PYTHONPATH`. Normal package installations also work and use their installed solver version.

The native mesh is cached in `constant/mesh.npz`; geometry, mesh-setting or mesher-code changes invalidate it automatically. Run `./allclean.sh --keep-mesh` to clear run outputs while retaining that cache; plain `./allclean.sh` also removes the mesh. Iteration snapshots stay in memory. Research replays and per-sweep archives are not part of the tutorial run.

The tutorial entry point writes directly into `samples/` and `solution/`. Before starting a new run, move any previous coupled outputs you want to keep; the launcher does not archive them. `reference_flow/` is separate.

`allplot.sh` writes PNG figures by default; `./allplot.sh pdf` selects PDF and `./allplot.sh png` explicitly selects PNG. Every figure is exactly 12.5 cm wide, with LaTeX-rendered NewPX text/math at 10.95 pt. LaTeX, dvipng, newpxtext and newpxmath must be available. Include PDFs at natural size, without rescaling or cropping.

The comparison always uses the registered fine reference in `reference_flow/samples/fine/` and `reference_flow/solution/fine/`. Both saved FVM solutions are sampled with the same 3D affine reconstruction, using 12 native cell-centroid neighbours. Derived fields are cached in `samples/comparison/`; original samples, reference results and running simulations are untouched. The cache checks source files and MPI pieces before reuse. Profiles and fields use exactly coincident saved states, currently at one-second intervals. Old generated frames without matching states are removed after successful plotting.

Before the first common saved state, `allplot.sh` reports that comparison plots
are not ready and stops before changing the figures. Run it again once the
simulations have written those outputs; no simulation restart is needed.

Field colours show the full three-component velocity difference normalized by freestream speed. Contours interpolate vectors for display; RMS and maximum differences use the original sample grid. Missing fluid coverage stays masked, and colour scales include the full sampled maximum. `reference_fvm_fields_*` compares the primary near-body solution. `reference_vpm_fields_*` compares the auxiliary VPM field inside the FVM region; it is not a whole-domain hybrid error map.

`figures/comparison_audit.md`, `comparison_audit.json` and `field_differences.csv` document the definitions, coverage and source checks. `reference_force_audit.*` shows raw drag and the accepted timestep at each force sample. The current fine reference has pressure/drag spikes coincident with very small timesteps; they remain visible, without smoothing or filtering. This needs resolution before its drag can validate hybrid accuracy. The schemes and local Cartesian sizing match, but fitted meshes and timestep histories differ.

Research plots and their generating scripts remain under `studies/coupler_accuracy/`. A full 20-second accuracy or speedup claim for this finer LES case requires validation beyond these instantaneous comparisons.
