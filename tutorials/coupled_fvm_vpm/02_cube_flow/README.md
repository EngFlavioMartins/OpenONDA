Run from your OpenONDA Python environment:

```bash
./allrun.sh
./allplot.sh
```

The cube occupies `[-0.5, 0.5]^3`; the FVM domain is `[-1.5, 1.5]^3`, giving one diameter of clearance on every side. Its mesh uses the same `12 × 0.06` background, `0.06` near-body box, and `0.06` cube-patch targets as `reference_flow/setup.py --name fine --dx 0.06`. The mesher resolves the nominal near-body and wall octree spacing to `0.045 m`; particle spacing is `0.06 m`. Four FVM processes use `dt=0.01 s`; RK2/GBD VPM and coupling use `dt=0.05 s`, through `t=20 s`. Both solvers retain equilibrium Smagorinsky LES at Re=1000.

The mixed vorticity boundary and buffered M4 renewal are iterated from the same FVM start and VPM predictor until the normal-velocity and tangential-gradient RMS residuals reach `1e-6`, with a maximum of 12 sweeps. Residuals and convergence status are recorded in `solution/coupler_diagnostics.jsonl`. Provisional sweeps do not write force or field samples. The body potential also enters particle transport; its analytical gradient and f64 differentiated boundary queries avoid single-precision differencing noise. The consistency source is disabled.

`allrun.sh` uses this checkout and limits thread oversubscription. The native mesh is cached in `constant/mesh.npz`; geometry, mesh-setting or mesher-code changes invalidate it automatically. Run `./allclean.sh --keep-mesh` to clear run outputs while retaining that cache; plain `./allclean.sh` also removes the mesh. Iteration snapshots stay in memory. Research replays and per-sweep archives are not part of the tutorial run.

The tutorial entry point writes directly into `samples/` and `solution/`. Before starting a new run, move any previous coupled outputs you want to keep; the launcher does not archive them. `reference_flow/` is separate.

`allplot.sh` compares the coupled samples with `reference_flow/samples/fine/` at exact shared times. The active fine reference setup records force and line samples but no plane slices. Before plotting, `allplot.sh` extracts the needed `z=0` slices from its saved `solution/fine/fine.pvd` field snapshots; it reuses slices already prepared. The reference field output cadence is one second, so field figures appear at shared one-second times. Other grid-study results remain in their separate subdirectories.

The three figure plotters called by `allplot.sh` are unchanged. Research plots and their generating scripts remain under `studies/coupler_accuracy/`. The candidate's demonstrated accuracy improvement is from the controlled short 3D DNS study; a full 20-second accuracy or speedup claim for this finer LES case requires the run itself.
