# Two heaving delta wings

Run `python setup.py` (or `./allrun.sh`), then `./allplot.sh`; use `./allplot.sh pdf` for vector figures. The installed OpenONDA package supplies the solver and plotting dependencies.

Two wings heave out of phase and pitch with the changing incident flow. The downstream wing crosses the upstream wake. Initial geometry pivots are specified before translation; prescribed-motion pivots are in the world frame. This distinction prevents the front wing from being translated twice.

Native force, motion, power and velocity samples are under `samples/delta_wing/`. Native coupled VPM+VLM backups and their VLM companion surface files are under `solution/`; the CSVs and wake-plane samples remain under `samples/delta_wing/`. The force figure compares both wings, their actual sampled centroids, motion input power and the last three complete cycles. The wake figure shows streamwise velocity and downwash averaged over the final heave period at three downstream planes. The sampled vertical window extends to z = −1.5 m to include the descending wake. The vector-strength magnitude has units m³/s and is not a conserved scalar circulation; its plot is a wake diagnostic.

Open `vlm.pvd` for the surface time series and the matching `vpm_*.xdmf`
files for the particles in the same solution directory. Their saved steps and
physical times match. `TimeValue` is VTK's reserved time-metadata name, used
when reading VTP files as a sequence; OpenONDA's own time field is `time`.

The authored run spans 10 heave cycles with 4000 accepted steps. Attached VLM
force and loading tables are recorded on every accepted step. The VPM-owned
backup clock is every 10 accepted steps (0.025 s, 40 coupled frames per cycle),
so each animation frame is a full coupled VPM+VLM restart state rather than a
separate VLM-only sample. `python assets/render_delta_wing_gif.py --fps 30`
selects the nearest coupled HDF5 backup for each physical 1/30 s target and
writes the source timestamp manifest beside the 30 fps GIF; it does not
synthesize intermediate solver states. The renderer uses one fixed oblique
x/y/z projection and one circulation color scale for the full sequence so the
span, chord and heave remain visible. Its `--fps` value is intentionally
constrained to 30 to match the centisecond duration pattern encoded in the GIF.
Plotters use native solver metadata and sampled data; no duplicate metadata or
checkpoint extraction is needed. By default, the plotters and native GIF
renderer read the accepted dense lineage (one fresh run or an ordered
fresh-prefix/continuation pair) in
`assets/delta_wing_accepted_lineage.json`. The lineage is finalized only after
the clean run reaches step 4000 / 10 s; before that point, plotting and GIF
generation must not present superseded sparse or interrupted output as current.
Pass `--samples`/`--solution` only for an explicit forensic source inspection.
Run `python assets/validate_results.py --pre-plot` to compare phase-resolved
loads between the final cycles and check the configured completion horizon.
Persistent cycle drift requires a longer run or an explicitly statistical
analysis. If a run resumes from a complete checkpoint, the manifest may
declare the fresh prefix and a separate continuation namespace; the same
finalizer validates both and the plotters/GIF renderer read their accepted
intervals in order.

The clean run has one logical fresh origin at step 0 / time 0, followed by full
coupled H5/XDMF/VTP states every 10 accepted steps (0.025 s) through step 4000.
The completion helper called by `allplot.sh` promotes the active lineage only
after the final native owner set, attached force/loading tables and endpoint
state pass their checks. `validate_results.py` checks completion, finite
samples and cycle behavior; the independent checkpoint audit covers exact
native owner membership and cadence. The wake-period plot uses trapezoidal time
weights across the one dense cadence:

```bash
python assets/validate_results.py --pre-plot
python assets/plot_delta_wing_forces.py
python assets/plot_delta_wing_circulation_history.py
python assets/plot_delta_wing_wake.py
python assets/render_delta_wing_gif.py --fps 30
```

The final GIF sidecar records the accepted source namespace(s), exact native
backup timestamps and encoded 30 fps timing. No interpolation or repeated
sparse-frame hold is permitted.

VLM supplies attached-flow circulation and Kutta–Joukowski loading, coupled to the trailing particle wake. This tutorial does not provide a separated leading-edge-vortex or viscous-stall model, and its 15 degree incidence should not be interpreted as experimental validation of those effects.

The fresh qualification run uses the CPU/FMM backend, Gaussian particles and
common wake-core overlap 2.5, with no Pedrizzetti relaxation. `setup.py` records
these same inputs. The current native output is assessed by the validator below;
a completed solver run is not automatically a converged solution.

The final dense native animation will be generated at
`assets/delta_wing_30fps.gif` only after the canonical run reaches step 4000
and the lineage manifest is finalized. Its sidecar will be
`assets/delta_wing_30fps.json`; no pre-completion animation is retained or
linked here.
