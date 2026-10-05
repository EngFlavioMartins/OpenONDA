# Flat plate: bound circulation and a particle wake

Compare a stationary plate in inclined flow with the same plate moving through still fluid. The case tests VLM loading, wake shedding and frame agreement. See [surface mesh setup and VLM assumptions](../../../docs/vpm.md#vlm-surfaces-and-wakes) and [LES/core spreading](../../../docs/vpm.md#diffusion-and-les).

## Run

From this directory:

```bash
python -m openonda.tutorial_runner . setup --mode static --angle 5
python -m openonda.tutorial_runner . setup --mode moving --angle 5
```

`./allrun.sh` removes previous output and runs ten angles in each frame, from −10° to 15°. `./allcontinue.sh` resumes saved native backups. `./allplot.sh` produces figures; add `pdf` for PDF exports.

## Geometry, flow and resolution

Edit [setup.py](setup.py): chord $c=1$ m, span $b=10$ m, reference speed $U=10$ m/s, density $\rho=1$ kg/m³ and viscosity $\nu=0.01$ m²/s. Thus $Re_c=Uc/\nu=1000$.

The generated surface has 8 chordwise and 14 spanwise panels per half-span, with geometric spacing ratio 4 toward the end. Change these counts, chord and span to create a new rectangular wing. The static plate remains at $x=0$ to 1 m while flow travels toward positive $x$. The moving plate accelerates toward negative $x$, with a 0.12 s ramp, and leaves the wake behind its trailing edge.

The wake uses Gaussian particles, direct induction, transposed stretching, core spreading and Smagorinsky LES with $C_s=0.30$. `wake_core_overlap=2.5` sets trailing and transverse wake cores from the larger span spacing or convected row length. The step is 0.0125 s; each plate travels 24 chord lengths (2.4 s static, approximately 2.46 s moving). Loads include the unsteady pressure contribution from changing bound potential jump.

## Results and reference

Force, spanwise/chordwise loading and flow-integral CSVs are in `samples/exp_<mode>_aoa<angle>/`. For geometry and wakes, open `solution/<case>/vlm.pvd` and `vpm.pvd` in ParaView; coupled checkpoints occur every 0.5 s.

Figures show settled lift, induced drag and pitching moment, spanwise loading and induced velocity, moving/static load histories, and the saved particle wake. The steady polar averages the final five chord lengths. The reference is rectangular-wing lifting-line theory, an attached-flow, small-incidence, high-aspect-ratio approximation. The wake uses a higher perspective view with the projected motion horizontal, the saved plate surface shaded in neutral grey, blue particle glyphs and one light-grey motion arrow. Editable geometry and its ParaView state are saved under `figures/auxiliary/flat_plate_wake/`.

Coupled mesh/time/core refinement is still needed to establish physical accuracy. VLM does not resolve skin friction, boundary layers, stall or separation; the 15° case does not validate those effects.
