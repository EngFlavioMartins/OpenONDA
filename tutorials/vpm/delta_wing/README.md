# Two heaving delta wings

Run `python setup.py` (or `./allrun.sh`), then `./allplot.sh`; use `./allplot.sh pdf` for vector figures. The installed OpenONDA package supplies the solver and plotting dependencies.

Two wings heave out of phase and pitch with the changing incident flow. The downstream wing crosses the upstream wake. Initial geometry pivots are specified before translation; prescribed-motion pivots are in the world frame. This distinction prevents the front wing from being translated twice.

Native force, motion, power and velocity samples are under `samples/delta_wing/`. The force figure compares both wings, their actual sampled centroids, motion input power and the last three complete cycles. The wake figure shows streamwise velocity and downwash averaged over the final heave period at three downstream planes. The sampled vertical window extends to z = −1.5 m to include the descending wake. The vector-strength magnitude has units m³/s and is not a conserved scalar circulation; its plot is a wake diagnostic.

The authored run spans 10 heave cycles with 400 steps, 100 force samples and 25 field snapshots per cycle. Checkpoints are saved once per cycle. Plotters use native solver metadata and sampled data; no duplicate metadata or checkpoint extraction is needed. `python assets/validate_results.py --pre-plot` compares phase-resolved loads between the final cycles and checks the configured completion horizon. Persistent cycle drift requires a longer run or an explicitly statistical analysis.

VLM supplies attached-flow circulation and Kutta–Joukowski loading, coupled to the trailing particle wake. This tutorial does not provide a separated leading-edge-vortex or viscous-stall model, and its 15 degree incidence should not be interpreted as experimental validation of those effects.

The current qualification run uses the CPU/FMM backend, Gaussian particles and
common wake-core overlap 2.5, with no Pedrizzetti relaxation. `setup.py` records
these same inputs. The ongoing run writes directly into this tutorial's native
output folders; its full ten-cycle horizon is still being assessed.
