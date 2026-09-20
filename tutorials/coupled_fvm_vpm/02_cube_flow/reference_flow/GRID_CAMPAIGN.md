# Cube grid and timestep assessment

The cube has side D = 1 m; U = 1 m/s and ν = 0.001 m²/s give Re = 1000.
All campaign grids share the domain
`[-6.48,12.96] × [-6.48,6.48] × [-6.48,6.48] m`.
The near-body region uses h, the wake uses 2h, and the background uses 8h.
The mesher anchors these sizes to h, so the requested wall spacings are exact.

| Level | h (m) | Maximum timestep (s) | Horizon (s) |
| --- | ---: | ---: | ---: |
| Coarse | 0.10125 | 0.005 | 120 |
| Medium | 0.0675 | 0.005 | 120 |
| Fine | 0.045 | 0.005 | 30 |
| Dense | 0.03 | 0.005 | 120 |
| Fine timestep control | 0.045 | 0.0025 | 120 |

Successive spatial levels have r = 1.5. Three monotone levels support an
observed-order/Richardson/GCI estimate; the fourth checks whether the fitted
order predicts the next difference. Equal domains and proportional refinement
are necessary, but statistical drift or temporal error can still invalidate
that estimate. See the [NASA spatial-convergence guidance](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html).

`./allrun.sh` lists five direct commands. Each uses two ranks, force samples
every 0.05 s and profiles every 0.25 s. The fine grid runs to 30 s and writes
visualization frames and restart checkpoints every 0.25 s. These are independent
schedules: every visualization frame is retained, while checkpoints retain the
latest two committed generations. All other levels retain their 120 s horizon,
initial/final visualization and five-second checkpoint cadence.

Fresh outputs are under `campaigns/geometric_r15_30s_fine`;
the timestep control uses its `temporal/` subdirectory
and loads the exact saved fine native mesh. Run fine before the temporal control.
The already-running `geometric_r15` campaign retains its originally submitted
120 s horizon and output schedules. Editing the launcher does not reconfigure
an active simulation or its already-submitted shell commands.

Campaign execution requires the source checkout. Its numerical-assessment
owner is [cube_reference_campaign.py](../../../../studies/panel_removal/cube_reference_campaign.py).
It validates realized wall spacing/domain and queues larger reference grids
until recorded coupled experiments finish. `CUBE_WAIT_FOR_COUPLED=0` explicitly
bypasses this resource queue on an adequately provisioned host. Historical
2.23-million-cell runs reached approximately 10.7 GiB summed rank peak RSS;
the dense level needs a quiet machine. The queue does not guarantee available
memory or disk space.

To continue an existing level, run its command with `--restart-from` pointing
to that level's `backup` directory. Restart loads its own saved mesh. Existing
native outputs and recorded numerical settings must agree with the requested run.

With the fine grid ending at 30 s, a common statistics window must end no later
than 30 s. The previously planned 60–120 s comparison is unavailable for this
fresh campaign unless the fine run is extended. A preliminary 15–30 s comparison
can be requested explicitly:

```bash
python postprocess_grid_study.py --samples-root campaigns/geometric_r15_30s_fine/samples --solution-root campaigns/geometric_r15_30s_fine/solution --output-dir figures --statistics-start 15 --statistics-end 30 --tolerance 0.01 --format png
python postprocess_temporal_control.py --campaign-root campaigns/geometric_r15_30s_fine --output-dir figures/auxiliary --statistics-start 15 --statistics-end 30
```

Use `--format pdf` for PDF figures. Inspect mean drag, force fluctuation RMS,
pressure/viscous contributions, whole-line and wake profiles, and stationarity.
Require less than 1% drag drift, resolved shedding and small changes on the
finest pair before interpreting GCI. The timestep screen requires mean-drag
change ≤0.25% and profile RMS change ≤0.5% of U; force fluctuation and frequency
screens remain separate. Missing data or unresolved frequencies are reported
as unqualified. Longer windows or further refinement may be necessary.

A geometric grid sequence and frequent snapshots do not prove GCI or grid
independence. The unchanged stationarity and spectral screens still apply;
the historical fine case had 11.9% drag drift over 15–30 s and did not qualify.
The new 30 s run must establish its own evidence, and a 15 s window may contain
too few shedding cycles. Matching nominal wall spacing with the coupled mesh
also does not mean the two solvers use the same mesh or outer domain.
