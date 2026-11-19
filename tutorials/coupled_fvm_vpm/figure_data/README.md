# Figure data for the paper and the thesis

Copies of the data files that the paper and thesis figure scripts read from the `data/`
folder beside each script. The folder layout below `paper/` and `thesis/` is the layout of
the two repositories, so a file is restored by copying it to the same relative path.
All files come from the runs on the final coupling algorithm (solver exports 37d7d7e3 and
b56a3c12); each file's header or `frame.json` names its source frame.

The particle frames of the baseline cylinder run are shared as well, in
`01_cylinder_shedding_flow/solution/sensitivity_analysis/box2.4_sub5_inset8.75/solution/vpm/`
(341 frames, `vpm_000000.vtu` to `vpm_002500.vtu`; flow time in seconds = frame number / 25).
