# Rejected hybrid cube-flow branch: 0.03 FVM core

This run tested only one change from `hybrid_filter10_t2p40_8de4709`: the
hybrid FVM maximum cell size was reduced from 0.04 to 0.03, matching the
nominal `referenceFlow` near-wake spacing.  The generated hybrid mesh had
1,012,160 cells.  The run was intentionally stopped after the first complete
field frame at 0.6 s because it failed the accuracy gate.

At 0.6 s, relative to `referenceFlow`:

- Cd error: +4.946% (the 0.04 baseline was -1.608%)
- stitched centerline Ux RMS error: 1.945% (baseline 1.153%)
- stitched off-axis Ux RMS error: 1.241% (baseline 0.623%)

The finer FVM generated stronger hand-off vorticity and degraded both the
interior solution and exterior VPM field.  Nominally matching the reference
wake spacing is therefore not a useful correction by itself.
