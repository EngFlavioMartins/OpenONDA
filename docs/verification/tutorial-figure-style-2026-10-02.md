# Tutorial figure migration, 2026-10-02

The approved Thesis palette and figure rules were read before this migration:
`Thesis/thesis_visuals/VISUAL_STYLE.md`, `styles/colors.py`, `styles/setup.py`,
`styles/fonts.py`, `styles/scientific_colormaps.py`, `styles/apply_openonda_theme.py`,
and `reference_style.py`.

The reusable implementation is `openonda/plotting.py`; the figure layout requirements is
[the tutorial figure standard](../tutorial-figure-style.md). All 18 active
`allplot.sh` launchers now default to PNG and PDF. Archived source snapshots under
`solution/` and `study_results/` were not migrated.

## Verification

- Regenerated **408 PDF/PNG pairs**, including the full coupled-cube time series.
- Checked every PDF page width against 125 mm (0.05 mm tolerance) and every PNG's
  physical width/height against its PDF (0.1 mm raster tolerance). No failures.
- Checked Matplotlib text size, clipping, overlap and symmetric margins during export.
- Inspected rendered examples from each available tutorial family, including the
  flat-plate and ring ParaView scenes, rotor panels, scalar maps and quadcopter.
- Rebuilt the rotor animation with the same signed/sequential palette.
- **198 targeted tests passed**: publication exports/layouts, plot inputs, launcher
  failure propagation, native scene source information, cube/cylinder plotting, and
  rotor/flat-plate comparison methodology. Ruff checks passed for the plotting files.
- The [machine-readable export inventory](tutorial-figure-exports-2026-10-02.json)
  records filenames, PDF dimensions, PNG dimensions and raster resolution.

The dual-format regression test also verifies that exporting `profile_t2.50`
preserves the decimal time in its filename and leaves the plotted arrays unchanged.

## Coverage and remaining source-data limits

| Tutorial | Export pairs | Status |
| --- | ---: | --- |
| Coupled cylinder, including reference flow | 4 | Rendered |
| Coupled cube, including reference grid figures | 355 | Rendered |
| Coupled NACA 4412 | — | No native force samples or local result bundle |
| FVM airfoil | 3 | Rendered from restored local bundle |
| FVM boundary layer | 2 | Rendered from restored local bundle |
| FVM cube | — | No native samples/snapshots or local result bundle |
| FVM cylinder IBM | 4 | Rendered from restored local bundle |
| FVM step profile | 2 | Rendered from restored local bundle |
| FVM Taylor–Green | 1 | Rendered |
| Lamb–Oseen | 7 | Rendered |
| Vortex ring | 6 | Rendered, including ParaView scenes |
| Vortex interactions | 4 | Rendered |
| Flat plate | 8 | Rendered, including ParaView scene |
| Delta wing | 5 | Rendered under `figures/partial/` |
| Rotor | 4 | Rendered; animation also rebuilt |
| Quadcopter | 3 | Rendered |

The plotting commands referenced by the launchers were exercised separately so
one missing dataset did not prevent checking other cases. This is not a claim
that every complete `allplot.sh` currently exits successfully: scientific
validators and run completion checks remain in place. The delta-wing saved run
failed at 6.9 s, so its partial diagnostics are not final cycle-averaged results.
Missing-input cases were not populated with invented or substitute data.

Presentation changes do not establish numerical accuracy. Native samples,
normalizations, theoretical comparisons and solver settings were preserved.
Explanatory qualifications removed from the canvas remain in console output,
scene metadata or the existing scientific reports. In particular, the restored
boundary-layer results still report their out-of-band Blasius errors.

The local MPI-enabled ParaView build required execution outside the restricted
sandbox because it opens a local socket during startup. Both scene renders then
completed successfully; no renderer dependency or solver was replaced.
