# Figure conventions

Tutorial plot scripts use `openonda.plotting`. Call `set_thesis_style()` before creating a figure and `save_fig()` to export it. For custom layouts, use `fit_thesis_y_label_margins()`, `validate_thesis_figure()`, and `export_figure()`.

## Export

`./allplot.sh` writes PDF and PNG; append `pdf` or `png` for one format. Use a 125 mm canvas width, 10.95 pt New PX text, and 400 dpi PNG output (600 dpi for dense particle scenes). The installed environment provides the required LaTeX fonts and renderer.

Keep labels, units, time, and colour bars visible. Use inward ticks, 0.5 pt black axes, and no background grid. Share legends between panels with the same series. Avoid `bbox_inches="tight"`, which changes the physical canvas size.

## Series and fields

Use `method_style()` for consistent series:

| Result | Colour | Line | Marker |
| --- | --- | --- | --- |
| FVM | Ocean blue | Solid | Square |
| VPM/VLM | Light blue | Solid | Circle |
| FVM–VPM | Amber | Solid | Diamond |
| Theory, experiment, or reference solver | Gray | Dashed | Optional open marker |

Keep variant colours and markers consistent across a case's figures. Space markers with `markevery`; retain all samples in the line. Plot unordered surface samples with markers alone.

Use `thesis_blue` for magnitudes, `thesis_signed` for signed fields with a meaningful zero, and `thesis_amber` for streamlines over particle scenes. The colour bar must use the rendered field's normalization and limits. Figure styling does not change the physical data or establish simulation accuracy; see [flow comparisons](visualization.md).
