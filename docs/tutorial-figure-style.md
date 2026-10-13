# Figure conventions

Tutorial plot scripts use `openonda.plotting`. Call `set_thesis_style()` before creating a figure and `save_fig()` to export it. For custom layouts, use `fit_thesis_y_label_margins()`, `validate_thesis_figure()`, and `export_figure()`.

## Export

`./allplot.sh` writes PDF and PNG; append `pdf` or `png` for one format. Use a 125 mm canvas width, 10.95 pt New PX text, and 400 dpi PNG output (600 dpi for dense particle scenes). The one-command installation provides LaTeX and the New PX fonts; without LaTeX, `set_thesis_style()` draws the same figures with Matplotlib's math text in a Palatino-like serif.

Keep labels, units, time, and colour bars visible. Use inward ticks, 0.5 pt black axes, and no background grid. Share legends between panels with the same series. Avoid `bbox_inches="tight"`, which changes the physical canvas size.

## Series and fields

Keep series styles consistent:

| Result | Colour | Line | Marker |
| --- | --- | --- | --- |
| FVM | Ocean blue | Solid | Square |
| VPM/VLM | Light blue | Solid | Circle |
| FVM–VPM | Amber | Solid | Diamond |
| Theory, experiment, or reference solver | Gray | Dashed | Optional open marker |

Keep variant colours and markers consistent across a case's figures. Space markers with `markevery`; retain all samples in the line. Plot unordered surface samples with markers alone.

Use `thesis_blue` for magnitudes, `thesis_signed` for signed fields with a meaningful zero, and `thesis_amber` for streamlines over particle scenes. The colour bar must use the rendered field's normalization and limits. Figure styling does not change the physical data or establish simulation accuracy; see [flow comparisons](visualization.md).

## Layout

Author each layout at its native 125 mm width; do not let tight cropping or automatic layout resize the canvas.

- Measure the final left y-label at production font size and choose a left margin `x` that leaves 2–3 pt of clearance; set the right boundary to `1 - x`. Time-series families share one margin sized for their longest labels.
- Put titles 2–3 pt below the top; keep them short ("Aspect ratio", "Co-rotating"). Leave a small gap between the bottom x label and the legend frame.
- Hard-code subplot, colour-bar and legend positions per figure, compacting unused gaps while keeping room for labels and ticks.
- Draw visible curves, axes, ticks, marker edges and annotations at least 0.6 pt wide; ordinary curves use 1 pt.
- Show at most two significant digits in ticks, colour bars, annotations, legends and times; keep full precision in the data. Use shared exponents or offsets when rounding would hide small changes.
- Use snake_case file names. Rendered 3-D views fill the width on a shorter canvas, with colour limits, glyph scales and camera kept identical between compared states.

Validate exports with `validate_thesis_figure()` and inspect them at print size. Scientific renders use saved solver states; never rerun a simulation only to restyle a figure.
