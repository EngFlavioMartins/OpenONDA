# Visualization style and colour maps

OpenONDA's figures use the shared [`set_thesis_style()`](../openonda/plotting.py)
helper. It sets the font, 10.95 pt text, physical figure sizes, named
colours and export defaults. Plot scripts should call it before creating any
artists and should use `openonda.plotting.COLORS` rather than local hex values.

## Scientific colour maps for ParaView

The ParaView presets in [`assets/scientific_colour_maps/`](assets/scientific_colour_maps/)
are the upstream Scientific Colour Maps 8.0.1 XML files. They are kept beside
the documentation so a checkout is self-contained: import the files in
ParaView's colour-map preset editor.

Use sequential, perceptually ordered maps for non-negative magnitudes:

| Quantity | Preset | Reason |
| --- | --- | --- |
| velocity magnitude | `lajolla` | warm, monotonic lightness progression |
| vorticity magnitude | `navia` reversed (`navia_r`) | cool-to-dark progression that preserves low values |
| positive scalar fields | `batlow` | general-purpose sequential alternative |
| neutral body/background | `grayC` / white | keeps geometry visually quiet |

Use the diverging `vik` or `berlin` maps only for signed fields whose zero has
physical meaning. Do not use a rainbow map for a magnitude, and keep a common
range when comparing panels or time steps.

The presets were downloaded from Fabio Crameri's Scientific Colour Maps 8.0.1
release ([Zenodo record 8409685](https://doi.org/10.5281/zenodo.8409685)); the
upstream catalogue and usage notes are at
[fabiocrameri.ch/colourmaps](https://www.fabiocrameri.ch/colourmaps/). The
ParaView import workflow is described in its
[colour-map documentation](https://docs.paraview.org/en/latest/ReferenceManual/colorMapping.html).

Open `solution/fvm.pvd` or `solution/vpm.pvd` to load the saved time series.
The VPM collection references `.vtu` visualization frames; the `.h5` files
are numerical restart files. See [solution layout](solution_layout.md).
