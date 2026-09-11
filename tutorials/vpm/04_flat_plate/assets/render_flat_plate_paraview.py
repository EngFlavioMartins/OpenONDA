#!/usr/bin/env python3
"""Render the prepared flat-plate particle, VLM, and motion-arrow geometry."""

from __future__ import annotations

import argparse

from paraview.simple import (  # type: ignore[import-not-found]
    ColorBy,
    CreateView,
    GetColorTransferFunction,
    Glyph,
    HideScalarBarIfNotNeeded,
    ResetSession,
    SaveScreenshot,
    Show,
    XMLPolyDataReader,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--arrows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--omega-min", required=True, type=float)
    parser.add_argument("--omega-max", required=True, type=float)
    args = parser.parse_args()

    ResetSession()
    view = CreateView("RenderView")
    view.ViewSize = [1800, 900]
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = [-11.6, 0.0, -0.75]
    view.CameraPosition = [-11.6, -28.0, 12.0]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelScale = 7.0

    particles = XMLPolyDataReader(FileName=[args.particles])
    glyphs = Glyph(Input=particles, GlyphType="Sphere")
    glyphs.OrientationArray = ["POINTS", "No orientation array"]
    glyphs.ScaleArray = ["POINTS", "glyph_radius"]
    glyphs.ScaleFactor = 1.0
    glyphs.GlyphMode = "All Points"
    glyphs.GlyphType.Radius = 1.0
    glyphs.GlyphType.ThetaResolution = 8
    glyphs.GlyphType.PhiResolution = 8
    particle_display = Show(glyphs, view)
    particle_display.Representation = "Surface"
    ColorBy(particle_display, ("POINTS", "vorticity_magnitude"))
    omega_lut = GetColorTransferFunction("vorticity_magnitude")
    viridis = [
        (0.000, 0.267004, 0.004874, 0.329415),
        (0.125, 0.278826, 0.175490, 0.483397),
        (0.250, 0.229739, 0.322361, 0.545706),
        (0.375, 0.172719, 0.448791, 0.557885),
        (0.500, 0.127568, 0.566949, 0.550556),
        (0.625, 0.157851, 0.683765, 0.501686),
        (0.750, 0.369214, 0.788888, 0.382914),
        (0.875, 0.678489, 0.863742, 0.189503),
        (1.000, 0.993248, 0.906157, 0.143936),
    ]
    span = args.omega_max - args.omega_min
    omega_lut.RGBPoints = [
        component
        for fraction, red, green, blue in viridis
        for component in (args.omega_min + fraction * span, red, green, blue)
    ]
    omega_lut.ColorSpace = "RGB"
    particle_display.LookupTable = omega_lut
    HideScalarBarIfNotNeeded(omega_lut, view)

    surface = XMLPolyDataReader(FileName=[args.surface])
    surface_display = Show(surface, view)
    surface_display.Representation = "Surface With Edges"
    surface_display.ColorArrayName = [None, ""]
    surface_display.DiffuseColor = [0.78, 0.79, 0.80]
    surface_display.AmbientColor = [0.78, 0.79, 0.80]
    surface_display.EdgeColor = [0.22, 0.24, 0.27]
    surface_display.LineWidth = 0.7

    arrows = XMLPolyDataReader(FileName=[args.arrows])
    arrow_display = Show(arrows, view)
    arrow_display.Representation = "Surface"
    arrow_display.ColorArrayName = [None, ""]
    arrow_display.DiffuseColor = [0.78, 0.31, 0.08]
    arrow_display.AmbientColor = [0.78, 0.31, 0.08]

    view.Update()
    SaveScreenshot(
        args.output,
        view,
        ImageResolution=[1800, 900],
        TransparentBackground=0,
        CompressionLevel=0,
    )


if __name__ == "__main__":
    main()
