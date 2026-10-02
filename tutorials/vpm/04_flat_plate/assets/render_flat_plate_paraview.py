#!/usr/bin/env python3
"""Render the prepared flat-plate particle, VLM, and motion-arrow geometry."""

from __future__ import annotations

import argparse
import json

from paraview.simple import (  # type: ignore[import-not-found]
    AssignViewToLayout,
    ColorBy,
    CreateLayout,
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
    parser.add_argument("--camera", required=True)
    parser.add_argument("--color-points", required=True)
    args = parser.parse_args()
    camera = json.loads(args.camera)
    color_points = json.loads(args.color_points)

    ResetSession()
    view = CreateView("RenderView")
    layout = CreateLayout("Flat plate render")
    AssignViewToLayout(view=view, layout=layout)
    view.ViewSize = [1800, 900]
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = camera["focal_point_m"]
    view.CameraPosition = camera["position_m"]
    view.CameraViewUp = camera["view_up"]
    view.CameraParallelScale = camera["parallel_scale"]

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
    span = args.omega_max - args.omega_min
    omega_lut.RGBPoints = [
        component
        for fraction, red, green, blue in color_points
        for component in (args.omega_min + fraction * span, red, green, blue)
    ]
    omega_lut.ColorSpace = "RGB"
    particle_display.LookupTable = omega_lut
    HideScalarBarIfNotNeeded(omega_lut, view)

    surface = XMLPolyDataReader(FileName=[args.surface])
    surface_display = Show(surface, view)
    surface_display.Representation = "Surface With Edges"
    surface_display.ColorArrayName = [None, ""]
    surface_display.DiffuseColor = [191 / 255, 35 / 255, 38 / 255]
    surface_display.AmbientColor = [191 / 255, 35 / 255, 38 / 255]
    surface_display.EdgeColor = [0.22, 0.24, 0.27]
    surface_display.LineWidth = 3.1

    arrows = XMLPolyDataReader(FileName=[args.arrows])
    arrow_display = Show(arrows, view)
    arrow_display.Representation = "Surface"
    arrow_display.ColorArrayName = [None, ""]
    arrow_display.DiffuseColor = [239 / 255, 156 / 255, 31 / 255]
    arrow_display.AmbientColor = [239 / 255, 156 / 255, 31 / 255]

    view.Update()
    written = SaveScreenshot(
        args.output,
        layout,
        ImageResolution=[1800, 900],
        TransparentBackground=0,
        CompressionLevel=0,
    )
    if not written:
        raise RuntimeError(f"ParaView failed to save screenshot: {args.output}")


if __name__ == "__main__":
    main()
