"""Render the saved flat-plate surface, wake particles and motion arrow."""

from __future__ import annotations

import argparse
import json

from paraview.simple import (
    AssignViewToLayout,
    ColorBy,
    CreateLayout,
    CreateView,
    GetColorTransferFunction,
    Glyph,
    HideScalarBarIfNotNeeded,
    ResetSession,
    SaveScreenshot,
    SaveState,
    Show,
    XMLPolyDataReader,
)


def reference_lighting(view):
    """Use the author's schematic light kit and filmic tone mapping."""
    for name, value in {
        "UseEnvironmentLighting": 0,
        "UseLight": 1,
        "UseToneMapping": 1,
        "ToneMappingType": 3,
        "UseFXAA": 1,
        "Exposure": 1.5,
        "Contrast": 1.6773,
        "Shoulder": 0.9714,
        "MidIn": 0.18,
        "MidOut": 0.18,
        "KeyLightIntensity": 0.75,
        "KeyLightAzimuth": 10,
        "KeyLightElevation": 50,
        "KeyLightWarmth": 0.6,
        "FillLightAzimuth": -10,
        "FillLightElevation": -75,
        "FillLightWarmth": 0.4,
        "BackLightAzimuth": 110,
        "BackLightElevation": 0,
        "BackLightWarmth": 0.5,
        "HeadLightWarmth": 0.5,
        "FillLightKFRatio": 3,
        "BackLightKBRatio": 3.5,
        "HeadLightKHRatio": 3,
    }.items():
        setattr(view, name, value)


def shaded_material(display, colour=None):
    display.Representation = "Surface"
    display.Interpolation = "PBR"
    display.Metallic = 0.0
    display.Roughness = 0.3
    display.CoatStrength = 0.0
    if colour is not None:
        display.ColorArrayName = [None, ""]
        display.DiffuseColor = colour
        display.AmbientColor = colour


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--arrows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-output", required=True)
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
    view.ViewSize = camera["image_pixels"]
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    reference_lighting(view)
    view.CameraParallelProjection = 0
    view.CameraFocalPoint = camera["focal_point_m"]
    view.CameraPosition = camera["position_m"]
    view.CameraViewUp = camera["view_up"]
    view.CameraViewAngle = camera["view_angle_degrees"]
    particles = XMLPolyDataReader(FileName=[args.particles])
    glyphs = Glyph(Input=particles, GlyphType="Sphere")
    glyphs.OrientationArray = ["POINTS", "No orientation array"]
    glyphs.ScaleArray = ["POINTS", "glyph_radius"]
    glyphs.ScaleFactor = 1.0
    glyphs.GlyphMode = "All Points"
    glyphs.GlyphType.Radius = 1.0
    glyphs.GlyphType.ThetaResolution = 16
    glyphs.GlyphType.PhiResolution = 12
    particle_display = Show(glyphs, view)
    shaded_material(particle_display)
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
    shaded_material(surface_display, [96 / 255, 104 / 255, 108 / 255])
    arrows = XMLPolyDataReader(FileName=[args.arrows])
    arrow_display = Show(arrows, view)
    shaded_material(arrow_display, [0.64, 0.64, 0.64])
    view.Update()
    SaveScreenshot(
        args.output,
        layout,
        ImageResolution=camera["image_pixels"],
        TransparentBackground=0,
        CompressionLevel=0,
    )
    SaveState(args.state_output)


if __name__ == "__main__":
    main()
