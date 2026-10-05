"""ParaView batch renderer; invoked by plot_vortex_ring_scenes.py.

All labels are overlaid later by LaTeX. Geometry arrives as VTK PolyData.
"""

import json
import sys
from pathlib import Path

from paraview.simple import (
    AssignViewToLayout,
    ColorBy,
    CreateLayout,
    CreateView,
    GetColorTransferFunction,
    Glyph,
    Hide,
    SaveScreenshot,
    Show,
    XMLPolyDataReader,
)

from openonda.results import read_json

out = Path(sys.argv[1])
rgb_points = json.loads(sys.argv[2])
scene = read_json(out / "scene.json")
schematic_only = "--schematic-only" in sys.argv[3:]
view = CreateView("RenderView")
layout = CreateLayout("Vortex ring render")
AssignViewToLayout(view=view, layout=layout)
view.ViewSize = [1500, 1000]
view.UseColorPaletteForBackground = 0
view.Background = [1, 1, 1]
view.OrientationAxesVisibility = 0
focus = scene["camera_focal_point"]
view.CameraPosition = scene["camera_position"]
view.CameraFocalPoint = focus
view.CameraViewUp = [0, 0, 1]
view.CameraParallelProjection = 1
view.CameraParallelScale = scene["parallel_scale"]
view.ViewSize = scene["snapshot_view_size_px"]


def save_screenshot(name, resolution):
    path = out / name
    SaveScreenshot(str(path), layout, ImageResolution=resolution)


def read(name):
    return XMLPolyDataReader(FileName=[str(out / name)])


def particles(name, colored):
    g = Glyph(Input=read(name), GlyphType="Sphere")
    g.GlyphType.Radius = 1.0
    g.GlyphType.ThetaResolution = 20
    g.GlyphType.PhiResolution = 20
    g.OrientationArray = ["POINTS", "No orientation array"]
    g.ScaleArray = ["POINTS", "radius" if colored else "schematic_radius"]
    g.ScaleFactor = 1.0
    g.GlyphMode = "All Points"
    d = Show(g, view)
    d.Ambient = 0.25
    d.Diffuse = 0.75
    d.Specular = 0.28
    d.SpecularPower = 30
    if colored:
        ColorBy(d, ("POINTS", "strength"))
        lut = GetColorTransferFunction("strength")
        lut.RGBPoints = rgb_points
        lut.ColorSpace = "RGB"
        lut.RescaleTransferFunction(0.04, 0.8)
        d.SetScalarBarVisibility(view, False)
    else:
        d.ColorArrayName = ["POINTS", ""]
        d.DiffuseColor = [21 / 255, 96 / 255, 116 / 255]
        d.AmbientColor = d.DiffuseColor
    return g


def geometry(name, color, opacity=1):
    src = read(name + ".vtp")
    d = Show(src, view)
    d.ColorArrayName = ["POINTS", ""]
    d.DiffuseColor = color
    d.AmbientColor = color
    d.Ambient = 0.22
    d.Diffuse = 0.78
    d.Specular = 0.6
    d.SpecularPower = 40
    d.Opacity = opacity
    return src


for i in [] if schematic_only else [0, 1]:
    g = particles(f"particles_{i}.vtp", True)
    motion = geometry("motion_arrow", scene["arrow_color"])
    save_screenshot(f"particles_{i}.png", scene["snapshot_view_size_px"])
    Hide(g, view)
    Hide(motion, view)
focus = [0.045185642, -0.018074257, -0.087358927]
view.CameraPosition = [5 + focus[0], -2 + focus[1], 3 + focus[2]]
view.CameraFocalPoint = focus
view.CameraParallelScale = 1.3
view.ViewSize = [1880, 1640]
view.DepthPeeling = 1
objects = [particles("particles_0.vtp", False)]
objects.append(geometry("core_envelope", [0.72, 0.74, 0.76], 0.44))
for name in ["cut_edge", "other_edge"]:
    objects.append(geometry(name, [0.48, 0.51, 0.55]))
objects.append(geometry("core_scale", [0, 0.55, 0.7]))
for name in ["radius_arrow", "speed_arrow", "circulation_arrow"]:
    objects.append(geometry(name, scene["arrow_color"]))
save_screenshot("schematic.png", [2820, 2460])
for obj in objects:
    Hide(obj, view)
view.ViewSize = [1500, 1000]
view.CameraParallelScale = 0.11
view.CameraPosition = [5, -2, 3]
view.CameraFocalPoint = [0, 0, 0]
particles("core_particles.vtp", False)
geometry("detail_scale", [0, 0.55, 0.7])
geometry("core_radius_arrow", scene["arrow_color"])
save_screenshot("core_detail.png", [1500, 1000])
