# state file generated using paraview version 6.1.0
from pathlib import Path

import paraview

paraview.compatibility.major = 6
paraview.compatibility.minor = 1

from paraview.simple import *


# Resolve inputs from the tutorial rather than the machine where this state
# file was generated.  ParaView accepts ``Path`` values only inconsistently
# across releases, so keep the public reader arguments as strings.
CASE_DIR = Path(__file__).resolve().parent.parent
SOLUTION_DIR = CASE_DIR / "solution"

# Keep the saved camera when displaying the first source.
paraview.simple._DisableFirstRenderCameraReset()

# setup views used in the visualization

renderView1 = CreateView("RenderView")
renderView1.Set(
    ViewSize=[1920, 1080],
    OrientationAxesVisibility=0,
    CenterOfRotation=[5.231250152587891, 0.0, 0.0],
    UseToneMapping=1,
    UseAmbientOcclusion=1,
    CameraPosition=[15.009017210607619, -19.65209775934984, 20.01461842555845],
    CameraFocalPoint=[4.955890763484405, 0.24291572660225969, 0.35246784658047137],
    CameraViewUp=[-0.34967806985877253, 0.5632158884176522, 0.7486742352276845],
    CameraViewAngle=13.971051406860353,
    UseColorPaletteForBackground=0,
    Background=[1.0, 1.0, 1.0],
)

SetActiveView(None)

# setup view layouts

layout1 = CreateLayout(name="Layout #1")
layout1.AssignView(0, renderView1)
layout1.SetSize(1920, 1080)

SetActiveView(renderView1)


# setup the selections

selectionSource0 = CreateSelection(
    proxyname="IDSelectionSource",
    registrationname="SelectionSource0",
    groupname="selections",
    IDs=[0, 2655],
)

appendSelections = CreateSelection(
    proxyname="AppendSelections",
    registrationname="AppendSelections",
    groupname="selections",
    Input=selectionSource0,
    Expression="s0",
    SelectionNames=["s0"],
)

# setup the data processing pipelines

vpmpvd = PVDReader(
    registrationName="vpm.pvd",
    FileName=str(SOLUTION_DIR / "vpm.pvd"),
)
vpmpvd.PointArrays = [
    "core_radius",
    "eddy_viscosity",
    "effective_viscosity",
    "group_id",
    "kinematic_viscosity",
    "particle_volume",
    "velocity",
    "vortex_strength",
    "vorticity",
    "zone_id",
]

clip1 = Clip(registrationName="Clip1", Input=vpmpvd)
clip1.Set(
    ClipType="Box",
    Invert=0,
)

clip1.ClipType.Set(
    Position=[-1.5, -2.5, -0.05],
    Length=[3.0, 2.3850001096725464, 1.5],
)

clip1.HyperTreeGridClipper.Origin = [
    5.647500157356262,
    -5.960464477539063e-08,
    -5.960464477539063e-08,
]

cubestl = STLReader(
    registrationName="cube.stl",
    FileNames=[str(CASE_DIR / "assets" / "cube.stl")],
)

fvmpvd = PVDReader(
    registrationName="fvm.pvd",
    FileName=str(SOLUTION_DIR / "fvm.pvd"),
)
fvmpvd.CellArrays = [
    "velocity",
    "kinematic_pressure",
    "courant_number",
    "vorticity",
    "cell_size",
    "refinement_level",
    "boundary_layer_index",
    "cell_volume",
    "cell_equivalent_size",
    "global_cell_id",
]

cellDatatoPointData1 = CellDatatoPointData(registrationName="CellDatatoPointData1", Input=fvmpvd)

slice1 = Slice(registrationName="Slice1", Input=cellDatatoPointData1)
slice1.SliceOffsetValues = [0.0]

slice1.SliceType.Set(
    Origin=[1.125, 0.0, 0.0],
    Normal=[0.0, 0.0, 1.0],
)

slice1.HyperTreeGridSlicer.Origin = [1.125, 0.0, 0.0]

appendSelections.SetSelectionId(slice1.GetGlobalID())
appendSelections.SetSelectionPort(0)

# setup the visualization in view 'renderView1'

fvmpvdDisplay = Show(fvmpvd, renderView1, "UnstructuredGridRepresentation")

fvmpvdDisplay.Set(
    Representation="Outline",
    ColorArrayName=["POINTS", ""],
    LineWidth=2.0,
    RenderLinesAsTubes=1,
)

slice1Display = Show(slice1, renderView1, "GeometryRepresentation")

velocityLUT = GetColorTransferFunction("velocity")
velocityLUT.Set(
    RGBPoints=[
        # scalar, red, green, blue
        0.0016831935004858933,
        0.0,
        0.0,
        0.0,
        0.0788327186378327,
        0.0,
        0.062745,
        0.0,
        0.1559822437751795,
        0.0,
        0.12549,
        0.0,
        0.23313238369876538,
        0.0,
        0.188235,
        0.0,
        0.3102819088361122,
        0.0,
        0.25098,
        0.0,
        0.38743143397345897,
        0.0,
        0.313725,
        0.0,
        0.46458095911080577,
        0.0,
        0.376471,
        0.0,
        0.5417304842481526,
        0.094118,
        0.439216,
        0.0,
        0.6188803598136556,
        0.196078,
        0.501961,
        0.0,
        0.6960301493090854,
        0.294118,
        0.564706,
        0.0,
        0.7731796744464321,
        0.396078,
        0.627451,
        0.0,
        0.850329199583779,
        0.498039,
        0.690196,
        0.0,
        0.9274787247211258,
        0.6,
        0.752941,
        0.145098,
        1.0046288646447115,
        0.701961,
        0.815686,
        0.364706,
        1.0817783897820583,
        0.8,
        0.878431,
        0.580392,
        1.1589279149194054,
        0.901961,
        0.941176,
        0.796078,
        1.2312556715839478,
        1.0,
        1.0,
        1.0,
    ],
    NanColor=[1.0, 0.0, 0.0],
    ScalarRangeInitialized=1.0,
)

slice1Display.Set(
    Representation="Surface",
    ColorArrayName=["POINTS", "velocity"],
    LookupTable=velocityLUT,
    EdgeColor=[0.9176470637321472, 0.9176470637321472, 0.9176470637321472],
)

slice1Display.ScaleTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

slice1Display.OpacityTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

cubestlDisplay = Show(cubestl, renderView1, "GeometryRepresentation")

sTLSolidLabelingLUT = GetColorTransferFunction("STLSolidLabeling")
sTLSolidLabelingLUT.Set(
    RGBPoints=GenerateRGBPoints(
        range_min=0.0,
        range_max=1.1757813367477812e-38,
    ),
    ScalarRangeInitialized=1.0,
)

cubestlDisplay.Set(
    Representation="Surface",
    ColorArrayName=["POINTS", ""],
    LookupTable=sTLSolidLabelingLUT,
    Interpolation="PBR",
    Luminosity=27.0,
    Roughness=0.42,
    Metallic=0.04,
)

clip1Display = Show(clip1, renderView1, "UnstructuredGridRepresentation")

vorticityLUT = GetColorTransferFunction("vorticity")
vorticityLUT.Set(
    AutomaticRescaleRangeMode="Never",
    RGBPoints=GenerateRGBPoints(
        preset_name="Viridis",
        range_min=0.0,
        range_max=5.0,
    ),
    NanColor=[1.0, 0.0, 0.0],
    ScalarRangeInitialized=1.0,
)

clip1Display.Set(
    Representation="Point Gaussian",
    ColorArrayName=["POINTS", "vorticity"],
    LookupTable=vorticityLUT,
    InterpolateScalarsBeforeMapping=0,
    PointSize=3.0,
    DisableLighting=1,
    BackfaceRepresentation="Points",
    Pickable=0,
    OSPRayUseScaleArray="All Exact",
    GaussianRadius=0.02,
    ShaderPreset="Plain circle",
    OpacityByArray=1,
    OpacityArray=["POINTS", "vorticity"],
    OpacityArrayComponent="Magnitude",
)

clip1Display.ScaleTransferFunction.Points = [0.000133218, 0.0, 0.5, 0.0, 5.0, 1.0, 0.5, 0.0]

clip1Display.OpacityTransferFunction.Points = [0.000133218, 0.0, 0.5, 0.0, 5.0, 1.0, 0.5, 0.0]

vpmpvdDisplay = Show(vpmpvd, renderView1, "UnstructuredGridRepresentation")

vpmpvdDisplay.Set(
    Representation="Point Gaussian",
    ColorArrayName=[None, ""],
    GaussianRadius=0.1269000029563904,
)

vpmpvdDisplay.ScaleTransferFunction.Points = [
    0.04724999889731407,
    0.0,
    0.5,
    0.0,
    0.04725762829184532,
    1.0,
    0.5,
    0.0,
]

vpmpvdDisplay.OpacityTransferFunction.Points = [
    0.04724999889731407,
    0.0,
    0.5,
    0.0,
    0.04725762829184532,
    1.0,
    0.5,
    0.0,
]

# setup the color legend parameters for each legend in this view

vorticityLUTColorBar = GetScalarBar(vorticityLUT, renderView1)
vorticityLUTColorBar.Set(
    Title="vorticity",
    ComponentTitle="Magnitude",
)

vorticityLUTColorBar.Visibility = 0

Hide(vpmpvd, renderView1)

# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed

velocityPWF = GetOpacityTransferFunction("velocity")
velocityPWF.Set(
    Points=[0.0016831935004858933, 0.0, 0.5, 0.0, 1.2312556715839478, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

sTLSolidLabelingPWF = GetOpacityTransferFunction("STLSolidLabeling")
sTLSolidLabelingPWF.Set(
    Points=[0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

vorticityPWF = GetOpacityTransferFunction("vorticity")
vorticityPWF.Set(
    Points=[0.0, 0.0, 0.5, 0.0, 5.0, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed

timeKeeper1 = GetTimeKeeper()

timeKeeper1.SuppressedTimeSources = [fvmpvd, cubestl]

timeAnimationCue1 = GetTimeTrack()


animationScene1 = GetAnimationScene()

animationScene1.Set(
    ViewModules=renderView1,
    Cues=timeAnimationCue1,
    AnimationTime=20.0,
    StartTime=0.5,
    EndTime=20.0,
    PlayMode="Snap To TimeSteps",
)


SetActiveSource(fvmpvd)
