# state file generated using paraview version 6.1.0
from pathlib import Path

import paraview

paraview.compatibility.major = 6
paraview.compatibility.minor = 1

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView("RenderView")
renderView1.Set(
    ViewSize=[2046, 782],
    InteractionMode="2D",
    CenterOfRotation=[3.125, 0.0, 0.0],
    CameraPosition=[0.5283767468872254, 0.006883336534572343, 35.53307527973966],
    CameraFocalPoint=[0.5283767468872254, 0.006883336534572343, 0.0],
)

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name="Layout #1")
layout1.AssignView(0, renderView1)
layout1.SetSize(2046, 782)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVD Reader'
reference_flowpvd = PVDReader(
    registrationName="reference_flow.pvd",
    FileName=str(Path(__file__).resolve().parent / "solution" / "reference_flow.pvd"),
)
reference_flowpvd.Set(
    CellArrays=[
        "velocity",
        "kinematic_pressure",
        "courant_number",
        "vorticity",
        "vtkGhostType",
        "global_cell_id",
    ],
    PointArrays=["global_point_id"],
)

# create a new 'Slice'
slice1 = Slice(registrationName="Slice1", Input=reference_flowpvd)
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Set(
    Origin=[3.125, 0.0, 0.0],
    Normal=[0.0, 0.0, 1.0],
)

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [3.125, 0.0, 0.0]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from slice1
slice1Display = Show(slice1, renderView1, "GeometryRepresentation")

# get color transfer function/color map for 'velocity'
uLUT = GetColorTransferFunction("velocity")
uLUT.Set(
    RGBPoints=GenerateRGBPoints(
        range_min=0.03424185738753899,
        range_max=1.7542101839235682,
    ),
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
slice1Display.Set(
    Representation="Surface",
    ColorArrayName=["CELLS", "velocity"],
    LookupTable=uLUT,
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
slice1Display.ScaleTransferFunction.Points = [411.0, 0.0, 0.5, 0.0, 592810.0, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
slice1Display.OpacityTransferFunction.Points = [411.0, 0.0, 0.5, 0.0, 592810.0, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for uLUT in view renderView1
uLUTColorBar = GetScalarBar(uLUT, renderView1)
uLUTColorBar.Set(
    Title="velocity",
    ComponentTitle="Magnitude",
)

# set color bar visibility
uLUTColorBar.Visibility = 1

# show color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'velocity'
uPWF = GetOpacityTransferFunction("velocity")
uPWF.Set(
    Points=[0.03424185738753899, 0.0, 0.5, 0.0, 1.7542101839235682, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation scene

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.Set(
    ViewModules=renderView1,
    Cues=timeAnimationCue1,
    AnimationTime=0.15,
    EndTime=0.15,
    PlayMode="Snap To TimeSteps",
)

# ----------------------------------------------------------------
# restore active source
SetActiveSource(reference_flowpvd)
# ----------------------------------------------------------------


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/nightly/python/
##--------------------------------------------
