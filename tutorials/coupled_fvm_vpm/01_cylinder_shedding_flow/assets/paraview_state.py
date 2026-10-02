# state file generated using paraview version 6.1.1
from pathlib import Path
import paraview
paraview.compatibility.major = 6
paraview.compatibility.minor = 1

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

SOLUTION_DIR = Path(__file__).resolve().parents[1] / "solution"
# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.Set(
    ViewSize=[1922, 542],
    InteractionMode='2D',
    CenterOfRotation=[1.109284200668335, 0.0, 0.0],
    CameraPosition=[1.109284200668335, 0.0, 14.034733410555077],
    CameraFocalPoint=[1.109284200668335, 0.0, 0.0],
)

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1922, 542)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVD Reader'
vpmpvd = PVDReader(registrationName='vpm.pvd', FileName=str(SOLUTION_DIR / 'vpm.pvd'))
vpmpvd.PointArrays = ['velocity', 'vortex_strength', 'vorticity', 'core_radius', 'particle_volume', 'kinematic_viscosity', 'eddy_viscosity', 'effective_viscosity', 'group_id', 'zone_id']

# create a new 'PVD Reader'
fvmpvd = PVDReader(registrationName='fvm.pvd', FileName=str(SOLUTION_DIR / 'fvm.pvd'))
fvmpvd.CellArrays = ['velocity', 'kinematic_pressure', 'courant_number', 'vorticity', 'cell_size', 'refinement_level', 'boundary_layer_index', 'cell_volume', 'cell_equivalent_size', 'global_cell_id']

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=vpmpvd)

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Set(
    Origin=[0.07856833934783936, -0.001432955265045166, 0.1],
    Normal=[0.0, 0.0, 1.0],
)

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [0.07856833934783936, -0.001432955265045166, 0.0]

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=fvmpvd)
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Normal = [0.0, 0.0, 1.0]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from fvmpvd
fvmpvdDisplay = Show(fvmpvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
fvmpvdDisplay.Set(
    Representation='Surface',
    ColorArrayName=[None, ''],
)

# show data from slice1
slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'vorticity'
vorticityLUT = GetColorTransferFunction('vorticity')
vorticityLUT.Set(
    RGBPoints=GenerateRGBPoints(
        range_min=0.002230218640926978,
        range_max=18.30860138250205,
    ),
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
slice1Display.Set(
    Representation='Surface',
    ColorArrayName=['CELLS', 'vorticity'],
    LookupTable=vorticityLUT,
)

# show data from vpmpvd
vpmpvdDisplay = Show(vpmpvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
vpmpvdDisplay.Set(
    Representation='Point Gaussian',
    ColorArrayName=[None, ''],
    GaussianRadius=0.006600000262260437,
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
vpmpvdDisplay.ScaleTransferFunction.Points = [0.03999999910593033, 0.0, 0.5, 0.0, 0.04000762850046158, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
vpmpvdDisplay.OpacityTransferFunction.Points = [0.03999999910593033, 0.0, 0.5, 0.0, 0.04000762850046158, 1.0, 0.5, 0.0]

# show data from clip1
clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip1Display.Set(
    Representation='Point Gaussian',
    ColorArrayName=['POINTS', 'vorticity'],
    LookupTable=vorticityLUT,
    GaussianRadius=0.025,
    ShaderPreset='Plain circle',
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
clip1Display.ScaleTransferFunction.Points = [0.03999999910593033, 0.0, 0.5, 0.0, 0.04000762850046158, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
clip1Display.OpacityTransferFunction.Points = [0.03999999910593033, 0.0, 0.5, 0.0, 0.04000762850046158, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for vorticityLUT in view renderView1
vorticityLUTColorBar = GetScalarBar(vorticityLUT, renderView1)
vorticityLUTColorBar.Set(
    AutoOrient=0,
    WindowLocation='Any Location',
    Position=[0.14, 0.70],
    Title='vorticity',
    ComponentTitle='Magnitude',
    HorizontalTitle=1,
    TitleFontFamily='Times',
    TitleFontSize=24,
    LabelFontFamily='Times',
    LabelFontSize=24,
    ScalarBarThickness=18,
    ScalarBarLength=0.19999999999999984,
    DrawScalarBarOutline=1,
    ScalarBarOutlineColor=[0.0, 0.0, 0.0],
    ScalarBarOutlineThickness=2,
    AddRangeLabels=0,
)

# set color bar visibility
vorticityLUTColorBar.Visibility = 1

# hide data in view
Hide(fvmpvd, renderView1)

# show color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# hide data in view
Hide(vpmpvd, renderView1)

# show color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'vorticity'
vorticityPWF = GetOpacityTransferFunction('vorticity')
vorticityPWF.Set(
    Points=[0.002230218640926978, 0.0, 0.5, 0.0, 18.30860138250205, 1.0, 0.5, 0.0],
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
animationScene1.UpdateAnimationUsingDataTimeSteps()
# initialize the animation scene
animationScene1.Set(
    ViewModules=renderView1,
    Cues=timeAnimationCue1,
    AnimationTime=animationScene1.StartTime,
    PlayMode='Snap To TimeSteps',
)

# ----------------------------------------------------------------
# restore active source
SetActiveSource(clip1)
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
# SaveScreenshot("path/to/screenshot.png", layout1)
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
