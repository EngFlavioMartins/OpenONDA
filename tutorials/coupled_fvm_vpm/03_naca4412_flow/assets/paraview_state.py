# state file generated using paraview version 6.2.0
import paraview
paraview.compatibility.major = 6
paraview.compatibility.minor = 2

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.Set(
    ViewSize=[2172, 700],
    CenterOfRotation=[0.5593341351301969, -0.020000000000000018, 3.932313052956715e-08],
    CameraPosition=[7.475656848704529, 10.731863475765287, 14.205025200819005],
    CameraFocalPoint=[9.332121343705126, 0.9997930288770365, 0.4629332220183945],
    CameraViewUp=[-0.21251098824516296, 0.7836680907760127, -0.5836980412632404],
)

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(2172, 700)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVD Reader'
vpmpvd = PVDReader(registrationName='vpm.pvd', FileName='/home/flavio-martins/Projects/OpenONDA/tutorials/coupled_fvm_vpm/03_naca4412_flow/solution/vpm.pvd')
vpmpvd.PointArrays = ['velocity', 'vortex_strength', 'vorticity', 'core_radius', 'particle_volume', 'kinematic_viscosity', 'eddy_viscosity', 'effective_viscosity', 'group_id', 'zone_id']

# create a new 'STL Reader'
naca4412_wingstl = STLReader(registrationName='naca4412_wing.stl', FileNames=['/home/flavio-martins/Projects/OpenONDA/tutorials/coupled_fvm_vpm/03_naca4412_flow/assets/naca4412_wing.stl'])

# create a new 'PVD Reader'
fvmpvd = PVDReader(registrationName='fvm.pvd', FileName='/home/flavio-martins/Projects/OpenONDA/tutorials/coupled_fvm_vpm/03_naca4412_flow/solution/fvm.pvd')
fvmpvd.CellArrays = ['velocity', 'kinematic_pressure', 'courant_number', 'vorticity', 'cell_size', 'refinement_level', 'boundary_layer_index', 'cell_volume', 'cell_equivalent_size', 'global_cell_id']

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=fvmpvd)
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Set(
    Origin=[0.5593341351301969, -0.020000000000000018, 3.932313052956715e-08],
    Normal=[0.0, 0.0, 1.0],
)

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [0.5593341351301969, -0.020000000000000018, 3.932313052956715e-08]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from fvmpvd
fvmpvdDisplay = Show(fvmpvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
fvmpvdDisplay.Set(
    Representation='Outline',
    ColorArrayName=[None, ''],
)

# show data from vpmpvd
vpmpvdDisplay = Show(vpmpvd, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'vorticity'
vorticityLUT = GetColorTransferFunction('vorticity')
vorticityLUT.Set(
    RGBPoints=GenerateRGBPoints(
        preset_name='Viridis',
        range_min=0.04700850888802023,
        range_max=30.51438190040418,
    ),
    NanColor=[1.0, 0.0, 0.0],
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
vpmpvdDisplay.Set(
    Representation='Point Gaussian',
    ColorArrayName=['POINTS', 'vorticity'],
    LookupTable=vorticityLUT,
    GaussianRadius=0.04,
    ShaderPreset='Plain circle',
    ScaleByArray=1,
    SetScaleArray=['POINTS', 'vorticity'],
    ScaleArrayComponent='Magnitude',
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
vpmpvdDisplay.ScaleTransferFunction.Points = [0.00010155572958410664, 0.0, 0.5, 0.0, 30.51438190040418, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
vpmpvdDisplay.OpacityTransferFunction.Points = [0.041999999433755875, 0.0, 0.5, 0.0, 0.042007628828287125, 1.0, 0.5, 0.0]

# show data from slice1
slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
slice1Display.Set(
    Representation='Surface',
    ColorArrayName=['CELLS', 'vorticity'],
    LookupTable=vorticityLUT,
)

# show data from naca4412_wingstl
naca4412_wingstlDisplay = Show(naca4412_wingstl, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
naca4412_wingstlDisplay.Set(
    Representation='Surface',
    ColorArrayName=['POINTS', ''],
)

# setup the color legend parameters for each legend in this view

# get color legend/bar for vorticityLUT in view renderView1
vorticityLUTColorBar = GetScalarBar(vorticityLUT, renderView1)
vorticityLUTColorBar.Set(
    AutoOrient=0,
    WindowLocation='Any Location',
    Position=[0.7548545629601129, 0.5395915484838003],
    Title='vorticity',
    ComponentTitle='Magnitude',
    HorizontalTitle=1,
    TitleFontFamily='Times',
    TitleFontSize=41,
    LabelFontFamily='Times',
    LabelFontSize=41,
    ScalarBarThickness=25,
    ScalarBarLength=0.19999999999999984,
    DrawScalarBarOutline=1,
    ScalarBarOutlineColor=[0.0, 0.0, 0.0],
    ScalarBarOutlineThickness=2,
    AddRangeLabels=0,
)

# set color bar visibility
vorticityLUTColorBar.Visibility = 1

# get color transfer function/color map for 'STLSolidLabeling'
sTLSolidLabelingLUT = GetColorTransferFunction('STLSolidLabeling')
sTLSolidLabelingLUT.Set(
    RGBPoints=GenerateRGBPoints(
        range_min=0.0,
        range_max=1.1757813367477812e-38,
    ),
    ScalarRangeInitialized=1.0,
)

# get color legend/bar for sTLSolidLabelingLUT in view renderView1
sTLSolidLabelingLUTColorBar = GetScalarBar(sTLSolidLabelingLUT, renderView1)
sTLSolidLabelingLUTColorBar.Set(
    AutoOrient=0,
    WindowLocation='Any Location',
    Position=[0.047211837361586295, 0.7238772627695146],
    Title='STLSolidLabeling',
    ComponentTitle='',
    HorizontalTitle=1,
    TitleFontFamily='Times',
    TitleFontSize=41,
    LabelFontFamily='Times',
    LabelFontSize=41,
    ScalarBarThickness=25,
    ScalarBarLength=0.19999999999999984,
    DrawScalarBarOutline=1,
    ScalarBarOutlineColor=[0.0, 0.0, 0.0],
    ScalarBarOutlineThickness=2,
    AddRangeLabels=0,
)

# set color bar visibility
sTLSolidLabelingLUTColorBar.Visibility = 0

# show color legend
vpmpvdDisplay.SetScalarBarVisibility(renderView1, True)

# show color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'STLSolidLabeling'
sTLSolidLabelingPWF = GetOpacityTransferFunction('STLSolidLabeling')
sTLSolidLabelingPWF.Set(
    Points=[0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# get opacity transfer function/opacity map for 'vorticity'
vorticityPWF = GetOpacityTransferFunction('vorticity')
vorticityPWF.Set(
    Points=[0.04700850888802023, 0.0, 0.5, 0.0, 30.51438190040418, 1.0, 0.5, 0.0],
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
    AnimationTime=1.92,
    EndTime=1.92,
    PlayMode='Snap To TimeSteps',
)

# ----------------------------------------------------------------
# restore active source
SetActiveSource(vpmpvd)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup camera links
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