# state file generated using paraview version 6.1.1
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
renderView1 = CreateView('RenderView')
renderView1.Set(
    ViewSize=[2184, 700],
    CenterOfRotation=[6.689284026622772, -0.0014328956604003906, 0.0],
    CameraPosition=[7.996528563871198, -8.422852715427569, 6.407914481299701],
    CameraFocalPoint=[6.68928402662277, -0.0014328956604003963, -5.4711726939477697e-17],
    CameraViewUp=[-0.2010644155553002, 0.5732169552836452, 0.7943522033536213],
)

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(2184, 700)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'STL Reader'
cylinder_longstl = STLReader(registrationName='cylinder_long.stl', FileNames=['/home/flavio-martins/Projects/OpenONDA/tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/assets/cylinder_long.stl'])

# create a new 'PVD Reader'
vpmpvd = PVDReader(registrationName='vpm.pvd', FileName='/home/flavio-martins/Projects/OpenONDA/tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution/vpm.pvd')

# create a new 'PVD Reader'
fvmpvd = PVDReader(registrationName='fvm.pvd', FileName='/home/flavio-martins/Projects/OpenONDA/tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution/fvm.pvd')
fvmpvd.CellArrays = ['velocity', 'kinematic_pressure', 'courant_number', 'vorticity', 'cell_size', 'refinement_level', 'boundary_layer_index', 'cell_volume', 'cell_equivalent_size']

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=fvmpvd)
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Set(
    Origin=[0.3999999999999999, 0.0, 0.0],
    Normal=[0.0, 0.0, 1.0],
)

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [0.3999999999999999, 0.0, 0.0]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from fvmpvd
fvmpvdDisplay = Show(fvmpvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
fvmpvdDisplay.Set(
    Representation='Outline',
    ColorArrayName=['POINTS', ''],
)

# show data from vpmpvd
vpmpvdDisplay = Show(vpmpvd, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'vorticity'
vorticityLUT = GetColorTransferFunction('vorticity')
vorticityLUT.Set(
    RGBPoints=GenerateRGBPoints(
        range_min=0.0008865025592967868,
        range_max=3.241889238357544,
    ),
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
vpmpvdDisplay.Set(
    Representation='Point Gaussian',
    ColorArrayName=['POINTS', 'vorticity'],
    LookupTable=vorticityLUT,
    GaussianRadius=0.016,
    ShaderPreset='Plain circle',
    ScaleByArray=1,
    SetScaleArray=['POINTS', 'vorticity'],
    ScaleArrayComponent='Magnitude',
)

# show data from cylinder_longstl
cylinder_longstlDisplay = Show(cylinder_longstl, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'STLSolidLabeling'
sTLSolidLabelingLUT = GetColorTransferFunction('STLSolidLabeling')
sTLSolidLabelingLUT.Set(
    RGBPoints=GenerateRGBPoints(
        range_min=0.0,
        range_max=1.1757813367477812e-38,
    ),
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
cylinder_longstlDisplay.Set(
    Representation='Surface',
    ColorArrayName=['CELLS', 'STLSolidLabeling'],
    LookupTable=sTLSolidLabelingLUT,
)

# show data from slice1
slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
slice1Display.Set(
    Representation='Surface',
    ColorArrayName=['CELLS', 'vorticity'],
    LookupTable=vorticityLUT,
)

# setup the color legend parameters for each legend in this view

# get color legend/bar for vorticityLUT in view renderView1
vorticityLUTColorBar = GetScalarBar(vorticityLUT, renderView1)
vorticityLUTColorBar.Set(
    AutoOrient=0,
    WindowLocation='Any Location',
    Position=[0.767907808057557, 0.5695915484838003],
    Title='vorticity',
    ComponentTitle='Magnitude',
    HorizontalTitle=1,
    TitleFontFamily='Times',
    TitleFontSize=41,
    LabelFontFamily='Times',
    LabelFontSize=41,
    ScalarBarThickness=25,
    ScalarBarLength=0.1999999999999995,
    DrawScalarBarOutline=1,
    ScalarBarOutlineColor=[0.0, 0.0, 0.0],
    ScalarBarOutlineThickness=2,
    AddRangeLabels=0,
)

# set color bar visibility
vorticityLUTColorBar.Visibility = 1

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

# get color transfer function/color map for 'velocity'
velocityLUT = GetColorTransferFunction('velocity')
velocityLUT.Set(
    RGBPoints=GenerateRGBPoints(
        range_min=0.004135256335276704,
        range_max=1.4281685185791768,
    ),
    ScalarRangeInitialized=1.0,
)

# get color legend/bar for velocityLUT in view renderView1
velocityLUTColorBar = GetScalarBar(velocityLUT, renderView1)
velocityLUTColorBar.Set(
    AutoOrient=0,
    WindowLocation='Any Location',
    Position=[0.047211837361586295, 0.7238772627695146],
    Title='velocity',
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
velocityLUTColorBar.Visibility = 0

# show color legend
vpmpvdDisplay.SetScalarBarVisibility(renderView1, True)

# hide data in view
Hide(cylinder_longstl, renderView1)

# show color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'velocity'
velocityPWF = GetOpacityTransferFunction('velocity')
velocityPWF.Set(
    Points=[0.004135256335276704, 0.0, 0.5, 0.0, 1.4281685185791768, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# get opacity transfer function/opacity map for 'vorticity'
vorticityPWF = GetOpacityTransferFunction('vorticity')
vorticityPWF.Set(
    Points=[0.0008865025592967868, 0.0, 0.5, 0.0, 3.241889238357544, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# get opacity transfer function/opacity map for 'STLSolidLabeling'
sTLSolidLabelingPWF = GetOpacityTransferFunction('STLSolidLabeling')
sTLSolidLabelingPWF.Set(
    Points=[0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper
timeKeeper1.SuppressedTimeSources = [vpmpvd, cylinder_longstl]

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.Set(
    ViewModules=renderView1,
    Cues=timeAnimationCue1,
    AnimationTime=0.0,
    EndTime=99.99999999998809,
    PlayMode='Snap To TimeSteps',
)

# initialize the animation scene

# ----------------------------------------------------------------
# restore active source
SetActiveSource(vpmpvd)
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