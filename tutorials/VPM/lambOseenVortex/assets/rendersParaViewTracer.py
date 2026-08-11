# trace generated using paraview version 6.0.1-1233-gf6d296c8ae
# import paraview
# paraview.compatibility.major = 6
# paraview.compatibility.minor = 0

from pathlib import Path

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

TUTORIAL_DIR = Path(__file__).resolve().parents[1]

# get active source.
vpm_dipole_cs_000010xdmf = GetActiveSource()

# get active view
renderView1 = GetActiveViewOrCreate("RenderView")

# get display properties
vpm_dipole_cs_000010xdmfDisplay = GetRepresentation(vpm_dipole_cs_000010xdmf, view=renderView1)

# change representation type
vpm_dipole_cs_000010xdmfDisplay.SetRepresentationType("Point Gaussian")

# set scalar coloring
ColorBy(vpm_dipole_cs_000010xdmfDisplay, ("POINTS", "Vorticity", "Magnitude"))

# get color transfer function/color map for 'Radius'
radiusLUT = GetColorTransferFunction("Radius")

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(radiusLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
vpm_dipole_cs_000010xdmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
vpm_dipole_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Vorticity'
vorticityLUT = GetColorTransferFunction("Vorticity")

# get opacity transfer function/opacity map for 'Vorticity'
vorticityPWF = GetOpacityTransferFunction("Vorticity")

# get 2D transfer function for 'Vorticity'
vorticityTF2D = GetTransferFunction2D("Vorticity")

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
vorticityLUT.ApplyPreset("Viridis", True)

# Rescale transfer function
vorticityLUT.RescaleTransferFunction(1.0988134145736694, 6.900094985961914)

# Rescale transfer function
vorticityPWF.RescaleTransferFunction(1.0988134145736694, 6.900094985961914)

# get color legend/bar for vorticityLUT in view renderView1
vorticityLUTColorBar = GetScalarBar(vorticityLUT, renderView1)

# change scalar bar placement
vorticityLUTColorBar.Set(
    Position=[0.6094698873781611, 0.08881167409492528],
    ScalarBarLength=0.20000000000000007,
)

# create a new 'PVD Reader'
dipole_cs_z0pvd = PVDReader(
    registrationName="dipole_cs_z0.pvd",
    FileName=str(TUTORIAL_DIR / "samples/dipole_cs/dipole_cs_z0.pvd"),
)

# show data in view
dipole_cs_z0pvdDisplay = Show(dipole_cs_z0pvd, renderView1, "StructuredGridRepresentation")

# trace defaults for the display properties.
dipole_cs_z0pvdDisplay.Representation = "Surface"

# show color bar/color legend
dipole_cs_z0pvdDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
vorticityLUT.RescaleTransferFunction(1.0988134145736694, 10.495820045471191)

# Rescale transfer function
vorticityPWF.RescaleTransferFunction(1.0988134145736694, 10.495820045471191)

# get color transfer function/color map for 'Velocity'
velocityLUT = GetColorTransferFunction("Velocity")

# get opacity transfer function/opacity map for 'Velocity'
velocityPWF = GetOpacityTransferFunction("Velocity")

# get 2D transfer function for 'Velocity'
velocityTF2D = GetTransferFunction2D("Velocity")

# set scalar coloring
ColorBy(dipole_cs_z0pvdDisplay, ("POINTS", "StrainRate", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(velocityLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
dipole_cs_z0pvdDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
dipole_cs_z0pvdDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'StrainRate'
strainRateLUT = GetColorTransferFunction("StrainRate")

# get opacity transfer function/opacity map for 'StrainRate'
strainRatePWF = GetOpacityTransferFunction("StrainRate")

# get 2D transfer function for 'StrainRate'
strainRateTF2D = GetTransferFunction2D("StrainRate")

# set scalar coloring
ColorBy(dipole_cs_z0pvdDisplay, ("POINTS", "VelocityGradient", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(strainRateLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
dipole_cs_z0pvdDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
dipole_cs_z0pvdDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'VelocityGradient'
velocityGradientLUT = GetColorTransferFunction("VelocityGradient")

# get opacity transfer function/opacity map for 'VelocityGradient'
velocityGradientPWF = GetOpacityTransferFunction("VelocityGradient")

# get 2D transfer function for 'VelocityGradient'
velocityGradientTF2D = GetTransferFunction2D("VelocityGradient")

# hide data in view
Hide(vpm_dipole_cs_000010xdmf, renderView1)

# get animation scene
animationScene1 = GetAnimationScene()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

# set scalar coloring
ColorBy(dipole_cs_z0pvdDisplay, ("POINTS", "Vorticity", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(velocityGradientLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
dipole_cs_z0pvdDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
dipole_cs_z0pvdDisplay.SetScalarBarVisibility(renderView1, True)

# set scalar coloring
ColorBy(dipole_cs_z0pvdDisplay, ("POINTS", "VorticityMagnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(vorticityLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
dipole_cs_z0pvdDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
dipole_cs_z0pvdDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'VorticityMagnitude'
vorticityMagnitudeLUT = GetColorTransferFunction("VorticityMagnitude")

# get opacity transfer function/opacity map for 'VorticityMagnitude'
vorticityMagnitudePWF = GetOpacityTransferFunction("VorticityMagnitude")

# get 2D transfer function for 'VorticityMagnitude'
vorticityMagnitudeTF2D = GetTransferFunction2D("VorticityMagnitude")

# set scalar coloring
ColorBy(dipole_cs_z0pvdDisplay, ("POINTS", "StrainRate", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(vorticityMagnitudeLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
dipole_cs_z0pvdDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
dipole_cs_z0pvdDisplay.SetScalarBarVisibility(renderView1, True)

# set scalar coloring
ColorBy(dipole_cs_z0pvdDisplay, ("POINTS", "Velocity", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(strainRateLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
dipole_cs_z0pvdDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
dipole_cs_z0pvdDisplay.SetScalarBarVisibility(renderView1, True)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# show data in view
vpm_dipole_cs_000010xdmfDisplay = Show(
    vpm_dipole_cs_000010xdmf, renderView1, "UnstructuredGridRepresentation"
)

# show color bar/color legend
vpm_dipole_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# hide data in view
Hide(vpm_dipole_cs_000010xdmf, renderView1)

# show data in view
vpm_dipole_cs_000010xdmfDisplay = Show(
    vpm_dipole_cs_000010xdmf, renderView1, "UnstructuredGridRepresentation"
)

# show color bar/color legend
vpm_dipole_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView1, True)

ExtendFileSeries(vpm_dipole_cs_000010xdmf)

# change scalar bar placement
vorticityLUTColorBar.Set(
    Position=[0.29537175330767046, 0.05911818750488697],
    ScalarBarLength=0.20000000000000007,
)

# set active source
SetActiveSource(dipole_cs_z0pvd)

# create a new 'Stream Tracer'
streamTracer1 = StreamTracer(
    registrationName="StreamTracer1", Input=dipole_cs_z0pvd, SeedType="Line"
)

# show data in view
streamTracer1Display = Show(streamTracer1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
streamTracer1Display.Representation = "Surface"

# hide data in view
Hide(dipole_cs_z0pvd, renderView1)

# show color bar/color legend
streamTracer1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView1.Update()

animationScene1.GoToNext()

animationScene1.GoToNext()

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=streamTracer1.SeedType)

ExtendFileSeries(vpm_dipole_cs_000010xdmf)

# set active source
SetActiveSource(dipole_cs_z0pvd)

ReloadFiles(dipole_cs_z0pvd)

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

# set active source
SetActiveSource(streamTracer1)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=streamTracer1.SeedType)

# Properties modified on streamTracer1
streamTracer1.SurfaceStreamlines = 1

# update the view to ensure updated data information
renderView1.Update()

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
velocityLUT.ApplyPreset("Inferno", True)

# get color legend/bar for velocityLUT in view renderView1
velocityLUTColorBar = GetScalarBar(velocityLUT, renderView1)

# change scalar bar placement
velocityLUTColorBar.Set(
    Position=[0.4183848839227361, 0.06103389631714751],
    ScalarBarLength=0.19999999999999996,
)

# change scalar bar placement
vorticityLUTColorBar.Set(
    Position=[0.11292531239543825, 0.023677574478067023],
    ScalarBarLength=0.2,
)

# change scalar bar placement
velocityLUTColorBar.Set(
    Position=[0.3288893759752586, 0.013141176010634081],
    ScalarBarLength=0.19999999999999982,
)

# Properties modified on streamTracer1
streamTracer1.SurfaceStreamlines = 0

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=streamTracer1.SeedType)

ExtendFileSeries(vpm_dipole_cs_000010xdmf)

# set active source
SetActiveSource(dipole_cs_z0pvd)

ReloadFiles(dipole_cs_z0pvd)

animationScene1.GoToLast()

# hide data in view
Hide(streamTracer1, renderView1)

# set active source
SetActiveSource(streamTracer1)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=streamTracer1.SeedType)

# show data in view
streamTracer1Display = Show(streamTracer1, renderView1, "GeometryRepresentation")

# show color bar/color legend
streamTracer1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView1.Update()

# change scalar bar placement
velocityLUTColorBar.Set(
    Position=[0.21105938288610865, 0.0],
    ScalarBarLength=0.1999999999999998,
)

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView1.Update()

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=streamTracer1.SeedType)

# change scalar bar placement
vorticityLUTColorBar.Set(
    Position=[0.7304097629828605, 0.6022216357807489],
    ScalarBarLength=0.20000000000000007,
)

# change scalar bar placement
velocityLUTColorBar.Set(
    Position=[0.7573620781176222, 0.16475095785440616],
    ScalarBarLength=0.1999999999999997,
)

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# Properties modified on streamTracer1Display
streamTracer1Display.LineWidth = 2.0

# Rescale transfer function
velocityLUT.RescaleTransferFunction(0.0016060901747159285, 0.3383681346349004)

# Rescale transfer function
velocityPWF.RescaleTransferFunction(0.0016060901747159285, 0.3383681346349004)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# Properties modified on vpm_dipole_cs_000010xdmfDisplay
vpm_dipole_cs_000010xdmfDisplay.GaussianRadius = 0.02

# Properties modified on vpm_dipole_cs_000010xdmfDisplay
vpm_dipole_cs_000010xdmfDisplay.GaussianRadius = 0.015

# Properties modified on vpm_dipole_cs_000010xdmfDisplay
vpm_dipole_cs_000010xdmfDisplay.GaussianRadius = 0.01

# Rescale transfer function
vorticityLUT.RescaleTransferFunction(3.5301316965159583e-07, 0.9587439327527179)

# Rescale transfer function
vorticityPWF.RescaleTransferFunction(3.5301316965159583e-07, 0.9587439327527179)

# set active source
SetActiveSource(streamTracer1)

# Properties modified on streamTracer1
streamTracer1.Set(
    InitialStepLength=0.1,
    MaximumStepLength=0.1,
)

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
velocityLUT.RescaleTransferFunction(0.0015906440319744628, 0.3383681346349004)

# Rescale transfer function
velocityPWF.RescaleTransferFunction(0.0015906440319744628, 0.3383681346349004)

# Properties modified on streamTracer1
streamTracer1.IntegrationDirection = "FORWARD"

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on streamTracer1
streamTracer1.IntegrationDirection = "BOTH"

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(dipole_cs_z0pvd)

# create a new 'Calculator'
calculator1 = Calculator(registrationName="Calculator1", Input=dipole_cs_z0pvd)

# Properties modified on calculator1
calculator1.Set(
    ResultArrayName="u_norm",
    Function="VelocityMagnitude / 1.273",
)

# show data in view
calculator1Display = Show(calculator1, renderView1, "StructuredGridRepresentation")

# trace defaults for the display properties.
calculator1Display.Representation = "Surface"

# hide data in view
Hide(dipole_cs_z0pvd, renderView1)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color transfer function/color map for 'u_norm'
u_normLUT = GetColorTransferFunction("u_norm")

# get opacity transfer function/opacity map for 'u_norm'
u_normPWF = GetOpacityTransferFunction("u_norm")

# get 2D transfer function for 'u_norm'
u_normTF2D = GetTransferFunction2D("u_norm")

# set active source
SetActiveSource(streamTracer1)

# Properties modified on streamTracer1
streamTracer1.Input = calculator1

# hide data in view
Hide(calculator1, renderView1)

# set active source
SetActiveSource(streamTracer1)

# set scalar coloring
ColorBy(streamTracer1Display, ("POINTS", "u_norm"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(velocityLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
streamTracer1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
streamTracer1Display.SetScalarBarVisibility(renderView1, True)

# Properties modified on streamTracer1
streamTracer1.MaximumStreamlineLength = 5.0

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on streamTracer1
streamTracer1.MaximumStreamlineLength = 2.0

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on streamTracer1
streamTracer1.MaximumStreamlineLength = 3.0

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on streamTracer1
streamTracer1.MaximumStreamlineLength = 2.0

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on streamTracer1
streamTracer1.MaximumStreamlineLength = 1.5

# update the view to ensure updated data information
renderView1.Update()

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Inferno", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Inferno", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Inferno", True)

# get color legend/bar for u_normLUT in view renderView1
u_normLUTColorBar = GetScalarBar(u_normLUT, renderView1)

# change scalar bar placement
u_normLUTColorBar.Set(
    Position=[0.6654477726580506, 0.039961099382281595],
    ScalarBarLength=0.19999999999999987,
)

# get layout
layout1 = GetLayout()

# Enter preview mode
layout1.PreviewMode = [1476, 945]

# Properties modified on renderView1
renderView1.Set(
    CenterOfRotation=[0.0, 0.0, 0.19999999999999996],
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# set active source
SetActiveSource(dipole_cs_z0pvd)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# Properties modified on vpm_dipole_cs_000010xdmfDisplay
vpm_dipole_cs_000010xdmfDisplay.ScaleByArray = 1

# Properties modified on vpm_dipole_cs_000010xdmfDisplay
vpm_dipole_cs_000010xdmfDisplay.SetScaleArray = ["POINTS", "Vorticity"]

# Properties modified on vpm_dipole_cs_000010xdmfDisplay
vpm_dipole_cs_000010xdmfDisplay.ScaleArrayComponent = "Magnitude"

# Rescale transfer function
vpm_dipole_cs_000010xdmfDisplay.ScaleTransferFunction.RescaleTransferFunction(
    0.28740522265434265, 0.9581561088562012
)

# Properties modified on renderView1
renderView1.UseToneMapping = 1

# Properties modified on renderView1
renderView1.UseToneMapping = 0

# Properties modified on renderView1
renderView1.UseToneMapping = 1

# change scalar bar placement
vorticityLUTColorBar.Position = [0.09355339441917489, 0.7198063815434608]

# change scalar bar placement
u_normLUTColorBar.Position = [0.8856374745550695, 0.04949499768736634]

# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# change scalar bar placement
u_normLUTColorBar.Set(
    Position=[0.8958000761810857, 0.03996109938228161],
    ScalarBarLength=0.1999999999999999,
)

# change scalar bar placement
vorticityLUTColorBar.Position = [0.08203577924302312, 0.7420521442553252]

# Properties modified on vorticityLUTColorBar
vorticityLUTColorBar.Set(
    TitleFontSize=41,
    LabelFontSize=41,
)

# Properties modified on vorticityLUTColorBar
vorticityLUTColorBar.Set(
    ScalarBarThickness=25,
    ScalarBarLength=0.2,
)

# set active source
SetActiveSource(streamTracer1)

# change scalar bar placement
vorticityLUTColorBar.Set(
    Position=[0.09084336731890387, 0.7367555340858337],
    ScalarBarLength=0.19999999999999996,
)

# Properties modified on u_normLUTColorBar
u_normLUTColorBar.Set(
    TitleFontSize=41,
    LabelFontSize=41,
    ScalarBarThickness=25,
)

# Properties modified on u_normLUTColorBar
u_normLUTColorBar.Title = "$\\|\\mathbf{u} \\| / U_{c,0}$"

# Properties modified on u_normLUTColorBar
u_normLUTColorBar.ScalarBarOutlineThickness = 2

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# Properties modified on vorticityLUTColorBar
vorticityLUTColorBar.ScalarBarOutlineThickness = 2

animationScene1.GoToFirst()

# Rescale transfer function
vorticityLUT.RescaleTransferFunction(7.163449290996035e-09, 10.495798110961914)

# Rescale transfer function
vorticityPWF.RescaleTransferFunction(7.163449290996035e-09, 10.495798110961914)

animationScene1.GoToLast()

# Rescale transfer function
vorticityLUT.RescaleTransferFunction(3.4227836661730737e-06, 0.9590354515556503)

# Rescale transfer function
vorticityPWF.RescaleTransferFunction(3.4227836661730737e-06, 0.9590354515556503)

# create a new 'Calculator'
calculator2 = Calculator(registrationName="Calculator2", Input=vpm_dipole_cs_000010xdmf)

# Properties modified on calculator2
calculator2.Set(
    ResultArrayName="omega_norm",
    Function="Vorticity_X",
)

# show data in view
calculator2Display = Show(calculator2, renderView1, "UnstructuredGridRepresentation")

# trace defaults for the display properties.
calculator2Display.Representation = "Surface"

# hide data in view
Hide(vpm_dipole_cs_000010xdmf, renderView1)

# show color bar/color legend
calculator2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color transfer function/color map for 'omega_norm'
omega_normLUT = GetColorTransferFunction("omega_norm")

# get opacity transfer function/opacity map for 'omega_norm'
omega_normPWF = GetOpacityTransferFunction("omega_norm")

# get 2D transfer function for 'omega_norm'
omega_normTF2D = GetTransferFunction2D("omega_norm")

# Properties modified on calculator2
calculator2.Function = "Vorticity_Y"

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on calculator2
calculator2.Function = "Vorticity_Z"

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(-0.9581561088562012, 0.9555851817131042)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(-0.9581561088562012, 0.9555851817131042)

animationScene1.GoToFirst()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(-10.481795310974121, 10.495798110961914)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(-10.481795310974121, 10.495798110961914)

animationScene1.GoToLast()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(-0.9581450819969177, 0.9555851817131042)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(-0.9581450819969177, 0.9555851817131042)

# Properties modified on calculator2
calculator2.Function = "Vorticity_Z / 20.37"

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(-0.04703706833563661, 0.04691139821861091)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(-0.04703706833563661, 0.04691139821861091)

# Properties modified on renderView1
renderView1.CameraPosition = [9.802984298688976, 2.5219327558780518, 4.092327580292105]

# get color legend/bar for omega_normLUT in view renderView1
omega_normLUTColorBar = GetScalarBar(omega_normLUT, renderView1)

# change scalar bar placement
omega_normLUTColorBar.Set(
    Position=[0.047211837361586295, 0.7238772627695146],
    ScalarBarLength=0.19999999999999984,
)

# change representation type
calculator2Display.SetRepresentationType("Point Gaussian")

# Properties modified on calculator2Display
calculator2Display.GaussianRadius = 0.015674999952316283

# Properties modified on calculator2Display
calculator2Display.GaussianRadius = 0.007837499976158142

# Properties modified on calculator2Display
calculator2Display.ScaleByArray = 1

# Rescale transfer function
calculator2Display.ScaleTransferFunction.RescaleTransferFunction(
    -0.047037609664025586, 0.04691139821861091
)

# Properties modified on calculator2Display
calculator2Display.UseScaleFunction = 0

# Properties modified on calculator2Display
calculator2Display.SetScaleArray = ["POINTS", "Vorticity"]

# Properties modified on calculator2Display
calculator2Display.ScaleArrayComponent = "Magnitude"

# Properties modified on calculator2Display
calculator2Display.UseScaleFunction = 1

# Properties modified on calculator2Display
calculator2Display.UseScaleFunction = 0

# Properties modified on calculator2Display
calculator2Display.UseScaleFunction = 1

# Rescale transfer function
calculator2Display.ScaleTransferFunction.RescaleTransferFunction(
    0.28740522265434265, 0.9581561088562012
)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
omega_normLUT.ApplyPreset("Blue Orange (divergent)", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
omega_normLUT.ApplyPreset("Gray and Red", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
omega_normLUT.ApplyPreset("Warm to Cool (Extended)", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
omega_normLUT.ApplyPreset("PuOr", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
omega_normLUT.ApplyPreset("PiYG", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
omega_normLUT.ApplyPreset("BrBG", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
omega_normLUT.ApplyPreset("GYPi", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
omega_normLUT.ApplyPreset("GnYlRd", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
omega_normLUT.ApplyPreset("Warm to Cool (Extended)", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
omega_normLUT.ApplyPreset("PuOr", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
omega_normLUT.ApplyPreset("BrBG", True)

# Properties modified on calculator2Display
calculator2Display.GaussianRadius = 0.01

# Properties modified on calculator2Display
calculator2Display.ScaleByArray = 0

# Properties modified on calculator2Display
calculator2Display.ScaleByArray = 1

# Properties modified on renderView1
renderView1.UseToneMapping = 0

# Properties modified on renderView1
renderView1.UseToneMapping = 1

# Properties modified on renderView1
renderView1.UseToneMapping = 0

# Properties modified on renderView1
renderView1.UseToneMapping = 1

# Properties modified on renderView1
renderView1.Exposure = 2.0

# Properties modified on renderView1
renderView1.Exposure = 3.1

# Properties modified on renderView1
renderView1.Exposure = 3.0

# Properties modified on renderView1
renderView1.UseAmbientOcclusion = 1

# Properties modified on renderView1
renderView1.UseAmbientOcclusion = 0

# Properties modified on renderView1
renderView1.UseAmbientOcclusion = 1

# Properties modified on renderView1
renderView1.UseToneMapping = 0

# Properties modified on renderView1
renderView1.UseToneMapping = 1

# Properties modified on renderView1
renderView1.UseToneMapping = 0

# Properties modified on renderView1
renderView1.UseToneMapping = 1

# Properties modified on renderView1
renderView1.Exposure = 2.0

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.Set(
    TitleFontSize=41,
    LabelFontSize=41,
    ScalarBarThickness=25,
    ScalarBarOutlineThickness=2,
)

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.Title = "$ \\omega_z / \\omega_{c,0}$"

# Properties modified on renderView1
renderView1.UseToneMapping = 0

# Properties modified on renderView1
renderView1.UseToneMapping = 1

# Properties modified on renderView1
renderView1.UseToneMapping = 0

# Properties modified on renderView1
renderView1.UseToneMapping = 1

# Properties modified on renderView1
renderView1.UseToneMapping = 0

# Properties modified on renderView1
renderView1.UseToneMapping = 1

# Properties modified on renderView1
renderView1.Exposure = 1.1

# Properties modified on renderView1
renderView1.UseToneMapping = 0

# Properties modified on renderView1
renderView1.UseToneMapping = 1

# Properties modified on renderView1
renderView1.UseToneMapping = 0

# Properties modified on renderView1
renderView1.UseToneMapping = 1

# Properties modified on renderView1
renderView1.UseToneMapping = 0

# find settings proxy
generalSettings = GetSettingsProxy("GeneralSettings")

# find settings proxy
iOSettings = GetSettingsProxy("IOSettings")

# find settings proxy
renderViewInteractionSettings = GetSettingsProxy("RenderViewInteractionSettings")

# find settings proxy
renderViewSettings = GetSettingsProxy("RenderViewSettings")

# find settings proxy
representedArrayListSettings = GetSettingsProxy("RepresentedArrayListSettings")

# find settings proxy
colorPalette = GetSettingsProxy("ColorPalette")

# Properties modified on colorPalette
colorPalette.Background2 = [1.0, 1.0, 0.49803921580314636]

# Properties modified on colorPalette
colorPalette.Background = [0.0, 0.0, 0.0]

# Properties modified on calculator2Display
calculator2Display.Emissive = 1

# Properties modified on calculator2Display
calculator2Display.Emissive = 0

# Properties modified on calculator2Display
calculator2Display.Emissive = 1

# Properties modified on calculator2Display
calculator2Display.GaussianRadius = 0.015

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.ScalarBarOutlineColor = [1.0, 1.0, 1.0]

# set active source
SetActiveSource(streamTracer1)

# Properties modified on u_normLUTColorBar
u_normLUTColorBar.ScalarBarOutlineColor = [1.0, 1.0, 1.0]

# change scalar bar placement
u_normLUTColorBar.Position = [0.8876699948802727, 0.028308557009400256]

# change scalar bar placement
omega_normLUTColorBar.Position = [0.02417660700928277, 0.7196399746339213]

# layout/tab size in pixels
layout1.SetSize(1476, 944)

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[9.802984298688976, 2.5219327558780518, 4.092327580292105],
    CameraFocalPoint=[-1.015601139278945, -3.3751019450226045, -3.453370564148839],
    CameraViewUp=[-0.2530668474186389, 0.9042747689630575, -0.3438667081216217],
    CameraParallelScale=3.739504136868496,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/dipoleCs.png",
    viewOrLayout=layout1,
    location=16,
    ImageResolution=[1476, 945],
    FontScaling="Do not scale fonts",
    TransparentBackground=1,
    # PNG options
    CompressionLevel="1",
)

# layout/tab size in pixels
layout1.SetSize(1476, 944)

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[9.802984298688976, 2.5219327558780518, 4.092327580292105],
    CameraFocalPoint=[-1.015601139278945, -3.3751019450226045, -3.453370564148839],
    CameraViewUp=[-0.2530668474186389, 0.9042747689630575, -0.3438667081216217],
    CameraParallelScale=3.739504136868496,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/dipoleCs.png",
    viewOrLayout=layout1,
    location=16,
    ImageResolution=[1476, 945],
    FontScaling="Do not scale fonts",
    # PNG options
    CompressionLevel="1",
)

# Properties modified on u_normLUTColorBar
u_normLUTColorBar.ScalarBarLength = 0.25

# layout/tab size in pixels
layout1.SetSize(1476, 944)

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[9.802984298688976, 2.5219327558780518, 4.092327580292105],
    CameraFocalPoint=[-1.015601139278945, -3.3751019450226045, -3.453370564148839],
    CameraViewUp=[-0.2530668474186389, 0.9042747689630575, -0.3438667081216217],
    CameraParallelScale=3.739504136868496,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/dipoleCs.png",
    viewOrLayout=layout1,
    location=16,
    ImageResolution=[1476, 945],
    FontScaling="Do not scale fonts",
    TransparentBackground=1,
    # PNG options
    CompressionLevel="1",
)

# Properties modified on u_normLUTColorBar
u_normLUTColorBar.ScalarBarLength = 0.2

# Properties modified on u_normLUTColorBar
u_normLUTColorBar.AddRangeLabels = 1

# Properties modified on u_normLUTColorBar
u_normLUTColorBar.Set(
    DrawTickMarks=0,
    DrawTickLabels=0,
)

# set active source
SetActiveSource(calculator2)

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.Set(
    DrawTickMarks=0,
    DrawTickLabels=0,
    AddRangeLabels=1,
)

# layout/tab size in pixels
layout1.SetSize(1476, 944)

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[9.802984298688976, 2.5219327558780518, 4.092327580292105],
    CameraFocalPoint=[-1.015601139278945, -3.3751019450226045, -3.453370564148839],
    CameraViewUp=[-0.2530668474186389, 0.9042747689630575, -0.3438667081216217],
    CameraParallelScale=3.739504136868496,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/dipoleCs.png",
    viewOrLayout=layout1,
    location=16,
    ImageResolution=[1476, 945],
    FontScaling="Do not scale fonts",
    TransparentBackground=1,
    # PNG options
    CompressionLevel="1",
)

# layout/tab size in pixels
layout1.SetSize(1476, 944)

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[9.802984298688976, 2.5219327558780518, 4.092327580292105],
    CameraFocalPoint=[-1.015601139278945, -3.3751019450226045, -3.453370564148839],
    CameraViewUp=[-0.2530668474186389, 0.9042747689630575, -0.3438667081216217],
    CameraParallelScale=3.739504136868496,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/i.png",
    viewOrLayout=layout1,
    location=16,
    ImageResolution=[1476, 945],
    FontScaling="Do not scale fonts",
    # PNG options
    CompressionLevel="1",
)

# set active source
SetActiveSource(calculator1)

# set active source
SetActiveSource(streamTracer1)

# Properties modified on streamTracer1Display
streamTracer1Display.DisableLighting = 1

# Properties modified on streamTracer1Display
streamTracer1Display.DisableLighting = 0

# Properties modified on streamTracer1Display
streamTracer1Display.Interpolation = "PBR"

# Properties modified on streamTracer1Display
streamTracer1Display.Luminosity = 7.000000000000001

# Properties modified on streamTracer1Display
streamTracer1Display.Luminosity = 17.0

# Properties modified on streamTracer1Display
streamTracer1Display.Luminosity = 32.0

# Properties modified on streamTracer1Display
streamTracer1Display.Luminosity = 15.0

# Properties modified on streamTracer1Display
streamTracer1Display.DisableLighting = 1

# Properties modified on streamTracer1Display
streamTracer1Display.DisableLighting = 0

# Properties modified on streamTracer1Display
streamTracer1Display.Metallic = 0.22

# Properties modified on streamTracer1Display
streamTracer1Display.Metallic = 0.36

# Properties modified on streamTracer1Display
streamTracer1Display.Metallic = 0.47

# Properties modified on streamTracer1Display
streamTracer1Display.Metallic = 0.0

# Properties modified on streamTracer1Display
streamTracer1Display.DisableLighting = 1

# Properties modified on streamTracer1Display
streamTracer1Display.DisableLighting = 0

# Properties modified on streamTracer1Display
streamTracer1Display.Interpolation = "Gouraud"

# Properties modified on streamTracer1Display
streamTracer1Display.Interpolation = "Flat"

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Viridis", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Fast (Reds)", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("GnBu", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("erdc_blue_BW", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("BLUE-WHITE", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Linear Green (Gr4L)", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("GnBuPu", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("BuGnYl", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("blue2cyan", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("erdc_blue2gold_BW", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("PuRd", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("RED-PURPLE", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Oranges", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Fast (Blues)", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("GnBuPu", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("erdc_blue2green_muted", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("erdc_cyan2orange", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Fast (Blues)", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Inferno", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Linear Green (Gr4L)", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Viridis", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Blues", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("erdc_blue_BW", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("erdc_magenta_BW", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("GnBu", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Viridis", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("magenta", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("erdc_blue2yellow", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("BuGnYl", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("erdc_blue2yellow", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("GnBu", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("GnBu", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("magenta", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("magenta", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Blues", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Blues", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Linear Green (Gr4L)", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Linear Green (Gr4L)", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("blue2cyan", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("blue2cyan", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Plasma", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_normLUT.ApplyPreset("Plasma", True)

# set active source
SetActiveSource(calculator2)

# Properties modified on calculator2Display
calculator2Display.GaussianRadius = 0.017

# set active source
SetActiveSource(streamTracer1)

# set active source
SetActiveSource(calculator2)

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.RangeLabelFormat = "{:<#6.2f}"

# layout/tab size in pixels
layout1.SetSize(1476, 944)

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[9.802984298688976, 2.5219327558780518, 4.092327580292105],
    CameraFocalPoint=[-1.015601139278945, -3.3751019450226045, -3.453370564148839],
    CameraViewUp=[-0.2530668474186389, 0.9042747689630575, -0.3438667081216217],
    CameraParallelScale=3.739504136868496,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/dipoleRender.png",
    viewOrLayout=layout1,
    location=16,
    ImageResolution=[2952, 1890],
    # PNG options
    CompressionLevel="1",
)

# layout/tab size in pixels
layout1.SetSize(1476, 944)

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[9.802984298688976, 2.5219327558780518, 4.092327580292105],
    CameraFocalPoint=[-1.015601139278945, -3.3751019450226045, -3.453370564148839],
    CameraViewUp=[-0.2530668474186389, 0.9042747689630575, -0.3438667081216217],
    CameraParallelScale=3.739504136868496,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/dipoleRender.png",
    viewOrLayout=layout1,
    location=16,
    ImageResolution=[2952, 1890],
    FontScaling="Do not scale fonts",
    # PNG options
    CompressionLevel="1",
)

# layout/tab size in pixels
layout1.SetSize(1476, 944)

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[9.802984298688976, 2.5219327558780518, 4.092327580292105],
    CameraFocalPoint=[-1.015601139278945, -3.3751019450226045, -3.453370564148839],
    CameraViewUp=[-0.2530668474186389, 0.9042747689630575, -0.3438667081216217],
    CameraParallelScale=3.739504136868496,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/dipoleRender.png",
    viewOrLayout=layout1,
    location=16,
    ImageResolution=[1476, 945],
    FontScaling="Do not scale fonts",
    # PNG options
    CompressionLevel="1",
)

# create new layout object 'Layout #2'
layout2 = CreateLayout(name="Layout #2")

# set active view
SetActiveView(None)

# get active view
renderView2 = GetActiveViewOrCreate("RenderView")

# Create a new 'Render View'
renderView2_1 = CreateView("RenderView")
renderView2_1.Set(
    StereoType="Crystal Eyes",
    ANARIRendererParameters=["", "", ""],
    BackEnd="OSPRay raycaster",
    OSPRayMaterialLibrary=materialLibrary1,
)

# assign view to a particular cell in the layout
AssignViewToLayout(view=renderView2_1, layout=layout2, hint=0)

# create a new 'XDMF Reader'
vpm_merging_cs_000010xdmf = XDMFReader(
    registrationName="vpm_merging_cs_000010.xdmf*",
    FileNames=[
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000010.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000020.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000030.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000040.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000050.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000060.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000070.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000080.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000090.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000100.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000110.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000120.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000130.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000140.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000150.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000160.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000170.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000180.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000190.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000200.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000210.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000220.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000230.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000240.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000250.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000260.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000270.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000280.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000290.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000300.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000310.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_cs/vpm_merging_cs_000320.xdmf",
    ],
)

# show data in view
vpm_merging_cs_000010xdmfDisplay = Show(
    vpm_merging_cs_000010xdmf, renderView2_1, "UnstructuredGridRepresentation"
)

# trace defaults for the display properties.
vpm_merging_cs_000010xdmfDisplay.Representation = "Surface"

# reset view to fit data
renderView2_1.ResetCamera(False, 0.9)

# show color bar/color legend
vpm_merging_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, True)

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView2_1.Update()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(-0.047037609664025586, 0.04691139821861091)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(-0.047037609664025586, 0.04691139821861091)

# get opacity transfer function/opacity map for 'Radius'
radiusPWF = GetOpacityTransferFunction("Radius")

# get 2D transfer function for 'Radius'
radiusTF2D = GetTransferFunction2D("Radius")

animationScene1.GoToLast()

animationScene1.GoToFirst()

animationScene1.Play()

# create a new 'XDMF Reader'
vpm_merging_dvh_000010xdmf = XDMFReader(
    registrationName="vpm_merging_dvh_000010.xdmf*",
    FileNames=[
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000010.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000020.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000030.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000040.xdmf",
    ],
)

# show data in view
vpm_merging_dvh_000010xdmfDisplay = Show(
    vpm_merging_dvh_000010xdmf, renderView2_1, "UnstructuredGridRepresentation"
)

# trace defaults for the display properties.
vpm_merging_dvh_000010xdmfDisplay.Representation = "Surface"

# show color bar/color legend
vpm_merging_dvh_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, True)

# update the view to ensure updated data information
renderView2_1.Update()

# set scalar coloring
ColorBy(vpm_merging_dvh_000010xdmfDisplay, ("POINTS", "Vorticity", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(radiusLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
vpm_merging_dvh_000010xdmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
vpm_merging_dvh_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, True)

# set active source
SetActiveSource(vpm_merging_cs_000010xdmf)

# set scalar coloring
ColorBy(vpm_merging_cs_000010xdmfDisplay, ("POINTS", "Vorticity", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(radiusLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
vpm_merging_cs_000010xdmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
vpm_merging_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, True)

animationScene1.Play()

ExtendFileSeries(vpm_merging_cs_000010xdmf)

# set active source
SetActiveSource(vpm_merging_dvh_000010xdmf)

ExtendFileSeries(vpm_merging_dvh_000010xdmf)

# destroy vpm_merging_dvh_000010xdmf
Delete(vpm_merging_dvh_000010xdmf)
del vpm_merging_dvh_000010xdmf

# get color legend/bar for vorticityLUT in view renderView2_1
vorticityLUTColorBar_1 = GetScalarBar(vorticityLUT, renderView2_1)

# change scalar bar placement
vorticityLUTColorBar_1.Position = [0.047211837361586295, 0.6453332014668326]

# change scalar bar placement
vorticityLUTColorBar_1.Position = [0.04790292236504173, 0.629049676562618]

# set active source
SetActiveSource(vpm_merging_cs_000010xdmf)

# create a new 'Calculator'
calculator3 = Calculator(registrationName="Calculator3", Input=vpm_merging_cs_000010xdmf)

# Properties modified on calculator3
calculator3.Set(
    ResultArrayName="omega_norm",
    Function="Vorticity_Z / 20",
)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# trace defaults for the display properties.
calculator3Display.Representation = "Surface"

# hide data in view
Hide(vpm_merging_cs_000010xdmf, renderView2_1)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView2_1.Update()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(-0.1946840089621003, 0.19828487634658815)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(-0.1946840089621003, 0.19828487634658815)

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(dipole_cs_z0pvd)

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(calculator3)

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(calculator3)

# change representation type
calculator3Display.SetRepresentationType("Point Gaussian")

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(calculator3)

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(calculator3)

# set active source
SetActiveSource(calculator2)

# set active view
SetActiveView(renderView1)

# set active source
SetActiveSource(streamTracer1)

# set active view
SetActiveView(renderView2_1)

# set active source
SetActiveSource(calculator3)

animationScene1.GoToFirst()

animationScene1.Play()

# create a new 'PVD Reader'
merging_cs_z0pvd = PVDReader(
    registrationName="merging_cs_z0.pvd",
    FileName=str(TUTORIAL_DIR / "samples/merging_cs/merging_cs_z0.pvd"),
)

# show data in view
merging_cs_z0pvdDisplay = Show(merging_cs_z0pvd, renderView2_1, "StructuredGridRepresentation")

# trace defaults for the display properties.
merging_cs_z0pvdDisplay.Representation = "Surface"

# show color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, True)

# update the view to ensure updated data information
renderView2_1.Update()

# hide data in view
Hide(calculator3, renderView2_1)

# set active source
SetActiveSource(calculator3)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# create a new 'Stream Tracer'
streamTracer2 = StreamTracer(registrationName="StreamTracer2", Input=calculator3, SeedType="Line")

# show data in view
streamTracer2Display = Show(streamTracer2, renderView2_1, "GeometryRepresentation")

# trace defaults for the display properties.
streamTracer2Display.Representation = "Surface"

# update the view to ensure updated data information
renderView2_1.Update()

# set active source
SetActiveSource(merging_cs_z0pvd)

# set active source
SetActiveSource(streamTracer2)

# hide data in view
Hide(streamTracer2, renderView2_1)

# set active source
SetActiveSource(streamTracer2)

# show data in view
streamTracer2Display = Show(streamTracer2, renderView2_1, "GeometryRepresentation")

# hide data in view
Hide(calculator3, renderView2_1)

# set active source
SetActiveSource(streamTracer2)

# hide data in view
Hide(merging_cs_z0pvd, renderView2_1)

# Properties modified on renderView2_1
renderView2_1.UseColorPaletteForBackground = 0

# hide data in view
Hide(streamTracer2, renderView2_1)

# set active source
SetActiveSource(streamTracer2)

# show data in view
streamTracer2Display = Show(streamTracer2, renderView2_1, "GeometryRepresentation")

# reset view to fit data
renderView2_1.ResetCamera(False, 0.9)

# set active source
SetActiveSource(calculator3)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# set active source
SetActiveSource(streamTracer2)

# set active source
SetActiveSource(calculator3)

# hide data in view
Hide(streamTracer2, renderView2_1)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# destroy streamTracer2
Delete(streamTracer2)
del streamTracer2

# set active source
SetActiveSource(merging_cs_z0pvd)

# create a new 'Stream Tracer'
streamTracer2 = StreamTracer(
    registrationName="StreamTracer2", Input=merging_cs_z0pvd, SeedType="Line"
)

# show data in view
streamTracer2Display = Show(streamTracer2, renderView2_1, "GeometryRepresentation")

# trace defaults for the display properties.
streamTracer2Display.Representation = "Surface"

# hide data in view
Hide(merging_cs_z0pvd, renderView2_1)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView2_1.Update()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(-0.30538744357618913, 0.30544104971719477)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(-0.30538744357618913, 0.30544104971719477)

# get color legend/bar for omega_normLUT in view renderView2_1
omega_normLUTColorBar_1 = GetScalarBar(omega_normLUT, renderView2_1)

# change scalar bar placement
omega_normLUTColorBar_1.Set(
    Position=[0.748317573367115, 0.19801519380399735],
    ScalarBarLength=0.19999999999999984,
)

# Properties modified on streamTracer2
streamTracer2.MaximumSteps = 100

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView2_1.Update()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(-0.30538744357618913, 0.30544104971719477)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(-0.30538744357618913, 0.30544104971719477)

# hide data in view
Hide(streamTracer2, renderView2_1)

# set active source
SetActiveSource(streamTracer2)

# show data in view
streamTracer2Display = Show(streamTracer2, renderView2_1, "GeometryRepresentation")

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# update the view to ensure updated data information
renderView2_1.Update()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(-0.30538744357618913, 0.30544104971719477)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(-0.30538744357618913, 0.30544104971719477)

# set active source
SetActiveSource(streamTracer1)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=streamTracer2.SeedType)

# set active view
SetActiveView(renderView1)

# set active view
SetActiveView(renderView2_1)

# set active source
SetActiveSource(streamTracer2)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=streamTracer2.SeedType)

# hide data in view
Hide(streamTracer2, renderView2_1)

# set active source
SetActiveSource(streamTracer2)

# show data in view
streamTracer2Display = Show(streamTracer2, renderView2_1, "GeometryRepresentation")

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(streamTracer2, renderView2_1)

# show data in view
streamTracer2Display = Show(streamTracer2, renderView2_1, "GeometryRepresentation")

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# update the view to ensure updated data information
renderView2_1.Update()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(-0.30538744357618913, 0.30544104971719477)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(-0.30538744357618913, 0.30544104971719477)

# set active source
SetActiveSource(merging_cs_z0pvd)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=streamTracer2.SeedType)

# set active source
SetActiveSource(calculator3)

# set active source
SetActiveSource(merging_cs_z0pvd)

# set active source
SetActiveSource(streamTracer2)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=streamTracer2.SeedType)

# set active source
SetActiveSource(merging_cs_z0pvd)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=streamTracer2.SeedType)

# create a new 'Calculator'
calculator4 = Calculator(registrationName="Calculator4", Input=merging_cs_z0pvd)

# Properties modified on calculator4
calculator4.Set(
    ResultArrayName="u_norm",
    Function="VelocityMagnitude / 1.273",
)

# show data in view
calculator4Display = Show(calculator4, renderView2_1, "StructuredGridRepresentation")

# trace defaults for the display properties.
calculator4Display.Representation = "Surface"

# hide data in view
Hide(merging_cs_z0pvd, renderView2_1)

# show color bar/color legend
calculator4Display.SetScalarBarVisibility(renderView2_1, True)

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView2_1.Update()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(-0.30538744357618913, 0.30544104971719477)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(-0.30538744357618913, 0.30544104971719477)

# set active source
SetActiveSource(streamTracer2)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=streamTracer2.SeedType)

# Properties modified on streamTracer2
streamTracer2.Input = calculator4

# hide data in view
Hide(calculator4, renderView2_1)

# hide data in view
Hide(streamTracer2, renderView2_1)

# set active source
SetActiveSource(streamTracer2)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=streamTracer2.SeedType)

# show data in view
streamTracer2Display = Show(streamTracer2, renderView2_1, "GeometryRepresentation")

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# set active view
SetActiveView(renderView1)

animationScene1.GoToLast()

# set active source
SetActiveSource(streamTracer1)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=streamTracer2.SeedType)

# set active source
SetActiveSource(calculator2)

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(-0.04703515174052969, 0.04688709988919437)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(-0.04703515174052969, 0.04688709988919437)

# set active source
SetActiveSource(streamTracer1)

# Rescale transfer function
u_normLUT.RescaleTransferFunction(0.02950305476651026, 0.2658037216365694)

# Rescale transfer function
u_normPWF.RescaleTransferFunction(0.02950305476651026, 0.2658037216365694)

# set active view
SetActiveView(renderView2_1)

# Properties modified on renderView2_1
renderView2_1.Set(
    CenterOfRotation=[5.3000030517578125, 0.006647884845733643, 0.009999990463256836],
    CameraPosition=[9.802984298688976, 2.5219327558780518, 4.092327580292105],
    CameraFocalPoint=[-1.015601139278945, -3.3751019450226045, -3.453370564148839],
    CameraViewUp=[-0.2530668474186389, 0.9042747689630575, -0.3438667081216217],
    CameraParallelScale=3.739504136868496,
)

# set active source
SetActiveSource(calculator3)

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(0.015056930632350135, 0.07900468964874012)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(0.015056930632350135, 0.07900468964874012)

# set active source
SetActiveSource(streamTracer2)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=streamTracer2.SeedType)

# Rescale transfer function
u_normLUT.RescaleTransferFunction(0.009982657009416801, 0.1938155260602841)

# Rescale transfer function
u_normPWF.RescaleTransferFunction(0.009982657009416801, 0.1938155260602841)

# Properties modified on streamTracer2
streamTracer2.SurfaceStreamlines = 1

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView2_1.Update()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(-0.047037609664025586, 0.0790281042961379)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(-0.047037609664025586, 0.0790281042961379)

# Rescale transfer function
u_normLUT.RescaleTransferFunction(0.009071407165123318, 0.2658037216365694)

# Rescale transfer function
u_normPWF.RescaleTransferFunction(0.009071407165123318, 0.2658037216365694)

# get color legend/bar for u_normLUT in view renderView2_1
u_normLUTColorBar_1 = GetScalarBar(u_normLUT, renderView2_1)

# change scalar bar placement
u_normLUTColorBar_1.Position = [0.08833139506718408, 0.6424596382484419]

# Enter preview mode
layout2.PreviewMode = [1476, 945]

# change scalar bar placement
omega_normLUTColorBar_1.Set(
    Position=[0.8092931831232126, 0.029582990414166815],
    ScalarBarLength=0.19999999999999976,
)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=streamTracer2.SeedType)

# Properties modified on renderView2_1
renderView2_1.Set(
    CenterOfRotation=[0.0, 0.0, 0.19999999999999996],
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

animationScene1.GoToFirst()

# Rescale transfer function
u_normLUT.RescaleTransferFunction(0.025164689465969677, 0.5631573765454838)

# Rescale transfer function
u_normPWF.RescaleTransferFunction(0.025164689465969677, 0.5631573765454838)

# set active source
SetActiveSource(calculator3)

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(0.06143729009052439, 0.5151747720874754)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(0.06143729009052439, 0.5151747720874754)

animationScene1.Play()

animationScene1.GoToFirst()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(0.06143729009052439, 0.515985418665216)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(0.06143729009052439, 0.515985418665216)

# Properties modified on calculator3Display
calculator3Display.UseScaleFunction = 0

# Properties modified on calculator3Display
calculator3Display.UseScaleFunction = 1

# Properties modified on calculator3Display
calculator3Display.GaussianRadius = 0.015

# Properties modified on calculator3Display
calculator3Display.GaussianRadius = 0.01

# Properties modified on omega_normLUTColorBar_1
omega_normLUTColorBar_1.Set(
    TitleFontSize=41,
    LabelFontSize=41,
    ScalarBarThickness=20,
    ScalarBarLength=0.25,
)

# Properties modified on omega_normLUTColorBar_1
omega_normLUTColorBar_1.Set(
    DrawBackground=1,
    DrawScalarBarOutline=1,
)

# Properties modified on omega_normLUTColorBar_1
omega_normLUTColorBar_1.DrawBackground = 0

# Properties modified on omega_normLUTColorBar_1
omega_normLUTColorBar_1.ScalarBarOutlineThickness = 2

# Properties modified on omega_normLUTColorBar_1
omega_normLUTColorBar_1.Set(
    TitleFontFamily="Times",
    LabelFontFamily="Times",
)

# Properties modified on omega_normLUTColorBar_1
omega_normLUTColorBar_1.Set(
    DrawTickMarks=0,
    DrawTickLabels=0,
    RangeLabelFormat="{:<#6.1f}",
)

# Properties modified on omega_normLUTColorBar_1
omega_normLUTColorBar_1.HorizontalTitle = 1

# set active view
SetActiveView(renderView1)

animationScene1.GoToLast()

animationScene1.GoToFirst()

# set active source
SetActiveSource(streamTracer1)

# set active view
SetActiveView(renderView2_1)

# set active source
SetActiveSource(streamTracer2)

# Properties modified on u_normLUTColorBar_1
u_normLUTColorBar_1.Set(
    Title="$\\|\\mathbf{u} \\| / U_{c,0}$",
    ComponentTitle="",
)

# set active view
SetActiveView(renderView1)

# set active view
SetActiveView(renderView2_1)

# hide data in view
Hide(calculator3, renderView2_1)

# set active source
SetActiveSource(calculator3)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(calculator3, renderView2_1)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(streamTracer2, renderView2_1)

# set active source
SetActiveSource(streamTracer2)

# show data in view
streamTracer2Display = Show(streamTracer2, renderView2_1, "GeometryRepresentation")

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# set active source
SetActiveSource(merging_cs_z0pvd)

# show data in view
merging_cs_z0pvdDisplay = Show(merging_cs_z0pvd, renderView2_1, "StructuredGridRepresentation")

# show color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(merging_cs_z0pvd, renderView2_1)

# set active source
SetActiveSource(calculator3)

# hide data in view
Hide(calculator3, renderView2_1)

# set active source
SetActiveSource(calculator3)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# change scalar bar placement
u_normLUTColorBar_1.Position = [0.2556755685089185, 0.7187308246891199]

# set active source
SetActiveSource(streamTracer2)

# hide data in view
Hide(streamTracer2, renderView2_1)

# set active source
SetActiveSource(streamTracer2)

# show data in view
streamTracer2Display = Show(streamTracer2, renderView2_1, "GeometryRepresentation")

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# set scalar coloring
ColorBy(streamTracer2Display, ("CELLS", "ReasonForTermination"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(u_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
streamTracer2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# get color transfer function/color map for 'ReasonForTermination'
reasonForTerminationLUT = GetColorTransferFunction("ReasonForTermination")

# get opacity transfer function/opacity map for 'ReasonForTermination'
reasonForTerminationPWF = GetOpacityTransferFunction("ReasonForTermination")

# get 2D transfer function for 'ReasonForTermination'
reasonForTerminationTF2D = GetTransferFunction2D("ReasonForTermination")

# set scalar coloring
ColorBy(streamTracer2Display, ("CELLS", "SeedIds"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(reasonForTerminationLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
streamTracer2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# get color transfer function/color map for 'SeedIds'
seedIdsLUT = GetColorTransferFunction("SeedIds")

# get opacity transfer function/opacity map for 'SeedIds'
seedIdsPWF = GetOpacityTransferFunction("SeedIds")

# get 2D transfer function for 'SeedIds'
seedIdsTF2D = GetTransferFunction2D("SeedIds")

# set scalar coloring
ColorBy(streamTracer2Display, ("POINTS", "u_norm"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(seedIdsLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
streamTracer2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# change scalar bar placement
u_normLUTColorBar_1.Position = [0.8200387121403547, 0.625510485706069]

# change scalar bar placement
omega_normLUTColorBar_1.Set(
    WindowLocation="Any Location",
    Position=[0.8364244579945799, 0.015889830508474576],
)

# change scalar bar placement
omega_normLUTColorBar_1.Set(
    Position=[0.7551236449864498, 0.06885593220338983],
    ScalarBarLength=0.25000000000000006,
)

# set active source
SetActiveSource(calculator4)

# show data in view
calculator4Display = Show(calculator4, renderView2_1, "StructuredGridRepresentation")

# show color bar/color legend
calculator4Display.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(calculator4, renderView2_1)

# set active source
SetActiveSource(merging_cs_z0pvd)

# show data in view
merging_cs_z0pvdDisplay = Show(merging_cs_z0pvd, renderView2_1, "StructuredGridRepresentation")

# show color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(merging_cs_z0pvd, renderView2_1)

# show data in view
merging_cs_z0pvdDisplay = Show(merging_cs_z0pvd, renderView2_1, "StructuredGridRepresentation")

# show color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(merging_cs_z0pvd, renderView2_1)

# hide data in view
Hide(calculator3, renderView2_1)

# set active source
SetActiveSource(calculator3)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(calculator3, renderView2_1)

# set active source
SetActiveSource(calculator3)

# set active source
SetActiveSource(streamTracer2)

# change scalar bar placement
omega_normLUTColorBar_1.Set(
    Orientation="Horizontal",
    Position=[0.22268292682926816, 0.1427542372881357],
    ScalarBarLength=0.25,
)

# Properties modified on u_normLUTColorBar_1
u_normLUTColorBar_1.Set(
    TitleFontFamily="Times",
    TitleFontSize=41,
    LabelFontFamily="Times",
    LabelFontSize=41,
)

# Properties modified on u_normLUTColorBar_1
u_normLUTColorBar_1.Set(
    AutomaticLabelFormat=0,
    DrawTickMarks=0,
    DrawTickLabels=0,
    RangeLabelFormat="{:<#6.1f}",
)

# Properties modified on u_normLUTColorBar_1
u_normLUTColorBar_1.Set(
    Title="$\\|\\mathbf{u} \\| / U_{c,0}$",
    ComponentTitle="",
)

# Properties modified on u_normLUTColorBar_1
u_normLUTColorBar_1.Set(
    HorizontalTitle=1,
    ScalarBarThickness=20,
    ScalarBarLength=0.25,
)

# Properties modified on u_normLUTColorBar_1
u_normLUTColorBar_1.Set(
    DrawScalarBarOutline=1,
    ScalarBarOutlineThickness=2,
)

# Properties modified on u_normLUTColorBar_1
u_normLUTColorBar_1.ScalarBarThickness = 15

# Properties modified on u_normLUTColorBar_1
u_normLUTColorBar_1.Set(
    ScalarBarThickness=20,
    ScalarBarLength=0.2,
)

# change scalar bar placement
omega_normLUTColorBar_1.Position = [0.18406504065040635, 0.20313559322033908]

# set active source
SetActiveSource(calculator3)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "Circulation", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(omega_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# get color transfer function/color map for 'Circulation'
circulationLUT = GetColorTransferFunction("Circulation")

# get opacity transfer function/opacity map for 'Circulation'
circulationPWF = GetOpacityTransferFunction("Circulation")

# get 2D transfer function for 'Circulation'
circulationTF2D = GetTransferFunction2D("Circulation")

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "Vorticity", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(circulationLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# change scalar bar placement
vorticityLUTColorBar_1.Position = [0.11700861342195229, 0.6820157782575332]

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "Velocity", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(vorticityLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "Circulation", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(velocityLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "StrainRate", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(circulationLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# turn off scalar coloring
ColorBy(calculator3Display, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(strainRateLUT, renderView2_1)

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "BackgroundVelocity", "Magnitude"))

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# get color transfer function/color map for 'BackgroundVelocity'
backgroundVelocityLUT = GetColorTransferFunction("BackgroundVelocity")

# get opacity transfer function/opacity map for 'BackgroundVelocity'
backgroundVelocityPWF = GetOpacityTransferFunction("BackgroundVelocity")

# get 2D transfer function for 'BackgroundVelocity'
backgroundVelocityTF2D = GetTransferFunction2D("BackgroundVelocity")

# turn off scalar coloring
ColorBy(calculator3Display, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(backgroundVelocityLUT, renderView2_1)

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "GroupID"))

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# get color transfer function/color map for 'GroupID'
groupIDLUT = GetColorTransferFunction("GroupID")

# get opacity transfer function/opacity map for 'GroupID'
groupIDPWF = GetOpacityTransferFunction("GroupID")

# get 2D transfer function for 'GroupID'
groupIDTF2D = GetTransferFunction2D("GroupID")

# change scalar bar placement
omega_normLUTColorBar_1.Set(
    Position=[0.6027642276422763, 0.09508474576271196],
    ScalarBarLength=0.2499999999999999,
)

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "Vorticity", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(groupIDLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "Vorticity", "Z"))

# rescale color and/or opacity maps used to exactly fit the current data range
calculator3Display.RescaleTransferFunctionToDataRange(False, False)

# Update a scalar bar component title.
UpdateScalarBarsComponentTitle(vorticityLUT, calculator3Display)

# Rescale transfer function
vorticityLUT.RescaleTransferFunction(-0.06757276904600204, 5.177556037902832)

# Rescale transfer function
vorticityPWF.RescaleTransferFunction(-0.06757276904600204, 5.177556037902832)

# change scalar bar placement
vorticityLUTColorBar_1.Position = [0.0567105104409225, 0.6798971341897366]

# change scalar bar placement
omega_normLUTColorBar_1.Set(
    Position=[0.264010840108401, 0.6045338983050847],
    ScalarBarLength=0.2499999999999995,
)

# set active view
SetActiveView(renderView1)

# set active view
SetActiveView(renderView2_1)

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "BackgroundVelocity", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(vorticityLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "Circulation", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(backgroundVelocityLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "Radius"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(circulationLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "ViscosityTurbulent"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(radiusLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# get color transfer function/color map for 'ViscosityTurbulent'
viscosityTurbulentLUT = GetColorTransferFunction("ViscosityTurbulent")

# get opacity transfer function/opacity map for 'ViscosityTurbulent'
viscosityTurbulentPWF = GetOpacityTransferFunction("ViscosityTurbulent")

# get 2D transfer function for 'ViscosityTurbulent'
viscosityTurbulentTF2D = GetTransferFunction2D("ViscosityTurbulent")

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "Volume"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(viscosityTurbulentLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# get color transfer function/color map for 'Volume'
volumeLUT = GetColorTransferFunction("Volume")

# get opacity transfer function/opacity map for 'Volume'
volumePWF = GetOpacityTransferFunction("Volume")

# get 2D transfer function for 'Volume'
volumeTF2D = GetTransferFunction2D("Volume")

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "Vorticity", "Z"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(volumeLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "ZoneID"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(vorticityLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# get color transfer function/color map for 'ZoneID'
zoneIDLUT = GetColorTransferFunction("ZoneID")

# get opacity transfer function/opacity map for 'ZoneID'
zoneIDPWF = GetOpacityTransferFunction("ZoneID")

# get 2D transfer function for 'ZoneID'
zoneIDTF2D = GetTransferFunction2D("ZoneID")

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "omega_norm"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(zoneIDLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# get color legend/bar for omega_normLUT in view renderView2_1
omega_normLUTColorBar_2 = GetScalarBar(omega_normLUT, renderView2_1)

# change scalar bar placement
omega_normLUTColorBar_2.Position = [0.06550452028841555, 0.6677331949729044]

# change scalar bar placement
omega_normLUTColorBar_1.Set(
    Position=[0.4789972899728997, 0.0],
    ScalarBarLength=0.24999999999999956,
)

# change scalar bar placement
omega_normLUTColorBar_1.Set(
    Position=[0.34891598915989164, 0.97],
    ScalarBarLength=0.2499999999999991,
)

# set active source
SetActiveSource(calculator4)

# show data in view
calculator4Display = Show(calculator4, renderView2_1, "StructuredGridRepresentation")

# show color bar/color legend
calculator4Display.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(calculator4, renderView2_1)

# set active source
SetActiveSource(merging_cs_z0pvd)

# show data in view
merging_cs_z0pvdDisplay = Show(merging_cs_z0pvd, renderView2_1, "StructuredGridRepresentation")

# show color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(merging_cs_z0pvd, renderView2_1)

animationScene1.GoToFirst()

# show data in view
merging_cs_z0pvdDisplay = Show(merging_cs_z0pvd, renderView2_1, "StructuredGridRepresentation")

# show color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(merging_cs_z0pvd, renderView2_1)

# set active source
SetActiveSource(calculator4)

# show data in view
calculator4Display = Show(calculator4, renderView2_1, "StructuredGridRepresentation")

# show color bar/color legend
calculator4Display.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(calculator4, renderView2_1)

# change scalar bar placement
omega_normLUTColorBar_2.Position = [0.30398690511226373, 0.5967586187017179]

# get color legend/bar for u_normLUT in view renderView2_1
u_normLUTColorBar_2 = GetScalarBar(u_normLUT, renderView2_1)

# change scalar bar placement
u_normLUTColorBar_2.Position = [0.4625234904781174, 0.7334111610745994]

# change scalar bar placement
omega_normLUTColorBar_2.Position = [0.05601942543746698, 0.6603179407356161]

# change scalar bar placement
u_normLUTColorBar_2.Set(
    Position=[0.4306806720499331, 0.97],
    ScalarBarLength=0.1999999999999995,
)

# change scalar bar placement
omega_normLUTColorBar_2.Position = [0.080409669339906, 0.6857416695491755]

# set active source
SetActiveSource(calculator4)

# set active source
SetActiveSource(merging_cs_z0pvd)

# set active source
SetActiveSource(calculator4)

# show data in view
calculator4Display = Show(calculator4, renderView2_1, "StructuredGridRepresentation")

# show color bar/color legend
calculator4Display.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(calculator4, renderView2_1)

# set active source
SetActiveSource(merging_cs_z0pvd)

# show data in view
merging_cs_z0pvdDisplay = Show(merging_cs_z0pvd, renderView2_1, "StructuredGridRepresentation")

# show color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(merging_cs_z0pvd, renderView2_1)

# show data in view
merging_cs_z0pvdDisplay = Show(merging_cs_z0pvd, renderView2_1, "StructuredGridRepresentation")

# show color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(merging_cs_z0pvd, renderView2_1)

# show data in view
merging_cs_z0pvdDisplay = Show(merging_cs_z0pvd, renderView2_1, "StructuredGridRepresentation")

# show color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(merging_cs_z0pvd, renderView2_1)

# show data in view
merging_cs_z0pvdDisplay = Show(merging_cs_z0pvd, renderView2_1, "StructuredGridRepresentation")

# show color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(merging_cs_z0pvd, renderView2_1)

# change scalar bar placement
omega_normLUTColorBar_2.Position = [0.06685953383855099, 0.6889196356508703]

# set active source
SetActiveSource(calculator3)

animationScene1.GoToFirst()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(0.06143729009052439, 0.515985418665216)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(0.06143729009052439, 0.515985418665216)

# Properties modified on calculator3Display
calculator3Display.UseShaderReplacements = 1

# Properties modified on calculator3Display
calculator3Display.UseShaderReplacements = 0

# Properties modified on renderView2_1
renderView2_1.UseToneMapping = 1

# Properties modified on renderView2_1
renderView2_1.UseToneMapping = 0

# Properties modified on renderView2_1
renderView2_1.UseToneMapping = 1

# Properties modified on renderView2_1
renderView2_1.UseToneMapping = 0

# Properties modified on renderView2_1
renderView2_1.UseToneMapping = 1

# Properties modified on renderView2_1
renderView2_1.UseToneMapping = 0

# Properties modified on renderView2_1
renderView2_1.UseToneMapping = 1

# Properties modified on renderView2_1
renderView2_1.UseAmbientOcclusion = 1

# Properties modified on renderView2_1
renderView2_1.UseAmbientOcclusion = 0

# Properties modified on renderView2_1
renderView2_1.UseColorPaletteForBackground = 1

# hide data in view
Hide(calculator3, renderView2_1)

# set active source
SetActiveSource(calculator3)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# Properties modified on omega_normLUTColorBar_2
omega_normLUTColorBar_2.Set(
    TitleFontFamily="Times",
    TitleFontSize=41,
    LabelFontFamily="Times",
    LabelFontSize=41,
    ScalarBarThickness=25,
    ScalarBarLength=0.2,
)

# Properties modified on omega_normLUTColorBar_2
omega_normLUTColorBar_2.HorizontalTitle = 1

# change scalar bar placement
omega_normLUTColorBar_2.Set(
    WindowLocation="Any Location",
    Position=[0.06169546070460709, 0.6811440677966102],
    ScalarBarLength=0.19999999999999996,
)

# hide data in view
Hide(calculator3, renderView2_1)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(calculator3, renderView2_1)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(calculator3, renderView2_1)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# get color legend/bar for omega_normLUT in view renderView2_1
omega_normLUTColorBar_3 = GetScalarBar(omega_normLUT, renderView2_1)

# change scalar bar placement
omega_normLUTColorBar_3.Set(
    Position=[0.4259381246244589, 0.96364406779661],
    ScalarBarLength=0.1999999999999995,
)

# change scalar bar placement
omega_normLUTColorBar_2.Position = [0.06914803523035236, 0.6726694915254238]

# set active view
SetActiveView(renderView1)

# set active source
SetActiveSource(streamTracer1)

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(streamTracer1)

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(streamTracer1)

# set active view
SetActiveView(renderView2_1)

# set active source
SetActiveSource(calculator3)

# Properties modified on omega_normLUTColorBar_3
omega_normLUTColorBar_3.Title = "$ \\omega_z / \\omega_{c,0}$"

# change scalar bar placement
omega_normLUTColorBar_3.Set(
    Orientation="Horizontal",
    WindowLocation="Any Location",
    Position=[0.3258265582655824, 0.97],
    ScalarBarLength=0.3300000000000003,
)

# change scalar bar placement
omega_normLUTColorBar_2.Position = [0.07389058265582661, 0.6207627118644068]

# Properties modified on generalSettings
generalSettings.Set(
    AutoApplyInfo=0,
    PreservePropertyValuesInfo=0,
    ScalarBarMode="Automatically hide unused color bars",
)

# Properties modified on generalSettings
generalSettings.ScalarBarMode = "Manual (not recommended)"

# Properties modified on generalSettings
generalSettings.ScalarBarMode = "Automatically show and/or hide color bars"

# hide data in view
Hide(calculator3, renderView2_1)

# set active source
SetActiveSource(calculator3)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(calculator3, renderView2_1)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# get color legend/bar for None in view renderView2_1
noneColorBar = GetScalarBar(None, renderView2_1)

# Properties modified on generalSettings
generalSettings.ScalarBarMode = "Manual (not recommended)"

# change scalar bar placement
omega_normLUTColorBar_1.Set(
    Position=[0.33943089430894313, 0.8683050847457627],
    ScalarBarLength=0.24999999999999944,
)

# hide color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(streamTracer2)

# hide color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, False)

# Properties modified on generalSettings
generalSettings.ScalarBarMode = "Automatically show and/or hide color bars"

# set active source
SetActiveSource(calculator4)

# show color bar/color legend
calculator4Display.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
calculator4Display.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(merging_cs_z0pvd)

# show color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(vpm_merging_cs_000010xdmf)

# show color bar/color legend
vpm_merging_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
vpm_merging_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, False)

# show color bar/color legend
vpm_merging_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
vpm_merging_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(streamTracer2)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, False)

# change scalar bar placement
omega_normLUTColorBar_1.Set(
    Position=[0.48441734417344173, 0.08652542372881356],
    ScalarBarLength=0.24999999999999933,
)

# change scalar bar placement
omega_normLUTColorBar_3.Set(
    Position=[0.4172899728997287, 0.3227542372881355],
    ScalarBarLength=0.32999999999999996,
)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, False)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(calculator4)

# show color bar/color legend
calculator4Display.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
calculator4Display.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(merging_cs_z0pvd)

# show color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(calculator3)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(vpm_merging_cs_000010xdmf)

# show color bar/color legend
vpm_merging_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
vpm_merging_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, False)

# show color bar/color legend
vpm_merging_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
vpm_merging_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, False)

# change scalar bar placement
omega_normLUTColorBar_3.ScalarBarLength = 0.3686178861788618

# set active source
SetActiveSource(streamTracer1)

# set active source
SetActiveSource(calculator1)

# set active source
SetActiveSource(dipole_cs_z0pvd)

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# change scalar bar placement
omega_normLUTColorBar_3.Position = [0.500623306233062, 0.5176694915254236]

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# show data in view
vpm_dipole_cs_000010xdmfDisplay_1 = Show(
    vpm_dipole_cs_000010xdmf, renderView2_1, "UnstructuredGridRepresentation"
)

# trace defaults for the display properties.
vpm_dipole_cs_000010xdmfDisplay_1.Representation = "Surface"

# show color bar/color legend
vpm_dipole_cs_000010xdmfDisplay_1.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(vpm_dipole_cs_000010xdmf, renderView2_1)

# show color bar/color legend
vpm_dipole_cs_000010xdmfDisplay_1.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
vpm_dipole_cs_000010xdmfDisplay_1.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(calculator2)

# show data in view
calculator2Display_1 = Show(calculator2, renderView2_1, "UnstructuredGridRepresentation")

# trace defaults for the display properties.
calculator2Display_1.Representation = "Surface"

# hide color bar/color legend
calculator2Display_1.SetScalarBarVisibility(renderView2_1, False)

# hide data in view
Hide(calculator2, renderView2_1)

# show color bar/color legend
calculator2Display_1.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
calculator2Display_1.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(dipole_cs_z0pvd)

# set active source
SetActiveSource(dipole_cs_z0pvd)

# show data in view
dipole_cs_z0pvdDisplay_1 = Show(dipole_cs_z0pvd, renderView2_1, "StructuredGridRepresentation")

# trace defaults for the display properties.
dipole_cs_z0pvdDisplay_1.Representation = "Surface"

# hide color bar/color legend
dipole_cs_z0pvdDisplay_1.SetScalarBarVisibility(renderView2_1, False)

# hide data in view
Hide(dipole_cs_z0pvd, renderView2_1)

# show color bar/color legend
dipole_cs_z0pvdDisplay_1.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
dipole_cs_z0pvdDisplay_1.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(calculator1)

# set active source
SetActiveSource(calculator1)

# show data in view
calculator1Display_1 = Show(calculator1, renderView2_1, "StructuredGridRepresentation")

# trace defaults for the display properties.
calculator1Display_1.Representation = "Surface"

# hide color bar/color legend
calculator1Display_1.SetScalarBarVisibility(renderView2_1, False)

# hide data in view
Hide(calculator1, renderView2_1)

# show color bar/color legend
calculator1Display_1.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
calculator1Display_1.SetScalarBarVisibility(renderView2_1, False)

# show color bar/color legend
calculator1Display_1.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
calculator1Display_1.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(streamTracer1)

# set active source
SetActiveSource(streamTracer1)

# show data in view
streamTracer1Display_1 = Show(streamTracer1, renderView2_1, "GeometryRepresentation")

# trace defaults for the display properties.
streamTracer1Display_1.Representation = "Surface"

# hide color bar/color legend
streamTracer1Display_1.SetScalarBarVisibility(renderView2_1, False)

# hide data in view
Hide(streamTracer1, renderView2_1)

# show color bar/color legend
streamTracer1Display_1.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
streamTracer1Display_1.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(vpm_merging_cs_000010xdmf)

# show color bar/color legend
vpm_merging_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
vpm_merging_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(calculator3)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(merging_cs_z0pvd)

# show color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
merging_cs_z0pvdDisplay.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(calculator4)

# show color bar/color legend
calculator4Display.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
calculator4Display.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(streamTracer2)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(calculator3)

# change use separate color map
calculator3Display.UseSeparateColorMap = 1

# set scalar coloring using an separate color/opacity maps
ColorBy(calculator3Display, ("POINTS", "omega_norm"), True)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(omega_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# get separate color transfer function/color map for 'omega_norm'
separate_calculator3Display_omega_normLUT = GetColorTransferFunction(
    "omega_norm", calculator3Display, separate=True
)

# get separate opacity transfer function/opacity map for 'omega_norm'
separate_calculator3Display_omega_normPWF = GetOpacityTransferFunction(
    "omega_norm", calculator3Display, separate=True
)

# get separate 2D transfer function for 'omega_norm'
separate_calculator3Display_omega_normTF2D = GetTransferFunction2D(
    "omega_norm", calculator3Display, separate=True
)

# change use separate color map
calculator3Display.UseSeparateColorMap = 0

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "omega_norm"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(separate_calculator3Display_omega_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# change use separate color map
calculator3Display.UseSeparateColorMap = 1

# set scalar coloring using an separate color/opacity maps
ColorBy(calculator3Display, ("POINTS", "omega_norm"), True)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(omega_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# change use separate color map
calculator3Display.UseSeparateColorMap = 0

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "omega_norm"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(separate_calculator3Display_omega_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# change use separate color map
calculator3Display.UseSeparateColorMap = 1

# set scalar coloring using an separate color/opacity maps
ColorBy(calculator3Display, ("POINTS", "omega_norm"), True)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(omega_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# change use separate color map
calculator3Display.UseSeparateColorMap = 0

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "omega_norm"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(separate_calculator3Display_omega_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# hide color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, False)

# set active source
SetActiveSource(merging_cs_z0pvd)

# change use separate color map
merging_cs_z0pvdDisplay.UseSeparateColorMap = 1

# set scalar coloring using an separate color/opacity maps
ColorBy(merging_cs_z0pvdDisplay, ("POINTS", "Velocity", "Magnitude"), True)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(velocityLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
merging_cs_z0pvdDisplay.RescaleTransferFunctionToDataRange(True, False)

# get separate color transfer function/color map for 'Velocity'
separate_merging_cs_z0pvdDisplay_VelocityLUT = GetColorTransferFunction(
    "Velocity", merging_cs_z0pvdDisplay, separate=True
)

# get separate opacity transfer function/opacity map for 'Velocity'
separate_merging_cs_z0pvdDisplay_VelocityPWF = GetOpacityTransferFunction(
    "Velocity", merging_cs_z0pvdDisplay, separate=True
)

# get separate 2D transfer function for 'Velocity'
separate_merging_cs_z0pvdDisplay_VelocityTF2D = GetTransferFunction2D(
    "Velocity", merging_cs_z0pvdDisplay, separate=True
)

# set active source
SetActiveSource(calculator4)

# change use separate color map
calculator4Display.UseSeparateColorMap = 1

# set scalar coloring using an separate color/opacity maps
ColorBy(calculator4Display, ("POINTS", "u_norm"), True)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(u_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator4Display.RescaleTransferFunctionToDataRange(True, False)

# get separate color transfer function/color map for 'u_norm'
separate_calculator4Display_u_normLUT = GetColorTransferFunction(
    "u_norm", calculator4Display, separate=True
)

# get separate opacity transfer function/opacity map for 'u_norm'
separate_calculator4Display_u_normPWF = GetOpacityTransferFunction(
    "u_norm", calculator4Display, separate=True
)

# get separate 2D transfer function for 'u_norm'
separate_calculator4Display_u_normTF2D = GetTransferFunction2D(
    "u_norm", calculator4Display, separate=True
)

# set active source
SetActiveSource(streamTracer2)

# change use separate color map
streamTracer2Display.UseSeparateColorMap = 1

# set scalar coloring using an separate color/opacity maps
ColorBy(streamTracer2Display, ("POINTS", "u_norm"), True)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(u_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
streamTracer2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# get separate color transfer function/color map for 'u_norm'
separate_streamTracer2Display_u_normLUT = GetColorTransferFunction(
    "u_norm", streamTracer2Display, separate=True
)

# get separate opacity transfer function/opacity map for 'u_norm'
separate_streamTracer2Display_u_normPWF = GetOpacityTransferFunction(
    "u_norm", streamTracer2Display, separate=True
)

# get separate 2D transfer function for 'u_norm'
separate_streamTracer2Display_u_normTF2D = GetTransferFunction2D(
    "u_norm", streamTracer2Display, separate=True
)

# change use separate color map
streamTracer2Display.UseSeparateColorMap = 0

# set scalar coloring
ColorBy(streamTracer2Display, ("POINTS", "u_norm"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(separate_streamTracer2Display_u_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
streamTracer2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# change use separate color map
streamTracer2Display.UseSeparateColorMap = 1

# set scalar coloring using an separate color/opacity maps
ColorBy(streamTracer2Display, ("POINTS", "u_norm"), True)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(u_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
streamTracer2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# change use separate color map
streamTracer2Display.UseSeparateColorMap = 0

# set scalar coloring
ColorBy(streamTracer2Display, ("POINTS", "u_norm"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(separate_streamTracer2Display_u_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
streamTracer2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# change use separate color map
streamTracer2Display.UseSeparateColorMap = 1

# set scalar coloring using an separate color/opacity maps
ColorBy(streamTracer2Display, ("POINTS", "u_norm"), True)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(u_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
streamTracer2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# change use separate color map
streamTracer2Display.UseSeparateColorMap = 0

# set scalar coloring
ColorBy(streamTracer2Display, ("POINTS", "u_norm"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(separate_streamTracer2Display_u_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
streamTracer2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# change use separate color map
streamTracer2Display.UseSeparateColorMap = 1

# set scalar coloring using an separate color/opacity maps
ColorBy(streamTracer2Display, ("POINTS", "u_norm"), True)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(u_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
streamTracer2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# change use separate color map
streamTracer2Display.UseSeparateColorMap = 0

# set scalar coloring
ColorBy(streamTracer2Display, ("POINTS", "u_norm"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(separate_streamTracer2Display_u_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
streamTracer2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# change use separate color map
streamTracer2Display.UseSeparateColorMap = 1

# set scalar coloring using an separate color/opacity maps
ColorBy(streamTracer2Display, ("POINTS", "u_norm"), True)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(u_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
streamTracer2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# set active source
SetActiveSource(calculator4)

# change use separate color map
calculator4Display.UseSeparateColorMap = 0

# set scalar coloring
ColorBy(calculator4Display, ("POINTS", "u_norm"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(separate_calculator4Display_u_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
calculator4Display.RescaleTransferFunctionToDataRange(True, False)

# set active source
SetActiveSource(streamTracer2)

# change use separate color map
streamTracer2Display.UseSeparateColorMap = 0

# set scalar coloring
ColorBy(streamTracer2Display, ("POINTS", "u_norm"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(separate_streamTracer2Display_u_normLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
streamTracer2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# change scalar bar placement
omega_normLUTColorBar_3.Position = [0.3292140921409211, 0.46894067796610156]

# change scalar bar placement
omega_normLUTColorBar_3.Position = [0.2750135501355011, 0.289915254237288]

# change scalar bar placement
u_normLUTColorBar_2.Set(
    Position=[0.5458568238114507, 0.08440677966101694],
    ScalarBarLength=0.19999999999999918,
)

# change scalar bar placement
omega_normLUTColorBar_3.Set(
    Position=[0.3590243902439022, 0.19351694915254222],
    ScalarBarLength=0.3686178861788617,
)

# change scalar bar placement
omega_normLUTColorBar_1.Position = [0.483739837398374, 0.012372881355932203]

# change scalar bar placement
omega_normLUTColorBar_3.Position = [0.3983197831978318, 0.13949152542372864]

# change scalar bar placement
omega_normLUTColorBar_3.Position = [0.5162059620596204, 0.05474576271186424]

# change scalar bar placement
u_normLUTColorBar_2.Set(
    Position=[0.6014123793670063, 0.020847457627118635],
    ScalarBarLength=0.1999999999999993,
)

# change scalar bar placement
omega_normLUTColorBar_3.Position = [0.5582113821138209, 0.15114406779661]

# change scalar bar placement
omega_normLUTColorBar_1.Position = [0.46138211382113825, 0.0]

# hide data in view
Hide(streamTracer2, renderView2_1)

# hide data in view
Hide(calculator3, renderView2_1)

# change scalar bar placement
u_normLUTColorBar_1.Set(
    WindowLocation="Any Location",
    Position=[0.8974000677506775, 0.025423728813559324],
    ScalarBarLength=0.6438559322033897,
)

# change scalar bar placement
u_normLUTColorBar_1.Set(
    Position=[0.5992970867208672, 0.21716101694915252],
    ScalarBarLength=0.6438559322033894,
)

# change scalar bar placement
noneColorBar.Set(
    WindowLocation="Any Location",
    Position=[0.5074525745257453, 0.4629237288135593],
    ScalarBarLength=0.33,
)

# change scalar bar placement
omega_normLUTColorBar_1.Set(
    Position=[0.13414634146341464, 0.29449152542372886],
    ScalarBarLength=0.2499999999999994,
)

# change scalar bar placement
omega_normLUTColorBar_3.Set(
    Position=[0.416612466124661, 0.4149152542372879],
    ScalarBarLength=0.36861788617886165,
)

# create new layout object 'Layout #3'
layout3 = CreateLayout(name="Layout #3")

# set active view
SetActiveView(None)

# set active view
SetActiveView(renderView2_1)

# destroy renderView2_1
Delete(renderView2_1)
del renderView2_1

RemoveLayout(layout2)

# set active view
SetActiveView(None)

# get active view
renderView2 = GetActiveViewOrCreate("RenderView")

# Create a new 'Render View'
renderView2_1 = CreateView("RenderView")
renderView2_1.Set(
    StereoType="Crystal Eyes",
    ANARIRendererParameters=["", "", ""],
    BackEnd="OSPRay raycaster",
    OSPRayMaterialLibrary=materialLibrary1,
)

# assign view to a particular cell in the layout
AssignViewToLayout(view=renderView2_1, layout=layout3, hint=0)

# set active source
SetActiveSource(streamTracer2)

# show data in view
streamTracer2Display = Show(streamTracer2, renderView2_1, "GeometryRepresentation")

# trace defaults for the display properties.
streamTracer2Display.Representation = "Surface"

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# changing interaction mode based on data extents
renderView2_1.Set(
    InteractionMode="2D",
    CameraPosition=[4.291534423828125e-06, 1.5497207641601562e-06, 8.099143421649933],
    CameraFocalPoint=[4.291534423828125e-06, 1.5497207641601562e-06, 0.0],
)

# reset view to fit data
renderView2_1.ResetCamera(False, 0.9)

# set active source
SetActiveSource(calculator3)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# trace defaults for the display properties.
calculator3Display.Representation = "Surface"

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# get color legend/bar for u_normLUT in view renderView2_1
u_normLUTColorBar_1 = GetScalarBar(u_normLUT, renderView2_1)

# change scalar bar placement
u_normLUTColorBar_1.Position = [0.6550210979006326, 0.5399692167925031]

# get color legend/bar for omega_normLUT in view renderView2_1
omega_normLUTColorBar_1 = GetScalarBar(omega_normLUT, renderView2_1)

# change scalar bar placement
omega_normLUTColorBar_1.Position = [0.0641434199462442, 0.6223446957197062]

# Properties modified on omega_normLUTColorBar_1
omega_normLUTColorBar_1.Title = "$ \\omega_z / \\omega_{c,0}$"

# change scalar bar placement
u_normLUTColorBar_1.Set(
    Position=[0.6363618028073361, 0.33690408269288624],
    ScalarBarLength=0.1999999999999999,
)

# set active view
SetActiveView(renderView1)

# set active source
SetActiveSource(streamTracer1)

# set active view
SetActiveView(renderView2_1)

# set active source
SetActiveSource(streamTracer2)

# Rescale 2D transfer function
u_normTF2D.RescaleTransferFunction(0.0004675733152329219, 0.6164665401965168, 0.0, 1.0)

# Properties modified on u_normLUTColorBar_1
u_normLUTColorBar_1.Title = "$\\|\\mathbf{u} \\| / U_{c,0}$"

# set active source
SetActiveSource(calculator3)

# change representation type
calculator3Display.SetRepresentationType("Point Gaussian")

# change interaction mode for render view
renderView2_1.InteractionMode = "3D"

# change scalar bar placement
omega_normLUTColorBar_1.Position = [0.053431602392685115, 0.6520381823097445]

# change scalar bar placement
u_normLUTColorBar_1.Set(
    Position=[0.6322152927866036, 0.23058224361242646],
    ScalarBarLength=0.1999999999999999,
)

# Properties modified on renderView2_1
renderView2_1.UseColorPaletteForBackground = 0

# Properties modified on renderView2_1
renderView2_1.UseColorPaletteForBackground = 1

# Properties modified on renderView2_1
renderView2_1.UseColorPaletteForBackground = 0

# change scalar bar placement
omega_normLUTColorBar_1.Position = [0.024406032247557267, 0.664490289589438]

# change scalar bar placement
u_normLUTColorBar_1.Set(
    Position=[0.5948967026000107, 0.1031876075971008],
    ScalarBarLength=0.19999999999999982,
)

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(calculator3)

# set active source
SetActiveSource(calculator2)

# set active view
SetActiveView(renderView1)

# set active view
SetActiveView(renderView2_1)

# set active source
SetActiveSource(vpm_merging_cs_000010xdmf)

# set active source
SetActiveSource(calculator3)

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(0.06143729009052439, 0.5147135251868749)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(0.06143729009052439, 0.5147135251868749)

animationScene1.GoToFirst()

# Properties modified on renderView2_1
renderView2_1.Set(
    CenterOfRotation=[5.3000030517578125, 0.006647884845733643, 0.009999990463256836],
    CameraPosition=[9.802984298688976, 2.5219327558780518, 4.092327580292105],
    CameraFocalPoint=[-1.015601139278945, -3.3751019450226045, -3.453370564148839],
    CameraViewUp=[-0.2530668474186389, 0.9042747689630575, -0.3438667081216217],
    CameraParallelScale=3.739504136868496,
)

# Properties modified on renderView2_1
renderView2_1.Set(
    CenterOfRotation=[0.0, 0.0, 0.19999999999999996],
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

# change scalar bar placement
u_normLUTColorBar_1.Set(
    Position=[0.506783364659444, 0.05050561525993605],
    ScalarBarLength=0.1999999999999997,
)

# Properties modified on omega_normLUTColorBar_1
omega_normLUTColorBar_1.Set(
    TitleColor=[1.0, 1.0, 1.0],
    LabelColor=[1.0, 1.0, 1.0],
    ScalarBarOutlineColor=[1.0, 1.0, 1.0],
)

# set active source
SetActiveSource(streamTracer2)

# Properties modified on u_normLUTColorBar_1
u_normLUTColorBar_1.Set(
    TitleColor=[1.0, 1.0, 1.0],
    LabelColor=[1.0, 1.0, 1.0],
    ScalarBarOutlineColor=[1.0, 1.0, 1.0],
)

# Properties modified on u_normLUTColorBar_1
u_normLUTColorBar_1.Set(
    DrawTickMarks=0,
    DrawTickLabels=0,
    AddRangeLabels=1,
)

# set active source
SetActiveSource(calculator3)

# Properties modified on omega_normLUTColorBar_1
omega_normLUTColorBar_1.Set(
    DrawTickMarks=0,
    DrawTickLabels=0,
    AddRangeLabels=1,
)

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(0.06143729009052439, 0.515985418665216)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(0.06143729009052439, 0.515985418665216)

# Properties modified on calculator3Display.PolarAxes
calculator3Display.PolarAxes.Visibility = 1

# Properties modified on calculator3Display.PolarAxes
calculator3Display.PolarAxes.Visibility = 0

# Properties modified on calculator3Display.PolarAxes
calculator3Display.PolarAxes.Visibility = 1

# Properties modified on calculator3Display.PolarAxes
calculator3Display.PolarAxes.Visibility = 0

# Properties modified on renderView2_1
renderView2_1.UseColorPaletteForBackground = 1

# set active source
SetActiveSource(streamTracer2)

# Hide orientation axes
renderView2_1.OrientationAxesVisibility = 0

# Enter preview mode
layout3.PreviewMode = [1476, 945]

# change scalar bar placement
u_normLUTColorBar_1.Position = [0.7913362101878992, 0.02084459831078353]

# Properties modified on renderView2_1
renderView2_1.Set(
    CenterOfRotation=[5.3000030517578125, 0.006647884845733643, 0.009999990463256836],
    CameraPosition=[9.802984298688976, 2.5219327558780518, 4.092327580292105],
    CameraFocalPoint=[-1.015601139278945, -3.3751019450226045, -3.453370564148839],
    CameraViewUp=[-0.2530668474186389, 0.9042747689630575, -0.3438667081216217],
    CameraParallelScale=3.739504136868496,
)

# Properties modified on renderView2_1
renderView2_1.Set(
    CenterOfRotation=[0.0, 0.0, 0.19999999999999996],
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

# change scalar bar placement
omega_normLUTColorBar_1.Position = [0.04608624904972528, 0.6909733404368957]

# change scalar bar placement
u_normLUTColorBar_1.Set(
    Position=[0.8523118199439969, 0.026141208480275055],
    ScalarBarLength=0.19999999999999968,
)

animationScene1.GoToLast()

animationScene1.GoToFirst()

# set active source
SetActiveSource(calculator3)

# Properties modified on calculator3Display
calculator3Display.GaussianRadius = 0.015

# set active source
SetActiveSource(streamTracer2)

# Properties modified on streamTracer2Display
streamTracer2Display.RenderLinesAsTubes = 1

# Properties modified on streamTracer2Display
streamTracer2Display.RenderLinesAsTubes = 0

# Properties modified on streamTracer2Display
streamTracer2Display.LineWidth = 2.0

# set active view
SetActiveView(renderView1)

# set active source
SetActiveSource(streamTracer1)

# set active source
SetActiveSource(streamTracer2)

# set active view
SetActiveView(renderView2_1)

animationScene1.GoToLast()

# change scalar bar placement
omega_normLUTColorBar_1.Position = [0.030503593223167015, 0.6983885946741838]

# change scalar bar placement
u_normLUTColorBar_1.Set(
    Position=[0.8794120909467069, 0.022963242378580134],
    ScalarBarLength=0.1999999999999997,
)

# change scalar bar placement
u_normLUTColorBar_1.Set(
    Position=[0.8861871586973844, 0.03779375085315642],
    ScalarBarLength=0.19999999999999968,
)

animationScene1.GoToFirst()

# set active source
SetActiveSource(calculator3)

animationScene1.GoToFirst()

# layout/tab size in pixels
layout3.SetSize(1476, 944)

# current camera placement for renderView2_1
renderView2_1.Set(
    CameraPosition=[4.661829155108956, 2.2381249310079365, 4.512031664604395],
    CameraFocalPoint=[-1.2681266390188406, -0.7793947692982623, -0.14672798884850557],
    CameraViewUp=[-0.21690038784127927, 0.9218950549755336, -0.32103540204463515],
    CameraParallelScale=2.1022391505727165,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/mergingRenderT0.png",
    viewOrLayout=layout3,
    location=16,
    ImageResolution=[1476, 945],
    FontScaling="Do not scale fonts",
    # PNG options
    CompressionLevel="1",
)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
omega_normLUT.ApplyPreset("Viridis", True)

# layout/tab size in pixels
layout3.SetSize(1476, 944)

# current camera placement for renderView2_1
renderView2_1.Set(
    CameraPosition=[4.661829155108956, 2.2381249310079365, 4.512031664604395],
    CameraFocalPoint=[-1.2681266390188406, -0.7793947692982623, -0.14672798884850557],
    CameraViewUp=[-0.21690038784127927, 0.9218950549755336, -0.32103540204463515],
    CameraParallelScale=2.1022391505727165,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/mergingRenderT0.png",
    viewOrLayout=layout3,
    location=16,
    ImageResolution=[1476, 945],
    FontScaling="Do not scale fonts",
    # PNG options
    CompressionLevel="1",
)

animationScene1.GoToFirst()

animationScene1.Play()

animationScene1.GoToFirst()

animationScene1.Play()

animationScene1.GoToFirst()

# layout/tab size in pixels
layout3.SetSize(1476, 944)

# current camera placement for renderView2_1
renderView2_1.Set(
    CameraPosition=[4.914602071887978, 2.919102251518328, 4.786589747807204],
    CameraFocalPoint=[-0.7168317580553842, -0.5698831946028803, 0.0868263552085413],
    CameraViewUp=[-0.25898305752169487, 0.8978078382155344, -0.3561865544282423],
    CameraParallelScale=2.1022391505727165,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/mergingRenderT0.png",
    viewOrLayout=layout3,
    location=16,
    ImageResolution=[1476, 945],
    FontScaling="Do not scale fonts",
    # PNG options
    CompressionLevel="1",
)

# layout/tab size in pixels
layout3.SetSize(1476, 944)

# current camera placement for renderView2_1
renderView2_1.Set(
    CameraPosition=[4.914602071887978, 2.919102251518328, 4.786589747807204],
    CameraFocalPoint=[-0.7168317580553842, -0.5698831946028803, 0.0868263552085413],
    CameraViewUp=[-0.25898305752169487, 0.8978078382155344, -0.3561865544282423],
    CameraParallelScale=2.1022391505727165,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/mergingRenderT12.png",
    viewOrLayout=layout3,
    location=16,
    ImageResolution=[1476, 945],
    FontScaling="Do not scale fonts",
    # PNG options
    CompressionLevel="1",
)

# layout/tab size in pixels
layout3.SetSize(1476, 944)

# current camera placement for renderView2_1
renderView2_1.Set(
    CameraPosition=[4.914602071887978, 2.919102251518328, 4.786589747807204],
    CameraFocalPoint=[-0.7168317580553842, -0.5698831946028803, 0.0868263552085413],
    CameraViewUp=[-0.25898305752169487, 0.8978078382155344, -0.3561865544282423],
    CameraParallelScale=2.1022391505727165,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/mergingRenderT24.png",
    viewOrLayout=layout3,
    location=16,
    ImageResolution=[1476, 945],
    FontScaling="Do not scale fonts",
    # PNG options
    CompressionLevel="1",
)

# create new layout object 'Layout #2'
layout2_1 = CreateLayout(name="Layout #2")

# set active view
SetActiveView(None)

# get active view
renderView3 = GetActiveViewOrCreate("RenderView")

# Create a new 'Render View'
renderView3_1 = CreateView("RenderView")
renderView3_1.Set(
    StereoType="Crystal Eyes",
    ANARIRendererParameters=["", "", ""],
    BackEnd="OSPRay raycaster",
    OSPRayMaterialLibrary=materialLibrary1,
)

# assign view to a particular cell in the layout
AssignViewToLayout(view=renderView3_1, layout=layout2_1, hint=0)

# destroy renderView3_1
Delete(renderView3_1)
del renderView3_1

RemoveLayout(layout2_1)

# create a new 'XDMF Reader'
vpm_merging_dvh_000010xdmf = XDMFReader(
    registrationName="vpm_merging_dvh_000010.xdmf*",
    FileNames=[
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000010.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000020.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000030.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000040.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000050.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000060.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000070.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000080.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000090.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000100.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000110.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/merging_dvh/vpm_merging_dvh_000120.xdmf",
    ],
)

# show data in view
vpm_merging_dvh_000010xdmfDisplay = Show(
    vpm_merging_dvh_000010xdmf, renderView2_1, "UnstructuredGridRepresentation"
)

# trace defaults for the display properties.
vpm_merging_dvh_000010xdmfDisplay.Representation = "Surface"

# show color bar/color legend
vpm_merging_dvh_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, True)

# update the view to ensure updated data information
renderView2_1.Update()

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(-0.05154791098338573, 0.515985418665216)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(-0.05154791098338573, 0.515985418665216)

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

# hide data in view
Hide(streamTracer2, renderView2_1)

# set active source
SetActiveSource(streamTracer2)

# show data in view
streamTracer2Display = Show(streamTracer2, renderView2_1, "GeometryRepresentation")

# show color bar/color legend
streamTracer2Display.SetScalarBarVisibility(renderView2_1, True)

# hide data in view
Hide(calculator3, renderView2_1)

# set active source
SetActiveSource(vpm_merging_dvh_000010xdmf)

# set scalar coloring
ColorBy(vpm_merging_dvh_000010xdmfDisplay, ("POINTS", "Vorticity", "Z"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(radiusLUT, renderView2_1)

# rescale color and/or opacity maps used to include current data range
vpm_merging_dvh_000010xdmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
vpm_merging_dvh_000010xdmfDisplay.SetScalarBarVisibility(renderView2_1, True)

animationScene1.Play()

# change representation type
vpm_merging_dvh_000010xdmfDisplay.SetRepresentationType("Point Gaussian")

# destroy vpm_merging_dvh_000010xdmf
Delete(vpm_merging_dvh_000010xdmf)
del vpm_merging_dvh_000010xdmf

# Properties modified on renderView2_1
renderView2_1.Set(
    CameraPosition=[4.914602071887978, 2.919102251518328, 4.786589747807204],
    CameraFocalPoint=[-0.7168317580553842, -0.5698831946028803, 0.0868263552085413],
    CameraViewUp=[-0.25898305752169487, 0.8978078382155344, -0.3561865544282423],
)

# set active source
SetActiveSource(calculator3)

# show data in view
calculator3Display = Show(calculator3, renderView2_1, "UnstructuredGridRepresentation")

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView2_1, True)

# ================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
# ================================================================

# --------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1476, 944)

# layout/tab size in pixels
layout3.SetSize(1476, 944)

# -----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[9.802984298688976, 2.5219327558780518, 4.092327580292105],
    CameraFocalPoint=[-1.015601139278945, -3.3751019450226045, -3.453370564148839],
    CameraViewUp=[-0.2530668474186389, 0.9042747689630575, -0.3438667081216217],
    CameraParallelScale=3.739504136868496,
)

# current camera placement for renderView2_1
renderView2_1.Set(
    CameraPosition=[4.914602071887978, 2.919102251518328, 4.786589747807204],
    CameraFocalPoint=[-0.7168317580553842, -0.5698831946028803, 0.0868263552085413],
    CameraViewUp=[-0.25898305752169487, 0.8978078382155344, -0.3561865544282423],
    CameraParallelScale=2.1022391505727165,
)


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
