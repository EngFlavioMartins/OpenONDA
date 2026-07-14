# trace generated using paraview version 6.0.1-1233-gf6d296c8ae
# import paraview
# paraview.compatibility.major = 6
# paraview.compatibility.minor = 0

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XDMF Reader'
vpm_vortex_cs_000010xdmf = XDMFReader(
    registrationName="vpm_vortex_cs_000010.xdmf*",
    FileNames=[
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000010.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000020.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000030.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000040.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000050.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000060.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000070.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000080.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000090.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000100.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000110.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000120.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000130.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000140.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000150.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/vpm_vortex_cs_000160.xdmf",
    ],
)

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate("RenderView")

# show data in view
vpm_vortex_cs_000010xdmfDisplay = Show(
    vpm_vortex_cs_000010xdmf, renderView1, "UnstructuredGridRepresentation"
)

# trace defaults for the display properties.
vpm_vortex_cs_000010xdmfDisplay.Representation = "Surface"

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
vpm_vortex_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color transfer function/color map for 'Radius'
radiusLUT = GetColorTransferFunction("Radius")

# get opacity transfer function/opacity map for 'Radius'
radiusPWF = GetOpacityTransferFunction("Radius")

# get 2D transfer function for 'Radius'
radiusTF2D = GetTransferFunction2D("Radius")

animationScene1.GoToLast()

# set scalar coloring
ColorBy(vpm_vortex_cs_000010xdmfDisplay, ("POINTS", "Vorticity", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(radiusLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
vpm_vortex_cs_000010xdmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
vpm_vortex_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Vorticity'
vorticityLUT = GetColorTransferFunction("Vorticity")

# get opacity transfer function/opacity map for 'Vorticity'
vorticityPWF = GetOpacityTransferFunction("Vorticity")

# get 2D transfer function for 'Vorticity'
vorticityTF2D = GetTransferFunction2D("Vorticity")

# change representation type
vpm_vortex_cs_000010xdmfDisplay.SetRepresentationType("Point Gaussian")

# get color legend/bar for vorticityLUT in view renderView1
vorticityLUTColorBar = GetScalarBar(vorticityLUT, renderView1)

# change scalar bar placement
vorticityLUTColorBar.Set(
    Position=[0.8404178686360444, 0.06869673156618966],
    ScalarBarLength=0.20000000000000023,
)

# create a new 'Cone'
cone1 = Cone(registrationName="Cone1")

# show data in view
cone1Display = Show(cone1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
cone1Display.Representation = "Surface"

# update the view to ensure updated data information
renderView1.Update()

# destroy cone1
Delete(cone1)
del cone1

# create a new 'Cone'
cone1 = Cone(registrationName="Cone1")

# show data in view
cone1Display = Show(cone1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
cone1Display.Representation = "Surface"

# update the view to ensure updated data information
renderView1.Update()

# destroy cone1
Delete(cone1)
del cone1

# create a new 'Cylinder'
cylinder1 = Cylinder(registrationName="Cylinder1")

# show data in view
cylinder1Display = Show(cylinder1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
cylinder1Display.Representation = "Surface"

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on cylinder1
cylinder1.Resolution = 36

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on cylinder1
cylinder1.Radius = 0.25

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on cylinder1
cylinder1.Radius = 0.3

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on cylinder1Display
cylinder1Display.Orientation = [0.0, 1.0, 0.0]

# Properties modified on cylinder1Display.PolarAxes
cylinder1Display.PolarAxes.Orientation = [0.0, 1.0, 0.0]

# Properties modified on cylinder1Display
cylinder1Display.Orientation = [0.0, 0.0, 0.0]

# Properties modified on cylinder1Display.PolarAxes
cylinder1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]

# Properties modified on cylinder1Display
cylinder1Display.Orientation = [0.0, 90.0, 0.0]

# Properties modified on cylinder1Display.PolarAxes
cylinder1Display.PolarAxes.Orientation = [0.0, 90.0, 0.0]

# Properties modified on cylinder1Display
cylinder1Display.Orientation = [0.0, 0.0, 0.0]

# Properties modified on cylinder1Display.PolarAxes
cylinder1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]

# Properties modified on cylinder1Display
cylinder1Display.Orientation = [90.0, 0.0, 0.0]

# Properties modified on cylinder1Display.PolarAxes
cylinder1Display.PolarAxes.Orientation = [90.0, 0.0, 0.0]

# Properties modified on cylinder1Display
cylinder1Display.Scale = [5.0, 1.0, 1.0]

# Properties modified on cylinder1Display.DataAxesGrid
cylinder1Display.DataAxesGrid.Scale = [5.0, 1.0, 1.0]

# Properties modified on cylinder1Display.PolarAxes
cylinder1Display.PolarAxes.Scale = [5.0, 1.0, 1.0]

# Properties modified on cylinder1Display
cylinder1Display.Scale = [0.0, 1.0, 1.0]

# Properties modified on cylinder1Display.DataAxesGrid
cylinder1Display.DataAxesGrid.Scale = [0.0, 1.0, 1.0]

# Properties modified on cylinder1Display.PolarAxes
cylinder1Display.PolarAxes.Scale = [0.0, 1.0, 1.0]

# Properties modified on cylinder1Display
cylinder1Display.Scale = [0.0, 5.0, 1.0]

# Properties modified on cylinder1Display.DataAxesGrid
cylinder1Display.DataAxesGrid.Scale = [0.0, 5.0, 1.0]

# Properties modified on cylinder1Display.PolarAxes
cylinder1Display.PolarAxes.Scale = [0.0, 5.0, 1.0]

# Properties modified on cylinder1Display
cylinder1Display.Scale = [1.0, 5.0, 1.0]

# Properties modified on cylinder1Display.DataAxesGrid
cylinder1Display.DataAxesGrid.Scale = [1.0, 5.0, 1.0]

# Properties modified on cylinder1Display.PolarAxes
cylinder1Display.PolarAxes.Scale = [1.0, 5.0, 1.0]

# Properties modified on cylinder1Display
cylinder1Display.Scale = [1.0, 7.0, 1.0]

# Properties modified on cylinder1Display.DataAxesGrid
cylinder1Display.DataAxesGrid.Scale = [1.0, 7.0, 1.0]

# Properties modified on cylinder1Display.PolarAxes
cylinder1Display.PolarAxes.Scale = [1.0, 7.0, 1.0]

# Properties modified on cylinder1Display
cylinder1Display.BackfaceOpacity = 0.53

# Properties modified on cylinder1Display
cylinder1Display.BackfaceOpacity = 1.0

# Properties modified on cylinder1Display
cylinder1Display.Opacity = 0.8

# Properties modified on cylinder1Display
cylinder1Display.Opacity = 0.59

animationScene1.GoToFirst()

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

# Rescale transfer function
vorticityLUT.RescaleTransferFunction(1.5127851963043213, 6.738285541534424)

# Rescale transfer function
vorticityPWF.RescaleTransferFunction(1.5127851963043213, 6.738285541534424)

# turn off scalar coloring
ColorBy(vpm_vortex_cs_000010xdmfDisplay, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(vorticityLUT, renderView1)

# Properties modified on vpm_vortex_cs_000010xdmfDisplay
vpm_vortex_cs_000010xdmfDisplay.GaussianRadius = 0.0

# Properties modified on vpm_vortex_cs_000010xdmfDisplay
vpm_vortex_cs_000010xdmfDisplay.GaussianRadius = 0.01

# Properties modified on vpm_vortex_cs_000010xdmfDisplay
vpm_vortex_cs_000010xdmfDisplay.GaussianRadius = 0.05

# Properties modified on vpm_vortex_cs_000010xdmfDisplay
vpm_vortex_cs_000010xdmfDisplay.GaussianRadius = 0.02

# Properties modified on renderView1
renderView1.OrientationAxesVisibility = 0

# create a new 'Arrow'
arrow1 = Arrow(registrationName="Arrow1")

# show data in view
arrow1Display = Show(arrow1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow1Display.Representation = "Surface"

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on arrow1
arrow1.Set(
    TipResolution=36,
    ShaftResolution=36,
)

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on arrow1Display
arrow1Display.Orientation = [90.0, 0.0, 0.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Orientation = [90.0, 0.0, 0.0]

# Properties modified on arrow1Display
arrow1Display.Translation = [5.0, 0.0, 0.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Translation = [5.0, 0.0, 0.0]

# hide data in view
Hide(vpm_vortex_cs_000010xdmf, renderView1)

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

# show data in view
vpm_vortex_cs_000010xdmfDisplay = Show(
    vpm_vortex_cs_000010xdmf, renderView1, "UnstructuredGridRepresentation"
)

# hide data in view
Hide(cylinder1, renderView1)

# set active source
SetActiveSource(cylinder1)

# show data in view
cylinder1Display = Show(cylinder1, renderView1, "GeometryRepresentation")

# hide data in view
Hide(vpm_vortex_cs_000010xdmf, renderView1)

# set active source
SetActiveSource(arrow1)

# Properties modified on arrow1Display
arrow1Display.Translation = [0.0, 0.0, 0.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]

# Properties modified on arrow1Display
arrow1Display.Translation = [0.0, 5.0, 0.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Translation = [0.0, 5.0, 0.0]

# Properties modified on arrow1Display
arrow1Display.Translation = [0.0, 2.0, 0.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Translation = [0.0, 2.0, 0.0]

# Properties modified on arrow1Display
arrow1Display.Translation = [0.0, 1.0, 0.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Translation = [0.0, 1.0, 0.0]

# Properties modified on arrow1Display
arrow1Display.Translation = [0.0, 0.0, 0.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]

# Properties modified on arrow1Display
arrow1Display.Translation = [0.0, 0.0, 5.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Translation = [0.0, 0.0, 5.0]

# Properties modified on arrow1Display
arrow1Display.Translation = [0.0, 0.0, 3.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Translation = [0.0, 0.0, 3.0]

# Properties modified on arrow1Display
arrow1Display.Orientation = [0.0, 0.0, 0.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]

# Properties modified on arrow1Display
arrow1Display.Orientation = [0.0, 0.0, 90.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Orientation = [0.0, 0.0, 90.0]

# Properties modified on arrow1Display
arrow1Display.Orientation = [0.0, 0.0, 0.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]

# Properties modified on arrow1Display
arrow1Display.Orientation = [0.0, 90.0, 0.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Orientation = [0.0, 90.0, 0.0]

# Properties modified on arrow1Display
arrow1Display.Orientation = [0.0, -90.0, 0.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Orientation = [0.0, -90.0, 0.0]

# Properties modified on arrow1Display
arrow1Display.Translation = [0.0, 0.0, 3.2]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Translation = [0.0, 0.0, 3.2]

# Properties modified on arrow1Display
arrow1Display.Scale = [0.5, 1.0, 1.0]

# Properties modified on arrow1Display.DataAxesGrid
arrow1Display.DataAxesGrid.Scale = [0.5, 1.0, 1.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Scale = [0.5, 1.0, 1.0]

# Properties modified on arrow1Display
arrow1Display.Scale = [0.5, 0.5, 1.0]

# Properties modified on arrow1Display.DataAxesGrid
arrow1Display.DataAxesGrid.Scale = [0.5, 0.5, 1.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Scale = [0.5, 0.5, 1.0]

# Properties modified on arrow1Display
arrow1Display.Scale = [0.5, 0.5, 0.5]

# Properties modified on arrow1Display.DataAxesGrid
arrow1Display.DataAxesGrid.Scale = [0.5, 0.5, 0.5]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Scale = [0.5, 0.5, 0.5]

# Properties modified on arrow1Display
arrow1Display.Scale = [0.5, 0.5, 0.6]

# Properties modified on arrow1Display.DataAxesGrid
arrow1Display.DataAxesGrid.Scale = [0.5, 0.5, 0.6]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Scale = [0.5, 0.5, 0.6]

# Properties modified on arrow1Display
arrow1Display.Scale = [0.5, 0.6, 0.6]

# Properties modified on arrow1Display.DataAxesGrid
arrow1Display.DataAxesGrid.Scale = [0.5, 0.6, 0.6]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Scale = [0.5, 0.6, 0.6]

# Properties modified on arrow1Display
arrow1Display.Scale = [0.6, 0.6, 0.6]

# Properties modified on arrow1Display.DataAxesGrid
arrow1Display.DataAxesGrid.Scale = [0.6, 0.6, 0.6]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Scale = [0.6, 0.6, 0.6]

# Properties modified on arrow1Display
arrow1Display.Translation = [0.0, 0.0, 3.5]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Translation = [0.0, 0.0, 3.5]

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

# show data in view
vpm_vortex_cs_000010xdmfDisplay = Show(
    vpm_vortex_cs_000010xdmf, renderView1, "UnstructuredGridRepresentation"
)

# set active source
SetActiveSource(arrow1)

# set active source
SetActiveSource(cylinder1)

# set active source
SetActiveSource(arrow1)

# Properties modified on arrow1Display
arrow1Display.Translation = [0.0, 0.0, 3.0]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Translation = [0.0, 0.0, 3.0]

# Properties modified on arrow1Display
arrow1Display.Translation = [0.0, 0.0, 3.2]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Translation = [0.0, 0.0, 3.2]

# Properties modified on arrow1Display
arrow1Display.Translation = [0.0, 0.0, 3.3]

# Properties modified on arrow1Display.PolarAxes
arrow1Display.PolarAxes.Translation = [0.0, 0.0, 3.3]

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

# Properties modified on renderView1
renderView1.UseToneMapping = 1

# Properties modified on renderView1
renderView1.UseToneMapping = 0

# Properties modified on renderView1
renderView1.UseToneMapping = 1

# Properties modified on renderView1
renderView1.EnableRenderingwithANARI = 1

# Properties modified on renderView1
renderView1.EnableRenderingwithANARI = 0

# Properties modified on arrow1Display
arrow1Display.Interpolation = "PBR"

# Properties modified on arrow1Display
arrow1Display.Interpolation = "Flat"

# set active source
SetActiveSource(cylinder1)

# Properties modified on cylinder1Display
cylinder1Display.Interpolation = "Flat"

# Properties modified on cylinder1Display
cylinder1Display.Interpolation = "Gouraud"

# Properties modified on cylinder1Display
cylinder1Display.Interpolation = "PBR"

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

# Properties modified on vpm_vortex_cs_000010xdmfDisplay
vpm_vortex_cs_000010xdmfDisplay.Interpolation = "PBR"

renderView1.ApplyIsometricView()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# change interaction mode for render view
renderView1.InteractionMode = "2D"

# get layout
layout1 = GetLayout()

# Enter preview mode
layout1.PreviewMode = [1476, 945]

# Show orientation axes
renderView1.OrientationAxesVisibility = 1

# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# Exit preview mode
layout1.PreviewMode = [0, 0]

# change interaction mode for render view
renderView1.InteractionMode = "3D"

# Enter preview mode
layout1.PreviewMode = [1476, 945]

# set active source
SetActiveSource(cylinder1)

# set active source
SetActiveSource(arrow1)

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

# set scalar coloring
ColorBy(vpm_vortex_cs_000010xdmfDisplay, ("POINTS", "Vorticity", "Magnitude"))

# rescale color and/or opacity maps used to include current data range
vpm_vortex_cs_000010xdmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
vpm_vortex_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
vorticityLUT.ApplyPreset("Viridis", True)

animationScene1.GoToLast()

# Rescale transfer function
vorticityLUT.RescaleTransferFunction(0.7783187627792358, 1.8303581476211548)

# Rescale transfer function
vorticityPWF.RescaleTransferFunction(0.7783187627792358, 1.8303581476211548)

animationScene1.GoToFirst()

animationScene1.Play()

animationScene1.Play()

animationScene1.Play()

# hide color bar/color legend
vpm_vortex_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView1, False)

# show color bar/color legend
vpm_vortex_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# hide color bar/color legend
vpm_vortex_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView1, False)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
vorticityLUT.ApplyPreset("erdc_purple_BW", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
vorticityLUT.ApplyPreset("erdc_purple_BW", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
vorticityLUT.ApplyPreset("Black, Blue and White", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
vorticityLUT.ApplyPreset("Black, Blue and White", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
vorticityLUT.ApplyPreset("Plasma", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
vorticityLUT.ApplyPreset("Plasma", True)

animationScene1.GoToFirst()

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
vorticityLUT.ApplyPreset("X Ray", True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
vorticityLUT.ApplyPreset("X Ray", True)

# turn off scalar coloring
ColorBy(vpm_vortex_cs_000010xdmfDisplay, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(vorticityLUT, renderView1)

# change solid color
vpm_vortex_cs_000010xdmfDisplay.Set(
    AmbientColor=[0.3333333432674408, 1.0, 1.0],
    DiffuseColor=[0.3333333432674408, 1.0, 1.0],
)

# change solid color
vpm_vortex_cs_000010xdmfDisplay.Set(
    AmbientColor=[0.6666666865348816, 0.6666666865348816, 1.0],
    DiffuseColor=[0.6666666865348816, 0.6666666865348816, 1.0],
)

# change solid color
vpm_vortex_cs_000010xdmfDisplay.Set(
    AmbientColor=[1.0, 0.6666666865348816, 0.49803921580314636],
    DiffuseColor=[1.0, 0.6666666865348816, 0.49803921580314636],
)

# change solid color
vpm_vortex_cs_000010xdmfDisplay.Set(
    AmbientColor=[0.3333333432674408, 0.3333333432674408, 1.0],
    DiffuseColor=[0.3333333432674408, 0.3333333432674408, 1.0],
)

# change solid color
vpm_vortex_cs_000010xdmfDisplay.Set(
    AmbientColor=[0.3333333432674408, 0.0, 1.0],
    DiffuseColor=[0.3333333432674408, 0.0, 1.0],
)

# change solid color
vpm_vortex_cs_000010xdmfDisplay.Set(
    AmbientColor=[0.3333333432674408, 0.3333333432674408, 1.0],
    DiffuseColor=[0.3333333432674408, 0.3333333432674408, 1.0],
)

# Properties modified on vpm_vortex_cs_000010xdmfDisplay
vpm_vortex_cs_000010xdmfDisplay.GaussianRadius = 0.015

# Properties modified on vpm_vortex_cs_000010xdmfDisplay
vpm_vortex_cs_000010xdmfDisplay.GaussianRadius = 0.01

# Properties modified on vpm_vortex_cs_000010xdmfDisplay
vpm_vortex_cs_000010xdmfDisplay.GaussianRadius = 0.015

# layout/tab size in pixels
layout1.SetSize(1476, 944)

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[3.3416967540732747, 3.0675655719833514, 5.243336621965761],
    CameraFocalPoint=[-1.2011514556910963, -1.1163369540213413, -0.03220431395884908],
    CameraViewUp=[-0.3617892813152591, 0.8566276457547708, -0.367828210519568],
    CameraParallelScale=2.1022391505727165,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/lambOseenVortexSchematics.png",
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
    CameraPosition=[3.3416967540732747, 3.0675655719833514, 5.243336621965761],
    CameraFocalPoint=[-1.2011514556910963, -1.1163369540213413, -0.03220431395884908],
    CameraViewUp=[-0.3617892813152591, 0.8566276457547708, -0.367828210519568],
    CameraParallelScale=2.1022391505727165,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/lambOseenVortexSchematics.png",
    viewOrLayout=layout1,
    location=16,
    ImageResolution=[2952, 1890],
    FontScaling="Do not scale fonts",
    TransparentBackground=1,
    # PNG options
    CompressionLevel="1",
)

# hide data in view
Hide(vpm_vortex_cs_000010xdmf, renderView1)

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

# show data in view
vpm_vortex_cs_000010xdmfDisplay = Show(
    vpm_vortex_cs_000010xdmf, renderView1, "UnstructuredGridRepresentation"
)

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

# create a new 'XDMF Reader'
vpm_dipole_cs_000010xdmf = XDMFReader(
    registrationName="vpm_dipole_cs_000010.xdmf*",
    FileNames=[
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000010.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000020.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000030.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000040.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000050.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000060.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000070.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000080.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000090.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000100.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000110.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000120.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000130.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000140.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000150.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000160.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000170.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000180.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000190.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000200.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000210.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000220.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000230.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000240.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000250.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000260.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000270.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000280.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000290.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000300.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000310.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/vpm_dipole_cs_000320.xdmf",
    ],
)

# show data in view
vpm_dipole_cs_000010xdmfDisplay = Show(
    vpm_dipole_cs_000010xdmf, renderView1, "UnstructuredGridRepresentation"
)

# trace defaults for the display properties.
vpm_dipole_cs_000010xdmfDisplay.Representation = "Surface"

# show color bar/color legend
vpm_dipole_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# hide data in view
Hide(arrow1, renderView1)

# hide data in view
Hide(cylinder1, renderView1)

# hide data in view
Hide(vpm_vortex_cs_000010xdmf, renderView1)

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# set active source
SetActiveSource(cylinder1)

# show data in view
cylinder1Display = Show(cylinder1, renderView1, "GeometryRepresentation")

# hide data in view
Hide(cylinder1, renderView1)

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

# show data in view
vpm_vortex_cs_000010xdmfDisplay = Show(
    vpm_vortex_cs_000010xdmf, renderView1, "UnstructuredGridRepresentation"
)

# hide data in view
Hide(vpm_vortex_cs_000010xdmf, renderView1)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# create a new 'Cylinder'
cylinder2 = Cylinder(registrationName="Cylinder2")

# show data in view
cylinder2Display = Show(cylinder2, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
cylinder2Display.Representation = "Surface"

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# hide data in view
Hide(vpm_dipole_cs_000010xdmf, renderView1)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# show data in view
vpm_dipole_cs_000010xdmfDisplay = Show(
    vpm_dipole_cs_000010xdmf, renderView1, "UnstructuredGridRepresentation"
)

# set scalar coloring
ColorBy(vpm_dipole_cs_000010xdmfDisplay, ("POINTS", "Circulation", "Magnitude"))

# rescale color and/or opacity maps used to include current data range
vpm_dipole_cs_000010xdmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
vpm_dipole_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Circulation'
circulationLUT = GetColorTransferFunction("Circulation")

# get opacity transfer function/opacity map for 'Circulation'
circulationPWF = GetOpacityTransferFunction("Circulation")

# get 2D transfer function for 'Circulation'
circulationTF2D = GetTransferFunction2D("Circulation")

# turn off scalar coloring
ColorBy(vpm_dipole_cs_000010xdmfDisplay, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(circulationLUT, renderView1)

# set scalar coloring
ColorBy(vpm_dipole_cs_000010xdmfDisplay, ("POINTS", "Radius"))

# rescale color and/or opacity maps used to include current data range
vpm_dipole_cs_000010xdmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
vpm_dipole_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# set scalar coloring
ColorBy(vpm_dipole_cs_000010xdmfDisplay, ("POINTS", "GroupID"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(radiusLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
vpm_dipole_cs_000010xdmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
vpm_dipole_cs_000010xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'GroupID'
groupIDLUT = GetColorTransferFunction("GroupID")

# get opacity transfer function/opacity map for 'GroupID'
groupIDPWF = GetOpacityTransferFunction("GroupID")

# get 2D transfer function for 'GroupID'
groupIDTF2D = GetTransferFunction2D("GroupID")

# turn off scalar coloring
ColorBy(vpm_dipole_cs_000010xdmfDisplay, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(groupIDLUT, renderView1)

# set active source
SetActiveSource(cylinder2)

# set active source
SetActiveSource(cylinder1)

# set active source
SetActiveSource(cylinder2)

# set active source
SetActiveSource(cylinder1)

# set active source
SetActiveSource(cylinder2)

# set active source
SetActiveSource(cylinder1)

# set active source
SetActiveSource(cylinder2)

# Properties modified on cylinder2Display
cylinder2Display.Translation = [0.0, 2.0, 0.0]

# Properties modified on cylinder2Display.PolarAxes
cylinder2Display.PolarAxes.Translation = [0.0, 2.0, 0.0]

# Properties modified on cylinder2Display
cylinder2Display.Translation = [0.0, 1.0, 0.0]

# Properties modified on cylinder2Display.PolarAxes
cylinder2Display.PolarAxes.Translation = [0.0, 1.0, 0.0]

# Properties modified on cylinder2Display
cylinder2Display.Translation = [0.0, 0.5, 0.0]

# Properties modified on cylinder2Display.PolarAxes
cylinder2Display.PolarAxes.Translation = [0.0, 0.5, 0.0]

# Properties modified on cylinder2Display
cylinder2Display.Translation = [0.0, 0.4, 0.0]

# Properties modified on cylinder2Display.PolarAxes
cylinder2Display.PolarAxes.Translation = [0.0, 0.4, 0.0]

animationScene1.GoToFirst()

# Properties modified on cylinder2Display
cylinder2Display.Translation = [0.0, 0.45, 0.0]

# Properties modified on cylinder2Display.PolarAxes
cylinder2Display.PolarAxes.Translation = [0.0, 0.45, 0.0]

# Properties modified on cylinder2Display
cylinder2Display.Translation = [0.01, 0.45, 0.0]

# Properties modified on cylinder2Display.PolarAxes
cylinder2Display.PolarAxes.Translation = [0.01, 0.45, 0.0]

# Properties modified on cylinder2Display
cylinder2Display.Translation = [0.05, 0.45, 0.0]

# Properties modified on cylinder2Display.PolarAxes
cylinder2Display.PolarAxes.Translation = [0.05, 0.45, 0.0]

# Properties modified on cylinder2Display
cylinder2Display.Translation = [0.1, 0.45, 0.0]

# Properties modified on cylinder2Display.PolarAxes
cylinder2Display.PolarAxes.Translation = [0.1, 0.45, 0.0]

# Properties modified on cylinder2Display
cylinder2Display.Translation = [0.15, 0.45, 0.0]

# Properties modified on cylinder2Display.PolarAxes
cylinder2Display.PolarAxes.Translation = [0.15, 0.45, 0.0]

renderView1.ResetActiveCameraToNegativeZ()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# Exit preview mode
layout1.PreviewMode = [0, 0]

# change interaction mode for render view
renderView1.InteractionMode = "2D"

# Properties modified on cylinder2Display
cylinder2Display.Translation = [0.15, 0.47, 0.0]

# Properties modified on cylinder2Display.PolarAxes
cylinder2Display.PolarAxes.Translation = [0.15, 0.47, 0.0]

# Properties modified on cylinder2Display
cylinder2Display.Translation = [0.15, 0.48, 0.0]

# Properties modified on cylinder2Display.PolarAxes
cylinder2Display.PolarAxes.Translation = [0.15, 0.48, 0.0]

# Properties modified on cylinder2Display
cylinder2Display.Translation = [0.12, 0.48, 0.0]

# Properties modified on cylinder2Display.PolarAxes
cylinder2Display.PolarAxes.Translation = [0.12, 0.48, 0.0]

# Properties modified on cylinder2Display
cylinder2Display.Translation = [0.14, 0.48, 0.0]

# Properties modified on cylinder2Display.PolarAxes
cylinder2Display.PolarAxes.Translation = [0.14, 0.48, 0.0]

# create a new 'Cylinder'
cylinder3 = Cylinder(registrationName="Cylinder3")

# show data in view
cylinder3Display = Show(cylinder3, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
cylinder3Display.Representation = "Surface"

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(cylinder2)

# set active source
SetActiveSource(cylinder3)

# Properties modified on cylinder3Display
cylinder3Display.Translation = [0.14, -0.48, 0.0]

# Properties modified on cylinder3Display.PolarAxes
cylinder3Display.PolarAxes.Translation = [0.14, -0.48, 0.0]

# Properties modified on renderView1
renderView1.Set(
    CenterOfRotation=[0.0, 0.0, 0.19999999999999996],
    CameraPosition=[5.109684936214612, 2.595248598497172, 4.858944350771728],
    CameraFocalPoint=[-0.881938230588879, -0.6006831566002951, 0.4024530835897062],
    CameraViewUp=[-0.2356546083848252, 0.9115743219006995, -0.33689636566498377],
    CameraParallelScale=2.1022391505727165,
)

# change interaction mode for render view
renderView1.InteractionMode = "3D"

# create a new 'Arrow'
arrow2 = Arrow(registrationName="Arrow2")

# show data in view
arrow2Display = Show(arrow2, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow2Display.Representation = "Surface"

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(cylinder1)

# set active source
SetActiveSource(arrow2)

# set active source
SetActiveSource(arrow1)

# set active source
SetActiveSource(arrow2)

# set active source
SetActiveSource(arrow1)

# set active source
SetActiveSource(arrow2)

# change interaction mode for render view
renderView1.InteractionMode = "2D"

renderView1.ResetActiveCameraToNegativeZ()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# Properties modified on arrow2Display
arrow2Display.Translation = [0.0, 0.48, 3.3]

# Properties modified on arrow2Display.PolarAxes
arrow2Display.PolarAxes.Translation = [0.0, 0.48, 3.3]

# Properties modified on arrow2Display
arrow2Display.Translation = [0.12, 0.48, 3.3]

# Properties modified on arrow2Display.PolarAxes
arrow2Display.PolarAxes.Translation = [0.12, 0.48, 3.3]

# Properties modified on arrow2Display
arrow2Display.Translation = [0.13, 0.48, 3.3]

# Properties modified on arrow2Display.PolarAxes
arrow2Display.PolarAxes.Translation = [0.13, 0.48, 3.3]

# set active source
SetActiveSource(cylinder3)

# set active source
SetActiveSource(arrow2)

# Properties modified on arrow2Display
arrow2Display.Translation = [0.14, 0.48, 3.3]

# Properties modified on arrow2Display.PolarAxes
arrow2Display.PolarAxes.Translation = [0.14, 0.48, 3.3]

# create a new 'Arrow'
arrow3 = Arrow(registrationName="Arrow3")

# show data in view
arrow3Display = Show(arrow3, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow3Display.Representation = "Surface"

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(arrow2)

# set active source
SetActiveSource(arrow3)

# set active source
SetActiveSource(arrow2)

# set active source
SetActiveSource(arrow3)

# Properties modified on arrow3Display
arrow3Display.Translation = [0.14, -0.48, 3.3]

# Properties modified on arrow3Display.PolarAxes
arrow3Display.PolarAxes.Translation = [0.14, -0.48, 3.3]

# change interaction mode for render view
renderView1.InteractionMode = "3D"

# Properties modified on renderView1
renderView1.Set(
    CenterOfRotation=[0.0, 0.0, 0.19999999999999996],
    CameraPosition=[5.109684936214612, 2.595248598497172, 4.858944350771728],
    CameraFocalPoint=[-0.881938230588879, -0.6006831566002951, 0.4024530835897062],
    CameraViewUp=[-0.2356546083848252, 0.9115743219006995, -0.33689636566498377],
    CameraParallelScale=2.1022391505727165,
)

# Enter preview mode
layout1.PreviewMode = [1476, 945]

animationScene1.GoToFirst()

# create a new 'Arrow'
arrow4 = Arrow(registrationName="Arrow4")

# show data in view
arrow4Display = Show(arrow4, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow4Display.Representation = "Surface"

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Glyph'
glyph1 = Glyph(registrationName="Glyph1", Input=arrow4, GlyphType="Arrow")

# Properties modified on glyph1
glyph1.GlyphType = "2D Glyph"

# show data in view
glyph1Display = Show(glyph1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
glyph1Display.Representation = "Surface"

# update the view to ensure updated data information
renderView1.Update()

# hide data in view
Hide(arrow4, renderView1)

# Properties modified on glyph1
glyph1.GlyphType = "Arrow"

# update the view to ensure updated data information
renderView1.Update()

# hide data in view
Hide(glyph1, renderView1)

# set active source
SetActiveSource(glyph1)

# show data in view
glyph1Display = Show(glyph1, renderView1, "GeometryRepresentation")

# set active source
SetActiveSource(arrow4)

# hide data in view
Hide(glyph1, renderView1)

# show data in view
arrow4Display = Show(arrow4, renderView1, "GeometryRepresentation")

# destroy glyph1
Delete(glyph1)
del glyph1

# find source
glyph1 = FindSource("Glyph1")

# set active source
SetActiveSource(glyph1)

# get display properties
glyph1Display = GetRepresentation(glyph1, view=renderView1)

# set active source
SetActiveSource(arrow4)

# hide data in view
Hide(glyph1, renderView1)

# show data in view
arrow4Display = Show(arrow4, renderView1, "GeometryRepresentation")

# destroy glyph1
Delete(glyph1)
del glyph1

# set active source
SetActiveSource(arrow4)

# create a new 'Glyph'
glyph1 = Glyph(registrationName="Glyph1", Input=arrow4, GlyphType="Arrow")

# show data in view
glyph1Display = Show(glyph1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
glyph1Display.Representation = "Surface"

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.GlyphType = "2D Glyph"

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1Display
glyph1Display.RenderLinesAsTubes = 1

# set active source
SetActiveSource(arrow4)

# hide data in view
Hide(glyph1, renderView1)

# show data in view
arrow4Display = Show(arrow4, renderView1, "GeometryRepresentation")

# destroy glyph1
Delete(glyph1)
del glyph1

# Properties modified on renderView1
renderView1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
)

# set active source
SetActiveSource(arrow2)

# set active source
SetActiveSource(arrow4)

# set active source
SetActiveSource(arrow3)

# set active source
SetActiveSource(arrow4)

# Properties modified on arrow4Display
arrow4Display.Translation = [0.0, 0.48, 0.0]

# Properties modified on arrow4Display.PolarAxes
arrow4Display.PolarAxes.Translation = [0.0, 0.48, 0.0]

# Properties modified on arrow4Display
arrow4Display.Translation = [0.12, 0.48, 0.0]

# Properties modified on arrow4Display.PolarAxes
arrow4Display.PolarAxes.Translation = [0.12, 0.48, 0.0]

# Properties modified on arrow4Display
arrow4Display.Translation = [0.14, 0.48, 0.0]

# Properties modified on arrow4Display.PolarAxes
arrow4Display.PolarAxes.Translation = [0.14, 0.48, 0.0]

# Properties modified on arrow4Display
arrow4Display.Scale = [0.6, 1.0, 1.0]

# Properties modified on arrow4Display.DataAxesGrid
arrow4Display.DataAxesGrid.Scale = [0.6, 1.0, 1.0]

# Properties modified on arrow4Display.PolarAxes
arrow4Display.PolarAxes.Scale = [0.6, 1.0, 1.0]

# Properties modified on arrow4Display
arrow4Display.Scale = [0.6, 0.6, 1.0]

# Properties modified on arrow4Display.DataAxesGrid
arrow4Display.DataAxesGrid.Scale = [0.6, 0.6, 1.0]

# Properties modified on arrow4Display.PolarAxes
arrow4Display.PolarAxes.Scale = [0.6, 0.6, 1.0]

# Properties modified on arrow4Display
arrow4Display.Scale = [0.6, 0.6, 0.6]

# Properties modified on arrow4Display.DataAxesGrid
arrow4Display.DataAxesGrid.Scale = [0.6, 0.6, 0.6]

# Properties modified on arrow4Display.PolarAxes
arrow4Display.PolarAxes.Scale = [0.6, 0.6, 0.6]

# Properties modified on arrow4Display
arrow4Display.Translation = [0.28, 0.48, 0.0]

# Properties modified on arrow4Display.PolarAxes
arrow4Display.PolarAxes.Translation = [0.28, 0.48, 0.0]

# Properties modified on arrow4Display
arrow4Display.Scale = [0.8, 0.6, 0.6]

# Properties modified on arrow4Display.DataAxesGrid
arrow4Display.DataAxesGrid.Scale = [0.8, 0.6, 0.6]

# Properties modified on arrow4Display.PolarAxes
arrow4Display.PolarAxes.Scale = [0.8, 0.6, 0.6]

# Properties modified on arrow4Display
arrow4Display.Scale = [1.0, 0.6, 0.6]

# Properties modified on arrow4Display.DataAxesGrid
arrow4Display.DataAxesGrid.Scale = [1.0, 0.6, 0.6]

# Properties modified on arrow4Display.PolarAxes
arrow4Display.PolarAxes.Scale = [1.0, 0.6, 0.6]

# Properties modified on arrow4
arrow4.TipLength = 0.2

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on arrow4
arrow4.TipLength = 0.25

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(arrow3)

# set active source
SetActiveSource(arrow4)

# create a new 'Arrow'
arrow5 = Arrow(registrationName="Arrow5")

# show data in view
arrow5Display = Show(arrow5, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow5Display.Representation = "Surface"

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(arrow4)

# set active source
SetActiveSource(arrow5)

# Properties modified on arrow5Display
arrow5Display.Translation = [0.28, -0.48, 0.0]

# Properties modified on arrow5Display.PolarAxes
arrow5Display.PolarAxes.Translation = [0.28, -0.48, 0.0]

# Properties modified on arrow5
arrow5.TipLength = 0.25

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on renderView1
renderView1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
)

# layout/tab size in pixels
layout1.SetSize(1476, 944)

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/dipoleSchematics.png",
    viewOrLayout=layout1,
    location=16,
    ImageResolution=[2952, 1890],
    FontScaling="Do not scale fonts",
    TransparentBackground=1,
    # PNG options
    CompressionLevel="1",
)

# set active source
SetActiveSource(cylinder1)

# show data in view
cylinder1Display = Show(cylinder1, renderView1, "GeometryRepresentation")

# set active source
SetActiveSource(arrow1)

# show data in view
arrow1Display = Show(arrow1, renderView1, "GeometryRepresentation")

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

# show data in view
vpm_vortex_cs_000010xdmfDisplay = Show(
    vpm_vortex_cs_000010xdmf, renderView1, "UnstructuredGridRepresentation"
)

# hide data in view
Hide(vpm_dipole_cs_000010xdmf, renderView1)

# hide data in view
Hide(cylinder2, renderView1)

# hide data in view
Hide(cylinder3, renderView1)

# hide data in view
Hide(arrow2, renderView1)

# hide data in view
Hide(arrow3, renderView1)

# hide data in view
Hide(arrow4, renderView1)

# hide data in view
Hide(arrow5, renderView1)

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

# set active source
SetActiveSource(arrow5)

# show data in view
arrow5Display_1 = Show(arrow5, renderView2_1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow5Display_1.Representation = "Surface"

# reset view to fit data
renderView2_1.ResetCamera(False, 0.9)

# set active source
SetActiveSource(arrow4)

# show data in view
arrow4Display_1 = Show(arrow4, renderView2_1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow4Display_1.Representation = "Surface"

# hide data in view
Hide(arrow4, renderView2_1)

# show data in view
arrow4Display_1 = Show(arrow4, renderView2_1, "GeometryRepresentation")

# set active source
SetActiveSource(arrow3)

# show data in view
arrow3Display_1 = Show(arrow3, renderView2_1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow3Display_1.Representation = "Surface"

# set active source
SetActiveSource(arrow2)

# show data in view
arrow2Display_1 = Show(arrow2, renderView2_1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow2Display_1.Representation = "Surface"

# set active source
SetActiveSource(arrow2)

# set active source
SetActiveSource(cylinder3)

# show data in view
cylinder3Display_1 = Show(cylinder3, renderView2_1, "GeometryRepresentation")

# trace defaults for the display properties.
cylinder3Display_1.Representation = "Surface"

# set active source
SetActiveSource(cylinder2)

# show data in view
cylinder2Display_1 = Show(cylinder2, renderView2_1, "GeometryRepresentation")

# trace defaults for the display properties.
cylinder2Display_1.Representation = "Surface"

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

# set active view
SetActiveView(renderView1)

# set active source
SetActiveSource(arrow5)

# set active view
SetActiveView(renderView2_1)

# set active view
SetActiveView(renderView1)

# set active source
SetActiveSource(arrow4)

# set active view
SetActiveView(renderView2_1)

# set active view
SetActiveView(renderView1)

# set active source
SetActiveSource(arrow3)

# set active view
SetActiveView(renderView2_1)

# Properties modified on renderView2_1
renderView2_1.Set(
    CenterOfRotation=[0.0, 0.0, 0.19999999999999996],
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

# set active view
SetActiveView(renderView1)

# set active source
SetActiveSource(arrow2)

# set active view
SetActiveView(renderView2_1)

# set active source
SetActiveSource(cylinder3)

# set active view
SetActiveView(renderView1)

# set active view
SetActiveView(renderView2_1)

# set active view
SetActiveView(renderView1)

# set active source
SetActiveSource(cylinder2)

# set active view
SetActiveView(renderView2_1)

# set active view
SetActiveView(renderView1)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# set active view
SetActiveView(renderView2_1)

# set scalar coloring
ColorBy(vpm_dipole_cs_000010xdmfDisplay_1, ("POINTS", "Radius"))

# rescale color and/or opacity maps used to include current data range
vpm_dipole_cs_000010xdmfDisplay_1.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
vpm_dipole_cs_000010xdmfDisplay_1.SetScalarBarVisibility(renderView2_1, True)

# turn off scalar coloring
ColorBy(vpm_dipole_cs_000010xdmfDisplay_1, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(radiusLUT, renderView2_1)

# set active view
SetActiveView(renderView1)

# set active view
SetActiveView(renderView2_1)

# set active view
SetActiveView(renderView1)

# layout/tab size in pixels
layout1.SetSize(1476, 944)

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/lambOseenVortexSchematics.png",
    viewOrLayout=layout1,
    location=16,
    ImageResolution=[2952, 1890],
    FontScaling="Do not scale fonts",
    TransparentBackground=1,
    # PNG options
    CompressionLevel="1",
)

# set active view
SetActiveView(renderView2_1)

# Enter preview mode
layout2.PreviewMode = [1476, 945]

# layout/tab size in pixels
layout2.SetSize(1476, 944)

# current camera placement for renderView2_1
renderView2_1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/dipoleSchematics.png",
    viewOrLayout=layout2,
    location=16,
    ImageResolution=[1476, 945],
    FontScaling="Do not scale fonts",
    TransparentBackground=1,
    # PNG options
    CompressionLevel="1",
)

# layout/tab size in pixels
layout2.SetSize(1476, 944)

# current camera placement for renderView2_1
renderView2_1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/dipoleSchematics.png",
    viewOrLayout=layout2,
    location=16,
    ImageResolution=[2952, 1890],
    FontScaling="Do not scale fonts",
    TransparentBackground=1,
    # PNG options
    CompressionLevel="1",
)

# create new layout object 'Layout #3'
layout3 = CreateLayout(name="Layout #3")

# set active view
SetActiveView(None)

# get active view
renderView3 = GetActiveViewOrCreate("RenderView")

# assign view to a particular cell in the layout
AssignViewToLayout(view=renderView3_1, layout=layout3, hint=0)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# show data in view
vpm_dipole_cs_000010xdmfDisplay_2 = Show(
    vpm_dipole_cs_000010xdmf, renderView3_1, "UnstructuredGridRepresentation"
)

# trace defaults for the display properties.
vpm_dipole_cs_000010xdmfDisplay_2.Representation = "Surface"

# show color bar/color legend
vpm_dipole_cs_000010xdmfDisplay_2.SetScalarBarVisibility(renderView3_1, True)

# reset view to fit data
renderView3_1.ResetCamera(False, 0.9)

# hide data in view
Hide(vpm_dipole_cs_000010xdmf, renderView3_1)

# show data in view
vpm_dipole_cs_000010xdmfDisplay_2 = Show(
    vpm_dipole_cs_000010xdmf, renderView3_1, "UnstructuredGridRepresentation"
)

# show color bar/color legend
vpm_dipole_cs_000010xdmfDisplay_2.SetScalarBarVisibility(renderView3_1, True)

# reset view to fit data
renderView3_1.ResetCamera(False, 0.9)

# set active source
SetActiveSource(cylinder2)

# show data in view
cylinder2Display_2 = Show(cylinder2, renderView3_1, "GeometryRepresentation")

# trace defaults for the display properties.
cylinder2Display_2.Representation = "Surface"

# set active source
SetActiveSource(cylinder3)

# show data in view
cylinder3Display_2 = Show(cylinder3, renderView3_1, "GeometryRepresentation")

# trace defaults for the display properties.
cylinder3Display_2.Representation = "Surface"

# set active source
SetActiveSource(arrow2)

# show data in view
arrow2Display_2 = Show(arrow2, renderView3_1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow2Display_2.Representation = "Surface"

# set active source
SetActiveSource(arrow3)

# show data in view
arrow3Display_2 = Show(arrow3, renderView3_1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow3Display_2.Representation = "Surface"

# set active source
SetActiveSource(arrow4)

# show data in view
arrow4Display_2 = Show(arrow4, renderView3_1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow4Display_2.Representation = "Surface"

# set active source
SetActiveSource(arrow5)

# show data in view
arrow5Display_2 = Show(arrow5, renderView3_1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow5Display_2.Representation = "Surface"

# set active view
SetActiveView(renderView2_1)

# set active view
SetActiveView(renderView3_1)

# set active view
SetActiveView(renderView2_1)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# set active view
SetActiveView(renderView3_1)

# set active source
SetActiveSource(cylinder2)

# set active view
SetActiveView(renderView2_1)

# set active view
SetActiveView(renderView3_1)

# set active view
SetActiveView(renderView2_1)

# set active source
SetActiveSource(cylinder3)

# set active view
SetActiveView(renderView3_1)

# set active view
SetActiveView(renderView2_1)

# set active source
SetActiveSource(arrow2)

# set active view
SetActiveView(renderView3_1)

# set active source
SetActiveSource(arrow3)

# set active view
SetActiveView(renderView2_1)

# set active view
SetActiveView(renderView3_1)

# set active view
SetActiveView(renderView2_1)

# set active source
SetActiveSource(arrow4)

# set active view
SetActiveView(renderView3_1)

# set active view
SetActiveView(renderView2_1)

# set active source
SetActiveSource(arrow5)

# set active view
SetActiveView(renderView3_1)

# Properties modified on renderView3_1
renderView3_1.Set(
    CenterOfRotation=[0.0, 0.0, 0.19999999999999996],
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# set scalar coloring
ColorBy(vpm_dipole_cs_000010xdmfDisplay_2, ("POINTS", "Radius"))

# rescale color and/or opacity maps used to include current data range
vpm_dipole_cs_000010xdmfDisplay_2.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
vpm_dipole_cs_000010xdmfDisplay_2.SetScalarBarVisibility(renderView3_1, True)

# turn off scalar coloring
ColorBy(vpm_dipole_cs_000010xdmfDisplay_2, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(radiusLUT, renderView3_1)

# hide data in view
Hide(arrow4, renderView3_1)

# set active source
SetActiveSource(arrow4)

# show data in view
arrow4Display_2 = Show(arrow4, renderView3_1, "GeometryRepresentation")

# Properties modified on arrow4Display_2
arrow4Display_2.Scale = [-1.0, 0.6, 0.6]

# Properties modified on arrow4Display_2.DataAxesGrid
arrow4Display_2.DataAxesGrid.Scale = [-1.0, 0.6, 0.6]

# Properties modified on arrow4Display_2.PolarAxes
arrow4Display_2.PolarAxes.Scale = [-1.0, 0.6, 0.6]

# Properties modified on arrow4Display_2
arrow4Display_2.Translation = [0.0, 0.48, 0.0]

# Properties modified on arrow4Display_2.PolarAxes
arrow4Display_2.PolarAxes.Translation = [0.0, 0.48, 0.0]

# Properties modified on arrow4Display_2
arrow4Display_2.Translation = [-0.2, 0.48, 0.0]

# Properties modified on arrow4Display_2.PolarAxes
arrow4Display_2.PolarAxes.Translation = [-0.2, 0.48, 0.0]

# Properties modified on arrow4Display_2
arrow4Display_2.Translation = [-0.28, 0.48, 0.0]

# Properties modified on arrow4Display_2.PolarAxes
arrow4Display_2.PolarAxes.Translation = [-0.28, 0.48, 0.0]

# Properties modified on arrow4Display_2
arrow4Display_2.Translation = [-0.3, 0.48, 0.0]

# Properties modified on arrow4Display_2.PolarAxes
arrow4Display_2.PolarAxes.Translation = [-0.3, 0.48, 0.0]

# Enter preview mode
layout3.PreviewMode = [1476, 945]

# hide data in view
Hide(arrow3, renderView3_1)

# set active source
SetActiveSource(arrow3)

# show data in view
arrow3Display_2 = Show(arrow3, renderView3_1, "GeometryRepresentation")

# Properties modified on arrow3Display_2
arrow3Display_2.Orientation = [-90.0, -90.0, 0.0]

# Properties modified on arrow3Display_2.PolarAxes
arrow3Display_2.PolarAxes.Orientation = [-90.0, -90.0, 0.0]

# Properties modified on arrow3Display_2
arrow3Display_2.Orientation = [-180.0, -90.0, 0.0]

# Properties modified on arrow3Display_2.PolarAxes
arrow3Display_2.PolarAxes.Orientation = [-180.0, -90.0, 0.0]

# Properties modified on arrow3Display_2
arrow3Display_2.Translation = [0.14, -0.48, 3.6]

# Properties modified on arrow3Display_2.PolarAxes
arrow3Display_2.PolarAxes.Translation = [0.14, -0.48, 3.6]

# Properties modified on arrow3Display_2
arrow3Display_2.Translation = [0.14, -0.48, 4.2]

# Properties modified on arrow3Display_2.PolarAxes
arrow3Display_2.PolarAxes.Translation = [0.14, -0.48, 4.2]

# Properties modified on arrow3Display_2
arrow3Display_2.Translation = [0.14, -0.48, 4.0]

# Properties modified on arrow3Display_2.PolarAxes
arrow3Display_2.PolarAxes.Translation = [0.14, -0.48, 4.0]

# Properties modified on arrow3Display_2
arrow3Display_2.Translation = [0.14, -0.48, 3.8]

# Properties modified on arrow3Display_2.PolarAxes
arrow3Display_2.PolarAxes.Translation = [0.14, -0.48, 3.8]

# Properties modified on arrow3Display_2
arrow3Display_2.Translation = [0.14, -0.48, 3.9]

# Properties modified on arrow3Display_2.PolarAxes
arrow3Display_2.PolarAxes.Translation = [0.14, -0.48, 3.9]

# set active source
SetActiveSource(arrow3)

# set active view
SetActiveView(renderView2_1)

# set active view
SetActiveView(renderView3_1)

# set active source
SetActiveSource(arrow4)

# set active source
SetActiveSource(arrow3)

# Properties modified on arrow3Display_2
arrow3Display_2.Orientation = [0.0, -90.0, 0.0]

# Properties modified on arrow3Display_2.PolarAxes
arrow3Display_2.PolarAxes.Orientation = [0.0, -90.0, 0.0]

# Properties modified on arrow3Display_2
arrow3Display_2.Translation = [0.14, -0.48, 3.6]

# Properties modified on arrow3Display_2.PolarAxes
arrow3Display_2.PolarAxes.Translation = [0.14, -0.48, 3.6]

# Properties modified on arrow3Display_2
arrow3Display_2.Translation = [0.14, -0.48, 3.2]

# Properties modified on arrow3Display_2.PolarAxes
arrow3Display_2.PolarAxes.Translation = [0.14, -0.48, 3.2]

# Properties modified on arrow3Display_2
arrow3Display_2.Translation = [0.14, -0.48, 3.4]

# Properties modified on arrow3Display_2.PolarAxes
arrow3Display_2.PolarAxes.Translation = [0.14, -0.48, 3.4]

# Properties modified on arrow3Display_2
arrow3Display_2.Translation = [0.14, -0.48, 3.3]

# Properties modified on arrow3Display_2.PolarAxes
arrow3Display_2.PolarAxes.Translation = [0.14, -0.48, 3.3]

# set active source
SetActiveSource(arrow4)

# set active source
SetActiveSource(arrow2)

# set active source
SetActiveSource(arrow3)

# set active source
SetActiveSource(arrow2)

# set active source
SetActiveSource(arrow3)

# set active view
SetActiveView(renderView2_1)

# set active source
SetActiveSource(arrow5)

# create a new 'Arrow'
arrow6 = Arrow(registrationName="Arrow6")

# show data in view
arrow6Display = Show(arrow6, renderView2_1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow6Display.Representation = "Surface"

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView2_1.Update()

# Properties modified on arrow6
arrow6.Set(
    TipResolution=36,
    ShaftResolution=36,
)

# update the view to ensure updated data information
renderView2_1.Update()

# Properties modified on arrow6Display
arrow6Display.Orientation = [90.0, 0.0, 0.0]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Orientation = [90.0, 0.0, 0.0]

# hide data in view
Hide(arrow6, renderView2_1)

# set active source
SetActiveSource(arrow6)

# show data in view
arrow6Display = Show(arrow6, renderView2_1, "GeometryRepresentation")

# Properties modified on arrow6Display
arrow6Display.Orientation = [0.0, 0.0, 0.0]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]

# Properties modified on arrow6Display
arrow6Display.Orientation = [0.0, 90.0, 0.0]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Orientation = [0.0, 90.0, 0.0]

# Properties modified on arrow6Display
arrow6Display.Orientation = [0.0, 0.0, 0.0]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]

# Properties modified on arrow6Display
arrow6Display.Orientation = [0.0, 0.0, 90.0]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Orientation = [0.0, 0.0, 90.0]

# Properties modified on arrow6Display
arrow6Display.Scale = [0.6, 1.0, 1.0]

# Properties modified on arrow6Display.DataAxesGrid
arrow6Display.DataAxesGrid.Scale = [0.6, 1.0, 1.0]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Scale = [0.6, 1.0, 1.0]

# Properties modified on arrow6Display
arrow6Display.Scale = [0.6, 0.6, 1.0]

# Properties modified on arrow6Display.DataAxesGrid
arrow6Display.DataAxesGrid.Scale = [0.6, 0.6, 1.0]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Scale = [0.6, 0.6, 1.0]

# Properties modified on arrow6Display
arrow6Display.Scale = [0.6, 0.6, 0.6]

# Properties modified on arrow6Display.DataAxesGrid
arrow6Display.DataAxesGrid.Scale = [0.6, 0.6, 0.6]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Scale = [0.6, 0.6, 0.6]

# Properties modified on arrow6Display
arrow6Display.Translation = [3.3, 0.0, 0.0]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Translation = [3.3, 0.0, 0.0]

# Properties modified on arrow6Display
arrow6Display.Translation = [0.0, 0.0, 0.0]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Translation = [0.0, 0.0, 0.0]

# Properties modified on arrow6Display
arrow6Display.Translation = [0.0, 0.0, 1.0]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Translation = [0.0, 0.0, 1.0]

# Properties modified on arrow6Display
arrow6Display.Translation = [0.0, 0.0, 3.3]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Translation = [0.0, 0.0, 3.3]

# Properties modified on arrow6Display
arrow6Display.Translation = [0.0, 0.0, 3.5]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Translation = [0.0, 0.0, 3.5]

# Properties modified on arrow6Display
arrow6Display.Translation = [0.0, 0.48, 3.5]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Translation = [0.0, 0.48, 3.5]

# Properties modified on arrow6Display
arrow6Display.Translation = [0.0, 0.0, 3.5]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Translation = [0.0, 0.0, 3.5]

# Properties modified on arrow6Display
arrow6Display.Translation = [0.12, 0.0, 3.5]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Translation = [0.12, 0.0, 3.5]

# Properties modified on arrow6Display
arrow6Display.Translation = [0.12, -0.12, 3.5]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Translation = [0.12, -0.12, 3.5]

# Properties modified on arrow6Display
arrow6Display.Translation = [0.12, -0.12, 3.6]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Translation = [0.12, -0.12, 3.6]

# Properties modified on arrow6Display
arrow6Display.Translation = [0.14, -0.12, 3.6]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Translation = [0.14, -0.12, 3.6]

# Properties modified on renderView2_1
renderView2_1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
)

# Properties modified on arrow6
arrow6.Invert = 1

# update the view to ensure updated data information
renderView2_1.Update()

# Properties modified on arrow6
arrow6.Invert = 0

# update the view to ensure updated data information
renderView2_1.Update()

# create a new 'Arrow'
arrow7 = Arrow(registrationName="Arrow7")

# show data in view
arrow7Display = Show(arrow7, renderView2_1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow7Display.Representation = "Surface"

# update the view to ensure updated data information
renderView2_1.Update()

# set active source
SetActiveSource(arrow5)

# set active source
SetActiveSource(arrow6)

# set active source
SetActiveSource(arrow7)

# hide data in view
Hide(arrow6, renderView2_1)

# set active source
SetActiveSource(arrow6)

# show data in view
arrow6Display = Show(arrow6, renderView2_1, "GeometryRepresentation")

# set active source
SetActiveSource(arrow7)

# Properties modified on arrow7
arrow7.Invert = 1

# update the view to ensure updated data information
renderView2_1.Update()

# Properties modified on arrow7
arrow7.Invert = 0

# update the view to ensure updated data information
renderView2_1.Update()

# Properties modified on arrow7Display
arrow7Display.Orientation = [0.0, 0.0, -60.0]

# Properties modified on arrow7Display.PolarAxes
arrow7Display.PolarAxes.Orientation = [0.0, 0.0, -60.0]

# Properties modified on arrow7Display
arrow7Display.Orientation = [0.0, 0.0, -90.0]

# Properties modified on arrow7Display.PolarAxes
arrow7Display.PolarAxes.Orientation = [0.0, 0.0, -90.0]

# Properties modified on arrow7Display
arrow7Display.Translation = [0.14, -0.0, 3.6]

# Properties modified on arrow7Display.PolarAxes
arrow7Display.PolarAxes.Translation = [0.14, -0.0, 3.6]

# Properties modified on arrow7Display
arrow7Display.Translation = [0.14, 0.1, 3.6]

# Properties modified on arrow7Display.PolarAxes
arrow7Display.PolarAxes.Translation = [0.14, 0.1, 3.6]

# Properties modified on arrow7Display
arrow7Display.Translation = [0.14, 0.09, 3.6]

# Properties modified on arrow7Display.PolarAxes
arrow7Display.PolarAxes.Translation = [0.14, 0.09, 3.6]

# Properties modified on arrow7Display
arrow7Display.Translation = [0.14, 0.1, 3.6]

# Properties modified on arrow7Display.PolarAxes
arrow7Display.PolarAxes.Translation = [0.14, 0.1, 3.6]

# Properties modified on arrow7Display
arrow7Display.Translation = [0.14, 0.11, 3.6]

# Properties modified on arrow7Display.PolarAxes
arrow7Display.PolarAxes.Translation = [0.14, 0.11, 3.6]

# Properties modified on arrow7Display
arrow7Display.Translation = [0.14, 0.12, 3.6]

# Properties modified on arrow7Display.PolarAxes
arrow7Display.PolarAxes.Translation = [0.14, 0.12, 3.6]

# Properties modified on arrow7Display
arrow7Display.Translation = [0.14, 0.13, 3.6]

# Properties modified on arrow7Display.PolarAxes
arrow7Display.PolarAxes.Translation = [0.14, 0.13, 3.6]

# set active source
SetActiveSource(arrow6)

# Properties modified on arrow6Display
arrow6Display.Translation = [0.14, -0.13, 3.6]

# Properties modified on arrow6Display.PolarAxes
arrow6Display.PolarAxes.Translation = [0.14, -0.13, 3.6]

# Properties modified on renderView2_1
renderView2_1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
)

# layout/tab size in pixels
layout2.SetSize(1476, 944)

# current camera placement for renderView2_1
renderView2_1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/dipoleSchematics.png",
    viewOrLayout=layout2,
    location=16,
    ImageResolution=[2952, 1890],
    FontScaling="Do not scale fonts",
    TransparentBackground=1,
    # PNG options
    CompressionLevel="1",
)

# set active view
SetActiveView(renderView3_1)

# set active source
SetActiveSource(arrow6)

# show data in view
arrow6Display_1 = Show(arrow6, renderView3_1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow6Display_1.Representation = "Surface"

# set active source
SetActiveSource(arrow7)

# show data in view
arrow7Display_1 = Show(arrow7, renderView3_1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow7Display_1.Representation = "Surface"

# set active source
SetActiveSource(arrow6)

# set active view
SetActiveView(renderView2_1)

# set active view
SetActiveView(renderView3_1)

# set active view
SetActiveView(renderView2_1)

# set active source
SetActiveSource(arrow7)

# set active view
SetActiveView(renderView3_1)

# Properties modified on renderView3_1
renderView3_1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
)

# layout/tab size in pixels
layout3.SetSize(1476, 944)

# current camera placement for renderView3_1
renderView3_1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

# save screenshot
SaveScreenshot(
    filename="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/assets/mergingSchematics.png",
    viewOrLayout=layout3,
    location=16,
    ImageResolution=[2952, 1890],
    FontScaling="Do not scale fonts",
    TransparentBackground=1,
    # PNG options
    CompressionLevel="1",
)

# Properties modified on renderView3_1
renderView3_1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
)

# set active source
SetActiveSource(vpm_vortex_cs_000010xdmf)

ExtendFileSeries(vpm_vortex_cs_000010xdmf)

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

ExtendFileSeries(vpm_dipole_cs_000010xdmf)

animationScene1.Play()

# reset view to fit data
renderView3_1.ResetCamera(False, 0.9)

# set scalar coloring
ColorBy(vpm_dipole_cs_000010xdmfDisplay_2, ("POINTS", "Vorticity", "Magnitude"))

# rescale color and/or opacity maps used to include current data range
vpm_dipole_cs_000010xdmfDisplay_2.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
vpm_dipole_cs_000010xdmfDisplay_2.SetScalarBarVisibility(renderView3_1, True)

# Rescale transfer function
vorticityLUT.RescaleTransferFunction(0.28740522265434265, 0.9550838470458984)

# Rescale transfer function
vorticityPWF.RescaleTransferFunction(0.28740522265434265, 0.9550838470458984)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
vorticityLUT.ApplyPreset("Viridis", True)

# create a new 'PVD Reader'
vortex_cs_z0pvd = PVDReader(
    registrationName="vortex_cs_z0.pvd",
    FileName="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/vortex_cs/samples/vortex_cs_z0.pvd",
)

# show data in view
vortex_cs_z0pvdDisplay = Show(vortex_cs_z0pvd, renderView3_1, "StructuredGridRepresentation")

# trace defaults for the display properties.
vortex_cs_z0pvdDisplay.Representation = "Surface"

# show color bar/color legend
vortex_cs_z0pvdDisplay.SetScalarBarVisibility(renderView3_1, True)

# update the view to ensure updated data information
renderView3_1.Update()

# Rescale transfer function
vorticityLUT.RescaleTransferFunction(0.28740522265434265, 0.9581561088562012)

# Rescale transfer function
vorticityPWF.RescaleTransferFunction(0.28740522265434265, 0.9581561088562012)

# get color transfer function/color map for 'Velocity'
velocityLUT = GetColorTransferFunction("Velocity")

# get opacity transfer function/opacity map for 'Velocity'
velocityPWF = GetOpacityTransferFunction("Velocity")

# get 2D transfer function for 'Velocity'
velocityTF2D = GetTransferFunction2D("Velocity")

# destroy vortex_cs_z0pvd
Delete(vortex_cs_z0pvd)
del vortex_cs_z0pvd

# create a new 'PVD Reader'
dipole_cs_z0pvd = PVDReader(
    registrationName="dipole_cs_z0.pvd",
    FileName="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/lambOseenVortex/solution/dipole_cs/samples/dipole_cs_z0.pvd",
)

# show data in view
dipole_cs_z0pvdDisplay = Show(dipole_cs_z0pvd, renderView3_1, "StructuredGridRepresentation")

# trace defaults for the display properties.
dipole_cs_z0pvdDisplay.Representation = "Surface"

# show color bar/color legend
dipole_cs_z0pvdDisplay.SetScalarBarVisibility(renderView3_1, True)

# update the view to ensure updated data information
renderView3_1.Update()

# Rescale transfer function
velocityLUT.RescaleTransferFunction(0.003698638544641327, 0.45487244543401273)

# Rescale transfer function
velocityPWF.RescaleTransferFunction(0.003698638544641327, 0.45487244543401273)

animationScene1.GoToLast()

animationScene1.GoToLast()

animationScene1.GoToLast()

# set active source
SetActiveSource(vpm_dipole_cs_000010xdmf)

# turn off scalar coloring
ColorBy(vpm_dipole_cs_000010xdmfDisplay_2, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(vorticityLUT, renderView3_1)

# set active source
SetActiveSource(dipole_cs_z0pvd)

# destroy dipole_cs_z0pvd
Delete(dipole_cs_z0pvd)
del dipole_cs_z0pvd

animationScene1.GoToFirst()

# Properties modified on renderView3_1
renderView3_1.Set(
    CenterOfRotation=[0.0, 0.0, 0.19999999999999996],
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

# ================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
# ================================================================

# --------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout2.SetSize(1476, 944)

# layout/tab size in pixels
layout1.SetSize(1476, 944)

# layout/tab size in pixels
layout3.SetSize(1476, 944)

# -----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

# current camera placement for renderView2_1
renderView2_1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
    CameraParallelScale=2.1022391505727165,
)

# current camera placement for renderView3_1
renderView3_1.Set(
    CameraPosition=[5.140340443009666, 2.8744021063316954, 4.949527232308503],
    CameraFocalPoint=[-0.776275242088791, -0.5215924670370764, 0.5410442661189823],
    CameraViewUp=[-0.25519012504931765, 0.9007735408568369, -0.35140436560969224],
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
