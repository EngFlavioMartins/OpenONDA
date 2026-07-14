# trace generated using paraview version 6.0.1-1233-gf6d296c8ae
# import paraview
# paraview.compatibility.major = 6
# paraview.compatibility.minor = 0

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
slice_x9mpvd = FindSource("slice_x9m.pvd*")

# set active source
SetActiveSource(slice_x9mpvd)

# get color transfer function/color map for 'Velocity'
velocityLUT = GetColorTransferFunction("Velocity")

# get opacity transfer function/opacity map for 'Velocity'
velocityPWF = GetOpacityTransferFunction("Velocity")

# get 2D transfer function for 'Velocity'
velocityTF2D = GetTransferFunction2D("Velocity")

# get active view
renderView1 = GetActiveViewOrCreate("RenderView")

# get display properties
slice_x9mpvdDisplay = GetRepresentation(slice_x9mpvd, view=renderView1)

# create a new 'Calculator'
calculator1 = Calculator(registrationName="Calculator1", Input=slice_x9mpvd)

# find source
vlm_rotorpvd = FindSource("vlm_rotor.pvd")

# find source
vpm_rotor_000015xdmf = FindSource("vpm_rotor_000015.xdmf*")

# Properties modified on calculator1
calculator1.Set(
    ResultArrayName="u_zy",
    Function="Velocity_Y*jHat + Velocity_Z*kHat",
)

# show data in view
calculator1Display = Show(calculator1, renderView1, "StructuredGridRepresentation")

# trace defaults for the display properties.
calculator1Display.Representation = "Surface"

# hide data in view
Hide(slice_x9mpvd, renderView1)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# find source
slice_x18mpvd = FindSource("slice_x18m.pvd")

# find source
slice_x27mpvd = FindSource("slice_x27m.pvd")

# update the view to ensure updated data information
renderView1.Update()

# get color transfer function/color map for 'Vorticity'
vorticityLUT = GetColorTransferFunction("Vorticity")

# Rescale transfer function
vorticityLUT.RescaleTransferFunction(0.0046280805924543734, 146.872800859739)

# get opacity transfer function/opacity map for 'Vorticity'
vorticityPWF = GetOpacityTransferFunction("Vorticity")

# Rescale transfer function
vorticityPWF.RescaleTransferFunction(0.0046280805924543734, 146.872800859739)

# set scalar coloring
ColorBy(calculator1Display, ("POINTS", "u_zy", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(velocityLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
calculator1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'u_zy'
u_zyLUT = GetColorTransferFunction("u_zy")

# get opacity transfer function/opacity map for 'u_zy'
u_zyPWF = GetOpacityTransferFunction("u_zy")

# get 2D transfer function for 'u_zy'
u_zyTF2D = GetTransferFunction2D("u_zy")

# set active source
SetActiveSource(vlm_rotorpvd)

# get display properties
vlm_rotorpvdDisplay = GetRepresentation(vlm_rotorpvd, view=renderView1)

ReloadFiles(vlm_rotorpvd)

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

# get 2D transfer function for 'Vorticity'
vorticityTF2D = GetTransferFunction2D("Vorticity")

# get display properties
vpm_rotor_000015xdmfDisplay = GetRepresentation(vpm_rotor_000015xdmf, view=renderView1)

ExtendFileSeries(vpm_rotor_000015xdmf)

# set active source
SetActiveSource(slice_x9mpvd)

ReloadFiles(slice_x9mpvd)

# set active source
SetActiveSource(slice_x18mpvd)

# get display properties
slice_x18mpvdDisplay = GetRepresentation(slice_x18mpvd, view=renderView1)

ReloadFiles(slice_x18mpvd)

# set active source
SetActiveSource(slice_x27mpvd)

# get display properties
slice_x27mpvdDisplay = GetRepresentation(slice_x27mpvd, view=renderView1)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(calculator1)

# set active source
SetActiveSource(slice_x18mpvd)

# create a new 'Calculator'
calculator2 = Calculator(registrationName="Calculator2", Input=slice_x18mpvd)

# show data in view
calculator2Display = Show(calculator2, renderView1, "StructuredGridRepresentation")

# trace defaults for the display properties.
calculator2Display.Representation = "Surface"

# hide data in view
Hide(slice_x18mpvd, renderView1)

# show color bar/color legend
calculator2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(calculator2Display, ("POINTS", "u_zy", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(velocityLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
calculator2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator2Display.SetScalarBarVisibility(renderView1, True)

# set active source
SetActiveSource(slice_x27mpvd)

# create a new 'Calculator'
calculator3 = Calculator(registrationName="Calculator3", Input=slice_x27mpvd)

# show data in view
calculator3Display = Show(calculator3, renderView1, "StructuredGridRepresentation")

# trace defaults for the display properties.
calculator3Display.Representation = "Surface"

# hide data in view
Hide(slice_x27mpvd, renderView1)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(calculator3Display, ("POINTS", "u_zy", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(velocityLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
calculator3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator3Display.SetScalarBarVisibility(renderView1, True)

# get color legend/bar for u_zyLUT in view renderView1
u_zyLUTColorBar = GetScalarBar(u_zyLUT, renderView1)

# change scalar bar placement
u_zyLUTColorBar.Position = [0.05101280488059114, 0.6883532840839017]

# set active source
SetActiveSource(vlm_rotorpvd)

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

# Rescale transfer function
vorticityLUT.RescaleTransferFunction(0.0, 146.872800859739)

# Rescale transfer function
vorticityPWF.RescaleTransferFunction(0.0, 146.872800859739)

# set active source
SetActiveSource(vlm_rotorpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x9mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x18mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x27mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# get layout
layout1 = GetLayout()

# Enter preview mode
layout1.PreviewMode = [1476, 945]

# set active source
SetActiveSource(calculator1)

# Rescale transfer function
u_zyLUT.RescaleTransferFunction(2.2219614797736116e-05, 0.12737713874912346)

# Rescale transfer function
u_zyPWF.RescaleTransferFunction(2.2219614797736116e-05, 0.12737713874912346)

# Properties modified on calculator1
calculator1.Function = "Velocity_Y*jHat / 7 + Velocity_Z*kHat / 7"

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(slice_x27mpvd)

# set active source
SetActiveSource(calculator3)

# Rescale transfer function
u_zyLUT.RescaleTransferFunction(3.174230685390874e-06, 0.018196734107017637)

# Rescale transfer function
u_zyPWF.RescaleTransferFunction(3.174230685390874e-06, 0.018196734107017637)

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(vlm_rotorpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x9mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x18mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x27mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(calculator1)

# Rescale transfer function
u_zyLUT.RescaleTransferFunction(2.5756184087688322e-05, 0.3232298225015842)

# Rescale transfer function
u_zyPWF.RescaleTransferFunction(2.5756184087688322e-05, 0.3232298225015842)

# change scalar bar placement
u_zyLUTColorBar.Set(
    Position=[0.8416632113846562, 0.0],
    ScalarBarLength=0.1999999999999999,
)

# get color legend/bar for vorticityLUT in view renderView1
vorticityLUTColorBar = GetScalarBar(vorticityLUT, renderView1)

# change scalar bar placement
vorticityLUTColorBar.Set(
    Position=[0.8596290804006372, 0.4938896058468543],
    ScalarBarLength=0.2,
)

# change scalar bar placement
u_zyLUTColorBar.Set(
    Position=[0.24139220867462907, 0.6398305084745762],
    ScalarBarLength=0.19999999999999984,
)

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# change scalar bar placement
u_zyLUTColorBar.Position = [0.07472554200796241, 0.7182203389830508]

# change scalar bar placement
u_zyLUTColorBar.Set(
    Position=[0.07472554200796241, 0.7786016949152542],
    ScalarBarLength=0.13961864406779645,
)

# change scalar bar placement
u_zyLUTColorBar.Position = [0.07811307588330116, 0.7595338983050847]

# change scalar bar placement
vorticityLUTColorBar.Set(
    Position=[0.8128811129209624, 0.033084521101091585],
    ScalarBarLength=0.19999999999999987,
)

# change scalar bar placement
vorticityLUTColorBar.ScalarBarLength = 0.14915254237288122

# change scalar bar placement
vorticityLUTColorBar.Position = [0.8149136332461657, 0.033084521101091585]

# get animation scene
animationScene1 = GetAnimationScene()

animationScene1.GoToFirst()

animationScene1.Play()

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

# set active source
SetActiveSource(vlm_rotorpvd)

# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Properties modified on vlm_rotorpvdDisplay
vlm_rotorpvdDisplay.Interpolation = "PBR"

# Properties modified on vlm_rotorpvdDisplay
vlm_rotorpvdDisplay.Metallic = 0.1

# Properties modified on vlm_rotorpvdDisplay
vlm_rotorpvdDisplay.Metallic = 0.2

# Properties modified on vlm_rotorpvdDisplay
vlm_rotorpvdDisplay.Metallic = 0.3

# Properties modified on vlm_rotorpvdDisplay
vlm_rotorpvdDisplay.Roughness = 0.54

# Properties modified on vlm_rotorpvdDisplay
vlm_rotorpvdDisplay.Luminosity = 5.0

# Properties modified on vlm_rotorpvdDisplay
vlm_rotorpvdDisplay.Luminosity = 14.000000000000002

# Properties modified on vlm_rotorpvdDisplay
vlm_rotorpvdDisplay.Luminosity = 17.0

# Properties modified on vlm_rotorpvdDisplay
vlm_rotorpvdDisplay.Metallic = 0.68

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

# create a new 'Calculator'
calculator4 = Calculator(registrationName="Calculator4", Input=vpm_rotor_000015xdmf)

# Properties modified on calculator4
calculator4.Set(
    ResultArrayName="vort_norm ",
    Function="Vorticity / 8.17",
)

# show data in view
calculator4Display = Show(calculator4, renderView1, "UnstructuredGridRepresentation")

# trace defaults for the display properties.
calculator4Display.Representation = "Surface"

# hide data in view
Hide(vpm_rotor_000015xdmf, renderView1)

# show color bar/color legend
calculator4Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
u_zyLUT.RescaleTransferFunction(2.5756184087688322e-05, 0.40222122409135946)

# Rescale transfer function
u_zyPWF.RescaleTransferFunction(2.5756184087688322e-05, 0.40222122409135946)

# get color transfer function/color map for 'Radius'
radiusLUT = GetColorTransferFunction("Radius")

# get opacity transfer function/opacity map for 'Radius'
radiusPWF = GetOpacityTransferFunction("Radius")

# get 2D transfer function for 'Radius'
radiusTF2D = GetTransferFunction2D("Radius")

# Properties modified on calculator4
calculator4.ResultArrayName = "omega_norm "

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(calculator4Display, ("POINTS", "omega_norm ", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(radiusLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
calculator4Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator4Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'omega_norm'
omega_normLUT = GetColorTransferFunction("omega_norm")

# get opacity transfer function/opacity map for 'omega_norm'
omega_normPWF = GetOpacityTransferFunction("omega_norm")

# get 2D transfer function for 'omega_norm'
omega_normTF2D = GetTransferFunction2D("omega_norm")

# change scalar bar placement
u_zyLUTColorBar.Set(
    Position=[0.8586008807613499, 0.03813559322033892],
    ScalarBarLength=0.13961864406779634,
)

# get color legend/bar for omega_normLUT in view renderView1
omega_normLUTColorBar = GetScalarBar(omega_normLUT, renderView1)

# change scalar bar placement
omega_normLUTColorBar.Set(
    Position=[0.047211837361586295, 0.8139196356508706],
    ScalarBarLength=0.10995762711864387,
)

# change scalar bar placement
omega_normLUTColorBar.Position = [0.040436769610908786, 0.8393433644644298]

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(0.0019379428042752714, 9.082573398035624)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(0.0019379428042752714, 9.082573398035624)

animationScene1.GoToFirst()

# set active source
SetActiveSource(vlm_rotorpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x9mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x18mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x27mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

animationScene1.GoToLast()

animationScene1.GoToPrevious()

# set active source
SetActiveSource(calculator4)

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.Title = "$\\mathbs{\\omega} / \\text{TSR} U_\\infty / R$"

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.Title = """\\omega = \\frac{\\text{TSR} \\cdot U_\\infty}{R}
"""

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.Set(
    TitleColor=[1.0, 1.0, 1.0],
    LabelColor=[1.0, 1.0, 1.0],
)

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.ComponentTitle = ""

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.ScalarBarLength = 0.25

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.ScalarBarLength = 0.2

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.WindowLocation = "Upper Left Corner"

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.Title = """$\\omega = \\frac{\\text{TSR} \\cdot U_\\infty}{R}
$"""

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.Title = """$\\bm{\\omega} / \\text{TSR} = U_\\infty / R
$"""

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.Title = "$\\bm{\\omega} R / U_\\infty \\text{TSR}$"

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.Title = "$\\mathbf{\\omega} R / U_\\infty \\text{TSR}$"

# set active source
SetActiveSource(vlm_rotorpvd)

ReloadFiles(vlm_rotorpvd)

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

ExtendFileSeries(vpm_rotor_000015xdmf)

# set active source
SetActiveSource(slice_x9mpvd)

ReloadFiles(slice_x9mpvd)

# set active source
SetActiveSource(slice_x18mpvd)

ReloadFiles(slice_x18mpvd)

# set active source
SetActiveSource(slice_x27mpvd)

ReloadFiles(slice_x27mpvd)

animationScene1.GoToLast()

# set active source
SetActiveSource(calculator4)

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.Title = "$\\mathbf{\\omega} R / U_\\infty \\mathrm{TSR}$"

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.Title = "$\\mathbf{\\omega} R / \\mathrm{TSR} U_\\infty$"

# change scalar bar placement
omega_normLUTColorBar.Position = [0.04103150406504065, 0.7452330508474576]

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(0.00038368095090734225, 9.62679777729387)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(0.00038368095090734225, 9.62679777729387)

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.ScalarBarOutlineColor = [1.0, 1.0, 1.0]

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.ScalarBarLength = 0.15

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.ScalarBarLength = 0.2

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
omega_normLUT.ApplyPreset("Viridis", True)

# change representation type
calculator4Display.SetRepresentationType("Point Gaussian")

# Properties modified on calculator4Display
calculator4Display.Emissive = 1

# Properties modified on calculator4Display
calculator4Display.Emissive = 0

# Properties modified on calculator4Display
calculator4Display.ScaleByArray = 1

# Properties modified on calculator4Display
calculator4Display.SetScaleArray = ["POINTS", "Vorticity"]

# Properties modified on calculator4Display
calculator4Display.SetScaleArray = ["POINTS", "ZoneID"]

# Properties modified on calculator4Display
calculator4Display.SetScaleArray = ["POINTS", "omega_norm "]

# Properties modified on calculator4Display
calculator4Display.SetScaleArray = ["POINTS", "ZoneID"]

# Properties modified on calculator4Display
calculator4Display.SetScaleArray = ["POINTS", "Vorticity"]

# Properties modified on calculator4Display
calculator4Display.SetScaleArray = ["POINTS", "Volume"]

# Properties modified on calculator4Display
calculator4Display.SetScaleArray = ["POINTS", "Vorticity"]

# Properties modified on calculator4Display
calculator4Display.ScaleArrayComponent = "Magnitude"

# Rescale transfer function
calculator4Display.ScaleTransferFunction.RescaleTransferFunction(
    0.0031346733689129862, 94.042281411837
)

# Rescale transfer function
calculator4Display.ScaleTransferFunction.RescaleTransferFunction(0.00313467, 20.0)

# Rescale transfer function
calculator4Display.ScaleTransferFunction.RescaleTransferFunction(0.0, 20.0)

# Rescale transfer function
calculator4Display.ScaleTransferFunction.RescaleTransferFunction(0.0, 40.0)

# Properties modified on calculator4Display
calculator4Display.Emissive = 1

# Rescale transfer function
calculator4Display.ScaleTransferFunction.RescaleTransferFunction(0.0, 30.0)

# Rescale transfer function
calculator4Display.ScaleTransferFunction.RescaleTransferFunction(0.0, 20.0)

# Properties modified on calculator4Display
calculator4Display.OpacityByArray = 1

# Properties modified on calculator4Display
calculator4Display.OpacityArray = ["POINTS", "Vorticity"]

# Properties modified on calculator4Display
calculator4Display.OpacityArrayComponent = "Magnitude"

# Rescale transfer function
calculator4Display.OpacityTransferFunction.RescaleTransferFunction(
    0.0031346733689129862, 94.042281411837
)

# Rescale transfer function
calculator4Display.OpacityTransferFunction.RescaleTransferFunction(0.00313467, 20.0)

# Rescale transfer function
calculator4Display.OpacityTransferFunction.RescaleTransferFunction(0.0, 20.0)

# Properties modified on calculator4Display
calculator4Display.ShaderPreset = "Plain circle"

# Properties modified on calculator4Display
calculator4Display.GaussianRadius = 0.05

# Properties modified on calculator4Display
calculator4Display.ScaleByArray = 0

# Properties modified on calculator4Display
calculator4Display.ScaleByArray = 1

# Properties modified on calculator4Display
calculator4Display.ScaleByArray = 0

# Properties modified on calculator4Display
calculator4Display.ScaleByArray = 1

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

# set active source
SetActiveSource(calculator1)

# Properties modified on u_zyLUTColorBar
u_zyLUTColorBar.Set(
    ScalarBarLength=0.2,
    ScalarBarOutlineColor=[1.0, 1.0, 1.0],
)

# Properties modified on u_zyLUTColorBar
u_zyLUTColorBar.Set(
    TitleColor=[1.0, 1.0, 1.0],
    LabelColor=[1.0, 1.0, 1.0],
)

# Properties modified on u_zyLUTColorBar
u_zyLUTColorBar.Set(
    Title="$\\|u_{yx} \\| / U_\\infty$",
    ComponentTitle="",
)

# Properties modified on u_zyLUTColorBar
u_zyLUTColorBar.WindowLocation = "Lower Right Corner"

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_zyLUT.ApplyPreset("Plasma", True)

# Rescale transfer function
u_zyLUT.RescaleTransferFunction(0.00024809653759605177, 0.326506233851271)

# Rescale transfer function
u_zyPWF.RescaleTransferFunction(0.00024809653759605177, 0.326506233851271)

# set active source
SetActiveSource(calculator4)

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

# Rescale transfer function
vorticityLUT.RescaleTransferFunction(0.0, 80.12404147504044)

# Rescale transfer function
vorticityPWF.RescaleTransferFunction(0.0, 80.12404147504044)

animationScene1.GoToFirst()

animationScene1.Play()

# set active source
SetActiveSource(vlm_rotorpvd)

ReloadFiles(vlm_rotorpvd)

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

ExtendFileSeries(vpm_rotor_000015xdmf)

# set active source
SetActiveSource(slice_x9mpvd)

ReloadFiles(slice_x9mpvd)

# set active source
SetActiveSource(slice_x18mpvd)

ReloadFiles(slice_x18mpvd)

# set active source
SetActiveSource(slice_x27mpvd)

ReloadFiles(slice_x27mpvd)

animationScene1.GoToLast()

animationScene1.GoToFirst()

animationScene1.Play()

# set active source
SetActiveSource(vlm_rotorpvd)

ReloadFiles(vlm_rotorpvd)

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

ExtendFileSeries(vpm_rotor_000015xdmf)

animationScene1.GoToNext()

animationScene1.GoToFirst()

animationScene1.Play()

animationScene1.GoToFirst()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

# Properties modified on renderView1
renderView1.Set(
    CameraPosition=[-9.69291962332138, 12.235945997166828, 26.506865596621],
    CameraFocalPoint=[16.21371270593082, -3.3443951202549878, -4.745856688246357],
    CameraViewUp=[0.24539619200556087, 0.9333764841241101, -0.26189510845620023],
)

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToNext()

animationScene1.GoToFirst()

animationScene1.Play()

# set active source
SetActiveSource(calculator1)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_zyLUT.ApplyPreset("Inferno", True)

# set active source
SetActiveSource(calculator2)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
u_zyLUT.ApplyPreset("Inferno", True)

animationScene1.GoToFirst()

animationScene1.GoToLast()

# Rescale transfer function
u_zyLUT.RescaleTransferFunction(0.0002734577650830213, 0.3265419318825167)

# Rescale transfer function
u_zyPWF.RescaleTransferFunction(0.0002734577650830213, 0.3265419318825167)

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

# Rescale transfer function
vorticityLUT.RescaleTransferFunction(0.0, 80.04451106716921)

# Rescale transfer function
vorticityPWF.RescaleTransferFunction(0.0, 80.04451106716921)

# Properties modified on vpm_rotor_000015xdmfDisplay
vpm_rotor_000015xdmfDisplay.ShaderPreset = "Plain circle"

# set active source
SetActiveSource(calculator4)

# Rescale transfer function
calculator4Display.ScaleTransferFunction.RescaleTransferFunction(0.0, 30.0)

# Rescale transfer function
calculator4Display.OpacityTransferFunction.RescaleTransferFunction(0.0, 30.0)

animationScene1.GoToFirst()

animationScene1.Play()

animationScene1.GoToFirst()

# set active source
SetActiveSource(vlm_rotorpvd)

ReloadFiles(vlm_rotorpvd)

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

ExtendFileSeries(vpm_rotor_000015xdmf)

# set active source
SetActiveSource(slice_x9mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x18mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x27mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

animationScene1.GoToLast()

animationScene1.GoToFirst()

animationScene1.Play()

# set active source
SetActiveSource(calculator4)

# set active source
SetActiveSource(calculator1)

# Rescale transfer function
u_zyLUT.RescaleTransferFunction(0.00011587840518348304, 0.29201964895409915)

# Rescale transfer function
u_zyPWF.RescaleTransferFunction(0.00011587840518348304, 0.29201964895409915)

# Rescale transfer function
u_zyLUT.RescaleTransferFunction(0.0, 0.25)

# Rescale transfer function
u_zyPWF.RescaleTransferFunction(0.0, 0.25)

# Rescale 2D transfer function
u_zyTF2D.RescaleTransferFunction(0.0, 0.25, 0.0, 1.0)

# set active source
SetActiveSource(calculator4)

# Rescale transfer function
omega_normLUT.RescaleTransferFunction(0.0, 10.0)

# Rescale transfer function
omega_normPWF.RescaleTransferFunction(0.0, 10.0)

# Rescale 2D transfer function
omega_normTF2D.RescaleTransferFunction(0.0, 10.0, 0.0, 1.0)

# Properties modified on omega_normLUTColorBar
omega_normLUTColorBar.ScalarBarLength = 0.15

# set active source
SetActiveSource(calculator1)

# Properties modified on u_zyLUTColorBar
u_zyLUTColorBar.ScalarBarLength = 0.15

animationScene1.GoToFirst()

animationScene1.Play()

# set active source
SetActiveSource(vlm_rotorpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x9mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x18mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x27mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

animationScene1.GoToLast()

animationScene1.GoToFirst()

animationScene1.Play()

animationScene1.GoToFirst()

# set active source
SetActiveSource(vlm_rotorpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x9mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x18mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

# set active source
SetActiveSource(slice_x27mpvd)

ReloadFiles(vlm_rotorpvd)

ExtendFileSeries(vpm_rotor_000015xdmf)

ReloadFiles(slice_x9mpvd)

ReloadFiles(slice_x18mpvd)

ReloadFiles(slice_x27mpvd)

animationScene1.GoToLast()

animationScene1.GoToFirst()

animationScene1.Play()

# set active source
SetActiveSource(vlm_rotorpvd)

ReloadFiles(vlm_rotorpvd)

# set active source
SetActiveSource(vpm_rotor_000015xdmf)

ExtendFileSeries(vpm_rotor_000015xdmf)

animationScene1.GoToFirst()

animationScene1.Play()

animationScene1.GoToFirst()

# ================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
# ================================================================

# --------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1476, 944)

# -----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[-9.69291962332138, 12.235945997166828, 26.506865596621],
    CameraFocalPoint=[16.21371270593082, -3.3443951202549878, -4.745856688246357],
    CameraViewUp=[0.24539619200556087, 0.9333764841241101, -0.26189510845620023],
    CameraParallelScale=16.47671463728684,
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
