# state file generated using paraview version 6.0.1-1233-gf6d296c8ae
import paraview

paraview.compatibility.major = 6
paraview.compatibility.minor = 0

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
    ViewSize=[1730, 730],
    OrientationAxesVisibility=0,
    CenterOfRotation=[2.1964311599731445e-05, 8.815526962280273e-05, -0.5035135339712724],
    CameraPosition=[1.498740545306873, 0.44875930720859886, 0.5082019491742499],
    CameraFocalPoint=[0.10260539361229373, -0.11228411990951254, -0.8575335735066527],
    CameraViewUp=[-0.22858731418335262, 0.9601587993710814, -0.16075733197812442],
)

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name="Layout #1")
layout1.AssignView(0, renderView1)
layout1.SetSize(1730, 730)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML PolyData Reader'
surfacesvtp = XMLPolyDataReader(
    registrationName="surfaces.vtp",
    FileName=[
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000003.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000006.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000009.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000012.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000015.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000018.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000021.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000024.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000027.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000030.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000033.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000036.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000039.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000042.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000045.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000048.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000051.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000054.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000057.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000060.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000063.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000066.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000069.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000072.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000075.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000078.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000081.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000084.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000087.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000090.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000093.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000096.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000099.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000102.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000105.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000108.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000111.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000114.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000117.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000120.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000123.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000126.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000129.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000132.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000135.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000138.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000141.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000144.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000147.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000150.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000153.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000156.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000159.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000162.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000165.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000168.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000171.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000174.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000177.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000180.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000183.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000186.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000189.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000192.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000195.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000198.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000201.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000204.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000207.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000210.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000213.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000216.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000219.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000222.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000225.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000228.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000231.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000234.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000237.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000240.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000243.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000246.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000249.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000252.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000255.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000258.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000261.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000264.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000267.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000270.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000273.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000276.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000279.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000282.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000285.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000288.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000291.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000294.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000297.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000300.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000303.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000306.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000309.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000312.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000315.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000318.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000321.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000324.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000327.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000330.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000333.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000336.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000339.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000342.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000345.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000348.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000351.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000354.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000357.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000360.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000363.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000366.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000369.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000372.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000375.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000378.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000381.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000384.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000387.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000390.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000393.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000396.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000399.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000402.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000405.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000408.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000411.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000414.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000417.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000420.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000423.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000426.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000429.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000432.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000435.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000438.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000441.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000444.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000447.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000450.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000453.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000456.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000459.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000462.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000465.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000468.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000471.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000474.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000477.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000480.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000483.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000486.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000489.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000492.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000495.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000498.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000501.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000504.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000507.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000510.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000513.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000516.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000519.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000522.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000525.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000528.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000531.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000534.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000537.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000540.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000543.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000546.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000549.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000552.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000555.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000558.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000561.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000564.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000567.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000570.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000573.vtp",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/surface_particle_data_000576.vtp",
    ],
)
surfacesvtp.CellArrayStatus = [
    "Circulation",
    "PressureCoefficient",
    "Position",
    "KinematicVelocity",
]

# create a new 'Text'
text1 = Text(registrationName="Text1")
text1.Text = "$ \\| \\mathbf{u} \\|_{\\infty} $"

# create a new 'PVD Reader'
sampled_zplanepvd = PVDReader(
    registrationName="sampled_zplane.pvd",
    FileName="/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/samples/sampled_zplane.pvd",
)
sampled_zplanepvd.PointArrays = [
    "Velocity",
    "VelocityMagnitude",
    "Vorticity",
    "VorticityMagnitude",
    "PressureGradient",
    "PressureGradientMagnitude",
    "gradP_convective",
    "gradP_viscous",
    "gradP_temporal",
]

# create a new 'Calculator'
calculator1 = Calculator(registrationName="Calculator1", Input=sampled_zplanepvd)
calculator1.Set(
    ResultArrayName="plane_velocity ",
    Function="(Velocity_Y^2 + Velocity_Z^2)^0.5 / 0.8",
)

# create a new 'XDMF Reader'
particlesxdmf = XDMFReader(
    registrationName="particles.xdmf",
    FileNames=[
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000003.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000006.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000009.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000012.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000015.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000018.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000021.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000024.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000027.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000030.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000033.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000036.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000039.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000042.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000045.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000048.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000051.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000054.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000057.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000060.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000063.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000066.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000069.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000072.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000075.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000078.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000081.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000084.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000087.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000090.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000093.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000096.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000099.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000102.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000105.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000108.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000111.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000114.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000117.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000120.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000123.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000126.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000129.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000132.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000135.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000138.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000141.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000144.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000147.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000150.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000153.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000156.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000159.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000162.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000165.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000168.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000171.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000174.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000177.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000180.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000183.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000186.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000189.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000192.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000195.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000198.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000201.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000204.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000207.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000210.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000213.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000216.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000219.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000222.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000225.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000228.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000231.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000234.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000237.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000240.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000243.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000246.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000249.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000252.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000255.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000258.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000261.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000264.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000267.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000270.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000273.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000276.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000279.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000282.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000285.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000288.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000291.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000294.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000297.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000300.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000303.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000306.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000309.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000312.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000315.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000318.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000321.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000324.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000327.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000330.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000333.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000336.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000339.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000342.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000345.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000348.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000351.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000354.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000357.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000360.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000363.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000366.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000369.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000372.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000375.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000378.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000381.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000384.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000387.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000390.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000393.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000396.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000399.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000402.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000405.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000408.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000411.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000414.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000417.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000420.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000423.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000426.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000429.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000432.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000435.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000438.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000441.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000444.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000447.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000450.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000453.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000456.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000459.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000462.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000465.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000468.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000471.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000474.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000477.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000480.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000483.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000486.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000489.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000492.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000495.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000498.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000501.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000504.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000507.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000510.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000513.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000516.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000519.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000522.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000525.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000528.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000531.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000534.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000537.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000540.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000543.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000546.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000549.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000552.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000555.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000558.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000561.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000564.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000567.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000570.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000573.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/VPM/quadCopterVLM/solution/particle_data_000576.xdmf",
    ],
)
particlesxdmf.Set(
    PointArrayStatus=[
        "BackgroundVelocity",
        "Circulation",
        "GroupID",
        "Radius",
        "StrainRate",
        "Velocity",
        "VelocityGradient",
        "Viscosity",
        "ViscosityTurbulent",
        "Volume",
        "Vorticity",
    ],
    GridStatus=["VortexParticles"],
)

# create a new 'Arrow'
arrow1 = Arrow(registrationName="Arrow1")
arrow1.Set(
    TipResolution=36,
    ShaftResolution=36,
)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from particlesxdmf
particlesxdmfDisplay = Show(particlesxdmf, renderView1, "UnstructuredGridRepresentation")

# get color transfer function/color map for 'GroupID'
groupIDLUT = GetColorTransferFunction("GroupID")
groupIDLUT.Set(
    RGBPoints=GenerateRGBPoints(
        preset_name="nic_CubicYF",
        range_min=1.0,
        range_max=4.0,
    ),
    NanColor=[1.0, 1.0, 0.0],
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
particlesxdmfDisplay.Set(
    Representation="Point Gaussian",
    ColorArrayName=["POINTS", "GroupID"],
    LookupTable=groupIDLUT,
    GaussianRadius=0.0025,
    ShaderPreset="Gaussian Blur",
    Emissive=1,
    ScaleByArray=1,
    SetScaleArray=["POINTS", "Vorticity"],
    ScaleArrayComponent="Magnitude",
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
particlesxdmfDisplay.ScaleTransferFunction.Points = [-100.0, 0.0, 0.5, 0.0, 400.0, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
particlesxdmfDisplay.OpacityTransferFunction.Points = [
    0.012015827931463718,
    0.0,
    0.5,
    0.0,
    0.7980822920799255,
    1.0,
    0.5,
    0.0,
]

# show data from surfacesvtp
surfacesvtpDisplay = Show(surfacesvtp, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
surfacesvtpDisplay.Set(
    Representation="Surface",
    ColorArrayName=[None, ""],
)

# show data from calculator1
calculator1Display = Show(calculator1, renderView1, "StructuredGridRepresentation")

# get color transfer function/color map for 'plane_velocity'
plane_velocityLUT = GetColorTransferFunction("plane_velocity")
plane_velocityLUT.Set(
    RGBPoints=GenerateRGBPoints(
        preset_name="erdc_blue2green_BW",
        range_min=1.0,
        range_max=1.5,
    ),
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
calculator1Display.Set(
    Representation="Surface",
    ColorArrayName=["POINTS", "plane_velocity "],
    LookupTable=plane_velocityLUT,
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
calculator1Display.ScaleTransferFunction.Points = [
    0.8219946447099962,
    0.0,
    0.5,
    0.0,
    0.9525385322775234,
    1.0,
    0.5,
    0.0,
]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
calculator1Display.OpacityTransferFunction.Points = [
    0.8219946447099962,
    0.0,
    0.5,
    0.0,
    0.9525385322775234,
    1.0,
    0.5,
    0.0,
]

# show data from text1
text1Display = Show(text1, renderView1, "TextSourceRepresentation")

# trace defaults for the display properties.
text1Display.Set(
    WindowLocation="Any Location",
    Position=[0.001907514450867051, 0.47082191780821925],
    FontFamily="Times",
    FontSize=37,
)

# show data from arrow1
arrow1Display = Show(arrow1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow1Display.Set(
    Representation="Surface",
    ColorArrayName=[None, ""],
    Translation=[-0.0, 0.0, 0.2],
    Scale=[0.08, 0.08, 0.08],
    Orientation=[0.0, 90.0, 0.0],
)

# init the 'Polar Axes Representation' selected for 'PolarAxes'
arrow1Display.PolarAxes.Set(
    Translation=[-0.0, 0.0, 0.2],
    Scale=[0.08, 0.08, 0.08],
    Orientation=[0.0, 90.0, 0.0],
)

# setup the color legend parameters for each legend in this view

# get color legend/bar for plane_velocityLUT in view renderView1
plane_velocityLUTColorBar = GetScalarBar(plane_velocityLUT, renderView1)
plane_velocityLUTColorBar.Set(
    AutoOrient=0,
    Title="$ \\|\\mathbf{u}\\|_{xy} / \\|\\mathbf{u}\\|_\\infty$",
    ComponentTitle="",
    HorizontalTitle=1,
    TitleFontFamily="Times",
    TitleFontSize=37,
    LabelFontFamily="Times",
    LabelFontSize=37,
    ScalarBarThickness=20,
    ScalarBarLength=0.25,
    DrawScalarBarOutline=1,
    DrawTickLabels=0,
    RangeLabelFormat="{:<#6.1f}",
)

# set color bar visibility
plane_velocityLUTColorBar.Visibility = 1

# show color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'plane_velocity'
plane_velocityPWF = GetOpacityTransferFunction("plane_velocity")
plane_velocityPWF.Set(
    Points=[1.0, 0.0, 0.5, 0.0, 1.5, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# get opacity transfer function/opacity map for 'GroupID'
groupIDPWF = GetOpacityTransferFunction("GroupID")
groupIDPWF.Set(
    Points=[1.0, 0.0, 0.5, 0.0, 4.0, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.Set(
    ViewModules=renderView1,
    Cues=timeAnimationCue1,
    AnimationTime=0.0125,
    StartTime=0.0125,
    EndTime=2.4,
    PlayMode="Snap To TimeSteps",
)

# initialize the animation scene

# ----------------------------------------------------------------
# restore active source
SetActiveSource(text1)
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
