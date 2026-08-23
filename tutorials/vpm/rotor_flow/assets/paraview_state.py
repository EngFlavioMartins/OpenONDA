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
    CenterOfRotation=[0.12287281081080437, 1.0460753440856934, -0.4254031181335449],
    CameraPosition=[-3.153546492573666, 6.2765405335129065, 26.522279924725375],
    CameraFocalPoint=[11.804663370971234, -0.26886644419887934, 4.136504747954746],
    CameraViewUp=[0.12223626412344764, 0.9716399416374091, -0.20242065000388454],
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

# create a new 'PVD Reader'
surface_rotorpvd = PVDReader(
    registrationName="surface_rotor.pvd",
    FileName="/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/surface_rotor.pvd",
)
surface_rotorpvd.CellArrays = [
    "circulation",
    "pressure_jump_coefficient",
    "panel_centre",
    "bound_vortex_velocity",
]

# create a new 'XDMF Reader'
rotor_000004xdmf = XDMFReader(
    registrationName="rotor_000004.xdmf*",
    FileNames=[
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000004.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000008.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000012.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000016.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000020.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000024.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000028.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000032.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000036.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000040.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000044.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000048.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000052.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000056.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000060.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000064.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000068.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000072.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000076.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000080.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000084.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000088.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000092.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000096.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000100.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000104.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000108.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000112.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000116.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000120.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000124.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000128.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000132.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000136.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000140.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000144.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000148.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000152.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000156.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000160.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000164.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000168.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000172.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000176.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000180.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000184.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000188.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000192.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000196.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000200.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000204.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000208.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000212.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000216.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000220.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000224.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000228.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000232.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000236.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000240.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000244.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000248.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000252.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000256.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000260.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000264.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000268.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000272.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000276.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000280.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000284.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000288.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000292.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000296.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000300.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000304.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000308.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000312.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000316.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000320.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000324.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000328.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000332.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000336.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000340.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000344.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000348.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000352.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000356.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000360.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000364.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000368.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000372.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000376.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000380.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000384.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000388.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000392.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000396.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000400.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000404.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000408.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000412.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000416.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000420.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000424.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000428.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000432.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000436.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000440.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000444.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000448.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000452.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000456.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000460.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000464.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000468.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000472.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000476.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000480.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000484.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000488.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000492.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000496.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000500.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000504.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000508.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000512.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000516.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000520.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000524.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000528.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000532.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000536.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000540.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000544.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000548.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000552.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000556.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000560.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000564.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000568.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000572.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000576.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000580.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000584.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000588.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000592.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000596.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000600.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000604.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000608.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000612.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000616.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000620.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000624.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000628.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000632.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000636.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000640.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000644.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000648.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000652.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000656.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000660.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000664.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000668.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000672.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000676.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000680.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000684.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000688.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000692.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000696.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000700.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000704.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000708.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000712.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000716.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000720.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000724.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000728.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000732.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000736.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000740.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000744.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000748.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000752.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000756.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000760.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000764.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000768.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000772.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000776.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000780.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000784.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000788.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000792.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000796.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000800.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000804.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000808.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000812.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000816.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000820.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000824.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000828.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000832.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000836.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000840.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000844.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000848.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000852.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000856.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000860.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000864.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000868.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000872.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000876.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000880.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000884.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000888.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000892.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000896.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000900.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000904.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000908.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000912.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000916.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000920.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000924.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000928.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000932.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000936.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000940.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000944.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000948.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000952.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000956.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000960.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000964.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000968.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000972.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000976.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000980.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000984.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000988.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000992.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_000996.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001000.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001004.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001008.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001012.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001016.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001020.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001024.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001028.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001032.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001036.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001040.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001044.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001048.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001052.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001056.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001060.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001064.xdmf",
        "/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/rotorVLM/solution/rotor_001068.xdmf",
    ],
)
rotor_000004xdmf.Set(
    PointArrayStatus=[
        "freestream_velocity",
        "vortex_strength",
        "vortex_strength_magnitude",
        "group_id",
        "core_radius",
        "strain_rate",
        "velocity",
        "velocity_gradient",
        "velocity_magnitude",
        "kinematic_viscosity",
        "eddy_viscosity",
        "particle_volume",
        "vorticity",
        "vorticity_magnitude",
    ],
    GridStatus=["vortex_particles"],
)

# create a new 'Calculator'
calculator1 = Calculator(registrationName="Calculator1", Input=rotor_000004xdmf)
calculator1.Set(
    ResultArrayName="vorticity_magnitude",
    Function="vorticity_magnitude * 0.35 / 7",
)

# create a new 'Arrow'
arrow1 = Arrow(registrationName="Arrow1")
arrow1.Set(
    TipResolution=36,
    ShaftResolution=36,
)

# create a new 'Text'
text1 = Text(registrationName="Text1")
text1.Text = "$\\| \\mathbf{u} \\|_{\\infty}$"

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from surface_rotorpvd
surface_rotorpvdDisplay = Show(surface_rotorpvd, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
surface_rotorpvdDisplay.Set(
    Representation="Surface",
    ColorArrayName=[None, ""],
)

# show data from rotor_000004xdmf
rotor_000004xdmfDisplay = Show(rotor_000004xdmf, renderView1, "UnstructuredGridRepresentation")

# get color transfer function/color map for 'CirculationMagnitude'
circulationMagnitudeLUT = GetColorTransferFunction("vortex_strength_magnitude")
circulationMagnitudeLUT.Set(
    RGBPoints=GenerateRGBPoints(
        range_min=0.012105217203497887,
        range_max=0.3085739314556122,
    ),
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
rotor_000004xdmfDisplay.Set(
    Representation="Surface",
    ColorArrayName=["POINTS", "vortex_strength_magnitude"],
    LookupTable=circulationMagnitudeLUT,
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
rotor_000004xdmfDisplay.ScaleTransferFunction.Points = [
    0.012105217203497887,
    0.0,
    0.5,
    0.0,
    0.3085739314556122,
    1.0,
    0.5,
    0.0,
]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
rotor_000004xdmfDisplay.OpacityTransferFunction.Points = [
    0.012105217203497887,
    0.0,
    0.5,
    0.0,
    0.3085739314556122,
    1.0,
    0.5,
    0.0,
]

# show data from calculator1
calculator1Display = Show(calculator1, renderView1, "UnstructuredGridRepresentation")

# get color transfer function/color map for 'vorticity_magnitude'
norm_vorticityLUT = GetColorTransferFunction("vorticity_magnitude")
norm_vorticityLUT.Set(
    RGBPoints=GenerateRGBPoints(
        preset_name="Viridis",
        range_min=7.627697708482636e-15,
        range_max=3.994952774047851,
    ),
    NanColor=[1.0, 0.0, 0.0],
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
calculator1Display.Set(
    Representation="Point Gaussian",
    ColorArrayName=["POINTS", "vorticity_magnitude"],
    LookupTable=norm_vorticityLUT,
    GaussianRadius=0.04,
    ShaderPreset="Gaussian Blur",
    Emissive=1,
    ScaleByArray=1,
    SetScaleArray=["POINTS", "vorticity_magnitude"],
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
calculator1Display.ScaleTransferFunction.Points = [-2.0, 0.0, 0.5, 0.0, 2.0, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
calculator1Display.OpacityTransferFunction.Points = [
    1.4221741460266474e-14,
    0.0,
    0.5,
    0.0,
    7.99874267578125,
    1.0,
    0.5,
    0.0,
]

# show data from arrow1
arrow1Display = Show(arrow1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
arrow1Display.Set(
    Representation="Surface",
    ColorArrayName=[None, ""],
    Translation=[-2.0, 0.0, 0.0],
    Scale=[1.2, 1.2, 1.2],
)

# init the 'Polar Axes Representation' selected for 'PolarAxes'
arrow1Display.PolarAxes.Set(
    Translation=[-2.0, 0.0, 0.0],
    Scale=[1.2, 1.2, 1.2],
)

# show data from text1
text1Display = Show(text1, renderView1, "TextSourceRepresentation")

# trace defaults for the display properties.
text1Display.Set(
    WindowLocation="Any Location",
    Position=[0.01809248554913294, 0.47493150684931507],
    FontFamily="Times",
    FontSize=37,
)

# setup the color legend parameters for each legend in this view

# get color legend/bar for circulationMagnitudeLUT in view renderView1
circulationMagnitudeLUTColorBar = GetScalarBar(circulationMagnitudeLUT, renderView1)
circulationMagnitudeLUTColorBar.Set(
    Title="vortex_strength_magnitude",
    ComponentTitle="",
)

# set color bar visibility
circulationMagnitudeLUTColorBar.Visibility = 0

# get color legend/bar for norm_vorticityLUT in view renderView1
norm_vorticityLUTColorBar = GetScalarBar(norm_vorticityLUT, renderView1)
norm_vorticityLUTColorBar.Set(
    AutoOrient=0,
    Orientation="Horizontal",
    WindowLocation="Any Location",
    Position=[0.8323699421965318, 0.02054794520547945],
    Title="$ \\| \\mathbf{\\angular_velocity}\\|c  / \\| \\mathbf{u} \\|_\\infty $",
    ComponentTitle="",
    TitleFontFamily="Times",
    TitleFontSize=37,
    LabelFontFamily="Times",
    LabelFontSize=37,
    ScalarBarThickness=20,
    ScalarBarLength=0.15000000000000002,
    DrawScalarBarOutline=1,
    DrawTickLabels=0,
    RangeLabelFormat="{:<#6.1f}",
)

# set color bar visibility
norm_vorticityLUTColorBar.Visibility = 1

# hide data in view
Hide(rotor_000004xdmf, renderView1)

# show color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'CirculationMagnitude'
circulationMagnitudePWF = GetOpacityTransferFunction("vortex_strength_magnitude")
circulationMagnitudePWF.Set(
    Points=[0.012105217203497887, 0.0, 0.5, 0.0, 0.3085739314556122, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# get opacity transfer function/opacity map for 'vorticity_magnitude'
norm_vorticityPWF = GetOpacityTransferFunction("vorticity_magnitude")
norm_vorticityPWF.Set(
    Points=[7.627697708482636e-15, 0.0, 0.5, 0.0, 3.994952774047851, 1.0, 0.5, 0.0],
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
    AnimationTime=0.02,
    StartTime=0.02,
    EndTime=5.34,
    PlayMode="Snap To TimeSteps",
)

# initialize the animation scene

# ----------------------------------------------------------------
# restore active source
SetActiveSource(rotor_000004xdmf)
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
