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
    ViewSize=[1476, 945],
    OrientationAxesVisibility=0,
    CenterOfRotation=[5.231250152587891, 0.0, 0.0],
    UseToneMapping=1,
    UseAmbientOcclusion=1,
    CameraPosition=[35.17181585995478, -28.61535302281981, 22.62267563339554],
    CameraFocalPoint=[9.956924515612199, -0.8551330724730631, 0.5457414743554604],
    CameraViewUp=[-0.3513224633183323, 0.36616274340196586, 0.8616828721242434],
    CameraViewAngle=13.971051406860353,
    UseColorPaletteForBackground=0,
    Background=[1.0, 1.0, 1.0],
)

# init the 'Grid Axes 3D Actor' selected for 'AxesGrid'
renderView1.AxesGrid.Set(
    XTitleBold=0,
    XTitleFontSize=12,
    YTitleBold=0,
    YTitleFontSize=12,
    ZTitleBold=0,
    ZTitleFontSize=12,
    XLabelBold=0,
    XLabelFontSize=12,
    YLabelBold=0,
    YLabelFontSize=12,
    ZLabelBold=0,
    ZLabelFontSize=12,
)

# init the 'Polar Grid Actor' selected for 'PolarGrid'
renderView1.PolarGrid.Set(
    AxesLabelBold=0,
    AxesLabelFontSize=12,
)

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1476, 945)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the selections
# ----------------------------------------------------------------

# create a new 'ID Selection Source'
selectionSource0 = CreateSelection(proxyname='IDSelectionSource', registrationname='SelectionSource0', groupname='selections', IDs=[0, 2655])

# create a new 'Append Selections'
appendSelections = CreateSelection(proxyname='AppendSelections', registrationname='AppendSelections', groupname='selections', Input=selectionSource0,
    Expression='s0',
    SelectionNames=['s0'])

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVD Reader'
fvmpvd = PVDReader(registrationName='fvm.pvd', FileName='/home/flavio-martins/Projects/OpenONDA/tutorials/coupled_fvm_vpm/02_cube_flow/solution/fvm.pvd')
fvmpvd.CellArrays = ['velocity', 'kinematic_pressure', 'courant_number', 'vorticity', 'cell_size', 'refinement_level', 'boundary_layer_index', 'cell_volume', 'cell_equivalent_size', 'global_cell_id']

# create a new 'PVD Reader'
vpmpvd = PVDReader(registrationName='vpm.pvd', FileName='/home/flavio-martins/Projects/OpenONDA/tutorials/coupled_fvm_vpm/02_cube_flow/solution/vpm.pvd')
vpmpvd.PointArrays = ['velocity', 'vortex_strength', 'vorticity', 'core_radius', 'particle_volume', 'kinematic_viscosity', 'eddy_viscosity', 'effective_viscosity', 'group_id', 'zone_id']

# create a new 'STL Reader'
cubestl = STLReader(registrationName='cube.stl', FileNames=['/home/flavio-martins/Projects/OpenONDA/tutorials/coupled_fvm_vpm/02_cube_flow/assets/cube.stl'])

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=fvmpvd)

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=vpmpvd)
clip1.Set(
    ClipType='Box',
    Invert=0,
)

# init the 'Box' selected for 'ClipType'
clip1.ClipType.Set(
    Position=[-1.0, -2.5, -0.05],
    Length=[6.5, 2.3850001096725464, 1.5],
)

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [5.647500157356262, -5.960464477539063e-08, -5.960464477539063e-08]

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=cellDatatoPointData1)
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Normal = [0.0, 0.0, 1.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [1.125, 0.0, 0.0]

# create a new 'Glyph'
glyph1 = Glyph(registrationName='Glyph1', Input=slice1,
    GlyphType='Arrow')
glyph1.Set(
    OrientationArray=['POINTS', 'velocity'],
    ScaleArray=['POINTS', 'No scale array'],
    ScaleFactor=0.6885,
    MaximumNumberOfSamplePoints=500,
    Seed=100,
)

appendSelections.SetSelectionId(slice1.GetGlobalID())
appendSelections.SetSelectionPort(0)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from fvmpvd
fvmpvdDisplay = Show(fvmpvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
fvmpvdDisplay.Set(
    Representation='Outline',
    ColorArrayName=['POINTS', ''],
    LineWidth=2.0,
    RenderLinesAsTubes=1,
)

# init the 'Grid Axes Representation' selected for 'DataAxesGrid'
fvmpvdDisplay.DataAxesGrid.Set(
    XTitleBold=0,
    XTitleFontSize=12,
    YTitleBold=0,
    YTitleFontSize=12,
    ZTitleBold=0,
    ZTitleFontSize=12,
    XLabelBold=0,
    XLabelFontSize=12,
    YLabelBold=0,
    YLabelFontSize=12,
    ZLabelBold=0,
    ZLabelFontSize=12,
)

# init the 'Polar Axes Representation' selected for 'PolarAxes'
fvmpvdDisplay.PolarAxes.Set(
    PolarAxisTitleBold=0,
    PolarAxisTitleFontSize=12,
    PolarAxisLabelBold=0,
    PolarAxisLabelFontSize=12,
    LastRadialAxisTextBold=0,
    LastRadialAxisTextFontSize=12,
    SecondaryRadialAxesTextBold=0,
    SecondaryRadialAxesTextFontSize=12,
)

# show data from slice1
slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'vorticity'
vorticityLUT = GetColorTransferFunction('vorticity')
vorticityLUT.Set(
    AutomaticRescaleRangeMode='Never',
    RGBPoints=[
        # scalar, red, green, blue
        0.0, 0.267004, 0.004874, 0.329415,
        0.019610000000000002, 0.26851, 0.009605, 0.335427,
        0.039215, 0.269944, 0.014625, 0.341379,
        0.058824999999999995, 0.271305, 0.019942, 0.347269,
        0.07843, 0.272594, 0.025563, 0.353093,
        0.09804, 0.273809, 0.031497, 0.358853,
        0.117645, 0.274952, 0.037752, 0.364543,
        0.137255, 0.276022, 0.044167, 0.370164,
        0.15686499999999998, 0.277018, 0.050344, 0.375715,
        0.17647, 0.277941, 0.056324, 0.381191,
        0.19608, 0.278791, 0.062145, 0.386592,
        0.21568500000000002, 0.279566, 0.067836, 0.391917,
        0.23529499999999998, 0.280267, 0.073417, 0.397163,
        0.2549, 0.280894, 0.078907, 0.402329,
        0.27451, 0.281446, 0.08432, 0.407414,
        0.29412, 0.281924, 0.089666, 0.412415,
        0.313725, 0.282327, 0.094955, 0.417331,
        0.33333500000000005, 0.282656, 0.100196, 0.42216,
        0.35294, 0.28291, 0.105393, 0.426902,
        0.37255000000000005, 0.283091, 0.110553, 0.431554,
        0.39215500000000003, 0.283197, 0.11568, 0.436115,
        0.411765, 0.283229, 0.120777, 0.440584,
        0.431375, 0.283187, 0.125848, 0.44496,
        0.45098, 0.283072, 0.130895, 0.449241,
        0.47058999999999995, 0.282884, 0.13592, 0.453427,
        0.490195, 0.282623, 0.140926, 0.457517,
        0.509805, 0.28229, 0.145912, 0.46151,
        0.52941, 0.281887, 0.150881, 0.465405,
        0.54902, 0.281412, 0.155834, 0.469201,
        0.568625, 0.280868, 0.160771, 0.472899,
        0.5882350000000001, 0.280255, 0.165693, 0.476498,
        0.607845, 0.279574, 0.170599, 0.479997,
        0.62745, 0.278826, 0.17549, 0.483397,
        0.64706, 0.278012, 0.180367, 0.486697,
        0.6666650000000001, 0.277134, 0.185228, 0.489898,
        0.686275, 0.276194, 0.190074, 0.493001,
        0.70588, 0.275191, 0.194905, 0.496005,
        0.72549, 0.274128, 0.199721, 0.498911,
        0.7451000000000001, 0.273006, 0.20452, 0.501721,
        0.764705, 0.271828, 0.209303, 0.504434,
        0.784315, 0.270595, 0.214069, 0.507052,
        0.8039200000000001, 0.269308, 0.218818, 0.509577,
        0.82353, 0.267968, 0.223549, 0.512008,
        0.843135, 0.26658, 0.228262, 0.514349,
        0.8627450000000001, 0.265145, 0.232956, 0.516599,
        0.882355, 0.263663, 0.237631, 0.518762,
        0.90196, 0.262138, 0.242286, 0.520837,
        0.92157, 0.260571, 0.246922, 0.522828,
        0.9411750000000001, 0.258965, 0.251537, 0.524736,
        0.960785, 0.257322, 0.25613, 0.526563,
        0.98039, 0.255645, 0.260703, 0.528312,
        1.0, 0.253935, 0.265254, 0.529983,
        1.01961, 0.252194, 0.269783, 0.531579,
        1.039215, 0.250425, 0.27429, 0.533103,
        1.0588250000000001, 0.248629, 0.278775, 0.534556,
        1.07843, 0.246811, 0.283237, 0.535941,
        1.09804, 0.244972, 0.287675, 0.53726,
        1.117645, 0.243113, 0.292092, 0.538516,
        1.137255, 0.241237, 0.296485, 0.539709,
        1.156865, 0.239346, 0.300855, 0.540844,
        1.1764700000000001, 0.237441, 0.305202, 0.541921,
        1.19608, 0.235526, 0.309527, 0.542944,
        1.215685, 0.233603, 0.313828, 0.543914,
        1.235295, 0.231674, 0.318106, 0.544834,
        1.2549, 0.229739, 0.322361, 0.545706,
        1.27451, 0.227802, 0.326594, 0.546532,
        1.29412, 0.225863, 0.330805, 0.547314,
        1.313725, 0.223925, 0.334994, 0.548053,
        1.333335, 0.221989, 0.339161, 0.548752,
        1.35294, 0.220057, 0.343307, 0.549413,
        1.37255, 0.21813, 0.347432, 0.550038,
        1.3921549999999998, 0.21621, 0.351535, 0.550627,
        1.4117650000000002, 0.214298, 0.355619, 0.551184,
        1.431375, 0.212395, 0.359683, 0.55171,
        1.45098, 0.210503, 0.363727, 0.552206,
        1.47059, 0.208623, 0.367752, 0.552675,
        1.490195, 0.206756, 0.371758, 0.553117,
        1.5098049999999998, 0.204903, 0.375746, 0.553533,
        1.52941, 0.203063, 0.379716, 0.553925,
        1.54902, 0.201239, 0.38367, 0.554294,
        1.568625, 0.19943, 0.387607, 0.554642,
        1.588235, 0.197636, 0.391528, 0.554969,
        1.607845, 0.19586, 0.395433, 0.555276,
        1.62745, 0.1941, 0.399323, 0.555565,
        1.64706, 0.192357, 0.403199, 0.555836,
        1.666665, 0.190631, 0.407061, 0.556089,
        1.6862750000000002, 0.188923, 0.41091, 0.556326,
        1.7058799999999998, 0.187231, 0.414746, 0.556547,
        1.7254900000000002, 0.185556, 0.41857, 0.556753,
        1.7450999999999999, 0.183898, 0.422383, 0.556944,
        1.764705, 0.182256, 0.426184, 0.55712,
        1.7843149999999999, 0.180629, 0.429975, 0.557282,
        1.80392, 0.179019, 0.433756, 0.55743,
        1.8235299999999999, 0.177423, 0.437527, 0.557565,
        1.843135, 0.175841, 0.44129, 0.557685,
        1.862745, 0.174274, 0.445044, 0.557792,
        1.882355, 0.172719, 0.448791, 0.557885,
        1.90196, 0.171176, 0.45253, 0.557965,
        1.92157, 0.169646, 0.456262, 0.55803,
        1.9411749999999999, 0.168126, 0.459988, 0.558082,
        1.960785, 0.166617, 0.463708, 0.558119,
        1.9803899999999999, 0.165117, 0.467423, 0.558141,
        2.0, 0.163625, 0.471133, 0.558148,
        2.01961, 0.162142, 0.474838, 0.55814,
        2.039215, 0.160665, 0.47854, 0.558115,
        2.058825, 0.159194, 0.482237, 0.558073,
        2.07843, 0.157729, 0.485932, 0.558013,
        2.09804, 0.15627, 0.489624, 0.557936,
        2.117645, 0.154815, 0.493313, 0.55784,
        2.137255, 0.153364, 0.497, 0.557724,
        2.156865, 0.151918, 0.500685, 0.557587,
        2.17647, 0.150476, 0.504369, 0.55743,
        2.19608, 0.149039, 0.508051, 0.55725,
        2.215685, 0.147607, 0.511733, 0.557049,
        2.235295, 0.14618, 0.515413, 0.556823,
        2.2549, 0.144759, 0.519093, 0.556572,
        2.27451, 0.143343, 0.522773, 0.556295,
        2.29412, 0.141935, 0.526453, 0.555991,
        2.3137250000000003, 0.140536, 0.530132, 0.555659,
        2.333335, 0.139147, 0.533812, 0.555298,
        2.3529400000000003, 0.13777, 0.537492, 0.554906,
        2.37255, 0.136408, 0.541173, 0.554483,
        2.392155, 0.135066, 0.544853, 0.554029,
        2.411765, 0.133743, 0.548535, 0.553541,
        2.431375, 0.132444, 0.552216, 0.553018,
        2.45098, 0.131172, 0.555899, 0.552459,
        2.47059, 0.129933, 0.559582, 0.551864,
        2.490195, 0.128729, 0.563265, 0.551229,
        2.509805, 0.127568, 0.566949, 0.550556,
        2.5294100000000004, 0.126453, 0.570633, 0.549841,
        2.54902, 0.125394, 0.574318, 0.549086,
        2.568625, 0.124395, 0.578002, 0.548287,
        2.588235, 0.123463, 0.581687, 0.547445,
        2.6078449999999997, 0.122606, 0.585371, 0.546557,
        2.62745, 0.121831, 0.589055, 0.545623,
        2.6470599999999997, 0.121148, 0.592739, 0.544641,
        2.6666649999999996, 0.120565, 0.596422, 0.543611,
        2.686275, 0.120092, 0.600104, 0.54253,
        2.70588, 0.119738, 0.603785, 0.5414,
        2.7254899999999997, 0.119512, 0.607464, 0.540218,
        2.7451, 0.119423, 0.611141, 0.538982,
        2.764705, 0.119483, 0.614817, 0.537692,
        2.784315, 0.119699, 0.61849, 0.536347,
        2.8039199999999997, 0.120081, 0.622161, 0.534946,
        2.8235300000000003, 0.120638, 0.625828, 0.533488,
        2.843135, 0.12138, 0.629492, 0.531973,
        2.862745, 0.122312, 0.633153, 0.530398,
        2.8823549999999996, 0.123444, 0.636809, 0.528763,
        2.90196, 0.12478, 0.640461, 0.527068,
        2.92157, 0.126326, 0.644107, 0.525311,
        2.941175, 0.128087, 0.647749, 0.523491,
        2.9607850000000004, 0.130067, 0.651384, 0.521608,
        2.98039, 0.132268, 0.655014, 0.519661,
        3.0, 0.134692, 0.658636, 0.517649,
        3.0196099999999997, 0.137339, 0.662252, 0.515571,
        3.039215, 0.14021, 0.665859, 0.513427,
        3.058825, 0.143303, 0.669459, 0.511215,
        3.07843, 0.146616, 0.67305, 0.508936,
        3.09804, 0.150148, 0.676631, 0.506589,
        3.117645, 0.153894, 0.680203, 0.504172,
        3.1372549999999997, 0.157851, 0.683765, 0.501686,
        3.156865, 0.162016, 0.687316, 0.499129,
        3.17647, 0.166383, 0.690856, 0.496502,
        3.1960800000000003, 0.170948, 0.694384, 0.493803,
        3.2156849999999997, 0.175707, 0.6979, 0.491033,
        3.2352950000000003, 0.180653, 0.701402, 0.488189,
        3.2549, 0.185783, 0.704891, 0.485273,
        3.27451, 0.19109, 0.708366, 0.482284,
        3.29412, 0.196571, 0.711827, 0.479221,
        3.3137250000000003, 0.202219, 0.715272, 0.476084,
        3.333335, 0.20803, 0.718701, 0.472873,
        3.35294, 0.214, 0.722114, 0.469588,
        3.3725500000000004, 0.220124, 0.725509, 0.466226,
        3.392155, 0.226397, 0.728888, 0.462789,
        3.411765, 0.232815, 0.732247, 0.459277,
        3.431375, 0.239374, 0.735588, 0.455688,
        3.4509800000000004, 0.24607, 0.73891, 0.452024,
        3.47059, 0.252899, 0.742211, 0.448284,
        3.490195, 0.259857, 0.745492, 0.444467,
        3.5098049999999996, 0.266941, 0.748751, 0.440573,
        3.52941, 0.274149, 0.751988, 0.436601,
        3.54902, 0.281477, 0.755203, 0.432552,
        3.5686250000000004, 0.288921, 0.758394, 0.428426,
        3.588235, 0.296479, 0.761561, 0.424223,
        3.607845, 0.304148, 0.764704, 0.419943,
        3.6274499999999996, 0.311925, 0.767822, 0.415586,
        3.6470599999999997, 0.319809, 0.770914, 0.411152,
        3.666665, 0.327796, 0.77398, 0.40664,
        3.686275, 0.335885, 0.777018, 0.402049,
        3.7058799999999996, 0.344074, 0.780029, 0.397381,
        3.72549, 0.35236, 0.783011, 0.392636,
        3.7451, 0.360741, 0.785964, 0.387814,
        3.7647049999999997, 0.369214, 0.788888, 0.382914,
        3.784315, 0.377779, 0.791781, 0.377939,
        3.80392, 0.386433, 0.794644, 0.372886,
        3.82353, 0.395174, 0.797475, 0.367757,
        3.8431349999999997, 0.404001, 0.800275, 0.362552,
        3.8627450000000003, 0.412913, 0.803041, 0.357269,
        3.882355, 0.421908, 0.805774, 0.35191,
        3.90196, 0.430983, 0.808473, 0.346476,
        3.92157, 0.440137, 0.811138, 0.340967,
        3.9411750000000003, 0.449368, 0.813768, 0.335384,
        3.960785, 0.458674, 0.816363, 0.329727,
        3.98039, 0.468053, 0.818921, 0.323998,
        4.0, 0.477504, 0.821444, 0.318195,
        4.01961, 0.487026, 0.823929, 0.312321,
        4.0392149999999996, 0.496615, 0.826376, 0.306377,
        4.058825, 0.506271, 0.828786, 0.300362,
        4.07843, 0.515992, 0.831158, 0.294279,
        4.09804, 0.525776, 0.833491, 0.288127,
        4.1176449999999996, 0.535621, 0.835785, 0.281908,
        4.137255000000001, 0.545524, 0.838039, 0.275626,
        4.156865, 0.555484, 0.840254, 0.269281,
        4.17647, 0.565498, 0.84243, 0.262877,
        4.19608, 0.575563, 0.844566, 0.256415,
        4.215685000000001, 0.585678, 0.846661, 0.249897,
        4.235295, 0.595839, 0.848717, 0.243329,
        4.2549, 0.606045, 0.850733, 0.236712,
        4.27451, 0.616293, 0.852709, 0.230052,
        4.29412, 0.626579, 0.854645, 0.223353,
        4.313725, 0.636902, 0.856542, 0.21662,
        4.333335, 0.647257, 0.8584, 0.209861,
        4.35294, 0.657642, 0.860219, 0.203082,
        4.37255, 0.668054, 0.861999, 0.196293,
        4.392155, 0.678489, 0.863742, 0.189503,
        4.411765, 0.688944, 0.865448, 0.182725,
        4.431375, 0.699415, 0.867117, 0.175971,
        4.4509799999999995, 0.709898, 0.868751, 0.169257,
        4.47059, 0.720391, 0.87035, 0.162603,
        4.490195, 0.730889, 0.871916, 0.156029,
        4.509805, 0.741388, 0.873449, 0.149561,
        4.5294099999999995, 0.751884, 0.874951, 0.143228,
        4.54902, 0.762373, 0.876424, 0.137064,
        4.568625, 0.772852, 0.877868, 0.131109,
        4.588235, 0.783315, 0.879285, 0.125405,
        4.607845, 0.79376, 0.880678, 0.120005,
        4.6274500000000005, 0.804182, 0.882046, 0.114965,
        4.64706, 0.814576, 0.883393, 0.110347,
        4.666665, 0.82494, 0.88472, 0.106217,
        4.686275, 0.83527, 0.886029, 0.102646,
        4.7058800000000005, 0.845561, 0.887322, 0.099702,
        4.72549, 0.85581, 0.888601, 0.097452,
        4.7451, 0.866013, 0.889868, 0.095953,
        4.764705, 0.876168, 0.891125, 0.09525,
        4.784315, 0.886271, 0.892374, 0.095374,
        4.80392, 0.89632, 0.893616, 0.096335,
        4.82353, 0.906311, 0.894855, 0.098125,
        4.843135, 0.916242, 0.896091, 0.100717,
        4.862745, 0.926106, 0.89733, 0.104071,
        4.882355, 0.935904, 0.89857, 0.108131,
        4.90196, 0.945636, 0.899815, 0.112838,
        4.92157, 0.9553, 0.901065, 0.118128,
        4.941174999999999, 0.964894, 0.902323, 0.123941,
        4.960785, 0.974417, 0.90359, 0.130215,
        4.98039, 0.983868, 0.904867, 0.136897,
        5.0, 0.993248, 0.906157, 0.143936,
    ],
    NanColor=[1.0, 0.0, 0.0],
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
slice1Display.Set(
    Representation='Surface',
    ColorArrayName=['POINTS', 'vorticity'],
    LookupTable=vorticityLUT,
    EdgeColor=[0.9176470637321472, 0.9176470637321472, 0.9176470637321472],
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
slice1Display.ScaleTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
slice1Display.OpacityTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# init the 'Grid Axes Representation' selected for 'DataAxesGrid'
slice1Display.DataAxesGrid.Set(
    XTitleBold=0,
    XTitleFontSize=12,
    YTitleBold=0,
    YTitleFontSize=12,
    ZTitleBold=0,
    ZTitleFontSize=12,
    XLabelBold=0,
    XLabelFontSize=12,
    YLabelBold=0,
    YLabelFontSize=12,
    ZLabelBold=0,
    ZLabelFontSize=12,
)

# init the 'Polar Axes Representation' selected for 'PolarAxes'
slice1Display.PolarAxes.Set(
    PolarAxisTitleBold=0,
    PolarAxisTitleFontSize=12,
    PolarAxisLabelBold=0,
    PolarAxisLabelFontSize=12,
    LastRadialAxisTextBold=0,
    LastRadialAxisTextFontSize=12,
    SecondaryRadialAxesTextBold=0,
    SecondaryRadialAxesTextFontSize=12,
)

# show data from cubestl
cubestlDisplay = Show(cubestl, renderView1, 'GeometryRepresentation')

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
cubestlDisplay.Set(
    Representation='Surface',
    ColorArrayName=['POINTS', ''],
    LookupTable=sTLSolidLabelingLUT,
    Interpolation='PBR',
    Luminosity=27.0,
    Roughness=0.42,
    Metallic=0.04,
)

# init the 'Grid Axes Representation' selected for 'DataAxesGrid'
cubestlDisplay.DataAxesGrid.Set(
    XTitleBold=0,
    XTitleFontSize=12,
    YTitleBold=0,
    YTitleFontSize=12,
    ZTitleBold=0,
    ZTitleFontSize=12,
    XLabelBold=0,
    XLabelFontSize=12,
    YLabelBold=0,
    YLabelFontSize=12,
    ZLabelBold=0,
    ZLabelFontSize=12,
)

# init the 'Polar Axes Representation' selected for 'PolarAxes'
cubestlDisplay.PolarAxes.Set(
    PolarAxisTitleBold=0,
    PolarAxisTitleFontSize=12,
    PolarAxisLabelBold=0,
    PolarAxisLabelFontSize=12,
    LastRadialAxisTextBold=0,
    LastRadialAxisTextFontSize=12,
    SecondaryRadialAxesTextBold=0,
    SecondaryRadialAxesTextFontSize=12,
)

# show data from clip1
clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip1Display.Set(
    Representation='Point Gaussian',
    ColorArrayName=['POINTS', 'vorticity'],
    LookupTable=vorticityLUT,
    InterpolateScalarsBeforeMapping=0,
    PointSize=3.0,
    DisableLighting=1,
    BackfaceRepresentation='Points',
    Pickable=0,
    OSPRayUseScaleArray='All Exact',
    GaussianRadius=0.02,
    ShaderPreset='Plain circle',
    OpacityByArray=1,
    OpacityArray=['POINTS', 'vorticity'],
    OpacityArrayComponent='Magnitude',
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
clip1Display.ScaleTransferFunction.Points = [0.000133218, 0.0, 0.5, 0.0, 5.0, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
clip1Display.OpacityTransferFunction.Points = [0.000133218, 0.0, 0.5, 0.0, 5.0, 1.0, 0.5, 0.0]

# init the 'Grid Axes Representation' selected for 'DataAxesGrid'
clip1Display.DataAxesGrid.Set(
    XTitleBold=0,
    XTitleFontSize=12,
    YTitleBold=0,
    YTitleFontSize=12,
    ZTitleBold=0,
    ZTitleFontSize=12,
    XLabelBold=0,
    XLabelFontSize=12,
    YLabelBold=0,
    YLabelFontSize=12,
    ZLabelBold=0,
    ZLabelFontSize=12,
)

# init the 'Polar Axes Representation' selected for 'PolarAxes'
clip1Display.PolarAxes.Set(
    PolarAxisTitleBold=0,
    PolarAxisTitleFontSize=12,
    PolarAxisLabelBold=0,
    PolarAxisLabelFontSize=12,
    LastRadialAxisTextBold=0,
    LastRadialAxisTextFontSize=12,
    SecondaryRadialAxesTextBold=0,
    SecondaryRadialAxesTextFontSize=12,
)

# show data from vpmpvd
vpmpvdDisplay = Show(vpmpvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
vpmpvdDisplay.Set(
    Representation='Point Gaussian',
    ColorArrayName=['POINTS', ''],
    GaussianRadius=0.1269000029563904,
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
vpmpvdDisplay.ScaleTransferFunction.Points = [0.04724999889731407, 0.0, 0.5, 0.0, 0.04725762829184532, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
vpmpvdDisplay.OpacityTransferFunction.Points = [0.04724999889731407, 0.0, 0.5, 0.0, 0.04725762829184532, 1.0, 0.5, 0.0]

# init the 'Grid Axes Representation' selected for 'DataAxesGrid'
vpmpvdDisplay.DataAxesGrid.Set(
    XTitleBold=0,
    XTitleFontSize=12,
    YTitleBold=0,
    YTitleFontSize=12,
    ZTitleBold=0,
    ZTitleFontSize=12,
    XLabelBold=0,
    XLabelFontSize=12,
    YLabelBold=0,
    YLabelFontSize=12,
    ZLabelBold=0,
    ZLabelFontSize=12,
)

# init the 'Polar Axes Representation' selected for 'PolarAxes'
vpmpvdDisplay.PolarAxes.Set(
    PolarAxisTitleBold=0,
    PolarAxisTitleFontSize=12,
    PolarAxisLabelBold=0,
    PolarAxisLabelFontSize=12,
    LastRadialAxisTextBold=0,
    LastRadialAxisTextFontSize=12,
    SecondaryRadialAxesTextBold=0,
    SecondaryRadialAxesTextFontSize=12,
)

# show data from glyph1
glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'velocity'
velocityLUT = GetColorTransferFunction('velocity')
velocityLUT.Set(
    RGBPoints=[
        # scalar, red, green, blue
        0.0016831935004858933, 0.0, 0.0, 0.0,
        0.0800459967041357, 0.0, 0.062745, 0.0,
        0.1584087999077855, 0.0, 0.12549, 0.0,
        0.2367722275659991, 0.0, 0.188235, 0.0,
        0.31513503076964894, 0.0, 0.25098, 0.0,
        0.39349783397329874, 0.0, 0.313725, 0.0,
        0.4718606371769486, 0.0, 0.376471, 0.0,
        0.5502234403805984, 0.094118, 0.439216, 0.0,
        0.6285865995233495, 0.196078, 0.501961, 0.0,
        0.7069496712424619, 0.294118, 0.564706, 0.0,
        0.7853124744461117, 0.396078, 0.627451, 0.0,
        0.8636752776497615, 0.498039, 0.690196, 0.0,
        0.9420380808534113, 0.6, 0.752941, 0.145098,
        1.0204015085116247, 0.701961, 0.815686, 0.364706,
        1.0987643117152746, 0.8, 0.878431, 0.580392,
        1.1771271149189246, 0.901961, 0.941176, 0.796078,
        1.2505923209791667, 1.0, 1.0, 1.0,
    ],
    NanColor=[1.0, 0.0, 0.0],
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
glyph1Display.Set(
    Representation='Surface',
    ColorArrayName=['POINTS', 'velocity'],
    LookupTable=velocityLUT,
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
glyph1Display.ScaleTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
glyph1Display.OpacityTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for vorticityLUT in view renderView1
vorticityLUTColorBar = GetScalarBar(vorticityLUT, renderView1)
vorticityLUTColorBar.Set(
    AutoOrient=0,
    WindowLocation='Any Location',
    Position=[0.9074049978596246, 0.713639947706462],
    Title='$\\| {\\omega}U_\\infty \\| / D$',
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
    AllowOverlappingLabels=1,
)

# set color bar visibility
vorticityLUTColorBar.Visibility = 1

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
slice1Display.SetScalarBarVisibility(renderView1, True)

# show color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# hide data in view
Hide(vpmpvd, renderView1)

# hide data in view
Hide(glyph1, renderView1)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'vorticity'
vorticityPWF = GetOpacityTransferFunction('vorticity')
vorticityPWF.Set(
    Points=[0.0, 0.0, 0.5, 0.0, 5.0, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# get opacity transfer function/opacity map for 'velocity'
velocityPWF = GetOpacityTransferFunction('velocity')
velocityPWF.Set(
    Points=[0.0016831935004858933, 0.0, 0.5, 0.0, 1.2505923209791667, 1.0, 0.5, 0.0],
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

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation scene

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper
timeKeeper1.SuppressedTimeSources = [cubestl, fvmpvd]

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.Set(
    ViewModules=renderView1,
    Cues=timeAnimationCue1,
    AnimationTime=42.0,
    EndTime=42.0,
    PlayMode='Snap To TimeSteps',
)

# ----------------------------------------------------------------
# restore active source
SetActiveSource(fvmpvd)
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