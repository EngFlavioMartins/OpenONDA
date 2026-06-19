import gmsh
import numpy as np


def naca0012(x, c=1.0):
    return (
        5
        * 0.12
        * c
        * (
            0.2969 * np.sqrt(x / c)
            - 0.1260 * (x / c)
            - 0.3516 * (x / c) ** 2
            + 0.2843 * (x / c) ** 3
            - 0.1015 * (x / c) ** 4
        )
    )


def generate_mesh(output_file="airfoil.msh"):
    gmsh.initialize()
    gmsh.model.add("airfoil")

    c = 1.0
    x_in, x_out = -5.0, 15.0
    y_min, y_max = -5.0, 5.0
    depth = 0.1

    x_arr = np.linspace(0, c, 60)
    yu = naca0012(x_arr, c)

    pts_u = [gmsh.model.occ.addPoint(xi, yi, 0) for xi, yi in zip(x_arr, yu)]
    pts_l = [gmsh.model.occ.addPoint(xi, -yi, 0) for xi, yi in zip(x_arr, yu)]

    s_u = gmsh.model.occ.addSpline(pts_u)
    s_l = gmsh.model.occ.addSpline(pts_l)

    p1 = gmsh.model.occ.addPoint(x_in, y_max, 0)
    p2 = gmsh.model.occ.addPoint(x_out, y_max, 0)
    p3 = gmsh.model.occ.addPoint(x_out, y_min, 0)
    p4 = gmsh.model.occ.addPoint(x_in, y_min, 0)

    l_top = gmsh.model.occ.addLine(p1, p2)
    l_out = gmsh.model.occ.addLine(p2, p3)
    l_bot = gmsh.model.occ.addLine(p3, p4)
    l_in = gmsh.model.occ.addLine(p4, p1)

    loop_out = gmsh.model.occ.addCurveLoop([l_top, l_out, l_bot, l_in])
    l_te = gmsh.model.occ.addLine(pts_u[-1], pts_l[-1])
    loop_airfoil = gmsh.model.occ.addCurveLoop([s_u, l_te, -s_l])

    surf = gmsh.model.occ.addPlaneSurface([loop_out, loop_airfoil])

    ext = gmsh.model.occ.extrude(
        [(2, surf)], 0, 0, depth, numElements=[1], heights=[1], recombine=True
    )
    gmsh.model.occ.synchronize()

    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", [s_u, s_l, l_te])

    thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh, "InField", dist_field)
    gmsh.model.mesh.field.setNumber(thresh, "SizeMin", 0.03)
    gmsh.model.mesh.field.setNumber(thresh, "SizeMax", 1.0)
    gmsh.model.mesh.field.setNumber(thresh, "DistMin", 0.1)
    gmsh.model.mesh.field.setNumber(thresh, "DistMax", 3.0)

    gmsh.model.mesh.field.setAsBackgroundMesh(thresh)

    boundary_surfaces = gmsh.model.getBoundary(gmsh.model.getEntities(3), oriented=False)
    inlet, outlet, walls, airfoil, frontAndBack = [], [], [], [], []

    tol = 1e-4
    for dim, tag in boundary_surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)

        if abs(zmin) < tol and abs(zmax) < tol:
            frontAndBack.append(tag)
        elif abs(zmin - depth) < tol and abs(zmax - depth) < tol:
            frontAndBack.append(tag)
        elif abs(xmin - x_in) < tol and abs(xmax - x_in) < tol:
            inlet.append(tag)
        elif abs(xmin - x_out) < tol and abs(xmax - x_out) < tol:
            outlet.append(tag)
        elif abs(ymin - y_min) < tol or abs(ymax - y_max) < tol:
            walls.append(tag)
        else:
            airfoil.append(tag)

    gmsh.model.addPhysicalGroup(2, inlet, 1, "inlet")
    gmsh.model.addPhysicalGroup(2, outlet, 2, "outlet")
    gmsh.model.addPhysicalGroup(2, walls, 3, "walls")
    gmsh.model.addPhysicalGroup(2, airfoil, 4, "airfoil")
    gmsh.model.addPhysicalGroup(2, frontAndBack, 5, "frontAndBack")
    gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], 1, "fluid")

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(output_file)
    vtk_path = output_file.replace(".msh", ".vtk")
    gmsh.write(vtk_path)
    gmsh.finalize()
    print(f"Airfoil mesh: {output_file}")
    print(f"ParaView (VTK): {vtk_path}")


if __name__ == "__main__":
    generate_mesh()
