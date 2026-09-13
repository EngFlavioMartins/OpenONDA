"""Potential of a constant source triangle, with grad(phi) = source velocity.

Author: Flavio A. C. Martins, OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti


@ti.func
def _potential_edge(r0, r1, normal):
    """In-plane outward distance times the finite-edge logarithmic integral."""
    edge = r1-r0
    length = edge.norm()
    contribution = 0.0
    if length > 0:
        tangent = edge/length
        outward = tangent.cross(normal)
        transverse = r0.dot(outward)
        along = r0.dot(tangent)
        perpendicular2 = r0.cross(tangent).norm_sqr()
        a, b = r0.norm(), r1.norm()
        # a+b-length, without subtracting nearly equal lengths when the
        # target approaches the edge segment. The identities use a^2-s^2.
        first, last = a+along, b-along-length
        if along < 0:
            first = perpendicular2/(a-along)
        if along+length > 0:
            last = perpendicular2/(b+along+length)
        denominator = first+last
        if denominator > 0:
            contribution = transverse*ti.log(1+2*length/denominator)
        # On the segment, transverse*log(...) has limit zero. The scalar
        # potential is finite there even though its gradient is singular.
    return contribution


@ti.func
def compute_source_potential(p, v0, v1, v2, normal):
    """Return -integral_triangle 1/|p-y| dA_y / (4*pi).

    Vertices are ordered consistently with the supplied unit normal. For
    r_i = v_i-p and h = r_0 dot n, the positive surface integral is
    sum_edges (r_i dot m_i) log((|r_i|+|r_j|+L)/(|r_i|+|r_j|-L)) - h*Omega,
    where m_i = edge_unit cross n. Reversing vertices and normal leaves the
    scalar potential unchanged. Its spatial gradient is the velocity of a
    unit source panel; the potential tends to zero at infinity.
    """
    r0, r1, r2 = v0-p, v1-p, v2-p
    a, b, c = r0.norm(), r1.norm(), r2.norm()
    height = r0.dot(normal)
    determinant = r0.dot(r1.cross(r2))
    denominator = a*b*c+r0.dot(r1)*c+r1.dot(r2)*a+r2.dot(r0)*b
    omega = 0.0
    if height != 0:
        omega = 2*ti.atan2(determinant, denominator)
    integral = (_potential_edge(r0, r1, normal)+_potential_edge(r1, r2, normal)
                + _potential_edge(r2, r0, normal)-height*omega)
    return -integral/(4.0*3.141592653589793)
