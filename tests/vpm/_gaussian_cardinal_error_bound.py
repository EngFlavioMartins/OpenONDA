"""UNWIRED real-arithmetic smooth-field interpolation majorant, order10.

This bounds SOURCE and TARGET cardinal interpolation only, not the FFT,
Gaussian radial evaluation, deposition arithmetic, core correction or tail
completion. Both physical sources and targets must lie within one slab.
The bound uses absolute source strength, never a cancelling net circulation.

For ten consecutive nodes and central argument t in[4,5], the exact Lebesgue
maximum is Lambda=25609/16384 and max|prod(t-j)|/10!=63/262144. The latter
follows from prod((j+1/2)^2-y^2), |y|<=1/2. The former is the even polynomial
25609/16384-22349*y²/9216+3259*y⁴/4608-37*y⁶/576+y⁸/576;
its derivative in y² is negative throughout[0,1/4]. Telescoping six commuting
1D source/target interpolators gives C=sum_i Lambda^i*Omega*h_i^10/10!.
We order the larger h_i first to minimize this valid absolute majorant.

With Phi_tau's Fourier multiplier exp(-tau²|k|²/4)/|k|² and convention
(2pi)^-3 integral, absolute directional order10 derivatives obey
 ||D_a^10 u|| <= |Gamma| Gamma(6) (4/tau²)^6/(44pi²),
 ||D_a^10 J||F <= |Gamma| Gamma(13/2) (4/tau²)^(13/2)/(44pi²).
These follow by integrating |k_a|^10 and using |Gamma cross k|<=|Gamma||k|.

For separated interpolation hulls a second bound avoids multiplying that
near bound by every image. The Gaussian integral representation gives any
n unit-direction derivative of Phi <= A_n/r^(n+1), where
 A_n=n!/(4pi^(3/2))*sum_k 2^(n-2k)Gamma(n-k+1/2)/(k!(n-2k)!).
The resulting vector/Frobenius constants are sqrt(2)A_11 and sqrt(6)A_12:
sum_i|e_i cross Gamma|²=2|Gamma|². Every source/target stencil extends by
at most5h_z vertically, so D=image-to-physical-slab separation-10h_z is
a conservative separation of the complete two-stencil hulls.

All expressions here are ordinary binary64 evaluations of real-arithmetic
bounds, not outward-rounded numerical certificates or runtime admission.
"""

import math
from numbers import Integral

LEBESGUE_10 = 25609/16384
REMAINDER_10 = 63/262144


def _positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"positive finite {name} required")
    return float(value)


def gaussian_directional_constant(order):
    """Real-arithmetic Gaussian-potential derivative majorant at r=1."""
    if isinstance(order, bool) or not isinstance(order, Integral) or not 1 <= order <= 16:
        raise ValueError("bounded derivative order1..16 required")
    return math.factorial(order)/(4*math.pi**1.5)*sum(
        2**(order-2*k)*math.gamma(order-k+.5)/(math.factorial(k)*math.factorial(order-2*k))
        for k in range(order//2+1))


def slab_cardinal_interpolation_bound(absolute_strength, *, tau, spacing, width, shells):
    """Uniform per-target finite +/-K image-only interpolation error majorant.

    The finite family is (k,odd) for -K<=k<=K with (0,False) excluded.
    No physical coordinate rounding or mesh mutation is performed.
    """
    if (isinstance(absolute_strength, bool) or not math.isfinite(absolute_strength)
            or absolute_strength < 0):
        raise ValueError("finite nonnegative absolute strength required")
    tau, spacing, width = (_positive(value, name) for value, name in
                           ((tau, "tau"), (spacing, "spacing"), (width, "width")))
    if isinstance(shells, bool) or not isinstance(shells, Integral) or not 1 <= shells <= 16384:
        raise ValueError("bounded positive shells1..16384 required")
    ratio = width/spacing
    if not math.isfinite(ratio) or ratio > 2**30:
        raise ValueError("bounded slab grid required")
    cells = max(1, math.ceil(ratio))
    hz = width/cells
    axis_steps = sorted([spacing]*4+[hz]*2, reverse=True)
    interpolation_factor = REMAINDER_10*sum(LEBESGUE_10**i*h**10 for i, h in enumerate(axis_steps))
    uniform_u = math.gamma(6)*(4/tau**2)**6/(44*math.pi**2)
    uniform_j = math.gamma(6.5)*(4/tau**2)**6.5/(44*math.pi**2)
    far_u, far_j = math.sqrt(2)*gaussian_directional_constant(11), math.sqrt(6)*gaussian_directional_constant(12)
    near_count = 0
    sums = [0., 0.]
    far_sums = [0., 0.]
    for k in range(-shells, shells+1):
        for odd in (False, True):
            if k == 0 and not odd:
                continue
            lower, upper = ((2*k-1)*width, 2*k*width) if odd else (2*k*width, (2*k+1)*width)
            physical_separation = max(0., lower-width, -upper)
            hull_separation = physical_separation-10*hz
            values = (uniform_u, uniform_j)
            if hull_separation > 0:
                values = (min(uniform_u, far_u/hull_separation**12),
                          min(uniform_j, far_j/hull_separation**13))
                far_sums = [current+value for current, value in zip(far_sums, values, strict=True)]
            else:
                near_count += 1
            sums = [current+value for current, value in zip(sums, values, strict=True)]
    weight = absolute_strength*interpolation_factor
    count = 4*shells+1
    result = {
        "runtime_admissible": False,
        "scope": "real-arithmetic order10 double-cardinal interpolation only; all sources/targets inside physical slab",
        "shells": int(shells), "image_count": count, "near_hull_images": near_count,
        "absolute_strength": absolute_strength, "tau": tau, "spacing": spacing, "spacing_z": hz,
        "lebesgue_constant": LEBESGUE_10, "central_remainder_factor": REMAINDER_10,
        "six_operator_factor": interpolation_factor,
        "uniform_all_images_velocity": weight*count*uniform_u,
        "uniform_all_images_gradient": weight*count*uniform_j,
        "near_images_velocity": weight*near_count*uniform_u,
        "near_images_gradient": weight*near_count*uniform_j,
        "far_images_velocity": weight*far_sums[0], "far_images_gradient": weight*far_sums[1],
        "separated_total_velocity": weight*sums[0], "separated_total_gradient": weight*sums[1],
    }
    if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("interpolation majorant overflow")
    return result
