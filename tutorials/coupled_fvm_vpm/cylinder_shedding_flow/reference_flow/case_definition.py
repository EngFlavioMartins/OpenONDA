"""Machine-sized, quasi-2D Re=150 cylinder experiment (all lengths in D)."""

DOMAIN = (-8.0, 24.0, -10.0, 10.0, -0.5, 0.5)
GRIDS = {"coarse": 1.0 / 8.0, "medium": 1.0 / 16.0, "fine": 1.0 / 32.0}
REFINEMENT_RATIO = 2.0
# Fixed physical regions, all target spacings scale with the wall spacing.
# Keep the dense body region compact and put wake cells downstream.
REFINEMENT_REGIONS = (
    ("near_body", (-1.0, 2.0, -1.0, 1.0), 2.0),
    ("near_wake", (0.0, 6.0, -1.0, 1.0), 2.0),
    ("wake", (0.0, 12.0, -1.5, 1.5), 4.0),
)
CONTROL_DOMAINS = {
    "fine_domain": (-12.0, 28.0, -12.0, 12.0, -0.5, 0.5),
    "fine_span": (-8.0, 24.0, -10.0, 10.0, -0.375, 0.375),
}


def domain_for(case):
    return CONTROL_DOMAINS.get(case, DOMAIN)


def spacing_for(case):
    return GRIDS["fine"] if case in CONTROL_DOMAINS else GRIDS[case]


def construction_domain(dx, domain=DOMAIN):
    """Use four background cells through the temporary cfMesh source span."""
    half_span = 16.0 * dx
    return (*domain[:4], -half_span, half_span)


def extrusion_levels(dx, domain=DOMAIN):
    import numpy as np

    count = (domain[5] - domain[4]) / (4.0 * dx)
    if not np.isclose(count, round(count), atol=1e-10):
        raise ValueError("Physical span must contain an integer number of extrusion layers")
    return tuple(np.linspace(domain[4], domain[5], int(round(count)) + 1))
