"""Machine-sized, quasi-2D Re=150 cylinder experiment (all lengths in D)."""

DOMAIN = (-8.0, 24.0, -10.0, 10.0, -0.5, 0.5)
GRIDS = {"coarse": 1.0 / 8.0, "medium": 1.0 / 16.0, "fine": 1.0 / 32.0}
REFINEMENT_RATIO = 2.0
CONTROL_DOMAINS = {
    "fine_domain": (-12.0, 28.0, -12.0, 12.0, -0.5, 0.5),
    "fine_span": (-8.0, 24.0, -10.0, 10.0, -0.375, 0.375),
}


def domain_for(case):
    return CONTROL_DOMAINS.get(case, DOMAIN)


def spacing_for(case):
    return GRIDS["fine"] if case in CONTROL_DOMAINS else GRIDS[case]
