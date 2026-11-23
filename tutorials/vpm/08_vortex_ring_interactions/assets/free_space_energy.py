"""Kinetic energy of the saved particle fields in free space, for runs that sampled it periodically.

The solver measures the energy of a GBD run with the unbounded Gaussian integrals only while
every particle carries the same viscosity. The Smagorinsky runs give each particle its own
eddy viscosity, so their flow integrals hold the periodic Fourier energy instead, which loses
the far field of the rings (1 % less at t = 0 here). This script evaluates the unbounded energy
of each backup of such a run with the molecular viscosity, which does not enter the energy.

``python -m assets.free_space_energy`` writes ``samples/<case>/free_space_energy.csv`` for each
run whose ``flow_integrals.csv`` is not already the unbounded measurement.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from openonda.vpm.numerics.fourier_integrals import gaussian_fourier_integrals

from assets.postprocess import CASE_DIR, read_table
from assets.turbulence_statistics import CASES, backups, read_particles


def periodic(case):
    """Whether the run's flow integrals measured the energy on a periodic grid."""
    return set(read_table(case, "flow_integrals")["energy_measurement"]) != {"unbounded_energy"}


def free_space_energy(case):
    rows = []
    for time, path in backups(case):
        state = read_particles(path)
        spacing = float(np.median(np.cbrt(state["particle_volume"])))
        integrals = gaussian_fourier_integrals(
            state["position"],
            state["vortex_strength"],
            state["core_radius"],
            state["particle_volume"],
            effective_viscosity=state["kinematic_viscosity"],
            spacing=spacing,
            free_space=True,
        )
        rows.append({"time": time, "total_kinetic_energy": integrals.total_kinetic_energy})
        print(f"{case}: t = {time:.2f} s, E = {integrals.total_kinetic_energy:.6f}", flush=True)
    with open(CASE_DIR / "samples" / case / "free_space_energy.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", default=[case for case in CASES if periodic(case)])
    for name in parser.parse_args().cases:
        free_space_energy(name)
