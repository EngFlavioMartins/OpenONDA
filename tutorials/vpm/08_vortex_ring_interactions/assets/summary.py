"""The numbers the README quotes, from the samples of the six runs.

``python -m assets.summary`` writes ``figures/auxiliary/summary.json``; run
``assets.free_space_energy``, ``assets.turbulence_statistics`` and ``assets.energy_transfer``
first. Times are in s, energies in m^5/s^2, viscosities as multiples of nu.

- ``runs``: stop time, E/E_0 at COMPARE_TIME, the time and value of the lowest energy and the
  energy at the stop;
- ``spectra``: the peak of E(k) at COMPARE_TIME and E(k) near kR_0 = 10, where the rings
  hold their energy, and near kR_0 = 120, below the cut-off pi/h;
- ``closures``: each closure's effective viscosity nu_t(k)/nu at BUDGET_TIME, and for the
  spectral cusp its ratio at kR_0 = 120 to its plateau against the Chollet & Lesieur shape;
- ``cusp_sink``: the cusp's sink and the molecular dissipation nu Z at its last samples;
- ``energy_creation``: for each budget time where the energy rises, the numerical remainder
  and the share of the energy created below k_c/4.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import setup
from openonda.vpm.turbulence.spectral_cusp import CUSP_AMPLITUDE, CUSP_RATE, PLATEAU

from assets.postprocess import CASE_DIR, FLOWS, MODEL_STYLES, energy_budget, read_table

COMPARE_TIME = {"collision": 6.0, "leapfrog": 12.0}
BUDGET_TIME = {"collision": 5.5, "leapfrog": 11.5}
WAVENUMBERS = (10.0, 120.0)  # [1/m], kR_0 with R_0 = 1 m
CUTOFF = np.pi / setup.PARTICLE_SPACING


def nearest(values, target):
    return int(np.argmin(np.abs(np.asarray(values) - target)))


def runs():
    summary = {}
    for flow in FLOWS:
        for model in MODEL_STYLES:
            case = f"{flow}_{model}"
            table = energy_budget(case)
            time, energy = table["time"].to_numpy(), table["total_kinetic_energy"].to_numpy()
            lowest = int(np.argmin(energy))
            summary[case] = {
                "stop_time": float(time[-1]),
                "energy_ratio_at_compare_time": float(
                    energy[nearest(time, COMPARE_TIME[flow])] / energy[0]
                ),
                "compare_time": COMPARE_TIME[flow],
                "lowest_energy_time": float(time[lowest]),
                "lowest_energy": float(energy[lowest]),
                "stop_energy": float(energy[-1]),
            }
    return summary


def spectra():
    summary = {}
    for flow in FLOWS:
        for model in MODEL_STYLES:
            table = pd.read_csv(CASE_DIR / "samples" / f"{flow}_{model}" / "spectra.csv")
            rows = table[np.isclose(table["time"], COMPARE_TIME[flow])]
            k, energy = rows["wavenumber"].to_numpy(), rows["energy"].to_numpy()
            summary[f"{flow}_{model}"] = {
                "E_peak": float(energy.max()),
            } | {
                f"E_at_k{target:g}": {
                    "k": float(k[nearest(k, target)]),
                    "E": float(energy[nearest(k, target)]),
                }
                for target in WAVENUMBERS
            }
    return summary


def budget_rows(case, time):
    table = pd.read_csv(CASE_DIR / "samples" / case / "energy_transfer.csv")
    rows = table[np.isclose(table["time"], time) & (table["wavenumber"] > 0)]
    return rows[rows["wavenumber"] <= CUTOFF]


def closures():
    summary = {}
    for flow in FLOWS:
        for model in ("les", "les_cusp"):
            rows = budget_rows(f"{flow}_{model}", BUDGET_TIME[flow])
            k = rows["wavenumber"].to_numpy()
            scale = 2.0 * setup.KINEMATIC_VISCOSITY * k**2 * rows["energy"].to_numpy()
            ratio = (rows["subgrid_dissipation"] + rows["selective_dissipation"]).to_numpy() / scale
            resolved = ratio[k < 30.0]
            entry = {
                "time": BUDGET_TIME[flow],
                "nu_t_over_nu_below_k30": [float(resolved.min()), float(resolved.max())],
                "nu_t_over_nu_at_cutoff": float(ratio[-1]),
            }
            if model == "les_cusp":
                plateau = float(np.median(ratio[k < 0.25 * CUTOFF]))
                index = nearest(k, WAVENUMBERS[1])
                entry |= {
                    "plateau_nu_t_over_nu": plateau,
                    "k_shape": float(k[index]),
                    "shape_measured": float(ratio[index] / plateau),
                    "shape_chollet_lesieur": float(
                        1.0 + CUSP_AMPLITUDE / PLATEAU * np.exp(-CUSP_RATE * CUTOFF / k[index])
                    ),
                }
            summary[f"{flow}_{model}"] = entry
    return summary


def cusp_sink():
    summary = {}
    for flow in FLOWS:
        table = read_table(f"{flow}_les_cusp", "flow_integrals")
        rows = table.iloc[[nearest(table["time"], COMPARE_TIME[flow]), -1]]
        summary[f"{flow}_les_cusp"] = [
            {
                "time": float(row["time"]),
                "subgrid_dissipation": float(-row["subgrid_kinetic_energy_rate"]),
                "molecular_dissipation": float(setup.KINEMATIC_VISCOSITY * row["total_enstrophy"]),
                "cutoff_energy_spectrum": float(row["spectral_cutoff_energy_spectrum"]),
            }
            for _, row in rows.iterrows()
        ]
    return summary


def energy_creation():
    entries = []
    for flow in FLOWS:
        for model in MODEL_STYLES:
            case = f"{flow}_{model}"
            table = pd.read_csv(CASE_DIR / "samples" / case / "energy_transfer.csv")
            for time, rows in table.groupby("time"):
                k = rows["wavenumber"].to_numpy()
                width = np.gradient(k)
                rate = float((rows["energy_rate"] * width).sum())
                if rate <= 0.0:
                    continue
                remainder = rows["numerical_dissipation"].to_numpy() * width
                created = -remainder.clip(max=0.0)
                entries.append(
                    {
                        "case": case,
                        "time": float(time),
                        "energy_rate": rate,
                        "numerical_dissipation": float(remainder.sum()),
                        "molecular_dissipation": float(
                            (rows["molecular_dissipation"] * width).sum()
                        ),
                        "created_share_below_quarter_cutoff": float(
                            created[k < 0.25 * CUTOFF].sum() / created.sum()
                        ),
                    }
                )
    return entries


if __name__ == "__main__":
    output = CASE_DIR / "figures" / "auxiliary" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "runs": runs(),
        "spectra": spectra(),
        "closures": closures(),
        "cusp_sink": cusp_sink(),
        "energy_creation": energy_creation(),
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved: {output}")
