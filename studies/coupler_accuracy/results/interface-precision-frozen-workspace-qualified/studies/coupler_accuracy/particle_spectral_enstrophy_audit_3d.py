#!/usr/bin/env python3
"""Separate Gaussian and velocity-curl enstrophy in rejected 3D corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

from source.solvers.vpm.numerics import fourier_integrals as native
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays


def spectral_integrals(spectrum, shape, spacing):
    """Integrate the Fourier projection, retaining the zero mode separately."""
    wave = [2 * np.pi * fft.fftfreq(shape[0], spacing)[:, None, None],
            2 * np.pi * fft.fftfreq(shape[1], spacing)[None, :, None],
            2 * np.pi * fft.rfftfreq(shape[2], spacing)[None, None, :]]
    squared_wave = sum(value * value for value in wave)
    inverse = np.divide(1., squared_wave, out=np.zeros_like(squared_wave), where=squared_wave > 0)
    multiplicity = np.ones((1, 1, shape[2] // 2 + 1))
    multiplicity[:, :, 1:] = 2
    if shape[2] % 2 == 0:
        multiplicity[:, :, -1] = 1
    volume = float(np.prod(shape) * spacing**3)
    raw = physical = energy = helicity = 0.
    for axis in range(3):
        raw += float(np.sum(multiplicity * np.abs(spectrum[axis])**2))
        a, b = (axis + 1) % 3, (axis + 2) % 3
        cross = wave[a] * spectrum[b] - wave[b] * spectrum[a]
        cross_squared = np.abs(cross)**2
        physical += float(np.sum(multiplicity * cross_squared * inverse))
        energy += float(np.sum(multiplicity * cross_squared * inverse**2))
        velocity = 1j * cross * inverse
        helicity += float(np.sum(multiplicity * np.real(velocity * np.conjugate(spectrum[axis]))))
    longitudinal_spectrum = sum(wave[axis] * spectrum[axis] for axis in range(3))
    longitudinal = float(np.sum(multiplicity * np.abs(longitudinal_spectrum)**2 * inverse))
    zero = float(sum(abs(value[0, 0, 0])**2 for value in spectrum))
    identity = abs(raw - physical - longitudinal - zero) / max(raw, np.finfo(float).tiny)
    assert identity < 2e-13
    return {"gaussian_enstrophy": raw / volume, "velocity_curl_enstrophy": physical / volume,
            "longitudinal_enstrophy": longitudinal / volume, "zero_mode_enstrophy": zero / volume,
            "kinetic_energy": energy / (2 * volume), "helicity": helicity / volume,
            "orthogonal_decomposition_relative_error": identity,
            "non_curl_fraction_of_gaussian_enstrophy": (longitudinal + zero) / max(raw, np.finfo(float).tiny)}


def particle_spectrum(position, strength, radius, spacing, padding_factor):
    assert np.all(radius == radius[0])
    grid = native._grid_for_particles(position, spacing)
    shape = tuple(padding_factor * value for value in grid.shape)
    if np.prod(shape) > 8_000_000:
        raise ValueError("This diagnostic is bounded to eight million Fourier grid nodes")
    wave = [2 * np.pi * fft.fftfreq(shape[0], spacing)[:, None, None],
            2 * np.pi * fft.fftfreq(shape[1], spacing)[None, :, None],
            2 * np.pi * fft.rfftfreq(shape[2], spacing)[None, None, :]]
    gaussian = np.exp(-.25 * radius[0]**2 * sum(value * value for value in wave))
    compact = native._scatter_vortex_strength_m4(position, strength, grid)
    spectrum = [fft.rfftn(compact[..., axis], s=shape, workers=1) * gaussian for axis in range(3)]
    return spectral_integrals(spectrum, shape, spacing), shape


def manufactured_checks():
    shape, spacing = (27, 29, 31), .1
    length = np.array(shape) * spacing
    coordinates = np.meshgrid(*(spacing * np.arange(n) for n in shape), indexing="ij")
    modes = [np.array([1, 2, -3]), np.array([2, -1, 1])]
    amplitudes = [np.array([.3, -.7, .6]), np.array([-.2, .8, .5])]
    checks = []
    for kind in ("mixed", "longitudinal", "transverse"):
        field = np.zeros((*shape, 3))
        expected = {"gaussian_enstrophy": 0., "velocity_curl_enstrophy": 0., "kinetic_energy": 0.}
        for mode, amplitude in zip(modes, amplitudes, strict=True):
            wave = 2 * np.pi * mode / length
            parallel = wave * np.dot(wave, amplitude) / np.dot(wave, wave)
            value = {"mixed": amplitude, "longitudinal": parallel, "transverse": amplitude - parallel}[kind]
            transverse = value - wave * np.dot(wave, value) / np.dot(wave, wave)
            phase = sum(wave[axis] * coordinates[axis] for axis in range(3))
            field += np.cos(phase)[..., None] * value
            volume = float(np.prod(length))
            expected["gaussian_enstrophy"] += volume * np.dot(value, value) / 2
            expected["velocity_curl_enstrophy"] += volume * np.dot(transverse, transverse) / 2
            expected["kinetic_energy"] += volume * np.dot(transverse, transverse) / (4 * np.dot(wave, wave))
        spectrum = [fft.rfftn(field[..., axis], workers=1) * spacing**3 for axis in range(3)]
        actual = spectral_integrals(spectrum, shape, spacing)
        differences = {key: abs(actual[key] - target) for key, target in expected.items()}
        for key in expected:
            assert differences[key] < 5e-13 * max(expected["gaussian_enstrophy"], 1.)
        checks.append({"kind": kind, "shape": shape, "spacing": spacing, "actual": actual,
                       "analytical": expected, "absolute_differences": differences})
    # A single Gaussian exercises the physical particle Fourier multiplier.
    # On a cubic lattice, angular symmetry gives Z_curl = 2/3 (Z_blob - Z_0).
    shape, spacing, sigma = (48, 48, 48), .1, .3
    strength = np.array([[.7, -.3, 1.1]])
    grid = native.CartesianGrid(np.full(3, -1.2), spacing, (24, 24, 24))
    compact = native._scatter_vortex_strength_m4(np.zeros((1, 3)), strength, grid)
    wave = np.broadcast_arrays(2 * np.pi * fft.fftfreq(48, spacing)[:, None, None],
                               2 * np.pi * fft.fftfreq(48, spacing)[None, :, None],
                               2 * np.pi * fft.rfftfreq(48, spacing)[None, None, :])
    multiplier = np.exp(-.25 * sigma**2 * sum(value * value for value in wave))
    spectrum = [fft.rfftn(compact[..., axis], s=shape, workers=1) * multiplier for axis in range(3)]
    actual = spectral_integrals(spectrum, shape, spacing)
    exact_raw = float(np.sum(strength**2) / ((2 * np.pi)**1.5 * sigma**3))
    exact_curl = 2 / 3 * (actual["gaussian_enstrophy"] - actual["zero_mode_enstrophy"])
    assert abs(actual["gaussian_enstrophy"] - exact_raw) < 2e-12 * exact_raw
    assert abs(actual["velocity_curl_enstrophy"] - exact_curl) < 2e-13 * exact_raw
    reference = native.gaussian_fourier_integrals(np.zeros((1, 3)), strength, np.array([sigma]),
                                                np.ones(1), spacing=spacing, grid=grid)
    assert abs(actual["kinetic_energy"] - reference.total_kinetic_energy) < 2e-13
    assert abs(actual["gaussian_enstrophy"] - reference.total_enstrophy) < 2e-12
    checks.append({"kind": "single_gaussian", "shape": shape, "spacing": spacing,
                   "actual": actual, "analytic_unbounded_gaussian_enstrophy": exact_raw,
                   "cubic_symmetry_velocity_curl_enstrophy": exact_curl,
                   "native_periodic_energy_difference": abs(actual["kinetic_energy"] - reference.total_kinetic_energy)})
    return checks


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = json.loads(args.gate_audit.read_text())
    assert audit["schema"] == "openonda-particle-divergence-gate-audit-3d/1" and audit["status"] == "complete"
    assert audit["spatial_dimensions"] == 3 and audit["physical_time"] == 1.5
    assert audit["settings_unchanged"] and audit["prior_probe_outcome_reproduced"]
    sources = [hash_file(args.gate_audit), *audit["sources"], hash_file(Path(__file__).resolve()),
               hash_file(ROOT / "source/solvers/vpm/numerics/fourier_integrals.py")]
    for row in sources:
        assert hash_file(ROOT / row["path"]) == row
    args.output.mkdir(parents=True)
    report_path = args.output / "particle-spectral-enstrophy-audit-3d.json"
    result = {"schema": "openonda-particle-spectral-enstrophy-audit-3d/1", "status": "running",
              "spatial_dimensions": 3, "physical_time": 1.5, "configurations": [],
              "limitations": [
                  "These are instantaneous diagnostics of the unchanged baseline and rejected correction candidates. No acceptance gate, particle upload or advancing simulation changes.",
                  "The Fourier audit is periodic on a zero-padded temporary lattice and uses native M4 particle scatter. Grid-spacing and box-padding variations quantify those diagnostic sensitivities; this is not exact free-space quadrature.",
                  "Velocity-curl enstrophy is the norm of the solenoidal nonzero-frequency field. Gaussian enstrophy also includes longitudinal and zero-frequency terms. Enstrophy is not a conserved physical-time invariant of general 3D flow.",
                  "This assesses the meaning of the correction diagnostics, not the full-FVM force/profile error or the admissibility of a changed correction policy."]}
    started = time.perf_counter()
    try:
        result["manufactured_checks"] = manufactured_checks()
        field_records = audit["candidate_fields"]
        fields = [read_arrays(ROOT / row["path"]) for row in field_records]
        position, strength, radius = (fields[0][key] for key in ("position", "vortex_strength", "core_radius"))
        assert len(position) == 28441 and np.all(radius == .0625)
        attempts = [next(a for a in audit["attempts"] if a.get("fields") == row) for row in field_records]
        controls = [attempts[0]["before_integrals"], *[row["after_integrals"] for row in attempts]]
        states = [strength, *[value["relaxed"] for value in fields]]
        for value in fields:
            np.testing.assert_array_equal(value["position"], position)
            np.testing.assert_array_equal(value["vortex_strength"], strength)
            np.testing.assert_array_equal(value["core_radius"], radius)
        native_differences = []
        for spacing, padding in ((.0625, 2), (.03125, 2), (.0625, 4)):
            configuration = {"spacing": spacing, "padding_factor": padding, "states": []}
            for index, state in enumerate(states):
                integrals, shape = particle_spectrum(position, state, radius, spacing, padding)
                row = {"name": "baseline" if index == 0 else f"rejected_candidate_{index - 1}",
                       "integrals": integrals, "shape": shape}
                if index:
                    row.update(fields=field_records[index - 1], correction_scale=attempts[index - 1]["correction_scale"])
                    before = configuration["states"][0]["integrals"]
                    row["relative_changes"] = {key: (integrals[key] - before[key]) / before[key]
                                                for key in ("gaussian_enstrophy", "velocity_curl_enstrophy", "kinetic_energy")}
                if spacing == .0625 and padding == 2:
                    control = controls[index]
                    for key, old_key in (("gaussian_enstrophy", "total_enstrophy"), ("kinetic_energy", "total_kinetic_energy"),
                                         ("helicity", "total_helicity")):
                        difference = abs(integrals[key] - control[old_key])
                        assert difference < 2e-12 * max(abs(control[old_key]), 1.)
                        native_differences.append(difference)
                configuration["states"].append(row)
            result["configurations"].append(configuration)
            report_path.write_text(json.dumps(result, indent=2) + "\n")
        result["native_integral_checks"] = len(native_differences)
        result["native_integral_maximum_absolute_difference"] = max(native_differences)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for configuration in result["configurations"]:
            states = sorted(configuration["states"][1:], key=lambda row: row["correction_scale"])
            label = f"h={configuration['spacing']:g}, padding ×{configuration['padding_factor']}"
            for axis, key in zip(axes, ("gaussian_enstrophy", "velocity_curl_enstrophy"), strict=True):
                axis.plot([row["correction_scale"] for row in states],
                          [100 * row["relative_changes"][key] for row in states], "o-", label=label)
        for axis, title in zip(axes, ("Gaussian-vorticity norm", "Physical velocity-curl norm"), strict=True):
            axis.axhspan(-.01, .01, color=".85", label="±0.01% reference band")
            axis.axhline(0, color=".5", linewidth=.7)
            axis.set(title=title, xlabel="Trial correction amplitude", ylabel="Enstrophy change (%)")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        fig.suptitle("Fully 3D saved cube · rejected correction candidates · no particle update")
        figures = []
        for extension in ("png", "svg"):
            path = args.output / ("enstrophy-comparison." + extension)
            fig.savefig(path, dpi=170)
            figures.append(hash_file(path))
        plt.close(fig)
        sources += figures
        frozen = json.loads((ROOT / "frozen-workspace.json").read_text())
        for row in frozen["records"]:
            assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
        unique = {row["path"]: row for row in sources}
        for row in unique.values():
            path = ROOT / row["path"]
            assert hash_file(path) == row
            if path.suffix == ".py":
                archive = args.output / "sources" / row["path"]
                archive.parent.mkdir(parents=True, exist_ok=True)
                archive.write_bytes(path.read_bytes())
        result.update(status="complete", frozen_original_files_verified=len(frozen["records"]),
                      sources=list(unique.values()), figures=figures)
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "native_integral_checks", "native_integral_maximum_absolute_difference")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.gate_audit, args.output = args.gate_audit.resolve(), args.output.resolve()
    run(args)
