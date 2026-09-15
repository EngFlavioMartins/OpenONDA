"""Measure an exact FFT workspace optimization for the research scalar potential.

The free-space kernel only needs displacements between the first N cells.
An embedding of at least 2*N-1 per axis therefore suffices. Cache its transforms
and sum the three component products before a single inverse transform. This
changes neither the gauge nor the sampled Gaussian kernel. It is not used by
the trajectory controls, whose implementation and timing remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from cube_covector_flux_control import potential_on_lattice
from cube_covector_flux_step import SourceCells
import numpy as np
from scipy import fft
from scipy.special import erf
from threadpoolctl import threadpool_limits


class LatticePotential:
    """Fixed-shape free-space convolution; source strengths remain call-time inputs."""

    def __init__(self, shape: tuple[int, ...], spacing: float, core_radius: float):
        self.shape = shape
        self.fft_shape = tuple(fft.next_fast_len(2 * n - 1, real=True) for n in shape)
        axes = []
        for size in self.fft_shape:
            index = np.arange(size)
            axes.append(np.where(index <= size // 2, index, index - size) * spacing)
        coordinates = (axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :])
        radius = np.sqrt(sum(coordinate**2 for coordinate in coordinates))
        scaled = radius / core_radius
        factor = np.zeros_like(radius)
        np.divide(
            erf(scaled) - 2 / np.sqrt(np.pi) * scaled * np.exp(-(scaled**2)),
            4 * np.pi * radius**3,
            out=factor,
            where=radius > 0,
        )
        self.spectra = [fft.rfftn(factor * coordinate) for coordinate in coordinates]

    def evaluate(self, coefficient: np.ndarray) -> np.ndarray:
        """Return psi [m/s] at the source lattice using zero-extended coefficients."""
        if coefficient.shape != (*self.shape, 3):
            raise ValueError("Potential workspace requires its exact configured source shape")
        spectrum = sum(
            fft.rfftn(coefficient[..., component], s=self.fft_shape) * kernel
            for component, kernel in enumerate(self.spectra)
        )
        potential = fft.irfftn(spectrum, s=self.fft_shape)
        return potential[tuple(slice(0, n) for n in self.shape)].copy()


def run(state_path: Path, output: Path) -> None:
    """Check odd/even embeddings and a real cloud, then time warmed evaluations."""
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    rng = np.random.default_rng(20260915)
    for shape in ((3, 4, 5), (6, 7, 8), (9, 5, 4)):
        coefficient = rng.normal(size=(*shape, 3)) * 0.06**3
        expected = potential_on_lattice(coefficient, 0.06, 0.066)
        candidate = LatticePotential(shape, 0.06, 0.066).evaluate(coefficient)
        rows.append(
            {"shape": shape, "max_absolute_difference": float(np.max(abs(candidate - expected)))}
        )
    with np.load(state_path) as data:
        position, strength, radius = (
            data[key].astype(float) for key in ("position", "vortex_strength", "core_radius")
        )
    cells = SourceCells.from_particles(position, radius)
    coefficient = np.zeros((*cells.identifiers.shape, 3))
    coefficient[tuple(cells.index.T)] = strength
    started = time.perf_counter()
    workspace = LatticePotential(cells.identifiers.shape, 0.06, cells.core_radius)
    preparation = time.perf_counter() - started
    expected = potential_on_lattice(coefficient, 0.06, cells.core_radius)
    candidate = workspace.evaluate(coefficient)
    error = float(np.max(abs(expected - candidate)))
    if max(error, *(row["max_absolute_difference"] for row in rows)) > 1e-12:
        raise AssertionError("FFT embedding changed the free-space potential")
    timings = {"original": [], "cached_embedding": []}
    for _ in range(5):
        started = time.perf_counter()
        potential_on_lattice(coefficient, 0.06, cells.core_radius)
        timings["original"].append(time.perf_counter() - started)
        started = time.perf_counter()
        workspace.evaluate(coefficient)
        timings["cached_embedding"].append(time.perf_counter() - started)
    report = {
        "scope": __doc__,
        "random_checks": rows,
        "actual_cloud_max_absolute_difference": error,
        "shape": cells.identifiers.shape,
        "cached_fft_shape": workspace.fft_shape,
        "preparation_seconds": preparation,
        "kernel_cache_bytes": sum(spectrum.nbytes for spectrum in workspace.spectra),
        "timings_seconds": timings,
        "median_speedup": float(
            np.median(timings["original"]) / np.median(timings["cached_embedding"])
        ),
        "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2), fft.set_workers(2):
        run(args.state, args.output)
