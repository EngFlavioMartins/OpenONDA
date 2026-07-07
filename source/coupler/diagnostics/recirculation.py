"""
Centerline recirculation diagnostics for bluff-body coupling cases.

The FVM-VPM interface can look healthy by particle-count and conservation
signals while still damaging the wake if the numerical boundary cuts through
the recirculation bubble.  These helpers turn centerline samples into a few
scalar quantities that are easy to gate after a cubeFlow run:

* minimum streamwise velocity behind the body,
* first reattachment location, and
* streamwise velocity at a probe near the downstream interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RecirculationMetrics:
    time: float
    min_ux: float
    x_min_ux: float
    reattachment_x: float | None
    recirculation_length: float
    ux_probe: float | None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True)
class RecirculationComparison:
    time: float
    hybrid: RecirculationMetrics
    reference: RecirculationMetrics
    reattachment_error: float | None
    ux_probe_error: float | None
    min_ux_error: float

    def to_dict(self) -> dict:
        out = asdict(self)
        out["hybrid"] = self.hybrid.to_dict()
        out["reference"] = self.reference.to_dict()
        return out


def _read_named_csv(path: Path) -> dict[str, np.ndarray]:
    """Read an OpenFOAM or VPM centerline CSV into named arrays."""
    data = np.genfromtxt(path, delimiter=",", names=True, comments="#", dtype=float)
    if data.size == 0 or data.dtype.names is None:
        return {}
    if data.ndim == 0:
        data = data.reshape(1)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


def _streamwise_column(columns: dict[str, np.ndarray]) -> str | None:
    for name in ("Ux", "U_0", "U:0", "U"):
        if name in columns:
            return name
    return None


def read_centerline(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``x, Ux`` from a centerline CSV.

    Supports the OpenFOAM set format (``x,p,U_0,U_1,U_2``) and the VPM sampler
    format (``x,y,z,Ux,Uy,Uz,...``).
    """
    columns = _read_named_csv(path)
    if "x" not in columns:
        raise ValueError(f"{path} has no 'x' column")
    ux_name = _streamwise_column(columns)
    if ux_name is None:
        raise ValueError(f"{path} has no streamwise velocity column")
    return columns["x"], columns[ux_name]


def recirculation_metrics(
    x: np.ndarray,
    ux: np.ndarray,
    *,
    time: float,
    cube_back_x: float = 0.5,
    x_max: float | None = None,
    probe_x: float | None = 1.45,
) -> RecirculationMetrics:
    """Compute first-bubble centerline recirculation metrics behind the body."""
    x = np.asarray(x, dtype=float).reshape(-1)
    ux = np.asarray(ux, dtype=float).reshape(-1)
    mask = (x >= cube_back_x) & np.isfinite(x) & np.isfinite(ux)
    if x_max is not None:
        mask &= x <= x_max
    x = x[mask]
    ux = ux[mask]
    order = np.argsort(x)
    x = x[order]
    ux = ux[order]
    if x.size == 0:
        return RecirculationMetrics(time, np.nan, np.nan, None, 0.0, None)

    i_min = int(np.argmin(ux))
    ux_probe = None
    if probe_x is not None and x[0] <= probe_x <= x[-1]:
        ux_probe = float(np.interp(probe_x, x, ux))

    negative = ux < 0.0
    if not negative.any():
        return RecirculationMetrics(
            time=time,
            min_ux=float(ux[i_min]),
            x_min_ux=float(x[i_min]),
            reattachment_x=None,
            recirculation_length=0.0,
            ux_probe=ux_probe,
        )

    start = int(np.argmax(negative))
    end = None
    for idx in range(start + 1, x.size):
        if ux[idx] >= 0.0:
            end = idx
            break

    if end is None:
        reattachment_x = float(x[-1])
    else:
        x1, x2 = x[end - 1], x[end]
        u1, u2 = ux[end - 1], ux[end]
        if abs(u2 - u1) < 1e-30:
            reattachment_x = float(x2)
        else:
            reattachment_x = float(x1 + (0.0 - u1) * (x2 - x1) / (u2 - u1))

    return RecirculationMetrics(
        time=time,
        min_ux=float(ux[i_min]),
        x_min_ux=float(x[i_min]),
        reattachment_x=reattachment_x,
        recirculation_length=max(0.0, reattachment_x - cube_back_x),
        ux_probe=ux_probe,
    )


def load_centerline_series(
    sample_dir: Path,
    *,
    cube_back_x: float = 0.5,
    probe_x: float | None = 1.45,
    x_max: float | None = None,
) -> list[RecirculationMetrics]:
    """Load all ``postProcessing/centerlineSampling/<time>/centerline*.csv`` files."""
    out: list[RecirculationMetrics] = []
    if not sample_dir.exists():
        return out
    for time_dir in sample_dir.iterdir():
        if not time_dir.is_dir():
            continue
        try:
            t = float(time_dir.name)
        except ValueError:
            continue
        files = sorted(time_dir.glob("centerline*.csv"))
        if not files:
            continue
        x, ux = read_centerline(files[0])
        out.append(
            recirculation_metrics(
                x,
                ux,
                time=t,
                cube_back_x=cube_back_x,
                x_max=x_max,
                probe_x=probe_x,
            )
        )
    return sorted(out, key=lambda item: item.time)


def compare_series(
    hybrid: list[RecirculationMetrics],
    reference: list[RecirculationMetrics],
    *,
    time_tolerance: float = 0.051,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> list[RecirculationComparison]:
    """Match hybrid and reference time series and compute scalar errors."""
    if not hybrid or not reference:
        return []
    ref_times = np.array([item.time for item in reference], dtype=float)
    comparisons: list[RecirculationComparison] = []
    for h in hybrid:
        if h.time < start_time:
            continue
        if end_time is not None and h.time > end_time:
            continue
        idx = int(np.argmin(np.abs(ref_times - h.time)))
        r = reference[idx]
        if abs(r.time - h.time) > time_tolerance:
            continue
        if h.reattachment_x is None or r.reattachment_x is None:
            x_err = None
        else:
            x_err = abs(h.reattachment_x - r.reattachment_x)
        if h.ux_probe is None or r.ux_probe is None:
            probe_err = None
        else:
            probe_err = abs(h.ux_probe - r.ux_probe)
        comparisons.append(
            RecirculationComparison(
                time=h.time,
                hybrid=h,
                reference=r,
                reattachment_error=x_err,
                ux_probe_error=probe_err,
                min_ux_error=abs(h.min_ux - r.min_ux),
            )
        )
    return comparisons


def first_exceedance(
    comparisons: list[RecirculationComparison],
    field: str,
    threshold: float,
) -> RecirculationComparison | None:
    for item in comparisons:
        value = getattr(item, field)
        if value is not None and value > threshold:
            return item
    return None


def load_interface_extent(metadata_path: Path) -> dict[str, float] | None:
    """Read downstream interface landmarks from a coupler run metadata file."""
    if not metadata_path.exists():
        return None
    data = json.loads(metadata_path.read_text())
    domain = data.get("fvm_solver", {}).get("fvm_domain", {})
    vpm = data.get("vpm_solver", {})
    xmax = domain.get("xmax")
    h = vpm.get("h", vpm.get("particle_spacing"))
    dead_zone_h = vpm.get("dead_zone_h")
    buffer_thickness = vpm.get("buffer_thickness")
    if xmax is None or h is None or dead_zone_h is None or buffer_thickness is None:
        return None
    xmax = float(xmax)
    dead_zone = float(dead_zone_h) * float(h)
    buffer_thickness = float(buffer_thickness)
    return {
        "xmax": xmax,
        "core_end_x": xmax - buffer_thickness,
        "ramp_start_x": xmax - buffer_thickness,
        "dead_zone_start_x": xmax - dead_zone,
    }


def first_reference_crossing(
    reference: list[RecirculationMetrics],
    x_limit: float,
    *,
    start_time: float = 0.0,
) -> RecirculationMetrics | None:
    for item in reference:
        if item.time < start_time or item.reattachment_x is None:
            continue
        if item.reattachment_x > x_limit:
            return item
    return None
