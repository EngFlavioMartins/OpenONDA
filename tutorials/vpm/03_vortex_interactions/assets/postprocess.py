"""Shared data loading and core tracking for vortex-interaction figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import defusedxml.ElementTree as ET
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

from .. import setup

CASE_DIR = Path(__file__).resolve().parents[1]
CASES = setup.CASES


def theme():
    """Return the repository plotting module without duplicating its settings."""
    from openonda import plotting

    return plotting


def case_style(name):
    """Return the established label, colour, and marker for one method."""
    label, palette, marker = {
        "baseline": ("Baseline", "TUDdark", "o"),
        "selective_eddy_viscosity": ("Selective eddy viscosity", "VPMpurple", "s"),
        "pedrizzetti_relaxation": ("Pedrizzetti relaxation", "TUDcyan", "D"),
        "particle_splitting": ("Particle splitting", "AccentGreen", "^"),
    }[name]
    return {"label": label, "color": theme().COLORS[palette], "marker": marker}


def figure_size(height_cm):
    """Return a thesis-width figure size in inches for ``height_cm``."""
    plotting = theme()
    return plotting.MAX_FIGURE_WIDTH_CM * plotting.CM, height_cm * plotting.CM


def plot_style_metadata():
    """Return the fixed dimensions, typography, and method styles used by figures."""
    plotting = theme()
    return {
        "width_cm": plotting.MAX_FIGURE_WIDTH_CM,
        "font_size_pt": plotting.THESIS_FONT_SIZE_PT,
        "cases": {name: case_style(name) for name in CASES},
    }


def comparison_legend(fig, handles, labels=None, *, location="top"):
    """Place a compact two-column method legend above or below a figure."""
    if location not in {"top", "bottom"}:
        raise ValueError("location must be 'top' or 'bottom'")
    return fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center" if location == "top" else "lower center",
        bbox_to_anchor=(0.5, 0.99 if location == "top" else 0.02),
        ncol=2,
        frameon=False,
        borderaxespad=0,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.25,
    )


def save_figure(fig, path, axes, formats=("png",), *, fit_margins=True):
    """Validate and save one figure using its script-matching base name."""
    plotting = theme()
    axes = (axes,) if hasattr(axes, "get_position") else tuple(axes)
    if fit_margins:
        plotting.fit_thesis_y_label_margins(fig, axes)
    plotting.validate_thesis_figure(fig, axes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for figure_format in formats:
        fig.savefig(
            path.with_suffix(f".{figure_format}"), dpi=plotting.DEFAULT_DPI, bbox_inches=None
        )


def load_metadata(name):
    """Read one native VPM metadata record, or return an empty dictionary."""
    path = CASE_DIR / "solution" / name / "vpm_metadata.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def file_sha256(path):
    """Return the SHA-256 digest of one source or generated file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_core_section(path):
    """Read one rectangular z=0 meridional vorticity plane.

    Returns nondimensional axial coordinate, radius, and azimuthal vorticity.
    The array shape is ``(N_x, N_r)`` independently of VTK point ordering.
    """
    grid = pv.read(path)
    points = np.asarray(grid.points)
    if not np.allclose(points[:, 2], 0, atol=1e-7) or np.min(points[:, 1]) < -1e-7:
        raise ValueError(f"Expected z=0, y>=0 meridional half-plane: {path}")
    x, ix = np.unique(points[:, 0], return_inverse=True)
    radius, ir = np.unique(points[:, 1], return_inverse=True)
    if len(points) != len(x) * len(radius):
        raise ValueError(f"Incomplete rectangular plane: {path}")
    if len(np.unique(ix * len(radius) + ir)) != len(points):
        raise ValueError(f"Duplicated rectangular-plane point: {path}")
    omega = np.empty((len(x), len(radius)))
    omega[ix, ir] = np.asarray(grid.point_data["vorticity"])[:, 2]
    if not np.isfinite(omega).all():
        raise ValueError(f"Non-finite sampled vorticity: {path}")
    return (
        x / setup.RING_RADIUS,
        radius / setup.RING_RADIUS,
        omega / (setup.RING_CIRCULATION / (np.pi * setup.CORE_RADIUS**2)),
    )


def discover_core_sections(samples_dir, runs=None):
    """List existing native core-section fields in time and requested-run order."""
    records = []
    for index in sorted(Path(samples_dir).glob("*/core_section.pvd")):
        run = index.parent.name
        if run not in CASES or (runs and run not in runs):
            continue
        for entry in ET.parse(index).findall(".//DataSet"):
            path = index.parent / entry.attrib["file"]
            if path.is_file():
                records.append(
                    {
                        "run": run,
                        "label": case_style(run)["label"],
                        "time": float(entry.attrib["timestep"]),
                        "path": path,
                    }
                )
    order = {name: index for index, name in enumerate(runs or CASES)}
    records.sort(key=lambda record: (record["time"], order[record["run"]]))
    return records


def sampled_peaks(x, radius, omega, merge_bridge=0.9):
    """Locate dominant positive vorticity maxima on one sampled plane.

    Secondary lobes connected to a stronger maximum above ``merge_bridge``
    times their own amplitude are treated as one core. Coordinates remain on
    the saved grid; no smoothing or new field evaluation is performed.
    """
    if not np.isfinite(omega).all() or not 0 < merge_bridge < 1:
        raise ValueError("Core samples must be finite and merge_bridge must lie in (0, 1)")
    maximum = float(omega.max())
    if maximum <= 0:
        return []
    indices = np.argwhere((omega == maximum_filter(omega, size=3)) & (omega > 0.1 * maximum))
    indices = sorted(indices, key=lambda index: -omega[tuple(index)])
    if not indices:
        return []
    clipped = any(i in (0, len(x) - 1) or j in (0, len(radius) - 1) for i, j in indices)
    raw_count = len(indices)
    interpolator = RegularGridInterpolator((x, radius), omega)
    parents = list(range(raw_count))

    def root(index):
        while parents[index] != index:
            index = parents[index]
        return index

    for child in range(1, raw_count):
        child_index = indices[child]
        child_value = float(omega[tuple(child_index)])
        candidates = []
        for stronger in range(child):
            strong_index = indices[stronger]
            line = np.linspace(
                [x[child_index[0]], radius[child_index[1]]],
                [x[strong_index[0]], radius[strong_index[1]]],
                101,
            )
            saddle = float(interpolator(line).min())
            if saddle >= merge_bridge * child_value:
                candidates.append((saddle, -np.linalg.norm(line[-1] - line[0]), stronger))
        if candidates:
            parents[child] = max(candidates)[2]

    representatives = sorted({root(index) for index in range(raw_count)})
    cores = [indices[index] for index in representatives]
    bridge_ratio = np.nan
    if len(cores) >= 2:
        pair = np.array([[x[i], radius[j]] for i, j in cores[:2]])
        bridge = interpolator(np.linspace(pair[0], pair[1], 101)).min()
        bridge_ratio = float(bridge / min(omega[tuple(index)] for index in cores[:2]))
    return [
        {
            "x": float(x[i]),
            "radius": float(radius[j]),
            "vorticity": float(omega[i, j]),
            "n_peaks": -1 if clipped else len(cores),
            "raw_n_peaks": raw_count,
            "bridge_ratio": bridge_ratio,
        }
        for i, j in cores
    ]


def core_peak_history(run, merge_bridge=0.9):
    """Extract grid-resolved core maxima from every saved plane for one run."""
    rows = []
    sources = []
    records = discover_core_sections(CASE_DIR / "samples", [run])
    for record in records:
        x, radius, omega = read_core_section(record["path"])
        step = int(record["path"].stem.rsplit("_", 1)[1])
        rows.extend(
            {"run": run, "time": record["time"], "step": step, **peak}
            for peak in sampled_peaks(x, radius, omega, merge_bridge)
        )
        sources.append(
            {
                "file": str(record["path"].relative_to(CASE_DIR)),
                "time": record["time"],
                "sha256": file_sha256(record["path"]),
            }
        )
    if not rows:
        raise ValueError(f"No usable core-section fields for {run}")
    return pd.DataFrame(rows), sources


def track_core_pair(peaks, bridge_limit=0.5, competing_peak_limit=0.5):
    """Track two separated cores until the saved field no longer resolves them."""
    previous = None
    rows = []
    reason = "all saved fields contain a separated pair"
    for step, group in peaks.sort_values("time").groupby("step", sort=False):
        group = group.sort_values("vorticity", ascending=False)
        if len(group) < 2 or (group.n_peaks < 2).any():
            reason = f"step {step}: fewer than two resolved peaks or clipped field"
            break
        if (
            len(group) > 2
            and group.iloc[2].vorticity >= competing_peak_limit * group.iloc[1].vorticity
        ):
            reason = f"step {step}: competing third peak"
            break
        group = group.iloc[:2]
        if group.bridge_ratio.max() >= bridge_limit:
            reason = f"step {step}: bridge ratio reached {bridge_limit:g}"
            break
        if previous is None:
            group = group.sort_values("x", ascending=False)
        else:
            points = group[["x", "radius"]].to_numpy()
            distance = np.linalg.norm(previous[:, None, :] - points[None, :, :], axis=2)
            _, assignment = linear_sum_assignment(distance)
            group = group.iloc[assignment]
        previous = group[["x", "radius"]].to_numpy()
        rows.extend(
            {**row.to_dict(), "core": core} for core, (_, row) in enumerate(group.iterrows(), 1)
        )
    return pd.DataFrame(rows), reason
