#!/usr/bin/env python3
"""Generate the rotorFlow turbine blade with OpenVSP and import it for VLM."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


@dataclass(frozen=True)
class RotorBladeDesign:
    radius: float = 6.0
    hub_radius: float = 1.0
    root_chord: float = 0.6
    tip_chord: float = 0.35
    freestream_velocity: float = 7.0
    tip_speed_ratio: float = 7.0
    axial_induction_design: float = 1.0 / 3.0
    alpha_design_deg: float = 5.0
    n_stations: int = 21
    chord_stations: int = 7

    @property
    def omega(self) -> float:
        return self.tip_speed_ratio * self.freestream_velocity / self.radius


def design_schedule(design: RotorBladeDesign) -> dict[str, np.ndarray]:
    r = np.linspace(design.hub_radius, design.radius, design.n_stations)
    chord = np.interp(r, [design.hub_radius, design.radius], [design.root_chord, design.tip_chord])
    phi = np.arctan2(
        design.freestream_velocity * (1.0 - design.axial_induction_design),
        design.omega * r,
    )
    theta = phi - np.radians(design.alpha_design_deg)
    beta = 90.0 - np.degrees(theta)
    return {
        "r": r,
        "chord": chord,
        "theta_deg": np.degrees(theta),
        "beta_deg": beta,
        "phi_deg": np.degrees(phi),
    }


def generate_rotorflow_openvsp_blade(
    output_dir: str | Path = "assets/openvsp",
    json_path: str | Path = "assets/blade.json",
    design: RotorBladeDesign | None = None,
) -> Path:
    design = design or RotorBladeDesign()
    output_dir = Path(output_dir)
    json_path = Path(json_path)
    vsp3_path, csv_path, schedule_path = export_openvsp_blade(output_dir, design)
    aircraft = import_openvsp_blade(csv_path, json_path)
    check_imported_geometry(aircraft, design, schedule_path)
    print(f"OpenVSP blade: {vsp3_path}")
    print(f"DegenGeom CSV: {csv_path}")
    print(f"Design schedule: {schedule_path}")
    print(f"OpenONDA VLM surface: {json_path}")
    return json_path


def export_openvsp_blade(output_dir: Path, design: RotorBladeDesign) -> tuple[Path, Path, Path]:
    vsp = _try_import_openvsp()
    if vsp is None:
        command = _openvsp_python_command()
        if command is None:
            raise ImportError(
                "OpenVSP Python API is not importable and no `openvsp-python` helper was found. "
                "Run `scripts/install/install_openvsp.sh` from the repository root."
            )
        args = [
            *command,
            str(Path(__file__).resolve()),
            "--output-dir",
            str(output_dir),
            "--radius",
            str(design.radius),
            "--hub-radius",
            str(design.hub_radius),
            "--root-chord",
            str(design.root_chord),
            "--tip-chord",
            str(design.tip_chord),
            "--freestream-velocity",
            str(design.freestream_velocity),
            "--tip-speed-ratio",
            str(design.tip_speed_ratio),
            "--axial-induction-design",
            str(design.axial_induction_design),
            "--alpha-design-deg",
            str(design.alpha_design_deg),
            "--n-stations",
            str(design.n_stations),
            "--chord-stations",
            str(design.chord_stations),
            "--export-only",
        ]
        subprocess.run(args, check=True, env=_openvsp_subprocess_env())
        return _blade_paths(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    vsp3_path, csv_path, schedule_path = _blade_paths(output_dir)
    schedule = design_schedule(design)
    _write_schedule(schedule_path, schedule)

    vsp.ClearVSPModel()
    wing_id = vsp.AddGeom("WING")
    vsp.SetGeomName(wing_id, "RotorFlow_OpenVSP_Blade")
    vsp.SetParmVal(wing_id, "Sym_Planar_Flag", "Sym", 0)
    vsp.SetParmVal(wing_id, "Y_Rel_Location", "XForm", design.hub_radius)
    # OpenVSP's wing chord starts along +X.  Rotating the whole wing by +90 deg
    # puts zero-twist chord along +Z; section Twist then equals rotor pitch theta.
    vsp.SetParmVal(wing_id, "Y_Rel_Rotation", "XForm", 90.0)
    vsp.SetParmVal(wing_id, "Tess_W", "Shape", max(5, 2 * design.chord_stations - 1))

    xsec_surf = vsp.GetXSecSurf(wing_id, 0)
    while vsp.GetNumXSec(xsec_surf) < design.n_stations:
        vsp.InsertXSec(wing_id, max(1, vsp.GetNumXSec(xsec_surf) - 1), vsp.XS_FOUR_SERIES)

    r = schedule["r"]
    chord = schedule["chord"]
    theta = schedule["theta_deg"]

    root_xsec = vsp.GetXSec(xsec_surf, 0)
    _set_xsec(vsp, root_xsec, "Twist", theta[0])
    _set_xsec(vsp, root_xsec, "Root_Chord", chord[0])
    _set_xsec(vsp, root_xsec, "Tip_Chord", chord[0])
    _set_xsec(vsp, root_xsec, "SectTess_U", 1)
    _set_airfoil(vsp, root_xsec)

    for i in range(1, design.n_stations):
        xsec = vsp.GetXSec(xsec_surf, i)
        _set_xsec(vsp, xsec, "Span", r[i] - r[i - 1])
        _set_xsec(vsp, xsec, "Root_Chord", chord[i - 1])
        _set_xsec(vsp, xsec, "Tip_Chord", chord[i])
        _set_xsec(vsp, xsec, "Sweep", 0.0)
        _set_xsec(vsp, xsec, "Dihedral", 0.0)
        _set_xsec(vsp, xsec, "Twist", theta[i])
        _set_xsec(vsp, xsec, "SectTess_U", 1)
        _set_airfoil(vsp, xsec)

    vsp.Update()
    vsp.WriteVSPFile(str(vsp3_path))
    vsp.SetComputationFileName(vsp.DEGEN_GEOM_CSV_TYPE, str(csv_path))
    vsp.ComputeDegenGeom(vsp.SET_ALL, vsp.DEGEN_GEOM_CSV_TYPE)
    if not csv_path.exists():
        raise RuntimeError(f"OpenVSP did not write DegenGeom CSV: {csv_path}")
    return vsp3_path, csv_path, schedule_path


def import_openvsp_blade(csv_path: Path, json_path: Path):
    from source.solvers.VPM.boundary_elements.vlm.geometry.openvsp_io import (
        OpenVSPImportConfig,
        load_degengeom_csv,
    )
    from source.solvers.VPM.boundary_elements.vlm.geometry.surface_io import save_surface

    config = OpenVSPImportConfig(
        preserve_vsp_paneling=True,
        target_surface_types=("wing", "rotor", "prop"),
    )
    aircraft = load_degengeom_csv(csv_path, config)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    save_surface(aircraft, str(json_path))
    return aircraft


def check_imported_geometry(aircraft, design: RotorBladeDesign, schedule_path: Path) -> None:
    schedule = design_schedule(design)
    rows = []
    for wing in aircraft.wings.values():
        for segment in wing.segments.values():
            le = 0.5 * (segment.vertices["a"] + segment.vertices["b"])
            te = 0.5 * (segment.vertices["d"] + segment.vertices["c"])
            chord_vec = te - le
            mid = 0.5 * (le + te)
            rows.append(
                (
                    float(mid[1]),
                    float(np.linalg.norm(chord_vec)),
                    float(np.degrees(np.arctan2(chord_vec[0], chord_vec[2]))),
                    int(segment.panels_chord),
                    int(segment.panels_span),
                )
            )

    if not rows:
        raise ValueError("OpenVSP import produced no blade segments.")

    rows.sort(key=lambda item: item[0])
    r_mid = np.array([row[0] for row in rows])
    chord = np.array([row[1] for row in rows])
    theta = np.array([row[2] for row in rows])
    target_chord = np.interp(r_mid, schedule["r"], schedule["chord"])
    target_theta = np.interp(r_mid, schedule["r"], schedule["theta_deg"])

    chord_err = float(np.max(np.abs(chord - target_chord)))
    theta_err = float(np.max(np.abs(theta - target_theta)))
    if chord_err > 0.04 or theta_err > 1.0:
        raise ValueError(
            "Imported OpenVSP blade does not match the design schedule: "
            f"max chord error={chord_err:.4f} m, max theta error={theta_err:.3f} deg. "
            f"Schedule: {schedule_path}"
        )

    for r_value, _, theta_value, _, _ in rows:
        relwind = np.array([design.freestream_velocity, 0.0, design.omega * r_value])
        chord_dir = np.array([np.sin(np.radians(theta_value)), 0.0, np.cos(np.radians(theta_value))])
        if float(np.dot(relwind, chord_dir)) <= 0.0:
            raise ValueError(
                f"Blade section at r={r_value:.3f} m is not leading-edge first "
                "for omega_vec=-omega*xhat."
            )

    print(
        "Imported OpenVSP blade QA: "
        f"{len(rows)} segments, {aircraft.total_num_panels()} VLM panels, "
        f"max chord error={chord_err:.4f} m, max theta error={theta_err:.3f} deg"
    )


def _set_xsec(vsp, xsec: str, name: str, value: float) -> None:
    parm = vsp.GetXSecParm(xsec, name)
    if not parm:
        raise RuntimeError(f"OpenVSP xsec parameter not found: {name}")
    vsp.SetParmVal(parm, float(value))


def _set_airfoil(vsp, xsec: str) -> None:
    # Symmetric NACA 0012: realistic finite-thickness visualization while the
    # DegenGeom plate/camberline remains compatible with the flat-plate VLM polar.
    _set_xsec(vsp, xsec, "Camber", 0.0)
    _set_xsec(vsp, xsec, "ThickChord", 0.12)
    _set_xsec(vsp, xsec, "SharpTEFlag", 1.0)


def _write_schedule(path: Path, schedule: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["r", "chord", "theta_deg", "beta_deg", "phi_deg"])
        for values in zip(
            schedule["r"],
            schedule["chord"],
            schedule["theta_deg"],
            schedule["beta_deg"],
            schedule["phi_deg"],
            strict=True,
        ):
            writer.writerow([f"{value:.10g}" for value in values])


def _blade_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    return (
        output_dir / "rotorflow_blade.vsp3",
        output_dir / "rotorflow_blade_degengeom.csv",
        output_dir / "rotorflow_blade_design.csv",
    )


def _try_import_openvsp():
    assets_dir = Path(__file__).resolve().parent
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != assets_dir]
    try:
        import openvsp as vsp
    except ImportError:
        return None
    if not hasattr(vsp, "ClearVSPModel"):
        return None
    return vsp


def _openvsp_python_command() -> list[str] | None:
    configured = os.environ.get("OPENONDA_OPENVSP_PYTHON") or os.environ.get("OPENVSP_PYTHON")
    candidates = [
        configured,
        shutil.which("openvsp-python"),
        str(Path.home() / "anaconda3/envs/openonda-openvsp/bin/python"),
        str(Path.home() / "miniforge3/envs/openonda-openvsp/bin/python"),
        str(Path.home() / "mambaforge/envs/openonda-openvsp/bin/python"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return [candidate]
    return None


def _openvsp_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    root = _openvsp_root()
    if root is not None:
        python_paths = [
            root / "python/openvsp",
            root / "python/degen_geom",
            root / "python/openvsp_config",
            root / "python/utilities",
        ]
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(path) for path in python_paths if path.exists()]
            + ([existing_pythonpath] if existing_pythonpath else [])
        )
        existing_ld_library_path = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(root / "lib")]
            + ([existing_ld_library_path] if existing_ld_library_path else [])
        )
    return env


def _openvsp_root() -> Path | None:
    candidates = [
        os.environ.get("OPENVSP_ROOT"),
        os.environ.get("OPENVSP_PATH"),
        str(Path.home() / "OpenVSP-3.51.0"),
    ]
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if (path / "python/openvsp").exists():
                return path
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rotorFlow's OpenVSP turbine blade.")
    parser.add_argument("--output-dir", default="assets/openvsp")
    parser.add_argument("--json", default="assets/blade.json")
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--hub-radius", type=float, default=1.0)
    parser.add_argument("--root-chord", type=float, default=0.6)
    parser.add_argument("--tip-chord", type=float, default=0.35)
    parser.add_argument("--freestream-velocity", type=float, default=7.0)
    parser.add_argument("--tip-speed-ratio", type=float, default=7.0)
    parser.add_argument("--axial-induction-design", type=float, default=1.0 / 3.0)
    parser.add_argument("--alpha-design-deg", type=float, default=5.0)
    parser.add_argument("--n-stations", type=int, default=21)
    parser.add_argument("--chord-stations", type=int, default=7)
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    design = RotorBladeDesign(
        radius=args.radius,
        hub_radius=args.hub_radius,
        root_chord=args.root_chord,
        tip_chord=args.tip_chord,
        freestream_velocity=args.freestream_velocity,
        tip_speed_ratio=args.tip_speed_ratio,
        axial_induction_design=args.axial_induction_design,
        alpha_design_deg=args.alpha_design_deg,
        n_stations=args.n_stations,
        chord_stations=args.chord_stations,
    )
    if args.export_only:
        export_openvsp_blade(Path(args.output_dir), design)
        return 0
    generate_rotorflow_openvsp_blade(args.output_dir, args.json, design)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
