"""Generate the rotor_flow turbine blade with OpenVSP and import it for VLM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openonda.results import write_csv_table


class RotorBladeDesign:
    def __init__(
        self,
        rotor_radius: float = 6.0,
        hub_radius: float = 1.0,
        root_chord: float = 0.6,
        tip_chord: float = 0.35,
        freestream_speed: float = 7.0,
        tip_speed_ratio: float = 7.0,
        design_axial_induction_factor: float = 1.0 / 3.0,
        design_angle_of_attack_degrees: float = 5.0,
        n_radial_stations: int = 21,
        n_chordwise_stations: int = 7,
    ):
        self.rotor_radius = rotor_radius
        self.hub_radius = hub_radius
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.freestream_speed = freestream_speed
        self.tip_speed_ratio = tip_speed_ratio
        self.design_axial_induction_factor = design_axial_induction_factor
        self.design_angle_of_attack_degrees = design_angle_of_attack_degrees
        self.n_radial_stations = n_radial_stations
        self.n_chordwise_stations = n_chordwise_stations
        self.angular_velocity = self.tip_speed_ratio * self.freestream_speed / self.rotor_radius


def design_schedule(design: RotorBladeDesign) -> dict[str, np.ndarray]:
    radial_position = np.linspace(design.hub_radius, design.rotor_radius, design.n_radial_stations)
    chord = np.interp(
        radial_position,
        [design.hub_radius, design.rotor_radius],
        [design.root_chord, design.tip_chord],
    )
    inflow_angle = np.arctan2(
        design.freestream_speed * (1.0 - design.design_axial_induction_factor),
        design.angular_velocity * radial_position,
    )
    twist_angle = inflow_angle - np.radians(design.design_angle_of_attack_degrees)
    openvsp_section_rotation = 90.0 - np.degrees(twist_angle)
    return {
        "radial_position": radial_position,
        "chord": chord,
        "twist_angle_degrees": np.degrees(twist_angle),
        "openvsp_section_rotation_degrees": openvsp_section_rotation,
        "inflow_angle_degrees": np.degrees(inflow_angle),
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
    import_openvsp_blade(csv_path, json_path)
    print(f"OpenVSP blade: {vsp3_path}")
    print(f"DegenGeom CSV: {csv_path}")
    print(f"Design schedule: {schedule_path}")
    print(f"OpenONDA VLM surface: {json_path}")
    return json_path


def export_openvsp_blade(output_dir: Path, design: RotorBladeDesign) -> tuple[Path, Path, Path]:
    import openvsp as vsp

    vsp3_path, csv_path, schedule_path = _blade_paths(output_dir)
    schedule = design_schedule(design)
    _write_schedule(schedule_path, schedule)
    vsp.ClearVSPModel()
    wing_id = vsp.AddGeom("WING")
    vsp.SetGeomName(wing_id, "RotorFlow_OpenVSP_Blade")
    vsp.SetParmVal(wing_id, "Sym_Planar_Flag", "Sym", 0)
    vsp.SetParmVal(wing_id, "Y_Rel_Location", "XForm", design.hub_radius)
    vsp.SetParmVal(wing_id, "Y_Rel_Rotation", "XForm", 90.0)
    vsp.SetParmVal(wing_id, "Tess_W", "Shape", max(5, 2 * design.n_chordwise_stations - 1))
    xsec_surf = vsp.GetXSecSurf(wing_id, 0)
    while vsp.GetNumXSec(xsec_surf) < design.n_radial_stations:
        vsp.InsertXSec(wing_id, max(1, vsp.GetNumXSec(xsec_surf) - 1), vsp.XS_FOUR_SERIES)
    radial_position = schedule["radial_position"]
    chord = schedule["chord"]
    twist_angle_degrees = schedule["twist_angle_degrees"]
    root_xsec = vsp.GetXSec(xsec_surf, 0)
    _set_xsec(vsp, root_xsec, "Twist", twist_angle_degrees[0])
    _set_xsec(vsp, root_xsec, "Root_Chord", chord[0])
    _set_xsec(vsp, root_xsec, "Tip_Chord", chord[0])
    _set_xsec(vsp, root_xsec, "SectTess_U", 1)
    _set_airfoil(vsp, root_xsec)
    for i in range(1, design.n_radial_stations):
        xsec = vsp.GetXSec(xsec_surf, i)
        _set_xsec(vsp, xsec, "Span", radial_position[i] - radial_position[i - 1])
        _set_xsec(vsp, xsec, "Root_Chord", chord[i - 1])
        _set_xsec(vsp, xsec, "Tip_Chord", chord[i])
        _set_xsec(vsp, xsec, "Sweep", 0.0)
        _set_xsec(vsp, xsec, "Dihedral", 0.0)
        _set_xsec(vsp, xsec, "Twist", twist_angle_degrees[i])
        _set_xsec(vsp, xsec, "SectTess_U", 1)
        _set_airfoil(vsp, xsec)
    vsp.Update()
    vsp.WriteVSPFile(str(vsp3_path))
    vsp.SetComputationFileName(vsp.DEGEN_GEOM_CSV_TYPE, str(csv_path))
    vsp.ComputeDegenGeom(vsp.SET_ALL, vsp.DEGEN_GEOM_CSV_TYPE)
    return (vsp3_path, csv_path, schedule_path)


def import_openvsp_blade(csv_path: Path, json_path: Path):
    from source.solvers.vpm.boundary_elements.vlm.geometry.openvsp_io import (
        OpenVSPImportConfig,
        load_degengeom_csv,
    )
    from source.solvers.vpm.boundary_elements.vlm.geometry.surface_io import (
        save_surface,
    )

    config = OpenVSPImportConfig(
        preserve_vsp_paneling=True, target_surface_types=("wing", "rotor", "prop")
    )
    aircraft = load_degengeom_csv(csv_path, config)
    save_surface(aircraft, str(json_path))
    return aircraft


def _set_xsec(vsp, xsec: str, name: str, value: float) -> None:
    parm = vsp.GetXSecParm(xsec, name)
    vsp.SetParmVal(parm, float(value))


def _set_airfoil(vsp, xsec: str) -> None:
    _set_xsec(vsp, xsec, "Camber", 0.0)
    _set_xsec(vsp, xsec, "ThickChord", 0.12)
    _set_xsec(vsp, xsec, "SharpTEFlag", 1.0)


def _write_schedule(path: Path, schedule: dict[str, np.ndarray]) -> None:
    columns = (
        "radial_position",
        "chord",
        "twist_angle_degrees",
        "openvsp_section_rotation_degrees",
        "inflow_angle_degrees",
    )
    rows = (
        [f"{value:.10g}" for value in values]
        for values in zip(*(schedule[name] for name in columns), strict=True)
    )
    write_csv_table(path, rows, columns=columns)


def _blade_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    return (
        output_dir / "rotorflow_blade.vsp3",
        output_dir / "rotorflow_blade_degengeom.csv",
        output_dir / "rotorflow_blade_design.csv",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rotor_flow's OpenVSP turbine blade.")
    parser.add_argument("--output-dir", default="assets/openvsp")
    parser.add_argument("--json", default="assets/blade.json")
    parser.add_argument("--rotor-radius", type=float, default=6.0)
    parser.add_argument("--hub-radius", type=float, default=1.0)
    parser.add_argument("--root-chord", type=float, default=0.6)
    parser.add_argument("--tip-chord", type=float, default=0.35)
    parser.add_argument("--freestream-speed", type=float, default=7.0)
    parser.add_argument("--tip-speed-ratio", type=float, default=7.0)
    parser.add_argument("--design-axial-induction-factor", type=float, default=1.0 / 3.0)
    parser.add_argument("--design-angle-of-attack-degrees", type=float, default=5.0)
    parser.add_argument("--n-radial-stations", type=int, default=21)
    parser.add_argument("--n-chordwise-stations", type=int, default=7)
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    design = RotorBladeDesign(
        rotor_radius=args.rotor_radius,
        hub_radius=args.hub_radius,
        root_chord=args.root_chord,
        tip_chord=args.tip_chord,
        freestream_speed=args.freestream_speed,
        tip_speed_ratio=args.tip_speed_ratio,
        design_axial_induction_factor=args.design_axial_induction_factor,
        design_angle_of_attack_degrees=args.design_angle_of_attack_degrees,
        n_radial_stations=args.n_radial_stations,
        n_chordwise_stations=args.n_chordwise_stations,
    )
    if args.export_only:
        export_openvsp_blade(Path(args.output_dir), design)
        return 0
    generate_rotorflow_openvsp_blade(args.output_dir, args.json, design)
    return 0


if __name__ == "__main__":
    main()
