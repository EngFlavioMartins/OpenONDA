#!/usr/bin/env python3
"""Fail when any repository surface uses superseded physical nomenclature.

This is a static repository gate, not a test runner. The default scan covers
source, configuration, documentation, and tracked reference text. Pass
``--generated`` to include every generated/archive text file plus NPZ and HDF5
schema names, including ignored workspace artifacts.
"""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

LEGACY_IDENTIFIERS = frozenset(
    {
        "U",
        "U_target",
        "u_target",
        "p",
        "phi",
        "Co",
        "nut",
        "dt",
        "Ux",
        "Uy",
        "Uz",
        "omega_x",
        "omega_y",
        "omega_z",
        "Fx",
        "Fy",
        "Fz",
        "Fpx",
        "Fpy",
        "Fpz",
        "Fvx",
        "Fvy",
        "Fvz",
        "Ftx",
        "Fty",
        "Ftz",
        "CL",
        "CD",
        "CC",
        "Cx",
        "Cy",
        "Cz",
        "Cd",
        "Cl",
        "Cm",
        "Cn",
        "Mx",
        "My",
        "Mz",
        "Velocity",
        "Vorticity",
        "VelocityMagnitude",
        "VorticityMagnitude",
        "VelocityGradient",
        "StrainRate",
        "FreestreamVelocity",
        "VortexStrength",
        "CoreRadius",
        "KinematicViscosity",
        "EddyViscosity",
        "EffectiveViscosity",
        "GroupID",
        "ZoneID",
        "BoundVelocity",
        "Pressure",
        "OpenONDASurfaceOrdering",
        "openonda_surface_ordering",
        "PanelCenter",
        "BoundLeg",
        "IsTE",
        "IsLE",
        "DeltaCp",
        "PanelForce",
        "TimeValue",
        "face_flux",
        "face_flux_old",
        "face_flux_older",
        "current_time_step_size",
        "cfl_max",
        "max_cfl",
        "cfl",
        "fvm_substeps",
        "vpm_bc",
        "flow_time",
        "time_step",
        "kinetic_energy",
        "energy",
        "enstrophy",
        "helicity",
        "processing_unit",
        "requested_processing_unit",
        "resolved_processing_unit",
        "particle_radius",
        "initial_particles",
        "final_particles",
        "sample_interval",
        "raw_backup_interval",
        "particle_spacing_min",
        "particle_spacing_max",
        "particle_spacing_mean",
        "velocity_gradient_magnitude_max",
        "molecular_kinematic_viscosity",
        "viscosity",
        "S_ref",
        "c_ref",
        "b_ref",
        "r_ref",
        "Cp_min",
        "Cp_max",
        "n_particles",
        "gamma",
        "Gamma",
        "nu",
        "rho",
        "dEdt",
        "neg_nu_enstrophy",
        "strength_magnitude",
        "max_gamma",
        "sum_gamma_magnitude",
        "core_radius_mean",
        "overlap_ratio",
        "overlap_ratio_max",
        "stabilization_events",
        "stabilization_last_mechanism",
        "stabilization_strength_growth",
        "stabilization_max_vorticity_growth",
        "global_cell_ids",
        "GlobalCellIds",
        "GlobalPointIds",
        "global_face_ids",
        "global_n_cells",
        "ranks",
        "velocity_present",
        "normal_velocity_present",
        "tangential_gradient_present",
        "boundary_mass_balance",
        "continuity_max",
        "continuity_sum",
        "pressure_max",
        "pressure_min",
        "turbulence_max",
        "turbulence_min",
        "velocity_max",
        "velocity_min",
        "momentum_residual",
        "pressure_residual",
        "vpm_bc_flux",
        "interface_normal_velocity",
        "pressure_datum_shift",
        "transfer_particle_count",
        "n_existing",
        "n_updated",
        "n_added",
        "n_support",
        "n_committed",
        "acceptance_counts",
        "dvh_fire_counter",
        "particle_regeneration_pending",
        "bc_resync_after_transfer",
        "pressure_anchor_to_freestream",
        "grad_p",
        "u_bc",
        "u_bc_prev",
        "u_bc_next",
        "freestream_velocity_mag",
        "bc_type",
        "viscosity_t",
    }
)

LEGACY_QUOTED = re.compile(
    rf"(?P<quote>[\"'])(?P<identifier>"
    rf"{'|'.join(map(re.escape, sorted(LEGACY_IDENTIFIERS, key=len, reverse=True)))})"
    rf"(?P=quote)"
)
LEGACY_API_TEXT = re.compile(
    r"(?:\bself\.face_flux\b|\b_current_time_step_size\b|\bU_target\b|\bu_target\b|"
    r"\bget_boundary_face_normals\b|\bbound_velocity\b|\brotation_deg\b|"
    r"\bvalue_p_field\b|\bglobal_n_cells\b|\b(?:LambOseenVPM|VortexRingVPM|"
    r"DoubletFlowVPM|TaylorGreenVortexVPM|IsotropicTurbulenceVPM|"
    r"ComputeOfflineDiagnostics|LinearSolveInfo)\b)"
)
LEGACY_EMBEDDED_OUTPUT = re.compile(
    r"(?:\b(?:Velocity|Vorticity|Pressure|VelocityMagnitude|VorticityMagnitude|"
    r"VelocityGradient|StrainRate|FreestreamVelocity|VortexStrength|CoreRadius|"
    r"KinematicViscosity|EddyViscosity|EffectiveViscosity|GroupID|ZoneID)"
    r"\.(?:PVLookupTable|PWF|LUT)\b|"
    r"\b(?:run_backups|samples_backup|vpm_bc_|fvm_centerline|vpm_centerline|centerline\.csv|"
    r"coupled_hybridFlow|coupled_FVM_VPM|cubeFlow|referenceFlow)\b)"
)

LEGACY_PATH_COMPONENTS = frozenset(
    {
        "FVM",
        "VPM",
        "VLM",
        "run_backups",
        "samples_backup",
        "checkpoint",
        "coupled_FVM_VPM",
        "lambOseenVortex",
        "vortexRing",
        "vortexInteractions",
        "quadCopter",
        "rotorFlow",
        "deltaWing",
        "flatPlate",
        "naca4412Flow",
        "cylinderSheddingFlow",
        "referenceFlow",
        "cubeFlow_setup.py",
        "referenceFlow_setup.py",
        "cylinderSheddingFlow_setup.py",
        "lambossen_setup.py",
        "quad_setup.py",
        "allplot.sh",
    }
)

COMPACT_API_NAMES = frozenset(
    {
        "U",
        "p",
        "phi",
        "Co",
        "nut",
        "dt",
        "Fx",
        "Fy",
        "Fz",
        "CL",
        "CD",
        "Mx",
        "My",
        "Mz",
        "gamma",
        "Gamma",
        "nu",
        "rho",
        "mdot",
        "n_particles",
        "time_step",
        "viscosity",
        "processing_unit",
        "grad_u",
        "source_pos",
        "source_rad",
        "target_pos",
    }
)
FORBIDDEN_EXACT_PYTHON_NAMES = frozenset(
    {
        "U_comp_star",
        "U_final",
        "Sij",
        "anti_diffuse_flag",
        "avg_particle_radius",
        "disturb_amp",
        "epsilon_W",
        "epsilon_w",
        "gradU",
        "grad_U",
        "grad_U_int",
        "max_disturb_modes",
        "molecular_viscosity",
        "new_volume",
        "normalize_circulation",
        "ring_strength",
        "ring_thickness",
        "viscosity_turbulent",
        "volume_cpu",
        "vortex_time",
        "_acceptance_counts",
        "_dvh_fire_counter",
        "_dvh_substeps",
        "_n_committed",
        "_particle_regeneration_pending",
        "bc_resync_after_transfer",
        "pressure_anchor_to_freestream",
        "grad_p",
        "u_bc",
        "u_bc_prev",
        "u_bc_next",
        "_velocity_bc_prev",
        "_normal_velocity_bc_prev",
        "_normal_velocity_bc_next",
        "_tangential_gradient_bc_prev",
        "_tangential_gradient_bc_next",
        "_pressure_gradient_bc_prev",
        "_pressure_gradient_bc_next",
        "freestream_velocity_mag",
        "bc_type",
        "viscosity_t",
    }
)

CSV_LEGACY_HEADERS = LEGACY_IDENTIFIERS | frozenset(
    {
        "x",
        "y",
        "z",
        "Sxx",
        "Sxy",
        "Sxz",
        "Syy",
        "Syz",
        "Szz",
        "dudx",
        "dudy",
        "dudz",
        "dvdx",
        "dvdy",
        "dvdz",
        "dwdx",
        "dwdy",
        "dwdz",
    }
)

# These files deliberately enumerate forbidden spellings to enforce their
# absence; they are not runtime compatibility fixtures.
NEGATIVE_ASSERTION_FILES = frozenset(
    {
        "scripts/check_nomenclature.py",
        "tests/test_public_api_has_no_legacy_aliases.py",
        "tests/test_tutorial_contracts.py",
    }
)
EXTERNAL_AMERICAN_API_NAMES = frozenset({"CenterOfRotation", "getCenterOfMass"})
TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".json",
        ".jsonl",
        ".csv",
        ".xdmf",
        ".xmf",
        ".vtu",
        ".vtp",
        ".vts",
        ".vtk",
        ".pvd",
        ".pvtu",
        ".pvts",
        ".pvsm",
        ".md",
        ".sh",
        ".yml",
        ".yaml",
        ".toml",
        ".log",
    }
)
GENERATED_COMPONENTS = frozenset(
    {"run_archives", "samples_archive", "solution", "samples", "grid_study"}
)
XML_FIELD_SUFFIXES = frozenset(
    {".xdmf", ".xmf", ".vtu", ".vtp", ".vts", ".vtk", ".pvd", ".pvtu", ".pvts", ".pvsm"}
)


def _is_generated(path: Path, root: Path) -> bool:
    return bool(set(path.relative_to(root).parts) & GENERATED_COMPONENTS)


def scan_text(root: Path, *, include_generated: bool = False) -> list[str]:
    """Scan source and text serialization surfaces."""
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        if relative in NEGATIVE_ASSERTION_FILES:
            continue
        if _is_generated(path, root) and not include_generated:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if 'replace(".", "p")' in line or '"marker"' in line:
                continue
            if (
                LEGACY_QUOTED.search(line)
                or LEGACY_API_TEXT.search(line)
                or LEGACY_EMBEDDED_OUTPUT.search(line)
            ):
                findings.append(f"{relative}:{line_number}:{line.strip()}")
    return findings


def scan_csv_headers(root: Path, *, include_generated: bool = False) -> list[str]:
    """Reject compact physical names in CSV serialization headers."""
    findings: list[str] = []
    for path in root.rglob("*.csv"):
        if _is_generated(path, root) and not include_generated:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            with path.open(newline="", encoding="utf-8-sig") as stream:
                header = next(csv.reader(stream), [])
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        for field_name in header:
            normalized = field_name.strip()
            if normalized in CSV_LEGACY_HEADERS:
                findings.append(f"{relative}:noncanonical CSV field {normalized!r}")
    return findings


def _python_identifiers(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id, node.lineno
        elif isinstance(node, ast.arg):
            yield node.arg, node.lineno
        elif isinstance(node, ast.Attribute):
            yield node.attr, node.lineno
        elif isinstance(node, ast.keyword) and node.arg is not None:
            yield node.arg, node.lineno
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield node.name, node.lineno


def scan_python_apis(root: Path) -> list[str]:
    """Reject compact public parameters and American geometry identifiers."""
    findings: list[str] = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        if relative in NEGATIVE_ASSERTION_FILES:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue

        for identifier, line_number in _python_identifiers(tree):
            if identifier in FORBIDDEN_EXACT_PYTHON_NAMES:
                findings.append(
                    f"{relative}:{line_number}:removed Python nomenclature {identifier!r}"
                )
            if "center" in identifier.lower() and identifier not in EXTERNAL_AMERICAN_API_NAMES:
                findings.append(
                    f"{relative}:{line_number}:identifier uses American geometry spelling "
                    f"{identifier!r}"
                )

        if relative.startswith("tests/"):
            continue
        is_kernel_module = "kernels" in path.parts or "numerics" in path.parts
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if node.name.startswith("_") and node.name != "__init__":
                continue
            is_decorated_kernel = any(
                isinstance(decorator, ast.Attribute) and decorator.attr in {"func", "kernel"}
                for decorator in node.decorator_list
            )
            if is_kernel_module or is_decorated_kernel:
                continue
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            for argument in arguments:
                if argument.arg in COMPACT_API_NAMES:
                    findings.append(
                        f"{relative}:{node.lineno}:public API {node.name} uses "
                        f"compact parameter {argument.arg!r}"
                    )
    return findings


def scan_paths(root: Path) -> list[str]:
    """Report old package, tutorial, archive, and American-centre paths."""
    findings: list[str] = []
    for path in root.rglob("*"):
        if ".git" in path.parts or "__pycache__" in path.parts or path.name.startswith("."):
            continue
        relative = path.relative_to(root).as_posix()
        relative_parts = path.relative_to(root).parts
        for component in relative_parts:
            if component in LEGACY_PATH_COMPONENTS or "centerline" in component.lower():
                findings.append(f"path:{relative}:noncanonical path component {component!r}")
            is_timestamp = bool(re.fullmatch(r"\d{8}T\d{6}Z", component))
            is_documentation_name = component in {
                "AGENTS.md",
                "CFMESH_ATTRIBUTION.md",
                "README.md",
                "REFERENCES.md",
            }
            if (
                relative_parts[0] in {"source", "tutorials"}
                and any(character.isupper() for character in component)
                and not is_timestamp
                and not is_documentation_name
            ):
                findings.append(
                    f"path:{relative}:solver/tutorial path is not lower snake case: {component!r}"
                )
    return findings


def scan_xml_field_names(root: Path, *, include_generated: bool = False) -> list[str]:
    """Reject generic particle-volume field names in XML-family artifacts."""
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in XML_FIELD_SUFFIXES:
            continue
        if _is_generated(path, root) and not include_generated:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            document = ET.parse(path)
        except (ET.ParseError, OSError) as error:
            findings.append(f"{relative}:cannot parse XML artifact: {error}")
            continue

        if path.suffix.lower() == ".pvsm":
            for property_element in document.iter("Property"):
                if property_element.get("name") in {"Representation", "RepresentationTypesInfo"}:
                    continue
                for element in property_element.iter():
                    if element.get("value") == "volume" or element.get("text") == "volume":
                        findings.append(
                            f"{relative}:ParaView property {property_element.get('name')!r} "
                            "uses noncanonical particle field 'volume'"
                        )
            continue

        for element in document.iter():
            if element.get("Name") == "volume":
                findings.append(f"{relative}:noncanonical XML physical field 'volume'")
    return findings


def _legacy_schema_name(name: str) -> bool:
    return name in LEGACY_IDENTIFIERS or bool(LEGACY_API_TEXT.search(name))


def scan_generated_binary_schemas(root: Path) -> list[str]:
    """Inspect field/dataset names in generated NPZ and HDF5 artifacts."""
    findings: list[str] = []
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - repository dependency
        return [f"generated NPZ audit unavailable: {error}"]

    for path in root.rglob("*.npz"):
        if not _is_generated(path, root):
            continue
        relative = path.relative_to(root).as_posix()
        try:
            with np.load(path, allow_pickle=False) as archive:
                for name in archive.files:
                    if _legacy_schema_name(name):
                        findings.append(f"{relative}:noncanonical NPZ field {name!r}")
        except (OSError, ValueError) as error:
            findings.append(f"{relative}:cannot inspect NPZ schema: {error}")

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - VPM dependency
        hdf5_paths = [path for path in root.rglob("*.h5") if _is_generated(path, root)]
        if hdf5_paths:
            findings.append(f"generated HDF5 audit unavailable: {error}")
        return findings

    for path in root.rglob("*.h5"):
        if not _is_generated(path, root):
            continue
        relative = path.relative_to(root).as_posix()
        try:
            with h5py.File(path, "r") as file:

                def inspect(name: str, item, *, relative: str = relative) -> None:
                    if name == "particles/volume":
                        findings.append(
                            f"{relative}:noncanonical VPM HDF5 object {name!r}; "
                            "use 'particles/particle_volume'"
                        )
                    for component in name.split("/"):
                        if _legacy_schema_name(component):
                            findings.append(f"{relative}:noncanonical HDF5 object {name!r}")
                    for attribute_name in item.attrs:
                        if _legacy_schema_name(attribute_name):
                            findings.append(
                                f"{relative}:noncanonical HDF5 attribute "
                                f"{attribute_name!r} on {name!r}"
                            )

                file.visititems(inspect)
        except (OSError, ValueError) as error:
            findings.append(f"{relative}:cannot inspect HDF5 schema: {error}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--paths", action="store_true", help="also scan directory names")
    parser.add_argument(
        "--generated",
        action="store_true",
        help="also scan all generated/reference artifacts, including ignored files",
    )
    args = parser.parse_args()
    findings = scan_text(args.root, include_generated=args.generated)
    findings.extend(scan_csv_headers(args.root, include_generated=args.generated))
    findings.extend(scan_python_apis(args.root))
    findings.extend(scan_xml_field_names(args.root, include_generated=args.generated))
    if args.paths:
        findings.extend(scan_paths(args.root))
    if args.generated:
        findings.extend(scan_generated_binary_schemas(args.root))
    if findings:
        print("Unapproved physical nomenclature:", file=sys.stderr)
        print("\n".join(sorted(set(findings))), file=sys.stderr)
        return 1
    print("Canonical nomenclature scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
