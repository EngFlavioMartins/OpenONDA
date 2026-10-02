"""Rebuild the ring schematic and paired particle view from the saved LES run.

Requires ParaView (PVPYTHON or installed pvpython), PyVista, and pdflatex;
PNG export also requires pdftoppm. The temporary build produces PNG (default)
or PDF figures, retaining their TeX sources and rendered image assets.
No simulation samples or solver parameters are modified. Both snapshots use
one orthographic field of view and the same strength-to-radius/color maps.
"""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"


from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, tempfile
import h5py, numpy as np, pyvista as pv
from matplotlib import colormaps
from openonda import plotting as theme
from .. import setup as s
from openonda.executables import find_executable
from source.solvers.vpm.io.manifest import _manifest_value

CASE = Path(__file__).resolve().parents[1]


def initial_cloud():
    """Rebuild the declarative initial cloud from the tutorial setup."""
    return s.build_case("les_transposed").initial_conditions[0].build()


def initial_configuration(configuration=None):
    """The complete declared initialization and its solver representation."""
    if configuration is None:
        case = s.build_case("les_transposed")
        configuration = {
            "initial_conditions": _manifest_value(case.initial_conditions),
            "initial_weak_particle_percent": case.initial_weak_particle_percent,
            "numerics": _manifest_value(case.numerics),
        }
    return {
        "initial_conditions": configuration["initial_conditions"],
        "initial_weak_particle_percent": configuration["initial_weak_particle_percent"],
        **{
            key: configuration["numerics"][key]
            for key in ("precision", "random_seed", "particle_kernel")
        },
    }


def validate_initial_cloud(cloud, path):
    """Reject a scene reconstructed from a different step-zero particle state.

    This check also applies to ``--schematic-only``: its initial geometry and
    normalization describe the saved run, not an unrelated current setup.
    The HDF5 field dtypes are the recorded solver precision.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved step-zero ring backup required for scene provenance: {path}"
        )
    with h5py.File(path) as initial:
        solver = initial["solver"].attrs
        if solver["step"] != 0 or solver["time"] != 0.0:
            raise ValueError("Initial ring backup is not the step-zero state.")
        count = solver["n_particles_total"]
        if count != len(cloud.position):
            raise ValueError("Initial ring backup particle count differs from reconstruction.")
        for name in ("position", "vortex_strength", "core_radius"):
            saved = initial[f"particles/{name}"][:]
            shape = (count,) if name == "core_radius" else (count, 3)
            if saved.shape != shape or saved.dtype not in (
                np.dtype("float32"),
                np.dtype("float64"),
            ):
                raise ValueError(f"Initial ring backup {name} has invalid shape or precision.")
            reconstructed = np.asarray(getattr(cloud, name))
            if reconstructed.shape != shape or not np.all(np.isfinite(reconstructed)):
                raise ValueError(f"Reconstructed initial ring {name} is invalid.")
            if not np.all(np.isfinite(saved)) or (name == "core_radius" and np.any(saved <= 0)):
                raise ValueError(f"Initial ring backup {name} is invalid.")
            if not np.array_equal(reconstructed.astype(saved.dtype), saved):
                raise ValueError(
                    f"Reconstructed initial ring {name} differs from the saved step-zero "
                    "state at its recorded precision; restore the matching setup before plotting."
                )


def validate_scene_initial_cloud(cloud, case_dir=CASE):
    """Prefer saved fields; otherwise require a recorded field fingerprint.

    The legacy published plotting archive omitted its step-zero backup. Its
    separate provenance sidecar was derived from that saved backup, not from
    a new reconstruction. The fallback authenticates the released metadata
    and each reconstructed field; it does not claim the old arrays are present.
    """
    case_dir = Path(case_dir)
    metadata_path = case_dir / "solution/les_transposed/vpm_metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    declared = initial_configuration()
    if initial_configuration(metadata["configuration"]) != declared:
        raise ValueError("Current ring initialization configuration differs from saved metadata.")
    initial_path = case_dir / "solution/les_transposed/vpm/vpm_000000.h5"
    if initial_path.is_file():
        validate_initial_cloud(cloud, initial_path)
        return "saved_step_zero_verified"

    sidecar_path = case_dir / "assets/initial_state_fingerprint.json"
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            "Saved ring step-zero state or its recorded fingerprint is required."
        )
    fingerprint = json.loads(sidecar_path.read_text())
    manifest = json.loads((case_dir / "assets/results/manifest.json").read_text())
    relative = metadata_path.relative_to(case_dir).as_posix()
    entries = [entry for entry in manifest["files"] if entry["path"] == relative]
    expected_metadata = {
        "path": relative,
        "sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "size": len(metadata_bytes),
    }
    if (
        type(fingerprint.get("schema_version")) is not int
        or fingerprint.get("schema_version") != 1
        or len(entries) != 1
        or entries[0] != expected_metadata
        or fingerprint.get("metadata") != expected_metadata
        or fingerprint.get("clock") != {"step": 0, "time": 0.0}
        or fingerprint.get("initial_configuration") != declared
    ):
        raise ValueError("Ring initial-state fingerprint or released metadata identity is invalid.")
    count = fingerprint.get("particle_count")
    if type(count) is not int or count != len(cloud.position) or count <= 0:
        raise ValueError("Ring initial-state fingerprint particle count is invalid.")
    fields = fingerprint.get("fields", {})
    if set(fields) != {"position", "vortex_strength", "core_radius"}:
        raise ValueError("Ring initial-state fingerprint fields are incomplete.")
    for name, record in fields.items():
        shape = [count] if name == "core_radius" else [count, 3]
        dtype_name = record.get("dtype")
        if (
            dtype_name not in ("<f4", "<f8", ">f4", ">f8")
            or record.get("shape") != shape
            or np.dtype(dtype_name).itemsize != {"f32": 4, "f64": 8}[declared["precision"]]
        ):
            raise ValueError(f"Ring initial-state fingerprint {name} shape or dtype is invalid.")
        array = np.asarray(getattr(cloud, name))
        if list(array.shape) != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"Reconstructed ring {name} is invalid.")
        if name == "core_radius" and np.any(array <= 0):
            raise ValueError("Reconstructed ring core radii must be positive.")
        stored = np.asarray(array, dtype=np.dtype(dtype_name), order="C")
        if not np.all(np.isfinite(stored)) or hashlib.sha256(
            stored.tobytes()
        ).hexdigest() != record.get("sha256"):
            raise ValueError(
                f"Reconstructed ring {name} differs from its recorded field fingerprint."
            )
    return "recorded_field_fingerprint_verified_reconstruction"


def tex_document(body, height):
    return (
        r"""\documentclass[11pt,tikz,border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{newpxtext,amsmath,newpxmath,graphicx}
\usetikzlibrary{arrows.meta}
\begin{document}
\begin{tikzpicture}[x=1mm,y=-1mm]
\path[use as bounding box] (0,0) rectangle (125,HEIGHT);
""".replace("HEIGHT", str(height))
        + body
        + "\n\\end{tikzpicture}\n\\end{document}\n"
    )


def export_document(work, output_dir, name, tex, figure_format, texbin):
    """Compile the labelled scene and retain the requested figure format."""
    (work / f"{name}.tex").write_text(tex)
    subprocess.run(
        [texbin, "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
        cwd=work,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    shutil.copy2(work / f"{name}.tex", output_dir / f"{name}.tex")
    if figure_format in ("png", "both"):
        pdftoppm = find_executable("pdftoppm")
        subprocess.run(
            [
                pdftoppm,
                "-png",
                "-r",
                "400",
                "-singlefile",
                str(work / f"{name}.pdf"),
                str(output_dir / name),
            ],
            check=True,
        )
    if figure_format in ("pdf", "both"):
        shutil.copy2(work / f"{name}.pdf", output_dir / f"{name}.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--available",
        action="store_true",
        help="Skip scenes until the final LES backup is available.",
    )
    parser.add_argument(
        "--schematic-only",
        action="store_true",
        help="Rebuild only the initial geometry diagram, still validating saved-run provenance.",
    )
    parser.add_argument("--output-dir", type=Path, default=CASE / "figures")
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    parser.add_argument("--pvpython")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pvbin = find_executable("pvpython", args.pvpython)
    texbin = find_executable("pdflatex")
    meta_path = CASE / "solution/les_transposed/vpm_metadata.json"
    if args.available and not meta_path.is_file():
        print("Skipping ring scenes until the LES run is available.")
        return
    meta = json.loads(meta_path.read_text())
    state = meta.get("state", {})
    status = meta.get("lifecycle", {}).get("status")
    completed_steps = int(state.get("step", -1))
    final_backup = CASE / "solution/les_transposed/vpm" / f"vpm_{completed_steps:06d}.h5"
    if args.available and (status != "completed" or not final_backup.is_file()):
        print("Skipping ring scenes until the final LES backup is available.")
        return
    if status != "completed":
        raise ValueError("A completed LES run is required for the final panel.")
    ic = initial_cloud()
    initial_provenance = validate_scene_initial_cloud(ic, CASE)
    print(f"Ring initial scene provenance: {initial_provenance}")
    path = final_backup
    with h5py.File(path) as f:
        p1 = f["particles/position"][:].astype(float)
        a1 = f["particles/vortex_strength"][:].astype(float)
        final_time = float(f["solver"].attrs["time"])
    if not np.isclose(final_time, float(state["time"])):
        raise ValueError("Snapshot time differs from run metadata.")
    if len(ic) != int(state["initial_n_particles_total"]):
        raise ValueError("Initial particle count differs from metadata.")
    strength_ref = np.linalg.norm(ic.vortex_strength, axis=1).max()
    colors = colormaps[theme.COLORMAPS["vorticity_magnitude"]](np.linspace(0, 1, 65))[:, :3]
    scene = {
        "variant": "les_transposed",
        "initial_provenance": initial_provenance,
        "initial_maximum_strength": float(strength_ref),
        "color_limits": [0.04, 0.80],
        "sphere_radius_rule": "0.025 R0 (|alpha|/max|alpha_0|)^0.65",
        "camera_position": [5, 0, 7],
        "camera_up": [0, 0, 1],
        "parallel_scale": 0.82,
        "normalized_times": [0, final_time * s.RING_STRENGTH / s.RING_RADIUS**2],
        "rgb_points": np.column_stack([np.linspace(0.04, 0.80, 65), colors]).ravel().tolist(),
        "centroids": [],
        "counts": [len(ic), len(p1)],
        "schematic_only": args.schematic_only,
    }
    with tempfile.TemporaryDirectory(prefix="vortex-ring-figures-") as temp:
        work = Path(temp)
        for i, (p, a) in enumerate([(ic.position, ic.vortex_strength), (p1, a1)]):
            weights = np.linalg.norm(a, axis=1)
            centroid = np.average(p, axis=0, weights=weights)
            scene["centroids"].append(centroid.tolist())
            data = pv.PolyData((p - centroid) / s.RING_RADIUS)
            data["strength"] = weights / strength_ref
            data["radius"] = 0.025 * (weights / strength_ref) ** 0.65
            data["schematic_radius"] = np.full(len(p), 0.008)
            data.save(work / f"particles_{i}.vtp")
        # This is an illustrative outer envelope, not a Gaussian isosurface.
        # Expose a real azimuthal particle layer and repeat it in the close-up.
        p = ic.position / s.RING_RADIUS
        angles = np.arctan2(p[:, 2], p[:, 1])
        sections = np.unique(np.round(angles, 5))
        cut = float(sections[np.argmin(abs(sections - 0.35))])
        selected = abs(angles - cut) < 1e-4
        radial = np.array([0, np.cos(cut), np.sin(cut)])
        axial = np.array([1.0, 0, 0])
        center = radial.copy()
        minor = s.CORE_RADIUS / s.RING_RADIUS
        envelope = 1.45 * minor
        theta = np.linspace(cut, 2 * np.pi - 0.60, 241)
        phi = np.linspace(0, 2 * np.pi, 81)
        tt, pp = np.meshgrid(theta, phi, indexing="ij")
        core = pv.StructuredGrid(
            envelope * np.cos(pp),
            (1 + envelope * np.sin(pp)) * np.cos(tt),
            (1 + envelope * np.sin(pp)) * np.sin(tt),
        ).extract_surface(algorithm="dataset_surface")
        core.save(work / "core_envelope.vtp")

        # Circular section boundaries inherit the same 3-D camera projection.
        def circle(origin, radius, name, tube):
            points = origin + radius * (
                np.cos(phi)[:, None] * axial + np.sin(phi)[:, None] * radial
            )
            pv.lines_from_points(points).tube(radius=tube, n_sides=16).save(work / name)

        circle(center, envelope, "cut_edge.vtp", 0.004)
        circle(center, minor, "core_scale.vtp", 0.0034)
        for angle in [-0.60]:
            r = np.array([0, np.cos(angle), np.sin(angle)])
            pts = r + envelope * (np.cos(phi)[:, None] * axial + np.sin(phi)[:, None] * r)
            pv.lines_from_points(pts).tube(radius=0.004, n_sides=16).save(work / "other_edge.vtp")
        detail = pv.PolyData(p[selected] - center)
        detail["schematic_radius"] = np.full(selected.sum(), 0.008)
        detail.save(work / "core_particles.vtp")
        circle(np.zeros(3), minor, "detail_scale.vtp", 0.0015)
        pv.Arrow(
            start=0.025 * np.array([5.0, -2, 3]) / np.sqrt(38),
            direction=axial,
            scale=minor,
            tip_length=0.22,
            tip_radius=0.045,
            shaft_radius=0.012,
        ).save(work / "core_radius_arrow.vtp")
        # A view-direction depth bias keeps the dimension arrow visible over particles.
        pv.Arrow(
            start=0.20 * np.array([5.0, -2, 3]) / np.sqrt(38),
            direction=[0, -1, 0],
            scale=1,
            tip_length=0.13,
            tip_radius=0.045,
            shaft_radius=0.012,
        ).save(work / "radius_arrow.vtp")
        pv.Arrow(
            start=[0, 0, 0],
            direction=[1, 0, 0],
            scale=1.45,
            tip_length=0.14,
            tip_radius=0.045,
            shaft_radius=0.011,
        ).save(work / "speed_arrow.vtp")
        theta = np.linspace(-2.8, -0.5, 121)
        pts = np.c_[np.full(len(theta), 0.23), 1.30 * np.cos(theta), 1.30 * np.sin(theta)]
        arc = pv.lines_from_points(pts).tube(radius=0.016, n_sides=24)
        direction = np.array([0, -np.sin(theta[-1]), np.cos(theta[-1])])
        tip = pv.Cone(
            center=pts[-1] - 0.08 * direction,
            direction=direction,
            height=0.16,
            radius=0.055,
            resolution=32,
        )
        arc.merge(tip).save(work / "circulation_arrow.vtp")
        scene["schematic"] = {
            "camera_position": [5, -2, 3],
            "parallel_scale": 1.30,
            "focal_offset_up": -0.10,
            "envelope_radius_over_R0": envelope,
            "core_radius_over_R0": minor,
            "cut_angle_radians": cut,
            "section_particles": int(selected.sum()),
            "envelope_meaning": "Illustrative outer surface; not the Gaussian core radius or a material boundary.",
        }
        # Orthographic projection, in millimetres on the final LaTeX canvas.
        camera = np.array([5.0, -2, 3])
        forward = -camera / np.linalg.norm(camera)
        right = np.cross(forward, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        def project(point, origin, width, height, scale):
            q = np.array(point) + 0.10 * up
            factor = height / (2 * scale)
            return np.array(origin) + [
                width / 2 + factor * np.dot(q, right),
                height / 2 - factor * np.dot(q, up),
            ]

        face = project(center, (0, 0), 94, 82, 1.30)
        scene["schematic"]["section_anchor_mm"] = face.tolist()
        render_command = [
            pvbin,
            str(Path(__file__).with_name("render_vortex_ring.py")),
            str(work),
            json.dumps(scene["rgb_points"]),
        ]
        if args.schematic_only:
            render_command.append("--schematic-only")
        subprocess.run(
            render_command,
            check=True,
        )
        # Keep image paths local so the exported LaTeX remains portable.
        for name in (
            ["schematic", "core_detail"]
            if args.schematic_only
            else ["particles_0", "particles_1", "schematic", "core_detail"]
        ):
            # The raw schematic must not collide with the labelled PNG export.
            asset_name = "schematic_raw" if name == "schematic" else name
            shutil.copy2(work / f"{name}.png", args.output_dir / f"vortex_ring_{asset_name}.png")
            shutil.copy2(work / f"{name}.png", work / f"vortex_ring_{asset_name}.png")
        body = r"""\node[anchor=north west,inner sep=0] at (0,0) {\includegraphics[width=62.5mm]{vortex_ring_particles_0.png}};
\node[anchor=north west,inner sep=0] at (62.5,0) {\includegraphics[width=62.5mm]{vortex_ring_particles_1.png}};
\node[anchor=north west] at (1,0.5) {(a)};
\node[anchor=north west] at (63.5,0.5) {(b)};
\node[anchor=south east,font=\normalsize] at (61.5,48.5) {$t\Gamma_0/R_0^2=0$};
\node[anchor=south east,font=\normalsize] at (124,48.5) {$t\Gamma_0/R_0^2=FINALTIME$};
\draw[line width=.6pt] (3,46.5) -- (15.70,46.5);
\draw[line width=.6pt] (3,45.8) -- (3,47.2) (15.70,45.8) -- (15.70,47.2);
\node[anchor=south,font=\normalsize] at (9.35,46) {$0.5R_0$};
""".replace("FINALTIME", f"{float(format(scene['normalized_times'][1], '.2g')):g}")
        # Vector colour bar from the identical ParaView transfer-function samples.
        for j, rgb in enumerate(colors):
            body += f"\\definecolor{{bar{j}}}{{rgb}}{{{rgb[0]:.6f},{rgb[1]:.6f},{rgb[2]:.6f}}}\n"
            body += f"\\fill[bar{j}] ({38 + j * 49 / 65:.5f},52) rectangle ({38 + (j + 1) * 49 / 65:.5f},54.5);\n"
        for value in [0.04, 0.2, 0.4, 0.6, 0.8]:
            x = 38 + (value - 0.04) / 0.76 * 49
            label = f"{value:g}"
            if value == 0.04:
                label = r"\leq0.04"
            if value == 0.8:
                label = r"\geq0.8"
            body += f"\\node[anchor=north,font=\\normalsize] at ({x},54.5) {{${label}$}};\n"
        body += r"\node[anchor=north,font=\normalsize] at (62.5,58.5) {$|\boldsymbol{\alpha}_p|/\max_q|\boldsymbol{\alpha}_{q,0}|$};"
        reconstructed = initial_provenance == "recorded_field_fingerprint_verified_reconstruction"
        if reconstructed:
            body += r"\node[anchor=south,font=\footnotesize] at (62.5,69) {Initial panel reconstructed; recorded field fingerprints verified.};"
        snapshot = tex_document(body, 71 if reconstructed else 66)
        schematic_body = r"""\node[anchor=north west,inner sep=0] at (0,0) {\includegraphics[width=94mm,height=82mm]{vortex_ring_schematic_raw.png}};
\node[anchor=north west,inner sep=0] at (90,8) {\includegraphics[width=35mm]{vortex_ring_core_detail.png}};
\node at (29,36) {$R_0$};
\node at (44,54) {$U_{\mathrm{ring}}\,\boldsymbol{e}_x$};
\node at (85,61) {$\boldsymbol{\omega}$};
\node[anchor=north] at (107.5,2) {Core section};
\node at (117,24) {$a_0$};
\node[align=center] at (106.5,45) {$a_0/R_0=0.10$};
\node at (106,57) {$\mathrm{Re}_\Gamma=\Gamma_0/\nu=3000$};
"""
        schematic_body += f"\\draw[cyan!65!black,densely dotted,line width=.6pt] ({face[0]:.3f},{face[1]:.3f}) -- (98,25.5);\n"
        if reconstructed:
            schematic_body += r"\node[anchor=south,font=\footnotesize] at (62.5,86) {Reconstructed initial state; recorded field fingerprints verified.};"
        schematic = tex_document(schematic_body, 88 if reconstructed else 82)
        documents = [("vortex_ring_schematic", schematic)]
        if not args.schematic_only:
            documents.insert(0, ("vortex_ring_snapshots", snapshot))
        for name, tex in documents:
            export_document(work, args.output_dir, name, tex, args.format, texbin)
        (args.output_dir / "vortex_ring_scene.json").write_text(json.dumps(scene, indent=2) + "\n")


if __name__ == "__main__":
    main()
