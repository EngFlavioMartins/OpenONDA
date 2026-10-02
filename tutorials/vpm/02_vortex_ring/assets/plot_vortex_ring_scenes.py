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
import argparse, json, shutil, subprocess, tempfile
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
    """Require the saved step-zero particle fields and matching configuration."""
    case_dir = Path(case_dir)
    metadata = json.loads((case_dir / "solution/les_transposed/vpm_metadata.json").read_text())
    if initial_configuration(metadata["configuration"]) != initial_configuration():
        raise ValueError("Current ring initialization configuration differs from saved metadata.")
    validate_initial_cloud(cloud, case_dir / "solution/les_transposed/vpm/vpm_000000.h5")
    return "saved_step_zero_verified"


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
        "camera_position": [4, -5, 1],
        "camera_up": [0, 0, 1],
        "parallel_scale": 1.18,
        "focal_offset_up": 0.02,
        "focal_offset_right": 0.20,
        "snapshot_panel_mm": [62.5, 40.0],
        "snapshot_view_size_px": [2500, 1600],
        "motion_arrow_direction": [1, 0, 0],
        "arrow_color": [0.64, 0.64, 0.64],
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
        # Side view: +x projects to the right, with a small downward slope.
        # Use the same physical arrow and orthographic scale in both panels.
        snapshot_camera = np.array(scene["camera_position"])
        forward = -snapshot_camera / np.linalg.norm(snapshot_camera)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        assert right[0] > 0
        snapshot_focus = scene["focal_offset_up"] * up + scene["focal_offset_right"] * right
        scene["camera_focal_point"] = snapshot_focus.tolist()
        scene["camera_position"] = (snapshot_camera + snapshot_focus).tolist()
        # Retain every sphere in the common, fixed orthographic viewport.
        for i in (0, 1):
            cloud = pv.read(work / f"particles_{i}.vtp")
            projected = (cloud.points - snapshot_focus) @ np.array([right, up]).T
            radii = cloud["radius"][:, None]
            half_extent = np.array([62.5 / 40.0, 1.0]) * scene["parallel_scale"]
            assert np.all(abs(projected) + radii < half_extent)
        # Each translated particle cloud is centred on its strength-weighted
        # centroid. The velocity vector starts at that same physical centre.
        arrow_start = np.zeros(3)
        scene["motion_arrow_start"] = arrow_start.tolist()
        scene["motion_arrow_length_over_R0"] = 1.50
        scene["snapshot_scale_bar_mm"] = 0.5 * 40.0 / (2 * scene["parallel_scale"])
        motion_arrow = pv.Arrow(
            start=arrow_start,
            direction=[1, 0, 0],
            scale=scene["motion_arrow_length_over_R0"],
            tip_length=0.24,
            tip_radius=0.070,
            shaft_radius=0.020,
        )
        projected_arrow = (motion_arrow.points - snapshot_focus) @ np.array([right, up]).T
        assert np.all(abs(projected_arrow) < half_extent)
        motion_arrow.save(work / "motion_arrow.vtp")
        tip = arrow_start + [scene["motion_arrow_length_over_R0"], 0, 0]
        factor = 40.0 / (2 * scene["parallel_scale"])
        tip_mm = np.array([31.25, 20.0]) + factor * np.array(
            [
                (tip - snapshot_focus) @ right,
                -(tip - snapshot_focus) @ up,
            ]
        )
        scene["motion_arrow_tip_mm"] = tip_mm.tolist()
        scene["motion_arrow_label_mm"] = (tip_mm + [-0.8, -3.5]).tolist()
        # Pass camera metadata to ParaView rather than maintaining a second copy.
        (work / "scene.json").write_text(json.dumps(scene))
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
            # The shaft ends inside the cone base, safely behind its tip.
            center=pts[-1] + 0.05 * direction,
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
\node[anchor=south east,font=\normalsize] at (61.5,45) {$t\Gamma_0/R_0^2=0$};
\node[anchor=south east,font=\normalsize] at (124,45) {$t\Gamma_0/R_0^2=FINALTIME$};
""".replace("FINALTIME", f"{float(format(scene['normalized_times'][1], '.2g')):g}")
        label_x, label_y = scene["motion_arrow_label_mm"]
        for offset in (0, 62.5):
            body += f"\\node[anchor=south west,inner sep=0,font=\\normalsize] at ({offset + label_x:.4f},{label_y:.4f}) {{$U_{{\\mathrm{{ring}}}}$}};\n"
        scale_start = 3.0
        scale_end = scale_start + scene["snapshot_scale_bar_mm"]
        body += f"\\draw[line width=.6pt] ({scale_start},44.5) -- ({scale_end:.5f},44.5);\n"
        body += f"\\draw[line width=.6pt] ({scale_start},43.8) -- ({scale_start},45.2) ({scale_end:.5f},43.8) -- ({scale_end:.5f},45.2);\n"
        body += f"\\node[anchor=south,font=\\normalsize] at ({(scale_start + scale_end) / 2:.5f},44) {{$0.5R_0$}};\n"
        # Vector colour bar from the identical ParaView transfer-function samples.
        for j, rgb in enumerate(colors):
            body += f"\\definecolor{{bar{j}}}{{rgb}}{{{rgb[0]:.6f},{rgb[1]:.6f},{rgb[2]:.6f}}}\n"
            body += f"\\fill[bar{j}] ({38 + j * 49 / 65:.5f},48) rectangle ({38 + (j + 1) * 49 / 65:.5f},50.4);\n"
        for value in [0.04, 0.2, 0.4, 0.6, 0.8]:
            x = 38 + (value - 0.04) / 0.76 * 49
            label = f"{value:g}"
            if value == 0.04:
                label = r"\leq0.04"
            if value == 0.8:
                label = r"\geq0.8"
            body += f"\\node[anchor=north,font=\\normalsize] at ({x},50.4) {{${label}$}};\n"
        body += r"\node[anchor=north,font=\normalsize] at (62.5,54.4) {$|\boldsymbol{\alpha}_p|/\max_q|\boldsymbol{\alpha}_{q,0}|$};"
        reconstructed = initial_provenance == "recorded_field_fingerprint_verified_reconstruction"
        if reconstructed:
            body += r"\node[anchor=north,font=\footnotesize] at (62.5,60.3) {Initial state reconstructed and fingerprint-verified.};"
        snapshot = tex_document(body, 64.5 if reconstructed else 61.0)
        schematic_body = r"""\node[anchor=north west,inner sep=0] at (0,0) {\includegraphics[width=94mm,height=82mm]{vortex_ring_schematic_raw.png}};
\node[anchor=north west,inner sep=0] at (90,8) {\includegraphics[width=35mm]{vortex_ring_core_detail.png}};
\node at (29,36) {$R_0$};
\node[anchor=east,inner sep=.5mm] at (57.5,55.5) {$U_{\mathrm{ring}}\,\boldsymbol{e}_x$};
\node at (85,61) {$\boldsymbol{\omega}$};
\node[anchor=north] at (107.5,2) {Core section};
\node at (117,24) {$a_0$};
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
