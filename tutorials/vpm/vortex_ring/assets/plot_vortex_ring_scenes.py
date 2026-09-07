"""Rebuild the ring schematic and paired particle view from the saved LES run.

Requires ParaView (PVPYTHON or installed pvpython), PyVista, and pdflatex.
Uses a temporary build directory and keeps only PNG, PDF, TeX and metadata.
No simulation samples or solver parameters are modified. Both snapshots use
one orthographic field of view and the same strength-to-radius/color maps.
"""

from pathlib import Path
import argparse, json, os, shutil, subprocess, tempfile
import h5py, numpy as np, pyvista as pv
from matplotlib import colormaps
from tutorials.vpm.vortex_ring import setup as s
import openonda.vpm as vpm

CASE = Path(__file__).resolve().parents[1]


def initial_cloud(meta):
    expected = {
        "ring_radius": s.RING_RADIUS,
        "ring_circulation": s.RING_STRENGTH,
        "core_radius": s.CORE_RADIUS,
        "particle_spacing": s.PARTICLE_SPACING,
        "particle_core_radius": 2 * s.PARTICLE_SPACING,
        "widnall_amplitude": s.DEFAULT_WIDNALL_AMPLITUDE,
        "widnall_modes": s.WIDNALL_MODES,
        "random_seed": s.RANDOM_SEED,
    }
    for key, value in expected.items():
        if not np.isclose(meta[key], value, rtol=0, atol=1e-12):
            raise ValueError(f"Setup differs from the saved run: {key}")
    sigma = meta["particle_core_radius"]
    distribution = vpm.ToroidalDistribution(
        ring_radius=s.RING_RADIUS,
        tube_radius=np.sqrt(s.CORE_RADIUS**2 - sigma**2)
        * np.sqrt(-np.log(s.TOROIDAL_TAIL_FRACTION)),
        spacing=s.PARTICLE_SPACING,
        core_radius_ratio=sigma / s.PARTICLE_SPACING,
    )
    return vpm.VortexRing(
        kinematic_viscosity=s.KINEMATIC_VISCOSITY,
        radius=s.RING_RADIUS,
        circulation=s.RING_STRENGTH,
        vortex_core_radius=s.CORE_RADIUS,
        disturbance=vpm.WidnallDisturbance.broadband(
            amplitude=s.DEFAULT_WIDNALL_AMPLITUDE,
            number_of_modes=s.WIDNALL_MODES,
            seed=s.RANDOM_SEED,
        ),
        core_compensation=vpm.ParticleCoreCompensation(),
        distribution=distribution,
        group_id=0,
    ).build()


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--available",
        action="store_true",
        help="Skip scenes until the final LES backup is available.",
    )
    parser.add_argument(
        "--schematic-only", action="store_true", help="Rebuild only the initial geometry diagram."
    )
    parser.add_argument("--output-dir", type=Path, default=CASE / "figures")
    parser.add_argument("--pvpython", default=os.environ.get("PVPYTHON"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pvbin = args.pvpython or shutil.which("pvpython")
    if not pvbin:
        candidates = sorted(Path("/Applications").glob("ParaView*.app/Contents/bin/pvpython"))
        if candidates:
            pvbin = str(candidates[-1])
    if not pvbin:
        raise FileNotFoundError("Set PVPYTHON to the ParaView pvpython executable.")
    texbin = shutil.which("pdflatex") or "/Library/TeX/texbin/pdflatex"
    meta_path = CASE / "samples/les_transposed/run_metadata.json"
    if args.available and not meta_path.is_file():
        print("Skipping ring scenes until the LES run is available.")
        return
    meta = json.loads(meta_path.read_text())
    if args.available and (
        meta.get("status") != "horizon_reached"
        or not (
            CASE / "solution/les_transposed" / f"vpm_{meta['completed_steps']:06d}.h5"
        ).is_file()
    ):
        print("Skipping ring scenes until the final LES backup is available.")
        return
    if meta["status"] != "horizon_reached":
        raise ValueError("A completed LES run is required for the final panel.")
    ic = initial_cloud(meta)
    path = CASE / "solution/les_transposed" / f"vpm_{meta['completed_steps']:06d}.h5"
    with h5py.File(path) as f:
        p1 = f["particles/position"][:].astype(float)
        a1 = f["particles/vortex_strength"][:].astype(float)
        final_time = float(f["solver"].attrs["time"])
    if not np.isclose(final_time, meta["final_time"]):
        raise ValueError("Snapshot time differs from run metadata.")
    if len(ic) != meta["initial_n_particles_total"]:
        raise ValueError("Initial particle count differs from metadata.")
    strength_ref = np.linalg.norm(ic.vortex_strength, axis=1).max()
    colors = colormaps["viridis"](np.linspace(0, 1, 65))[:, :3]
    scene = {
        "variant": "les_transposed",
        "initial_maximum_strength": float(strength_ref),
        "color_limits": [0.04, 0.80],
        "sphere_radius_rule": "0.025 R0 (|alpha|/max|alpha_0|)^0.65",
        "camera_position": [5, 0, 3],
        "camera_up": [0, 0, 1],
        "parallel_scale": 1.06,
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
        circle(center, minor, "core_scale.vtp", 0.003)
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
        (work / "scene.json").write_text(json.dumps(scene, indent=2))
        subprocess.run(
            [pvbin, str(Path(__file__).with_name("render_vortex_ring.py")), str(work)], check=True
        )
        # Keep image paths local so the exported LaTeX remains portable.
        for name in (
            ["schematic", "core_detail"]
            if args.schematic_only
            else ["particles_0", "particles_1", "schematic", "core_detail"]
        ):
            shutil.copy2(work / f"{name}.png", args.output_dir / f"vortex_ring_{name}.png")
            shutil.copy2(work / f"{name}.png", work / f"vortex_ring_{name}.png")
        body = r"""\node[anchor=north west,inner sep=0] at (0,0) {\includegraphics[width=62.5mm]{vortex_ring_particles_0.png}};
\node[anchor=north west,inner sep=0] at (62.5,0) {\includegraphics[width=62.5mm]{vortex_ring_particles_1.png}};
\node[anchor=north west] at (1,0) {(a)};
\node[anchor=north west] at (63.5,0) {(b)};
\node[anchor=south east,font=\normalsize] at (61.5,56.5) {$t\Gamma_0/R_0^2=0$};
\node[anchor=south east,font=\normalsize] at (124,56.5) {$t\Gamma_0/R_0^2=FINALTIME$};
\draw[line width=.6pt] (3,54.5) -- (15.28,54.5);
\draw[line width=.6pt] (3,53.8) -- (3,55.2) (15.28,53.8) -- (15.28,55.2);
\node[anchor=south,font=\normalsize] at (9.14,54) {$0.5R_0$};
""".replace("FINALTIME", f"{float(format(scene['normalized_times'][1], '.2g')):g}")
        # Vector colour bar from the identical ParaView transfer-function samples.
        for j, rgb in enumerate(colors):
            body += f"\\definecolor{{bar{j}}}{{rgb}}{{{rgb[0]:.6f},{rgb[1]:.6f},{rgb[2]:.6f}}}\n"
            body += f"\\fill[bar{j}] ({38 + j * 49 / 65:.5f},60) rectangle ({38 + (j + 1) * 49 / 65:.5f},62.5);\n"
        for value in [0.04, 0.2, 0.4, 0.6, 0.8]:
            x = 38 + (value - 0.04) / 0.76 * 49
            label = f"{value:g}"
            if value == 0.04:
                label = r"\leq0.04"
            if value == 0.8:
                label = r"\geq0.8"
            body += f"\\node[anchor=north,font=\\normalsize] at ({x},62.5) {{${label}$}};\n"
        body += r"\node[anchor=north,font=\normalsize] at (62.5,67) {$|\boldsymbol{\alpha}_p|/\max_q|\boldsymbol{\alpha}_{q,0}|$};"
        snapshot = tex_document(body, 74)
        schematic_body = r"""\node[anchor=north west,inner sep=0] at (0,0) {\includegraphics[width=94mm,height=82mm]{vortex_ring_schematic.png}};
\node[anchor=north west,inner sep=0] at (90,8) {\includegraphics[width=35mm]{vortex_ring_core_detail.png}};
\node at (29,36) {$R_0$};
\node at (44,54) {$U_{\mathrm{ring}}\,\boldsymbol{e}_x$};
\node at (85,61) {$\boldsymbol{\omega}$};
\node[anchor=north] at (107.5,2) {Core section};
\node at (117,32) {$a_0$};
\node[align=center] at (106.5,45) {$a_0/R_0=0.10$};
\node at (106,57) {$\mathrm{Re}_\Gamma=\Gamma_0/\nu=3000$};
"""
        schematic_body += f"\\draw[cyan!65!black,densely dotted,line width=.55pt] ({face[0]:.3f},{face[1]:.3f}) -- (98,25.5);\n"
        schematic = tex_document(schematic_body, 82)
        documents = [("vortex_ring_schematic", schematic)]
        if not args.schematic_only:
            documents.insert(0, ("vortex_ring_snapshots", snapshot))
        for name, tex in documents:
            (work / f"{name}.tex").write_text(tex)
            subprocess.run(
                [texbin, "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
                cwd=work,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            for ext in ["tex", "pdf"]:
                shutil.copy2(work / f"{name}.{ext}", args.output_dir / f"{name}.{ext}")
        (args.output_dir / "vortex_ring_scenes.json").write_text(json.dumps(scene, indent=2) + "\n")


if __name__ == "__main__":
    main()
