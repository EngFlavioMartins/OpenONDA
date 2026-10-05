"""Rebuild the ring schematic and paired particle view from the saved LES run.

Requires ParaView (PVPYTHON or installed pvpython), PyVista, and pdflatex;
PNG export also requires pdftoppm. The temporary build produces PNG (default)
or PDF figures, retaining their TeX sources and rendered image assets.
No simulation samples or solver parameters are modified. Both snapshots use
one orthographic field of view and the same strength-to-radius/color maps.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
from matplotlib import colormaps

from openonda import plotting as theme
from openonda.results import read_json, write_text
from openonda.scenes import export_scene
from source.solvers.vpm.io.postprocess import particle_state

from .. import setup as s

CASE = Path(__file__).resolve().parents[1]


def tex_document(body, height):
    return (
        "\\documentclass[11pt,tikz,border=0pt]{standalone}\n\\usepackage[T1]{fontenc}\n\\usepackage{newpxtext,amsmath,newpxmath,graphicx}\n\\usetikzlibrary{arrows.meta}\n\\begin{document}\n\\begin{tikzpicture}[x=1mm,y=-1mm]\n\\path[use as bounding box] (0,0) rectangle (125,HEIGHT);\n".replace(
            "HEIGHT", str(height)
        )
        + body
        + "\n\\end{tikzpicture}\n\\end{document}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schematic-only", action="store_true", help="Rebuild only the initial geometry diagram."
    )
    parser.add_argument("--output-dir", type=Path, default=CASE / "figures")
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    args = parser.parse_args()
    meta_path = CASE / "solution/les_transposed/vpm_metadata.json"
    meta = read_json(meta_path)
    completed_steps = int(meta["state"]["step"])
    final_backup = CASE / "solution/les_transposed/vpm" / f"vpm_{completed_steps:06d}.h5"
    initial = particle_state(CASE / "solution/les_transposed/vpm/vpm_000000.h5")
    final = particle_state(final_backup)
    ic = SimpleNamespace(
        position=initial["position"],
        vortex_strength=initial["vortex_strength"],
        core_radius=initial["core_radius"],
    )
    p1, a1, final_time = (final["position"], final["vortex_strength"], final["time"])
    strength_ref = np.linalg.norm(ic.vortex_strength, axis=1).max()
    colors = colormaps[theme.COLORMAPS["vorticity_magnitude"]](np.linspace(0, 1, 65))[:, :3]
    scene = {
        "variant": "les_transposed",
        "initial_maximum_strength": float(strength_ref),
        "color_limits": [0.04, 0.8],
        "sphere_radius_rule": "0.025 R0 (|alpha|/max|alpha_0|)^0.65",
        "camera_position": [4, -5, 1],
        "camera_up": [0, 0, 1],
        "parallel_scale": 1.18,
        "focal_offset_up": 0.02,
        "focal_offset_right": 0.2,
        "snapshot_panel_mm": [62.5, 40.0],
        "snapshot_view_size_px": [2500, 1600],
        "motion_arrow_direction": [1, 0, 0],
        "arrow_color": [0.64, 0.64, 0.64],
        "normalized_times": [0, final_time * s.RING_STRENGTH / s.RING_RADIUS**2],
        "rgb_points": np.column_stack([np.linspace(0.04, 0.8, 65), colors]).ravel().tolist(),
        "centroids": [],
        "counts": [len(ic.position), len(p1)],
        "schematic_only": args.schematic_only,
    }
    export_scene(
        args.output_dir,
        lambda work: scene_assets(work, args, ic, p1, a1, colors, strength_ref, scene),
        dpi=400,
    )


def scene_assets(work, args, ic, p1, a1, colors, strength_ref, scene):
    for i, (p, a) in enumerate([(ic.position, ic.vortex_strength), (p1, a1)]):
        weights = np.linalg.norm(a, axis=1)
        centroid = np.average(p, axis=0, weights=weights)
        scene["centroids"].append(centroid.tolist())
        data = pv.PolyData((p - centroid) / s.RING_RADIUS)
        data["strength"] = weights / strength_ref
        data["radius"] = 0.025 * (weights / strength_ref) ** 0.65
        data["schematic_radius"] = np.full(len(p), 0.008)
        data.save(work / f"particles_{i}.vtp")
    snapshot_camera = np.array(scene["camera_position"])
    forward = -snapshot_camera / np.linalg.norm(snapshot_camera)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    snapshot_focus = scene["focal_offset_up"] * up + scene["focal_offset_right"] * right
    scene["camera_focal_point"] = snapshot_focus.tolist()
    scene["camera_position"] = (snapshot_camera + snapshot_focus).tolist()
    arrow_start = np.zeros(3)
    scene["motion_arrow_start"] = arrow_start.tolist()
    scene["motion_arrow_length_over_R0"] = 1.5
    scene["snapshot_scale_bar_mm"] = 0.5 * 40.0 / (2 * scene["parallel_scale"])
    motion_arrow = pv.Arrow(
        start=arrow_start,
        direction=[1, 0, 0],
        scale=scene["motion_arrow_length_over_R0"],
        tip_length=0.24,
        tip_radius=0.07,
        shaft_radius=0.02,
    )
    motion_arrow.save(work / "motion_arrow.vtp")
    tip = arrow_start + [scene["motion_arrow_length_over_R0"], 0, 0]
    factor = 40.0 / (2 * scene["parallel_scale"])
    tip_mm = np.array([31.25, 20.0]) + factor * np.array(
        [(tip - snapshot_focus) @ right, -(tip - snapshot_focus) @ up]
    )
    scene["motion_arrow_tip_mm"] = tip_mm.tolist()
    scene["motion_arrow_label_mm"] = (tip_mm + [-0.8, -3.5]).tolist()
    write_text(work / "scene.json", json.dumps(scene))
    p = ic.position / s.RING_RADIUS
    angles = np.arctan2(p[:, 2], p[:, 1])
    sections = np.unique(np.round(angles, 5))
    cut = float(sections[np.argmin(abs(sections - 0.35))])
    selected = abs(angles - cut) < 0.0001
    radial = np.array([0, np.cos(cut), np.sin(cut)])
    axial = np.array([1.0, 0, 0])
    center = radial.copy()
    minor = s.CORE_RADIUS / s.RING_RADIUS
    envelope = 1.45 * minor
    theta = np.linspace(cut, 2 * np.pi - 0.6, 241)
    phi = np.linspace(0, 2 * np.pi, 81)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    core = pv.StructuredGrid(
        envelope * np.cos(pp),
        (1 + envelope * np.sin(pp)) * np.cos(tt),
        (1 + envelope * np.sin(pp)) * np.sin(tt),
    ).extract_surface(algorithm="dataset_surface")
    core.save(work / "core_envelope.vtp")

    def circle(origin, radius, name, tube):
        points = origin + radius * (np.cos(phi)[:, None] * axial + np.sin(phi)[:, None] * radial)
        pv.lines_from_points(points).tube(radius=tube, n_sides=16).save(work / name)

    circle(center, envelope, "cut_edge.vtp", 0.004)
    circle(center, minor, "core_scale.vtp", 0.0034)
    for angle in [-0.6]:
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
    pv.Arrow(
        start=0.2 * np.array([5.0, -2, 3]) / np.sqrt(38),
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
    pts = np.c_[np.full(len(theta), 0.23), 1.3 * np.cos(theta), 1.3 * np.sin(theta)]
    arc = pv.lines_from_points(pts).tube(radius=0.016, n_sides=24)
    direction = np.array([0, -np.sin(theta[-1]), np.cos(theta[-1])])
    tip = pv.Cone(
        center=pts[-1] + 0.05 * direction,
        direction=direction,
        height=0.16,
        radius=0.055,
        resolution=32,
    )
    arc.merge(tip).save(work / "circulation_arrow.vtp")
    scene["schematic"] = {
        "camera_position": [5, -2, 3],
        "parallel_scale": 1.3,
        "focal_offset_up": -0.1,
        "envelope_radius_over_R0": envelope,
        "core_radius_over_R0": minor,
        "cut_angle_radians": cut,
        "section_particles": int(selected.sum()),
        "envelope_meaning": "Illustrative outer surface; not the Gaussian core radius or a material boundary.",
    }
    camera = np.array([5.0, -2, 3])
    forward = -camera / np.linalg.norm(camera)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    def project(point, origin, width, height, scale):
        q = np.array(point) + 0.1 * up
        factor = height / (2 * scale)
        return np.array(origin) + [
            width / 2 + factor * np.dot(q, right),
            height / 2 - factor * np.dot(q, up),
        ]

    face = project(center, (0, 0), 94, 82, 1.3)
    scene["schematic"]["section_anchor_mm"] = face.tolist()
    render_arguments = [str(work), json.dumps(scene["rgb_points"])]
    if args.schematic_only:
        render_arguments.append("--schematic-only")
    images = {
        f"{name}.png": f"vortex_ring_{'schematic_raw' if name == 'schematic' else name}.png"
        for name in (
            ["schematic", "core_detail"]
            if args.schematic_only
            else ["particles_0", "particles_1", "schematic", "core_detail"]
        )
    }
    body = "\\node[anchor=north west,inner sep=0] at (0,0) {\\includegraphics[width=62.5mm]{vortex_ring_particles_0.png}};\n\\node[anchor=north west,inner sep=0] at (62.5,0) {\\includegraphics[width=62.5mm]{vortex_ring_particles_1.png}};\n\\node[anchor=north west] at (1,0.5) {(a)};\n\\node[anchor=north west] at (63.5,0.5) {(b)};\n\\node[anchor=south east,font=\\normalsize] at (61.5,45) {$t\\Gamma_0/R_0^2=0$};\n\\node[anchor=south east,font=\\normalsize] at (124,45) {$t\\Gamma_0/R_0^2=FINALTIME$};\n".replace(
        "FINALTIME", f"{float(format(scene['normalized_times'][1], '.2g')):g}"
    )
    label_x, label_y = scene["motion_arrow_label_mm"]
    for offset in (0, 62.5):
        body += f"\\node[anchor=south west,inner sep=0,font=\\normalsize] at ({offset + label_x:.4f},{label_y:.4f}) {{$U_{{\\mathrm{{ring}}}}$}};\n"
    scale_start = 3.0
    scale_end = scale_start + scene["snapshot_scale_bar_mm"]
    body += f"\\draw[line width=.6pt] ({scale_start},44.5) -- ({scale_end:.5f},44.5);\n"
    body += f"\\draw[line width=.6pt] ({scale_start},43.8) -- ({scale_start},45.2) ({scale_end:.5f},43.8) -- ({scale_end:.5f},45.2);\n"
    body += f"\\node[anchor=south,font=\\normalsize] at ({(scale_start + scale_end) / 2:.5f},44) {{$0.5R_0$}};\n"
    for j, rgb in enumerate(colors):
        body += f"\\definecolor{{bar{j}}}{{rgb}}{{{rgb[0]:.6f},{rgb[1]:.6f},{rgb[2]:.6f}}}\n"
        body += f"\\fill[bar{j}] ({38 + j * 49 / 65:.5f},48) rectangle ({38 + (j + 1) * 49 / 65:.5f},50.4);\n"
    for value in [0.04, 0.2, 0.4, 0.6, 0.8]:
        x = 38 + (value - 0.04) / 0.76 * 49
        label = f"{value:g}"
        if value == 0.04:
            label = "\\leq0.04"
        if value == 0.8:
            label = "\\geq0.8"
        body += f"\\node[anchor=north,font=\\normalsize] at ({x},50.4) {{${label}$}};\n"
    body += "\\node[anchor=north,font=\\normalsize] at (62.5,54.4) {$|\\boldsymbol{\\alpha}_p|/\\max_q|\\boldsymbol{\\alpha}_{q,0}|$};"
    snapshot = tex_document(body, 61.0)
    schematic_body = "\\node[anchor=north west,inner sep=0] at (0,0) {\\includegraphics[width=94mm,height=82mm]{vortex_ring_schematic_raw.png}};\n\\node[anchor=north west,inner sep=0] at (90,8) {\\includegraphics[width=35mm]{vortex_ring_core_detail.png}};\n\\node at (29,36) {$R_0$};\n\\node[anchor=east,inner sep=.5mm] at (57.5,55.5) {$U_{\\mathrm{ring}}\\,\\boldsymbol{e}_x$};\n\\node at (85,61) {$\\boldsymbol{\\omega}$};\n\\node[anchor=north] at (107.5,2) {Core section};\n\\node at (117,24) {$a_0$};\n"
    schematic_body += f"\\draw[cyan!65!black,densely dotted,line width=.6pt] ({face[0]:.3f},{face[1]:.3f}) -- (98,25.5);\n"
    schematic = tex_document(schematic_body, 82)
    documents = [("vortex_ring_schematic", schematic)]
    if not args.schematic_only:
        documents.insert(0, ("vortex_ring_snapshots", snapshot))
    return {
        "renderer": (Path(__file__).with_name("render_vortex_ring.py"), render_arguments),
        "images": images,
        "documents": documents,
        "format": args.format,
        "metadata": {"vortex_ring_scene.json": scene},
    }


if __name__ == "__main__":
    main()
