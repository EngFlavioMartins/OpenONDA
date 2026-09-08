"""Depth-tested sphere views of the GBD merger, with vector LaTeX labels.

Chart contract: compare the initial and final finite columns, particle vorticity
and instantaneous in-plane streamlines. Analytic ray-sphere intersections use
an orthographic camera, surface lighting and a depth buffer. Sphere radii
scale with particle-strength magnitude to power 0.65 within each frame. Colour limits are
clipped per frame and displayed explicitly. No particles are filtered out.
Both states share one field of view enclosing both clouds and the sampling plane.
The 600-dpi scene is embedded in a 125 x 85 mm PDF with vector labels.
Viridis/plasma retain the original scientific-field color conventions.
"""

from pathlib import Path
import json
import sys
import argparse

import h5py
import numpy as np
from scipy.spatial import cKDTree
from numba import njit, prange
from math import erf, exp, sqrt, pi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

CASE_DIR = Path(__file__).resolve().parents[1]
OUT = CASE_DIR / "figures"
from .. import setup

_initial_conditions = setup._initial_conditions

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from ..assets.postprocess import load_theme

load_theme()
plt.rcParams.update({"axes.linewidth": 0.45, "pdf.compression": 9})


@njit(parallel=True, cache=True)
def sample_velocity(points, position, strength, sigma):
    """Direct Gaussian Biot--Savart sum for these purely axial strengths."""
    result = np.zeros((len(points), 2))
    for i in prange(len(points)):
        vx = 0.0
        vy = 0.0
        for j in range(len(position)):
            rx = points[i, 0] - position[j, 0]
            ry = points[i, 1] - position[j, 1]
            rz = points[i, 2] - position[j, 2]
            d = sqrt(rx * rx + ry * ry + rz * rz)
            if d > 1e-12:
                rho = d / sigma[j]
                cutoff = erf(rho) - 2 / sqrt(pi) * rho * exp(-rho * rho)
                weight = cutoff * strength[j, 2] / (4 * pi * d * d * d)
                vx -= ry * weight
                vy += rx * weight
        result[i, 0] = vx
        result[i, 1] = vy
    return result


def sample_vorticity(points, position, strength, sigma):
    """Gaussian sum truncated only beyond five maximum particle radii.

    The absolute omitted-tail bound is recorded in the output metadata.
    """
    tree = cKDTree(position)
    radius = 5 * float(np.max(sigma))
    result = np.empty(len(points))
    for start in range(0, len(points), 256):
        neighbours = tree.query_ball_point(points[start : start + 256], radius, workers=-1)
        for offset, indices in enumerate(neighbours):
            indices = np.asarray(indices, dtype=int)
            r = points[start + offset] - position[indices]
            s = sigma[indices]
            result[start + offset] = np.sum(
                np.exp(-np.sum(r * r, axis=1) / s**2) * strength[indices, 2] / (np.pi**1.5 * s**3)
            )
    return result


def project(points):
    view = np.array([0.70, 0.61, -0.37])
    view /= np.linalg.norm(view)
    right = np.array([0.0, 0.0, 1.0]) - view[2] * view
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    angle = np.deg2rad(25)
    basis = np.array(
        [np.cos(angle) * right - np.sin(angle) * up, np.sin(angle) * right + np.cos(angle) * up]
    )
    return points @ basis.T, points @ view


@njit(cache=True)
def render_spheres(points, radii, colors, lower, upper, width, height):
    """Orthographic visible sphere surfaces, including mutual occlusion."""
    canvas = np.ones((height, width, 3), dtype=np.float64)
    zbuffer = np.full((height, width), -np.inf)
    scale = (width - 1) / (upper[0] - lower[0])
    light = np.array([-0.45, 0.65, 0.65])
    light /= np.sqrt(np.sum(light * light))
    half = light + np.array([0.0, 0.0, 1.0])
    half /= np.sqrt(np.sum(half * half))
    for i in np.argsort(points[:, 2]):
        px = (points[i, 0] - lower[0]) * scale
        py = (upper[1] - points[i, 1]) * scale
        rp = radii[i] * scale
        for yy in range(max(0, int(py - rp - 1)), min(height, int(py + rp + 2))):
            for xx in range(max(0, int(px - rp - 1)), min(width, int(px + rp + 2))):
                dx = (xx - px) / rp
                dy = (py - yy) / rp
                q = dx * dx + dy * dy
                coverage = min(1.0, max(0.0, (1.0 - sqrt(q)) * rp + 0.5))
                if coverage <= 0:
                    continue
                nz = sqrt(max(0.0, 1.0 - min(q, 1.0)))
                depth = points[i, 2] + radii[i] * nz
                if depth < zbuffer[yy, xx]:
                    continue
                diffuse = max(0.0, dx * light[0] + dy * light[1] + nz * light[2])
                specular = 0.28 * max(0.0, dx * half[0] + dy * half[1] + nz * half[2]) ** 30
                for c in range(3):
                    shaded = min(1.0, colors[i, c] * (0.38 + 0.66 * diffuse) + specular)
                    canvas[yy, xx, c] = coverage * shaded + (1.0 - coverage) * canvas[yy, xx, c]
                if coverage > 0.5:
                    zbuffer[yy, xx] = depth
    return canvas, zbuffer


@njit(cache=True)
def render_streamlines(canvas, zbuffer, segments, colors, lower, upper, line_radius):
    height, width = canvas.shape[:2]
    scale = (width - 1) / (upper[0] - lower[0])
    for i in range(len(segments)):
        a = segments[i, 0]
        b = segments[i, 1]
        ax = (a[0] - lower[0]) * scale
        ay = (upper[1] - a[1]) * scale
        bx = (b[0] - lower[0]) * scale
        by = (upper[1] - b[1]) * scale
        length2 = (bx - ax) ** 2 + (by - ay) ** 2
        if length2 < 1e-12:
            continue
        for yy in range(
            max(0, int(min(ay, by) - line_radius - 1)),
            min(height, int(max(ay, by) + line_radius + 2)),
        ):
            for xx in range(
                max(0, int(min(ax, bx) - line_radius - 1)),
                min(width, int(max(ax, bx) + line_radius + 2)),
            ):
                f = min(1.0, max(0.0, ((xx - ax) * (bx - ax) + (yy - ay) * (by - ay)) / length2))
                distance = sqrt((xx - ax - f * (bx - ax)) ** 2 + (yy - ay - f * (by - ay)) ** 2)
                coverage = min(1.0, max(0.0, line_radius + 0.5 - distance))
                depth = a[2] + f * (b[2] - a[2])
                if coverage <= 0 or depth < zbuffer[yy, xx]:
                    continue
                for c in range(3):
                    canvas[yy, xx, c] = (
                        coverage * colors[i, c] + (1.0 - coverage) * canvas[yy, xx, c]
                    )
                if coverage > 0.5:
                    zbuffer[yy, xx] = depth
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field-cache", type=Path, help="Optional reusable NPZ of the reconstructed fields."
    )
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--format", choices=("pdf", "png", "both"), default="pdf")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = json.loads((CASE_DIR / "samples/merging_gbd/run_metadata.json").read_text())
    viscosity = float(run["kinematic_viscosity"])
    a0 = float(run["velocity_peak_radius"])
    circulation = abs(float(run["circulations"][0]))
    sample_z = float(run["sample_plane_z"])
    # Reconstruction must describe the saved run even if setup.py was edited later.
    expected = {
        "velocity_peak_radius": setup.CORE_RADIUS,
        "gaussian_core_radius": setup.GAUSSIAN_CORE_RADIUS,
        "particle_spacing": setup.SPACING,
        "particle_core_radius": setup.PARTICLE_RADIUS,
        "vortex_separation": setup.SEPARATION,
        "column_length": setup.COLUMN_LENGTH,
        "circulations": setup.PHYSICS_CIRCULATIONS["merging"],
    }
    for key, value in expected.items():
        if not np.allclose(run[key], value, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Initial-condition setting {key} differs from the saved GBD run.")
    configs, *_ = _initial_conditions("merging", viscosity)
    clouds = [c.build() for c in configs]
    pos0 = np.concatenate([p.position for p in clouds])
    alpha0 = np.concatenate([p.vortex_strength for p in clouds])
    sigma0 = np.concatenate([p.core_radius for p in clouds])
    # Use the recorded final step, never a hard-coded backup filename.
    path = CASE_DIR / "solution/merging_gbd" / f"vpm_{int(run['number_of_steps']):06d}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"Final GBD particle backup required for sphere views: {path}")
    with h5py.File(path) as f:
        pos1 = f["particles/position"][:]
        alpha1 = f["particles/vortex_strength"][:]
        sigma1 = f["particles/core_radius"][:]
        time = float(f["solver"].attrs["time"])
    if not np.isclose(time, float(run["final_time"])):
        raise ValueError("Final GBD backup time does not match the sample metadata.")
    if len(pos0) != int(run["initial_n_particles_total"]):
        raise ValueError("Initial conditions no longer match the stored GBD run.")
    uc = circulation / (2 * np.pi * a0)
    wc = circulation / (np.pi * a0 * a0)
    x = np.linspace(-1.4, 1.4, 149)
    y = x.copy()
    xx, yy = np.meshgrid(x, y)
    query = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, sample_z)))
    cases = []
    tail_bounds = []
    cached = np.load(args.field_cache) if args.field_cache and args.field_cache.exists() else None
    cache_values = {"grid": x}
    for i, (pos, alpha, sigma, t) in enumerate(
        [(pos0, alpha0, sigma0, 0.0), (pos1, alpha1, sigma1, time)]
    ):
        assert np.all(alpha[:, :2] == 0), "This renderer requires purely axial strengths."
        if cached is not None:
            for field, value in [
                ("position", pos),
                ("strength", alpha),
                ("sigma", sigma),
                ("time", np.array(t)),
            ]:
                assert np.array_equal(cached[f"{field}_{i}"], value), f"Stale cache for {field}_{i}"
            assert np.array_equal(cached["grid"], x)
            omega = cached[f"omega_{i}"]
            vel = cached[f"velocity_{i}"]
        else:
            omega = sample_vorticity(pos, pos, alpha, sigma) / wc
            vel = sample_velocity(query, pos, alpha, sigma).reshape(len(y), len(x), 2)
        tail_bounds.append(
            float(
                np.exp(-25) * np.sum(np.abs(alpha[:, 2])) / (np.pi**1.5 * np.min(sigma) ** 3) / wc
            )
        )
        cases.append((pos, alpha, omega, vel, t))
        for field, value in [
            ("position", pos),
            ("strength", alpha),
            ("sigma", sigma),
            ("time", np.array(t)),
            ("omega", omega),
            ("velocity", vel),
        ]:
            cache_values[f"{field}_{i}"] = value
        print(
            f"Reconstructed nu*t/a_c0^2={t * viscosity / a0**2:.2g}: {len(pos)} particles",
            flush=True,
        )
    if args.field_cache and cached is None:
        np.savez_compressed(args.field_cache, **cache_values)
    # Fit the union once so a fixed world-space distance has the same display
    # size in both snapshots. Plane corners enclose every possible streamline.
    plane_corners = np.array([[px, py, sample_z] for px in [x[0], x[-1]] for py in [y[0], y[-1]]])
    bounds = np.concatenate([project(pos0)[0], project(pos1)[0], project(plane_corners)[0]])
    lower = bounds.min(axis=0)
    upper = bounds.max(axis=0)
    max_radius = max(
        float(
            np.max(
                0.035
                * (np.linalg.norm(alpha, axis=1) / np.linalg.norm(alpha, axis=1).max()) ** 0.65
            )
        )
        for _, alpha, _, _, _ in cases
    )
    padding = 0.018 * (upper - lower) + max_radius
    lower -= padding
    upper += padding
    width = 3000
    height = int(np.ceil((width - 1) * (upper[1] - lower[1]) / (upper[0] - lower[0]))) + 1
    # Equal world-space pixel scales in x and y avoid any shape distortion.
    upper[1] = lower[1] + (height - 1) * (upper[0] - lower[0]) / (width - 1)
    axes_rectangle = [0.005, 0.045, 0.99, 0.95]
    view_limits = []
    styles = []
    for (pos, alpha, omega, vel, t), name in zip(
        cases, ["mergingRenderT0.pdf", "mergingRenderFinal.pdf"]
    ):
        fig = plt.figure(figsize=(125 / 25.4, 85 / 25.4), facecolor="white")
        ax = fig.add_axes(axes_rectangle)
        ax.set_aspect("equal")
        ax.axis("off")
        projected, depth = project(pos)
        strength = np.linalg.norm(alpha, axis=1)
        radii = 0.035 * (strength / strength.max()) ** 0.65
        wmax = min(float(np.quantile(omega, 0.95)), 0.8 * float(np.max(omega)))
        wmin = 0.08 * wmax
        speed = np.linalg.norm(vel, axis=2) / uc
        umax = float(np.quantile(speed, 0.99))
        norms = [Normalize(wmin, wmax), Normalize(0, umax)]
        particle_cmap = plt.get_cmap("viridis").copy()
        particle_cmap.set_under((0.78, 0.82, 0.87, 1.0))
        colors = particle_cmap(norms[0](omega))[:, :3]
        # Streamplot integrates the actual instantaneous 3-D induced velocity on z=L/4.
        scratch, sax = plt.subplots()
        streams = sax.streamplot(
            x,
            y,
            vel[:, :, 0],
            vel[:, :, 1],
            density=0.75,
            arrowsize=0.001,
            broken_streamlines=False,
            color=speed,
            cmap="plasma",
            norm=norms[1],
            linewidth=0.3,
        )
        raw = streams.lines.get_segments()
        line_colors = plt.get_cmap("plasma")(norms[1](streams.lines.get_array()))[:, :3]
        segments = []
        for line in raw:
            points = np.column_stack((line, np.full(len(line), sample_z)))
            plane, plane_depth = project(points)
            segments.append(np.column_stack((plane, plane_depth)))
        plt.close(scratch)
        segments = np.array(segments)
        assert np.all(projected - radii[:, None] >= lower) and np.all(
            projected + radii[:, None] <= upper
        )
        assert np.all(segments[:, :, :2] >= lower) and np.all(segments[:, :, :2] <= upper)
        canvas, zbuffer = render_spheres(
            np.column_stack((projected, depth)), radii, colors, lower, upper, width, height
        )
        canvas = render_streamlines(canvas, zbuffer, segments, line_colors, lower, upper, 1.15)
        ax.imshow(
            np.uint8(np.clip(canvas, 0, 1) * 255),
            extent=[lower[0], upper[0], lower[1], upper[1]],
            interpolation="none",
        )
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        view_limits.append([list(lower), list(upper)])
        # Keep all labels as vector LaTeX text at the final physical size.
        for box, cmap, norm, label, extension in [
            (
                [0.055, 0.67, 0.023, 0.20],
                particle_cmap,
                norms[0],
                r"$\omega_z/\omega_{c,0}$",
                "both",
            ),
            ([0.89, 0.17, 0.023, 0.20], "plasma", norms[1], r"$|\mathbf{u}|/U_{c,0}$", "max"),
        ]:
            cax = fig.add_axes(box)
            cb = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cax,
                ticks=[norm.vmin, (norm.vmin + norm.vmax) / 2, norm.vmax],
                format=FormatStrFormatter("%.2g"),
                extend=extension,
            )
            cb.ax.tick_params(width=0.4, length=2, pad=2)
            cb.ax.set_title(label, fontsize=10.95, pad=7)
        tau = t * viscosity / a0**2
        time_relation = "=" if t == 0.0 else r"\approx"
        fig.text(
            0.98,
            0.025,
            rf"$\nu t/a_{{c,0}}^2{time_relation}{tau:.2g}$",
            ha="right",
            va="bottom",
            fontsize=10.95,
        )
        for fmt in ["pdf", "png"] if args.format == "both" else [args.format]:
            output = (args.output_dir / name).with_suffix("." + fmt)
            fig.savefig(output, dpi=600)
            print(output, flush=True)
        plt.close(fig)
        styles.append(
            {
                "time": t,
                "normalised_time": tau,
                "vorticity_limits": [wmin, wmax],
                "velocity_limits": [0, umax],
                "sphere_radius_range": list(map(float, [radii.min(), radii.max()])),
                "sphere_strength_reference": float(strength.max()),
                "scene_pixels": [width, height],
            }
        )
    assert view_limits[0] == view_limits[1]
    assert styles[0]["scene_pixels"] == styles[1]["scene_pixels"]
    metadata = args.output_dir / "render-lamb-oseen-snapshots.json"
    metadata.write_text(
        json.dumps(
            {
                "source_backup": str(path),
                "initial_particles": len(pos0),
                "final_particles": len(pos1),
                "final_time": time,
                "sample_plane_z": sample_z,
                "solution": "merging_gbd",
                "velocity_scale": uc,
                "vorticity_scale": wc,
                "styles": styles,
                "sphere_encoding": "Opaque shaded spheres with radius 0.035*(abs(strength)/frame_max_strength)^0.65. Glyph radius is not the Gaussian core radius.",
                "renderer": "CPU analytic ray-sphere intersections, Phong shading and a depth buffer; streamlines are depth tested against spheres.",
                "colour_clipping": "Particle upper limit min(95th percentile, 0.8*peak), lower limit 0.08*upper; values below lower are pale grey. Velocity upper limit 99th percentile. All limits are per frame and indicated by extended colour bars.",
                "vorticity_definition": "Gaussian reconstruction at particle positions, truncated beyond five maximum radii",
                "normalised_vorticity_absolute_tail_bounds": tail_bounds,
                "velocity_definition": "direct 3-D Gaussian Biot-Savart sum on z=L/4",
                "field_grid": [149, 149],
                "view_limits": view_limits,
                "framing": "Both states share the orthographic projection, field of view, raster dimensions and axes rectangle. Bounds enclose both particle clouds and the sampling plane. Colour limits and size normalisation are per frame.",
                "camera": {
                    "projection": "orthographic",
                    "view_vector_unnormalised": [0.70, 0.61, -0.37],
                    "in_plane_rotation_degrees": 25,
                    "axes_rectangle": axes_rectangle,
                    "pixels_per_length_unit": (width - 1) / (upper[0] - lower[0]),
                },
                "figure_size_mm": [125, 85],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
