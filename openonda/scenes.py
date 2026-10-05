"""Generic array rendering and labelled-scene export."""

from html import escape
from math import erf, exp, pi, sqrt
from pathlib import Path
import shutil
import subprocess
import tempfile

from numba import njit, prange
import numpy as np
from scipy.spatial import cKDTree

from openonda.executables import find_executable


@njit(parallel=True, cache=False)
def sample_axial_velocity(points, position, strength, sigma):
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


def sample_axial_vorticity(points, position, strength, sigma, cutoff):
    """Evaluate axial Gaussian vorticity within the supplied truncation radius."""
    tree = cKDTree(position)
    radius = cutoff * float(np.max(sigma))
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


@njit(cache=False)
def render_spheres(points, radii, colors, lower, upper, width, height, light):
    """Orthographic visible sphere surfaces, including mutual occlusion."""
    canvas = np.ones((height, width, 3), dtype=np.float64)
    zbuffer = np.full((height, width), -np.inf)
    scale = (width - 1) / (upper[0] - lower[0])
    light = light / np.sqrt(np.sum(light * light))
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
    return (canvas, zbuffer)


def export_document(work, output_dir, name, tex, figure_format, texbin=None, *, dpi):
    """Compile supplied vector labels and export the requested scene formats."""
    work, output_dir = Path(work), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    texbin = find_executable("pdflatex", texbin)
    (work / f"{name}.tex").write_text(tex)
    subprocess.run(
        [texbin, "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
        cwd=work,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    shutil.copy2(work / f"{name}.tex", output_dir / f"{name}.tex")
    if figure_format in ("png", "both"):
        subprocess.run(
            [
                find_executable("pdftoppm"),
                "-png",
                "-r",
                str(dpi),
                "-singlefile",
                str(work / f"{name}.pdf"),
                str(output_dir / name),
            ],
            check=True,
        )
    if figure_format in ("pdf", "both"):
        shutil.copy2(work / f"{name}.pdf", output_dir / f"{name}.pdf")


@njit(cache=False)
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


def export_scene(output_dir, prepare, *, dpi, pvpython=None):
    """Render the configured scene and remove its temporary files afterward."""
    from openonda.results import write_json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openonda-scene-") as temporary:
        work = Path(temporary)
        scene = prepare(work)
        script, arguments = scene["renderer"]
        subprocess.run(
            [find_executable("pvpython", pvpython), str(script), *map(str, arguments)], check=True
        )
        for source, destination in scene["images"].items():
            shutil.copy2(work / source, output_dir / destination)
            if source != destination:
                shutil.copy2(work / source, work / destination)
        for source, destination in scene.get("files", {}).items():
            target = output_dir / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.suffix == ".pvsm":
                text = (work / source).read_text()
                for input_name, output_name in scene["files"].items():
                    text = text.replace(
                        str(work / input_name), escape(str(output_dir / output_name), quote=True)
                    )
                target.write_text(text)
            else:
                shutil.copy2(work / source, target)
        for name, tex in scene["documents"]:
            export_document(work, output_dir, name, tex, scene["format"], dpi=dpi)
        for name, record in scene.get("metadata", {}).items():
            write_json(output_dir / name, record)


def figure_frame(fig):
    """Capture the authored canvas as one print-prepared raster frame."""
    from PIL import Image

    from openonda import plotting

    plotting.prepare_figure(fig)
    renderer = plotting.figure_renderer(fig)
    return Image.fromarray(np.asarray(renderer.buffer_rgba())[..., :3]).quantize(colors=256)


def export_animation(frames, output, *, fps):
    """Export the supplied frames at GIF's discrete clock resolution."""
    if not frames or not np.isfinite(fps) or fps <= 0 or fps > 100:
        raise ValueError("GIF export needs frames and a frame rate in (0, 100]")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ticks = np.rint(np.arange(len(frames) + 1) * 100 / fps).astype(int)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=(10 * np.diff(ticks)).tolist(),
        loop=0,
        disposal=2,
        optimize=False,
    )
    return output
