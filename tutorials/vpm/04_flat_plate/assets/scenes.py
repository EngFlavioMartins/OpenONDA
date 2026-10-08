"""Render particle scenes and figure animations for this tutorial."""

from html import escape
from pathlib import Path
import json
import shutil
import subprocess
import tempfile


def export_document(work, output_dir, name, tex, figure_format, texbin=None, *, dpi):
    """Compile supplied vector labels and export the requested scene formats."""
    work, output_dir = Path(work), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    texbin = shutil.which(texbin or "pdflatex")
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
                shutil.which("pdftoppm"),
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


def export_scene(output_dir, prepare, *, dpi, pvpython=None):
    """Render the configured scene and remove its temporary files afterward."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openonda-scene-") as temporary:
        work = Path(temporary)
        scene = prepare(work)
        script, arguments = scene["renderer"]
        subprocess.run(
            [shutil.which(pvpython or "pvpython"), str(script), *map(str, arguments)], check=True
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
            (output_dir / name).write_text(json.dumps(record, indent=2) + "\n")
