#!/usr/bin/env python3
"""Build docs/OpenONDA_manual.pdf from the user guides.

Each guide becomes a chapter. Links between guides become links inside the
PDF; links to tutorials and source files point at the GitHub repository.
Requires pandoc and XeLaTeX. Run from anywhere: python docs/manual/build_manual.py
"""

from pathlib import Path
import re
import subprocess
import tempfile

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parent
REPOSITORY = "https://github.com/EngFlavioMartins/OpenONDA/blob/development/"

# Reading order of the manual
CHAPTERS = (
    "introduction",
    "installation",
    "tutorials",
    "fvm",
    "vpm",
    "coupling",
    "continuation",
    "solution_layout",
    "visualization",
)


def introduction():
    """The README's overview, without its HTML banners and duplicated examples."""
    text = (ROOT / "README.md").read_text()
    overview = text[: text.index("## Installation")]
    overview = re.sub(r"<p[\s\S]*?</p>\n?", "", overview)
    overview = overview.replace("# OpenONDA", "# Introduction", 1)
    return overview + (
        "\nThis manual collects the user guides. Each chapter links to the tutorial cases "
        "that demonstrate it; the tutorials themselves live in the `tutorials/` directory "
        "of the repository.\n"
    )


def slug(text):
    """Pandoc-style identifier of a heading."""
    text = re.sub(r"[`*$\\{}]", "", text).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def chapter(name):
    """One guide with chapter-unique heading identifiers and resolved links."""
    text = introduction() if name == "introduction" else (DOCS / f"{name}.md").read_text()
    source = ROOT if name == "introduction" else DOCS

    def identify(match):
        hashes, title = match.group(1), match.group(2)
        identifier = name if len(hashes) == 1 else f"{name}-{slug(title)}"
        return f"{hashes} {title} {{#{identifier}}}"

    text = re.sub(r"^(#+)\s+(.*)$", identify, text, flags=re.MULTILINE)

    def link(match):
        label, target = match.group(1), match.group(2)
        if target.startswith(("http", "mailto")):
            return match.group(0)
        path, _, anchor = target.partition("#")
        if not path:
            return f"[{label}](#{name}-{anchor})"
        resolved = (source / path).resolve()
        if resolved.parent == DOCS and resolved.stem in CHAPTERS:
            other = resolved.stem
            return f"[{label}](#{other}-{anchor})" if anchor else f"[{label}](#{other})"
        relative = resolved.relative_to(ROOT).as_posix()
        return f"[{label}]({REPOSITORY}{relative}{'#' + anchor if anchor else ''})"

    return re.sub(r"\[([^\]]*)\]\(([^)\s]+)\)", link, text)


def main():
    manual = "\n\n".join(chapter(name) for name in CHAPTERS)
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "manual.md"
        source.write_text(manual)
        logos = Path(scratch) / "logos.tex"
        logos.write_text(f"\\graphicspath{{{{{(DOCS / 'logos').as_posix()}/}}}}\n")
        subprocess.run(
            [
                "pandoc",
                str(source),
                "--from=markdown+tex_math_dollars+pipe_tables",
                "--pdf-engine=xelatex",
                "--top-level-division=chapter",
                "--number-sections",
                "--toc",
                "--toc-depth=2",
                "--highlight-style=tango",
                f"--include-in-header={DOCS / 'manual' / 'header.tex'}",
                f"--include-in-header={logos}",
                f"--include-before-body={DOCS / 'manual' / 'title.tex'}",
                f"--resource-path={DOCS / 'logos'}",
                "--variable=documentclass:report",
                "--variable=classoption:openany",
                "--variable=papersize:a4",
                "--variable=fontsize:11pt",
                "--variable=geometry:margin=2.4cm",
                "--variable=mainfont:TeX Gyre Pagella",
                "--variable=mathfont:TeX Gyre Pagella Math",
                "--variable=sansfont:Noto Sans",
                "--variable=monofont:DejaVu Sans Mono",
                "--variable=monofontoptions:Scale=0.82",
                "--variable=linkcolor:[HTML]{1F5F99}",
                "--variable=urlcolor:[HTML]{1F5F99}",
                "--variable=toccolor:black",
                "--output",
                str(DOCS / "OpenONDA_manual.pdf"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
