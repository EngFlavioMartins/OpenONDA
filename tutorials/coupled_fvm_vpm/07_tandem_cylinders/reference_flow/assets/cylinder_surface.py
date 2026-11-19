"""The long cylinder of case 01, moved along the stream to the centre of one tandem cylinder."""

from pathlib import Path

from openonda.core.file_output import write_text_atomic

CYLINDER_STL = Path(__file__).resolve().parent / "cylinder_long.stl"


def write_cylinder_stl(path, centre):
    """Write the case 01 cylinder (axis along z through the origin) with its axis at ``centre`` (x, y)."""
    lines = []
    for line in CYLINDER_STL.read_text().splitlines():
        words = line.split()
        if words and words[0] == "vertex":
            x, y, z = (float(value) for value in words[1:])
            line = f"      vertex {x + centre[0]:.12e} {y + centre[1]:.12e} {z:.12e}"
        lines.append(line)
    write_text_atomic(path, "\n".join(lines) + "\n")
    return path
