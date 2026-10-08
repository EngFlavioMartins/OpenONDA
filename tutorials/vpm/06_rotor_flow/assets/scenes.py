"""Render particle scenes and figure animations for this tutorial."""

from pathlib import Path
import numpy as np


def figure_frame(fig):
    """Capture the authored canvas as one print-prepared raster frame."""
    from PIL import Image

    from openonda import plotting

    plotting.prepare_figure(fig)
    renderer = plotting.figure_renderer(fig)
    return Image.fromarray(np.asarray(renderer.buffer_rgba())[..., :3]).quantize(colors=256)


def export_animation(frames, output, *, fps):
    """Export the supplied frames at GIF's discrete clock resolution."""
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
