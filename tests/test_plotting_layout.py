"""Figure validation follows the tick labels Matplotlib actually paints."""

import matplotlib

matplotlib.use("Agg")
import pytest

from openonda import plotting


@pytest.mark.parametrize("scale", [1e-14, 1.0, 1e14])
def test_out_of_range_ticks_do_not_overlap_neighbouring_axes(scale):
    with plotting.plt.rc_context():
        plotting.plt.rcParams.update(
            {"text.usetex": False, "font.size": plotting.THESIS_FONT_SIZE_PT}
        )
        fig, axes = plotting.plt.subplots(2, 1, figsize=plotting.figure_size("stacked"))
        try:
            fig.subplots_adjust(left=0.2, right=0.8, bottom=0.15, top=0.85, hspace=0.5)
            for ax in axes:
                ax.set_ylim(-scale, scale)
                ax.set_yticks([-2 * scale, 0, 2 * scale], labels=["outside", "0", "outside"])
                ax.set_ylim(-scale, scale)
                ax.set_xticks([])
            # An out-of-range upper-axis tick lies inside the lower panel.
            plotting.validate_thesis_figure(fig, axes)
        finally:
            plotting.plt.close(fig)
