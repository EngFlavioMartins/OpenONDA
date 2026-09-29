"""Figure validation follows the tick labels Matplotlib actually paints."""

import matplotlib

matplotlib.use("Agg")
import pytest

from openonda import plotting


def test_corner_ticks_are_separated_by_rendered_bounds():
    with plotting.plt.rc_context():
        plotting.plt.rcParams.update(
            {
                "text.usetex": False,
                "font.size": plotting.THESIS_FONT_SIZE_PT,
                "axes.titlesize": plotting.THESIS_FONT_SIZE_PT,
            }
        )
        fig, ax = plotting.plt.subplots(figsize=plotting.figure_size("wide_short"))
        try:
            ax.set_xlim(-1.0, 3.0)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xticks([-1.0, 0.0, 1.0, 2.0, 3.0])
            ax.set_yticks([-1.5, -1.0, 0.0, 1.0, 1.5])
            ax.set_xlabel("x / c")
            ax.set_ylabel("y / c")
            fig.tight_layout()
            plotting.fit_thesis_y_label_margins(fig, ax)
            plotting.validate_thesis_figure(fig, ax)
            renderer = fig.canvas.get_renderer()
            x_label = ax.xaxis.get_major_ticks()[0].label1.get_window_extent(renderer)
            y_label = ax.yaxis.get_major_ticks()[0].label1.get_window_extent(renderer)
            assert not x_label.overlaps(y_label)
        finally:
            plotting.plt.close(fig)


def test_long_title_clears_top_y_tick_with_fixed_canvas():
    with plotting.plt.rc_context():
        plotting.plt.rcParams.update(
            {
                "text.usetex": False,
                "font.size": plotting.THESIS_FONT_SIZE_PT,
                "axes.titlesize": plotting.THESIS_FONT_SIZE_PT,
            }
        )
        fig, ax = plotting.plt.subplots(figsize=plotting.figure_size("wide_short"))
        try:
            field = ax.imshow([[0, 1], [1, 0]], extent=(-1.0, 3.0, -1.5, 1.5), origin="lower")
            fig.colorbar(field, ax=ax, label="velocity magnitude / freestream speed")
            ax.set_xlim(-1.0, 3.0)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlabel("x / c")
            ax.set_ylabel("y / c")
            ax.set_title(r"NACA 0012 velocity magnitude (Re = 1000, $\alpha$ = 0$^\circ$)")
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.tight_layout(pad=1.0)
            plotting.fit_thesis_y_label_margins(fig, fig.axes)
            renderer = fig.canvas.get_renderer()
            title = ax.title.get_window_extent(renderer)
            top_tick = max(
                (tick for tick in ax.yaxis.get_major_ticks() if tick.get_loc() <= 1.5),
                key=lambda tick: tick.get_loc(),
            ).label1.get_window_extent(renderer)
            assert not title.overlaps(top_tick)
            assert ax.title.get_fontsize() == plotting.THESIS_FONT_SIZE_PT
        finally:
            plotting.plt.close(fig)


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
