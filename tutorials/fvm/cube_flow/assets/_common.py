import argparse
from pathlib import Path

from openonda.results import latest_fvm_frame, read_grouped_csv, snapshot_vector_field

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = ASSETS_DIR.parent
FIGURES_DIR = SCRIPT_DIR / "figures"
SOLUTION_DIR = SCRIPT_DIR / "solution"


def _load_theme():
    from openonda import plotting as theme

    theme.set_thesis_style()
    return theme


THEME = _load_theme()
COLORS = THEME.COLORS
COLORMAPS = THEME.COLORMAPS
figure_size = THEME.figure_size

FREESTREAM_SPEED = 1.0
D_REF = 1.0

# Square cylinder, Re = 100, blockage 5%.
# Okajima, J. Fluid Mech. 123 (1982): experimental St ~ 0.14.
# Sohankar, Norberg & Davidson, IJNMF 26 (1998): St = 0.146, Cd = 1.48.
# Sen, Mittal & Biswas, IJNMF 67 (2011): St = 0.145, Cd = 1.53.
REFERENCES = {
    100.0: {
        "drag_coefficient": (1.45, 1.58),
        "strouhal_number": (0.140, 0.150),
    },
}


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=THEME.FORMAT_CHOICES, default="both")
    parser.add_argument("--dpi", type=int, default=THEME.DEFAULT_DPI)
    parser.add_argument("--Re", type=float, default=100.0)
    return parser


def load_forces_csv(solution_dir):
    return read_grouped_csv(Path(solution_dir).parent / "samples/forces_history.csv", "patch")


def strouhal_from_lift(t, cl):
    """Dominant lift frequency (Hz) from the second half of the signal."""
    n = len(t)
    if n < 32:
        return None
    t2, cl2 = t[n // 2 :], cl[n // 2 :]
    if np.ptp(cl2) < 1e-6:
        return None
    tu = np.linspace(t2[0], t2[-1], len(t2))
    clu = np.interp(tu, t2, cl2)
    clu -= clu.mean()
    freqs = np.fft.rfftfreq(len(tu), tu[1] - tu[0])
    amp = np.abs(np.fft.rfft(clu))
    if amp[1:].max() < 1e-8:
        return None
    return float(freqs[1:][np.argmax(amp[1:])])


def save_fig(fig, name, figures_dir, dpi=None, figure_format="both"):
    return THEME.save_fig(fig, Path(figures_dir) / name, figure_format=figure_format, dpi=dpi)
