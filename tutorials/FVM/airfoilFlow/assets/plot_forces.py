import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import COLORS, build_arg_parser, figure_size, load_forces_csv, save_fig, RE


def main():
    args = build_arg_parser().parse_args()
    data = load_forces_csv(args.solution_dir)
    if not data:
        return
    patch = "airfoil"
    if patch not in data:
        patches = list(data.keys())
        if patches:
            patch = patches[0]
        else:
            return
    d = data[patch]
    time = d.get("time", d.get("Time", None))
    if time is None:
        print("  No time column found")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figure_size("stacked"), sharex=True)
    if "Cl" in d:
        ax1.plot(time, d["Cl"], color=COLORS["TUDdark"], linestyle="-", linewidth=1.0)
        ax1.set_ylabel("$C_l$")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No Cl data", ha="center", va="center", transform=ax1.transAxes)
    if "Cd" in d:
        ax2.plot(time, d["Cd"], color=COLORS["AccentRed"], linestyle="-", linewidth=1.0)
        ax2.set_ylabel("$C_d$")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No Cd data", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("Time [s]")
    fig.suptitle(f"NACA0012  AoA=23$^\\circ$  Re={RE:.0f}")
    fig.tight_layout()
    save_fig(fig, "airfoil_forces.png", args.figures_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
