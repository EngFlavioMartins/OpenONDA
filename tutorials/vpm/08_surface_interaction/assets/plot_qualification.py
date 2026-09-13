"""Plot the compact tandem-surface qualification summary."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CASE_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    summary = pd.read_csv(CASE_DIR / "samples/tandem/qualification_summary.csv")
    figure_dir = CASE_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    axes = summary.plot(
        x="surface",
        y=["final_lift", "final_drag"],
        kind="bar",
        title="Real VPM--VLM tandem attached loads",
        ylabel="force [N]",
    )
    axes.figure.tight_layout()
    axes.figure.savefig(figure_dir / "tandem_loads.png", dpi=150)
    plt.close(axes.figure)
    print(f"Wrote qualification figure to {figure_dir / 'tandem_loads.png'}")


if __name__ == "__main__":
    main()
