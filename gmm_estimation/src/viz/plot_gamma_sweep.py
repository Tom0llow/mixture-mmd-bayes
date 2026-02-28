from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def plot_gamma_sweep(
    gamma_scales: Sequence[float],
    mean_ed2: Sequence[float],
    results_dir: Path,
) -> Path:
    """Plot mean ED2 against gamma scales and save the figure path."""
    gamma_scales = list(gamma_scales)
    mean_ed2 = list(mean_ed2)

    x = list(range(len(gamma_scales)))

    def format_gamma_label(value: float) -> str:
        if abs(value - 0.0) < 1e-12:
            return "0.0"
        if abs(value - 1.0) < 1e-12:
            return "1.0"
        if abs(value - 0.5) < 1e-12:
            return "0.5"
        if abs(value - 0.1) < 1e-12:
            return "0.1"

        label = f"{value:.0e}"
        if "e" not in label:
            return label
        mantissa, exp = label.split("e")
        return f"{mantissa}e{int(exp)}"

    plt.figure()
    plt.plot(x, mean_ed2, linestyle=":", marker="o")

    labels = [format_gamma_label(g) for g in gamma_scales]
    plt.xticks(x, labels, ha="center")

    plt.xlabel(r"gamma scale ($\gamma/\beta$)")
    plt.ylabel(r"Energy distance ($\mathcal{D}_E^2$)")
    plt.tight_layout()

    out_path = results_dir / "gamma_sweep.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
