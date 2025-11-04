import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path


def plot_cpl_vanilla_vs_salt() -> None:
    """
    Display a bar chart comparing CPL for Vanilla vs SALT across models.

    - Left cluster: Vanilla CPL for each model
    - Right cluster: SALT CPL for each model
    - Y axis is 0–100% (percent formatter)
    """

    model_names = [
        "Llama-3.1-8B",
        "QwQ-32B",
        "DeepSeek-R1-Distill-Qwen-1.5B",
    ]

    # CPL values (lower is better)
    vanilla_cpl = [0.385, 0.727, 0.824]
    salt_cpl = [0.316, 0.595, 0.718]

    # Two clusters: Vanilla (left) and SALT (right)
    cluster_centers = [0.0, 1.6]
    cluster_label_text = ["Vanilla", "SALT"]

    # Geometry for bars within each cluster
    cluster_width = 0.9
    bars_per_cluster = len(model_names)
    single_bar_width = cluster_width / bars_per_cluster

    # Distinct colors per model
    model_colors = ["#3B82F6", "#10B981", "#F59E0B"]

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)

    def bar_positions(center: float) -> list[float]:
        """Compute x positions for the set of bars around a cluster center."""
        left_edge = center - (cluster_width / 2.0)
        return [left_edge + (i + 0.5) * single_bar_width for i in range(bars_per_cluster)]

    # Plot Vanilla cluster
    vanilla_x = bar_positions(cluster_centers[0])
    vanilla_rects = []
    for i, x in enumerate(vanilla_x):
        rect = ax.bar(
            x,
            vanilla_cpl[i],
            width=single_bar_width * 0.9,
            color=model_colors[i],
        )
        vanilla_rects.extend(rect)

    # Plot SALT cluster
    salt_x = bar_positions(cluster_centers[1])
    salt_rects = []
    for i, x in enumerate(salt_x):
        rect = ax.bar(
            x,
            salt_cpl[i],
            width=single_bar_width * 0.9,
            color=model_colors[i],
        )
        salt_rects.extend(rect)

    # Annotate bars with percentages
    def annotate(rects, y_top: float) -> None:
        """Place percent labels without exceeding the axes top.

        If the default position would cross y_top, move the label inside the bar.
        """
        for r in rects:
            height = r.get_height()
            y = height + 0.02
            va = "bottom"
            if y > y_top - 0.005:
                y = max(0.01, height - 0.03)
                va = "top"
            ax.text(
                r.get_x() + r.get_width() / 2.0,
                y,
                f"{height * 100:.1f}%",
                ha="center",
                va=va,
                fontsize=9,
            )

    # Y-axis: cap at the tallest bar to highlight relative improvements
    y_top = max(max(vanilla_cpl), max(salt_cpl))
    ax.set_ylim(0.0, y_top)

    annotate(vanilla_rects, y_top)
    annotate(salt_rects, y_top)

    # Axis formatting
    ax.set_xticks(cluster_centers, cluster_label_text)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Contextual Privacy Leakage (CPL)")
    ax.set_title("CPL before and after SALT (lower is better)")

    # Legend mapping colors to models
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in model_colors]
    ax.legend(
        legend_handles,
        model_names,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout()

    # Save as vector PDF next to this script (usable in LaTeX)
    out_path = Path(__file__).parent / "cpl_vanilla_vs_salt.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved PDF: {out_path}")

    plt.show()


if __name__ == "__main__":
    plot_cpl_vanilla_vs_salt()


