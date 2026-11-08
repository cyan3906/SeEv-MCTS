import matplotlib.pyplot as plt
import numpy as np

# x-axis: number of samples or iterations
x = np.arange(2, 11)

# Keep data unchanged
data = {
    "MathVision": {
        "Cantor": [7.80, 7.90, 8.10, 8.30, 8.50, 8.80, 9.20, 9.70, 10.20],
        "ToT": [10.50, 11.00, 11.00, 11.00, 12.20, 12.20, 12.70, 14.60, 15.40],
        "VReST": [14.40, 14.80, 15.50, 16.90, 18.00, 18.40, 19.00, 19.90, 21.10],
        "SeEV-MCTS": [17.00, 17.70, 18.30, 18.70, 21.10, 21.70, 23.00, 24.60, 24.70],
    },
    "MathVista": {
        "Cantor": [43.60, 43.70, 43.90, 44.20, 44.20, 44.90, 45.00, 46.30, 47.10],
        "ToT": [44.50, 44.80, 45.40, 47.30, 47.60, 47.90, 48.20, 48.60, 49.60],
        "VReST": [44.00, 44.30, 46.70, 47.80, 48.20, 48.60, 49.10, 50.50, 51.70],
        "SeEV-MCTS": [47.00, 47.00, 49.80, 50.50, 52.30, 52.70, 54.00, 54.50, 54.80],
    },
    "CharXiv": {
        "Cantor": [26.10, 26.60, 26.70, 27.60, 27.90, 28.00, 28.50, 29.00, 29.20],
        "ToT": [28.30, 29.30, 29.80, 30.50, 30.60, 30.60, 32.20, 32.50, 33.20],
        "VReST": [28.50, 29.40, 30.30, 31.90, 32.60, 33.10, 34.50, 35.80, 36.70],
        "SeEV-MCTS": [31.00, 33.10, 33.20, 34.20, 34.50, 35.90, 37.00, 37.90, 38.30],
    },
}

# Style close to the reference figure
styles = {
    "Cantor": {
        "color": "#1f77b4",
        "label": "Cantor",
    },
    "ToT": {
        "color": "#ff7f0e",
        "label": "ToT",
    },
    "VReST": {
        "color": "#2ca02c",
        "label": "VReST",
    },
    "SeEV-MCTS": {
        "color": "#e41a1c",
        "label": "SeEV-MCTS\n(our)",
    },
}

# Dataset-specific title badges
title_badges = {
    "MathVista": "",
    "MathVision": "",
    "CharXiv": "",
}

# Keep your current order
dataset_order = ["MathVision", "MathVista", "CharXiv"]

# Adjusted y-axis limits according to current data
ylim_dict = {
    "MathVision": (6, 26),
    "MathVista": (42, 56),
    "CharXiv": (26, 40),
}

# Adjusted y-axis ticks
ytick_dict = {
    "MathVision": np.arange(6, 27, 4),  # 6, 10, 14, 18, 22, 26
    "MathVista": np.arange(42, 57, 2),  # 42, 44, 46, ..., 56
    "CharXiv": np.arange(26, 41, 2),  # 26, 28, 30, ..., 40
}

fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1))

for ax, dataset_name in zip(axes, dataset_order):
    for method, y in data[dataset_name].items():
        ax.plot(
            x,
            y,
            color=styles[method]["color"],
            marker="o",
            linewidth=2.0,
            markersize=4.2,
            markerfacecolor=(
                "white" if method == "SeEV-MCTS" else styles[method]["color"]
            ),
            markeredgewidth=1.6 if method == "SeEV-MCTS" else 1.0,
            markeredgecolor=styles[method]["color"],
            zorder=3 if method == "SeEV-MCTS" else 2,
        )

        # Right-side curve labels
        ax.text(
            x[-1] + 0.25,
            y[-1],
            styles[method]["label"],
            color=styles[method]["color"],
            fontsize=8,
            fontweight="bold" if method == "SeEV-MCTS" else "normal",
            va="center",
            ha="left",
            clip_on=False,
        )

    # Icon badge
    ax.text(
        0.34,
        1.085,
        title_badges[dataset_name],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#2F6DB3",
        bbox=dict(
            boxstyle="round,pad=0.18",
            edgecolor="#2F6DB3",
            facecolor="white",
            linewidth=1.2,
        ),
        clip_on=False,
    )

    # Dataset title
    ax.text(
        0.55,
        1.085,
        dataset_name,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#1f2d3d",
        clip_on=False,
    )

    ax.set_xlabel("Number of samples or iterations", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=10)

    ax.set_xlim(1.8, 11.25)
    ax.set_ylim(*ylim_dict[dataset_name])
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_yticks(ytick_dict[dataset_name])

    # Light dotted grid
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.55,
        alpha=0.35,
        color="#bdbdbd",
    )

    ax.tick_params(axis="both", labelsize=9, length=3)

    # Cleaner axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

# Caption-like note
fig.text(
    0.5,
    0.025,
    "Higher accuracy is better. SeEV-MCTS (our) consistently achieves the best performance across all datasets and sample sizes.",
    ha="center",
    fontsize=8,
    style="italic",
    color="#444444",
)

plt.subplots_adjust(
    left=0.065,
    right=0.93,
    top=0.83,
    bottom=0.20,
    wspace=0.42,
)

plt.savefig("rollout_analysis_demo_style.pdf", bbox_inches="tight")
plt.savefig("rollout_analysis_demo_style.png", dpi=300, bbox_inches="tight")
plt.show()
