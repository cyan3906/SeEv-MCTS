import matplotlib.pyplot as plt
import numpy as np

# x-axis: number of samples or iterations
x = np.arange(2, 11)

# Example data, replace with your real results
# data = {
#     "MathVision": {
#         "Cantor": [14.5, 15.5, 16.0, 16.7, 17.1, 17.7, 18.8, 21.7, 21.7],
#         "ToT": [14.5, 14.7, 14.8, 14.8, 16.0, 16.0, 16.3, 18.4, 19.0],
#         "VReST": [14.5, 15.4, 15.5, 17.0, 18.0, 18.4, 19.0, 20.0, 20.4],
#         "XXX": [17.1, 17.9, 18.4, 18.8, 21.1, 21.7, 23.0, 24.7, 28.2],
#     },
#     "MathVista": {
#         "Cantor": [54.6, 54.9, 55.8, 56.0, 56.3, 57.4, 57.7, 58.8, 62.3],
#         "ToT": [54.6, 54.7, 54.8, 55.1, 55.1, 55.8, 55.9, 57.3, 57.7],
#         "VReST": [54.5, 54.7, 55.9, 57.8, 58.1, 58.4, 58.7, 59.1, 60.2],
#         "XXX": [57.8, 57.8, 60.6, 61.3, 63.1, 63.5, 64.7, 65.3, 65.4],
#     },
#     "CharXiv": {
#         "Cantor": [27.2, 27.7, 27.8, 28.8, 29.0, 29.1, 29.7, 30.2, 30.9],
#         "ToT": [27.3, 28.2, 28.5, 29.5, 29.6, 29.6, 31.3, 31.5, 31.8],
#         "VReST": [27.2, 27.8, 28.9, 31.0, 31.2, 31.3, 31.5, 31.6, 32.0],
#         "XXX": [31.2, 33.1, 33.2, 34.3, 34.5, 35.9, 37.0, 37.9, 38.1],
#     },
# }

# data = {
#     "MathVision": {
#         "Cantor": [14.00, 14.40, 14.70, 15.00, 15.30, 15.60, 16.00, 16.40, 16.80],
#         "ToT": [14.20, 14.80, 15.50, 16.20, 16.90, 17.40, 18.10, 18.70, 19.20],
#         "VReST": [14.40, 15.20, 16.30, 17.10, 18.20, 19.00, 19.80, 20.50, 21.10],
#         "XXX": [17.00, 17.60, 18.30, 19.10, 20.00, 21.70, 22.90, 23.20, 24.70],
#     },
#     "MathVista": {
#         "Cantor": [54.50, 54.70, 55.80, 56.20, 56.60, 57.00, 57.50, 57.80, 58.10],
#         "ToT": [54.60, 55.90, 57.80, 58.20, 58.60, 59.00, 59.30, 59.70, 60.10],
#         "VReST": [54.80, 56.10, 57.20, 58.40, 59.20, 59.80, 60.30, 61.20, 62.30],
#         "XXX": [57.80, 60.60, 61.20, 63.10, 63.60, 64.20, 64.80, 65.20, 65.60],
#     },
#     "CharXiv": {
#         "Cantor": [27.30, 27.80, 28.30, 28.80, 29.10, 29.40, 29.70, 30.00, 30.30],
#         "ToT": [27.40, 28.10, 29.00, 30.80, 31.00, 31.30, 31.60, 31.90, 32.20],
#         "VReST": [28.50, 29.40, 30.30, 31.90, 32.60, 33.10, 34.50, 35.80, 36.70],
#         "XXX": [31.00, 33.20, 33.40, 34.40, 35.80, 36.40, 37.20, 37.90, 38.30],
#     },
# }

data = {
    "MathVision": {
        "Cantor": [14.40, 14.50, 14.70, 14.90, 15.10, 15.40, 15.80, 16.30, 16.80],
        "ToT": [14.30, 14.80, 14.80, 14.80, 16.00, 16.00, 16.50, 18.40, 19.20],
        "VReST": [14.40, 14.80, 15.50, 16.90, 18.00, 18.40, 19.00, 19.90, 21.10],
        "XXX": [17.00, 17.70, 18.30, 18.70, 21.10, 21.70, 23.00, 24.60, 24.70],
    },
    "MathVista": {
        "Cantor": [54.60, 54.70, 54.90, 55.20, 55.20, 55.90, 56.00, 57.30, 58.10],
        "ToT": [54.70, 55.00, 55.90, 57.80, 58.10, 58.40, 58.70, 59.10, 60.10],
        "VReST": [54.60, 54.90, 55.80, 56.00, 56.30, 57.20, 57.70, 58.10, 62.30],
        "XXX": [57.80, 57.80, 60.60, 61.30, 63.10, 63.50, 64.80, 65.30, 65.60],
    },
    "CharXiv": {
        "Cantor": [27.20, 27.70, 27.80, 28.70, 29.00, 29.10, 29.60, 30.10, 30.30],
        "ToT": [27.30, 28.30, 28.80, 29.50, 29.60, 29.60, 31.20, 31.50, 32.20],
        "VReST": [28.50, 29.40, 30.30, 31.90, 32.60, 33.10, 34.50, 35.80, 36.70],
        "XXX": [31.00, 33.10, 33.20, 34.20, 34.50, 35.90, 37.00, 37.90, 38.30],
    },
}


# Line styles
styles = {
    "Cantor": {
        "color": "#2F77B4",
        "marker": "o",
    },
    "ToT": {
        "color": "#F28E2B",
        "marker": "s",
    },
    "VReST": {
        "color": "#59A14F",
        "marker": "^",
    },
    "XXX": {
        "color": "#E15759",
        "marker": "d",
    },
}

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

for ax, dataset_name in zip(axes, ["MathVision", "MathVista", "CharXiv"]):
    for method, y in data[dataset_name].items():
        ax.plot(
            x,
            y,
            label=method,
            color=styles[method]["color"],
            marker=styles[method]["marker"],
            linewidth=1.5,
            markersize=4,
        )

    ax.set_title(dataset_name, fontsize=13)
    ax.set_xlabel("Number of samples or iterations", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(False)

# Put one shared legend above all subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=True,
    fontsize=12,
    bbox_to_anchor=(0.5, 1.05),
)

plt.tight_layout(rect=[0, 0, 1, 0.92])

# Save for paper
plt.savefig("rollout_analysis.pdf", bbox_inches="tight")
plt.savefig("rollout_analysis.png", dpi=300, bbox_inches="tight")

plt.show()
