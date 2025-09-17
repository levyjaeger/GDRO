"""
    author: jaegerl
    created: 2025-08-10
    scope: figures for concept shift simulation
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

print("drawing panel of losses for concept shift simulation...")

# load the losses dataframe
losses_df = pl.read_csv(REPO_ROOT / "results" / "tables" / "simulations" / "losses_concept_shift.csv")

guidance_names = {
    "ERM (LightGBM)": "ERM (LightGBM)",
    "None (simple DRO)": "DRO, no guidance",
    "Features": "DRO, features",
    "Outcome, marginal": "DRO, marginal outcome",
    "Outcome, conditional": "DRO, conditional outcome",
}

# directory where to save the plots
plotdir = REPO_ROOT / "results" / "figures" / "simulations"

# data for plots
k_order = sorted(losses_df.filter(pl.col("k").is_not_null())["k"].unique().to_list())
phi_order = [
    '$\\phi = \\pi/8$',
    '$\\phi = \\pi/4$',
    '$\\phi = 3\\pi/8$',
    '$\\phi = \\pi/2$',
    ]
df = losses_df.to_pandas()
df["rho"] = df["rho"].astype(float)
df["k"] = pd.Categorical(df["k"], categories=k_order, ordered=True)
df["phi_nice"] = pd.Categorical(df["phi_nice"], categories=phi_order, ordered=True)
df["guidance"] = df["guidance"].map(guidance_names)
df["guidance"] = pd.Categorical(df["guidance"], categories=list(guidance_names.values()), ordered=True)

# extract ERM baselines
erm = df[df["guidance"] == "ERM (LightGBM)"]
baseline = erm.groupby("phi_nice", observed=False)["loss"].first().to_dict()
data = df[df["guidance"] != "ERM (LightGBM)"].copy()

# latex fonts
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
# prepare facet grid
sns.set_context("talk", font_scale=1.5)
g = sns.FacetGrid(
    data,
    row="phi_nice",
    col="k",
    margin_titles=True,
    sharex=True,
    sharey=True,
    height=5,
    aspect=1.5,
)
axes = np.atleast_2d(g.axes)
non_erm_guidance = [g for g in data["guidance"].unique()]
markers_dict = {g: m for g, m in zip(non_erm_guidance, ["o", "s", "D", "^", "v", "P", "*"])}
colors_dict = {g: c for g, c in zip(non_erm_guidance, sns.color_palette(n_colors=len(non_erm_guidance)))}
for i, phi_val in enumerate(phi_order):
    for j, k_val in enumerate(k_order):
        ax = axes[i, j]
        # plot ERM horizontal line
        yerm_overall = baseline.get(phi_val, None)
        ax.axhline(y=yerm_overall, color="black", linestyle="--", linewidth=1.5)
        # plot other lines
        for guide in non_erm_guidance:
            subset = data[
                (data["phi_nice"] == phi_val) &
                (data["k"] == k_val) &
                (data["guidance"] == guide)
            ]
            ax.plot(
                subset["rho"],
                subset["loss"],
                label=guide,
                marker=markers_dict[guide],
                linestyle="-" if guide != "DRO, no guidance" else "--",
                color=colors_dict[guide]
            )
# set axes
g.set(xscale="log")
g.set_axis_labels(r"$\rho$", "Binary log-loss")
g.set_titles(row_template=r"{row_name}", col_template=r"$k = {col_name}$")
# g.fig.suptitle("Concept shift", y=1.02)
# make legend
custom_handles = []
custom_labels = []
erm_handle = mlines.Line2D([], [], color="black", linestyle="--", label="ERM (LightGBM)")
custom_handles.append(erm_handle)
custom_labels.append("ERM (LightGBM)")

for guide in non_erm_guidance:
    handle = mlines.Line2D([], [], color=colors_dict[guide],
                           marker=markers_dict[guide],
                           linestyle="-" if guide != "None (simple DRO)" else "--",
                           label=guide)
    custom_handles.append(handle)
    custom_labels.append(guide)
g.fig.legend(
    handles=custom_handles,
    labels=custom_labels,
    bbox_to_anchor=(0.5, -0.02),
    loc="upper center",
    ncol=3,
    title=None,
    frameon=False,
)
# finalize plot
plt.tight_layout()
g.fig.subplots_adjust(bottom=0.03,top=0.95)

# save figure
plt.savefig(plotdir + f"concept_shift_losses.png", bbox_inches="tight", dpi=300)

