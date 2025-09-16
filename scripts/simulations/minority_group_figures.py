"""
    author: jaegerl
    created: 2025-09-09
    scope: figures for hard subpopulation simulation
"""

import numpy as np
import pandas as pd
import polars as pl
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import matplotlib.lines as mlines
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


"""
    Losses versus rho panel
"""

print("drawing panel of losses for minority group simulation...")
# load the losses dataframe
losses_df = pl.read_csv(REPO_ROOT / "results"/ "tables"/ "simulations" / "losses_hard_subgroup.csv")

# replace "Simple DRO" with "No guidance" in the "guidance column"
losses_df = losses_df.with_columns([
    pl.when(pl.col("guidance") == "Simple DRO")
    .then(pl.lit("None (simple DRO)"))
    .otherwise(pl.col("guidance"))
    .alias("guidance")
])

# directory where to save the plots
plotdir = REPO_ROOT / "results" / "figures" / "simulations"

# data for plots
k_order = sorted(losses_df.filter(pl.col("k").is_not_null())["k"].unique().to_list())
prop_order = sorted(losses_df["prop_shift"].unique().to_list())
df = losses_df.to_pandas()
df["rho"] = df["rho"].astype(float)
df["k"] = pd.Categorical(df["k"], categories=k_order, ordered=True)
df["prop_shift"] = pd.Categorical(df["prop_shift"], categories=prop_order, ordered=True)

# extract ERM baselines
erm = df[df["guidance"] == "ERM (LightGBM)"]
baseline_overall = erm.groupby("prop_shift", observed=False)["loss_overall"].first().to_dict()
baseline_subgroup = erm.groupby("prop_shift", observed=False)["loss_subgroup"].first().to_dict()
data = df[df["guidance"] != "ERM (LightGBM)"].copy()

# latex fonts
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
# prepare facet grid
sns.set_context("talk", font_scale=1.2)
g = sns.FacetGrid(
    data,
    row="prop_shift",
    col="k",
    margin_titles=True,
    sharex=True,
    sharey=False,
    height=5,
    aspect=2,
)
axes = np.atleast_2d(g.axes)
non_erm_guidance = [g for g in data["guidance"].unique()]
markers_dict = {g: m for g, m in zip(non_erm_guidance, ["o", "s", "D", "^", "v", "P", "*"])}
colors_dict = {g: c for g, c in zip(non_erm_guidance, sns.color_palette(n_colors=len(non_erm_guidance)))}
for i, prop_val in enumerate(prop_order):
    for j, k_val in enumerate(k_order):
        ax = axes[i, j]
        # plot ERM horizontal line
        yerm_overall = baseline_overall.get(prop_val, None)
        ax.axhline(y=yerm_overall, color="black", linestyle="-", linewidth=1.5)
        yerm_subgroup = baseline_subgroup.get(prop_val, None)
        ax.axhline(y=yerm_subgroup, color="black", linestyle="--", linewidth=1.5)
        # plot other lines
        for guide in non_erm_guidance:
            subset = data[
                (data["prop_shift"] == prop_val) &
                (data["k"] == k_val) &
                (data["guidance"] == guide)
            ]
            ax.plot(
                subset["rho"],
                subset["loss_overall"],
                label=guide,
                marker=markers_dict[guide],
                linestyle="-",
                color=colors_dict[guide]
            )
            ax.plot(
                subset["rho"],
                subset["loss_subgroup"],
                label=guide,
                marker=markers_dict[guide],
                linestyle="--",
                color=colors_dict[guide]
            )
# set axes
g.set(xscale="log")
g.set_axis_labels(r"$\rho$", "MSE")
g.set_titles(row_template=r"$p = {row_name}$", col_template=r"$k = {col_name}$")
# g.fig.suptitle("Hard subpopulation", y=1.02)
# make legend
custom_handles = []
custom_labels = []
erm_handle_overall = mlines.Line2D([], [], color="black", linestyle="-", label="ERM, overall (LightGBM)")
erm_handle_subgroup = mlines.Line2D([], [], color="black", linestyle="--", label="ERM, minority group (LightGBM)")
custom_handles.append(erm_handle_overall)
custom_labels.append("ERM, overall (LightGBM)")
custom_handles.append(erm_handle_subgroup)
custom_labels.append("ERM, minority group (LightGBM)")
for guide in non_erm_guidance:
    handle = mlines.Line2D([], [], color=colors_dict[guide],
                           marker=markers_dict[guide],
                           linestyle="-",
                           label=guide)
    custom_handles.append(handle)
    custom_labels.append(guide + ", overall")
    handle = mlines.Line2D([], [], color=colors_dict[guide],
                           marker=markers_dict[guide],
                           linestyle="--",
                           label=guide)
    custom_handles.append(handle)
    custom_labels.append(guide + ", minority group")
g.fig.legend(
    handles=custom_handles,
    labels=custom_labels,
    bbox_to_anchor=(0.5, 0),
    loc="upper center",
    ncol=2,
    title=None,
    frameon=False,
)
# finalize plot
plt.tight_layout()
g.fig.subplots_adjust(bottom=0.08,top=0.95)

# save figure
plt.savefig(plotdir / "minority_group_losses.png", bbox_inches="tight", dpi=300)


"""
    Distribution of weights
"""

print("drawing panel of weights for minority group simulation...")

# load weights dataframe
weights_df = pl.read_csv(REPO_ROOT / "results/tables/simulations/weights_hard_subgroup.csv")

# select k=2, prop_shift=0.1, rho in {0.01, 1, 10} as an example
weights_subset = weights_df.filter(
    (pl.col("k") == 2) &
    (pl.col("prop_shift") == 0.2) &
    (pl.col("rho").is_in([1, 10, 100]))
).to_pandas()
# in the "subgroup" column, replace False with "Majority" and True with "Minority"
weights_subset["subgroup"] = weights_subset["subgroup"].map({False: "Majority", True: "Minority"})

sns.set_context("talk", font_scale=1.2)
# panels
g = sns.FacetGrid(
    weights_subset,
    row="rho",
    col="guidance",
    sharex="col",
    sharey="col",
    height=5,
    aspect=1.5,
    margin_titles=True,
    hue="subgroup",
    palette="Set2",
    legend_out=True
)
g.map_dataframe(sns.ecdfplot, x="weights")

# vertical line at equal weights
for ax in g.axes.flat:
    ax.axvline(x=1, color="black", linestyle="--", linewidth=1.5)
    
# axes
g.set_axis_labels(r"Weight (units of $1/n_S$)", "ECDF")
g.set_titles(row_template=r"$\rho={row_name}$", col_template="{col_name}")
g.set(xlim=(0, None))
for ax in g.axes.flat:
    ax.set_xticks(list(ax.get_xticks()) + [1])

# legend
if g._legend is not None:
    g._legend.remove()
handles, labels = g.axes[0,0].get_legend_handles_labels()
g.fig.legend(
    handles=handles,
    labels=labels,
    title="Group",
    loc="upper center",
    bbox_to_anchor=(0.5, 0.1),
    ncol=2,
    frameon=False,
)

# finalize plot
plt.tight_layout()
g.fig.subplots_adjust(bottom=0.2, top=0.9)

# save figure
plt.savefig(plotdir / "minority_group_weights.png", bbox_inches="tight", dpi=300)