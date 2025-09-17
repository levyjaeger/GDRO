"""
    author: jaegerl
    created: 2025-08-30
    scope: figures for model fitting process
"""

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def plot_training_loss_traces(
    source = "miiv",
    targets = ["hirid", "zigong"],
    outcome = "mortality",
    n_rounds=1000,
    lossesdir = REPO_ROOT / "results" / "tables" / "icu",
    plotdir = REPO_ROOT / "results" / "figures" / "icu" / "loss_traces",
    rhos = [0.01, 0.1, 1],
    k = 2,
    height=6,
    aspect=1.2,
    ):
    
    # load the losses
    print("Loading losses...")
    losses_list = {}
    for target in targets:
        losses_df = pd.read_csv(lossesdir / f"losses_{source}_nrounds{n_rounds}_{outcome}.csv")
        losses_df = losses_df[losses_df["target"] == target]
        # convert to polars dataframe
        losses_df = pl.from_pandas(losses_df)
        # filter for rhos and ks
        losses_df = losses_df.with_columns([
            pl.col("dataset").str.extract(r"rho([0-9\.]+)_").alias("rho"),
            pl.col("dataset").str.extract(r"k(\d+)_").alias("k"),
            pl.col("dataset").str.extract(fr"{outcome}_(.*?)_").alias("guidance"),
        ])
        losses_df = losses_df.filter(pl.col("rho").is_in([str(rho) for rho in rhos]))
        losses_df = losses_df.filter(pl.col("k") == str(k))
        # remove everything in parentheses in the guidance column
        losses_df = losses_df.with_columns([
            pl.col("guidance").str.replace_all(r" \(.*?\)", "")
        ])
        losses_list[target] = losses_df
    losses_df = pl.concat(losses_list.values())
    
    print("Losses loaded. Plotting...")
    # Add row index per subgroup
    losses_df_plot = losses_df.to_pandas()
    losses_df_plot = losses_df_plot.groupby(["target", "rho", "guidance"]).apply(
        lambda g: g.assign(idx=range(len(g)))
    ).reset_index(drop=True)
    losses_df_plot["idx"] = losses_df_plot["idx"] + 1

    # change names of guidance
    losses_df_plot["guidance"] = losses_df_plot["guidance"].str.replace("renal", "mortality")
    guidance_order = [
        "DRO, no guidance",
        "DRO, marginal demographics",
        "DRO, stratum mortality",
        "DRO, subpopulation prevalence",
        "DRO, subpopulation mortality"
    ]
    guidance_names = {
        "Simple DRO": "DRO, no guidance",
        "Demographic, marginal": "DRO, marginal demographics",
        "Demographic, conditional": "DRO, stratum mortality",
        "Elderly, prevalence": "DRO, subpopulation prevalence",
        "Elderly, mortality": "DRO, subpopulation mortality",
    }
    losses_df_plot["guidance"] = losses_df_plot["guidance"].map(guidance_names)

    # change names of targets
    target_names = {
        "miiv": "MIMIC-IV",
        "eicu": "eICU",
        "hirid": "HiRID",
        "nwicu": "NWICU",
        "zigong": "ZFPH",
    }
    losses_df_plot["target"] = losses_df_plot["target"].map(target_names)
    target_order = losses_df_plot["target"].unique()
    # reorder target_order according to target_names keys
    target_order = [tn for tn in target_names.values() if tn in target_order]
    # change order of targets according to names of target_names
    losses_df_plot["target"] = pd.Categorical(losses_df_plot["target"], categories=target_order, ordered=True)

    ylabel = "Binary log-loss" if outcome in ["mortality", "renal"] else "MSE"

    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
    sns.set_context("talk", font_scale=1.2)
    g = sns.FacetGrid(
        losses_df_plot, 
        row="target", 
        col="rho", 
        hue="guidance", hue_order=guidance_order,
        margin_titles=True, 
        height=height, aspect=aspect,
        sharey="row", sharex=True)

    g.map(plt.plot, "idx", "loss")
    
    for ax in g.axes.flat:
        for line, guidance in zip(ax.get_lines(), guidance_order):
            if guidance == "DRO, no guidance":
                line.set_linestyle("--")
            else:
                line.set_linestyle("-")

    for ax, (target, rho) in zip(g.axes.flat, losses_df_plot.groupby(["target", "rho"], observed=False).size().index):
        ax.set_title(f"Target: {target}, $\\rho={rho}$")
        ax.set_xlabel("Boosting round")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, n_rounds)
        ax.title.set_text(None)

    g.set_titles(
        col_template=r"$\rho = {col_name}$", 
        row_template=r"Target: {row_name}",
        )
    # add custom legend
    handles = []
    for guidance in guidance_order:
        handles.append(mlines.Line2D([], [], color=sns.color_palette()[len(handles)], label=guidance, linestyle="--" if guidance=="DRO, no guidance" else "-"))
    # place legend at bottom in the center outside figure
    g.fig.legend(
        handles=handles, 
        bbox_to_anchor=(0.5, -0.1), 
        loc="lower center", 
        ncol=3, 
        title=None,
        frameon=False
        )
    plt.savefig(plotdir / f"trainingloss_traces{source}_{outcome}_nrounds{n_rounds}_k{k}.png", bbox_inches="tight", dpi=300)


# example: hirid and zigong mortality
plot_training_loss_traces(
    source="miiv",
    targets = ["hirid", "zigong"],
    outcome="mortality",
    n_rounds=1000,
    rhos = [0.01, 0.1, 1],
    k = 2,
    height=6,
    aspect=1.2,
    )
