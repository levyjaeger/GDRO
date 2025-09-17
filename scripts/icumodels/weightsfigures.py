"""
    author: jaegerl
    created: 2025-08-30
    scope: figures for weights after final iteration
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from pathlib import Path
import matplotlib.lines as mlines

REPO_ROOT = Path(__file__).resolve().parents[2]


""" 
    Plotting weight ECDFs
"""
def plot_weights_ecdf(
    source: str = "miiv",
    targets: list[str] = ["hirid", "zigong"],
    outcome: str = "mortality",
    n_rounds: int = 1000,
    rhos: list[float] = [.01, .1, 1],
    k: int = 2,
    plotdir: str = REPO_ROOT / "results" / "figures" / "icu" / "weights_density",
    weightsdir: str = REPO_ROOT / "results" / "tables" / "icu",
    height: float = 6,
    aspect: float = 1.5,
    horizontal: bool = True,
    ):
    
    # load the csv file with the weights
    print("Loading weights...")
    weights_list = {}
    for target in targets:
        weights_df = pd.read_csv(weightsdir / f"weights_{source}_nrounds{n_rounds}_{outcome}.csv")
        weights_df = weights_df[weights_df["target"] == target]
        # convert to polars dataframe
        weights_df = pl.from_pandas(weights_df)
        # filter for rhos and ks
        weights_df = weights_df.with_columns([
            pl.col("dataset").str.extract(r"rho([0-9\.]+)_").alias("rho"),
            pl.col("dataset").str.extract(r"k(\d+)_").alias("k"),
            pl.col("dataset").str.extract(fr"{outcome}_(.*?)_").alias("guidance"),
        ])
        weights_df = weights_df.filter(pl.col("rho").is_in([str(rho) for rho in rhos]))
        weights_df = weights_df.filter(pl.col("k") == str(k))
        # remove everything in parentheses in the guidance column
        weights_df = weights_df.with_columns([
            pl.col("guidance").str.replace_all(r" \(.*?\)", "")
        ])
        weights_list[target] = weights_df
    weights_df_all = pl.concat(list(weights_list.values()), how="vertical")

    print("Weights loaded. Plotting...")
    
    # latex fonts
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    # initialize plot
    sns.set_context("talk", font_scale=1.2)  # "talk" or "poster" are good presets
    df = weights_df_all.to_pandas()
    rho_order = sorted(df["rho"].unique())
    df["rho"] = pd.Categorical(df["rho"], categories=rho_order, ordered=True)
    df["weight_sqrt"] = np.sqrt(df["weight"])
    # replace the string "renal" with "mortality" in the guidance column
    df["guidance"] = df["guidance"].str.replace("renal", "mortality")

    guidance_names = {
        "Simple DRO": "DRO, no guidance",
        "Demographic, marginal": "DRO, marginal demographics",
        "Demographic, conditional": "DRO, stratum mortality",
        "Elderly, prevalence": "DRO, subgroup prevalence",
        "Elderly, mortality": "DRO, subgroup mortality",
    }
    guidance_order = guidance_names.values()
    df["guidance"] = df["guidance"].map(guidance_names)
    
    # change names of targets
    target_names = {
        "miiv": "MIMIC-IV",
        "eicu": "eICU",
        "hirid": "HiRID",
        "nwicu": "NWICU",
        "zigong": "ZFPH",
    }
    df["target"] = df["target"].map(target_names)
    target_order = df["target"].unique()
    # reorder target_order according to target_names keys
    target_order = [tn for tn in target_names.values() if tn in target_order]
    df["target"] = pd.Categorical(df["target"], categories=target_order, ordered=True)

    g = sns.FacetGrid(df, 
                      col="rho" if horizontal else "target", 
                      row="target" if horizontal else "rho", 
                      sharex=False, sharey=False, 
                      margin_titles=True,
                      hue="guidance", hue_order=guidance_order,
                      height=height, aspect=aspect)
    g.map_dataframe(
        sns.ecdfplot,
        x="weight_sqrt",
    )
    for ax in g.axes.flat:
        for line, guidance in zip(ax.get_lines(), guidance_order):
            if guidance == "DRO, no guidance":
                line.set_linestyle("--")
            else:
                line.set_linestyle("-")
    for ax in g.axes.flatten():
        ax.axvline(1, color="black", linestyle="--", zorder=0)
        ax.set_xlim(left=0)
        # get current ticks
        xticks = list(ax.get_xticks())
        # add 1.0 if not already present
        if 1.0 not in xticks:
            xticks.append(1.0)
            xticks = sorted(xticks)
        ax.set_xticks(xticks)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, 1.02)
        ax.yaxis.set_major_locator(mticker.FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
    
    handles = []
    for guidance in guidance_order:
        handles.append(mlines.Line2D([], [], color=sns.color_palette()[len(handles)], label=guidance, linestyle="--" if guidance=="DRO, no guidance" else "-"))
    g.fig.legend(
        handles=handles, 
        bbox_to_anchor=(0.5, -0.1), 
        loc="lower center", 
        ncol=3, 
        title=None,
        frameon=False
        )
    g.set_axis_labels(r"$\sqrt{n_S \cdot \mathrm{Weight}}$", "ECDF")
    g.set_titles(
        col_template=r"$\rho$ = {col_name}" if horizontal else r"Target: {col_name}", 
        row_template=r"Target: {row_name}" if horizontal else r"$\rho$ = {row_name}",
        )
    plt.tight_layout()

    plt.savefig(plotdir / f"weightsecdf_{source}to{target}_{outcome}_nrounds{n_rounds}.png", bbox_inches='tight', dpi=300)


# mortality example
plot_weights_ecdf(
    source="miiv",
    targets=["hirid", "zigong"],
    outcome="mortality",
    n_rounds=1000,
    rhos=[.01, .1, 1],
    k= 2,
    height=6, 
    aspect=1.2,
    horizontal=True,
    )

# creatinine example
plot_weights_ecdf(
    source="miiv",
    targets=["hirid", "zigong"],
    outcome="creatinine",
    n_rounds=1000,
    rhos=[.01, .1, 1],
    k= 2,
    height=6, 
    aspect=1.2,
    horizontal=True,
    )

