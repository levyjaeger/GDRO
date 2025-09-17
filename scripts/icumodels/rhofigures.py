"""
    author: jaegerl
    created: 2025-08-30
    scope: figures for model evaluation (loss and AUROC, if applicable), versus log10(rho)
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


"""
    Panel of losses versus rho
"""

def plot_loss_rho(
    outcome: str = "renal",
    source: str = "miiv",
    targets: list[str] = ["eicu", "hirid", "miiv", "zigong", "nwicu"],
    tabledir: str = REPO_ROOT / "results" / "tables" / "icu",
    plotdir: str = REPO_ROOT / "results" / "figures" / "icu" / "loss_vs_rho/",
    n_rounds: int = 1000,
    ks: list[float] = [1, 2],
    rhos: list[float] = [0.01, 0.1, 0.2, 0.5, 1],
    height: float = 5,
    aspect: float = 2,
    horizontal: bool = True,
    losstype: str = "loss"
    ):
    
    # latex fonts
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    
    target_tables = []
    # get the names of the model files
    for target in targets:
        print(f"processing target {target}\n")
        model_tables = [
            f for f in os.listdir(f"{tabledir}/{source}to{target}_{outcome}") 
            if f.startswith(f"subgroupeval_{source}to{target}") 
            and f.endswith(".csv")
            and f"nrounds{n_rounds}" in f
            ] 
        # load all tables into a dictionary
        table_list = {}
        for table_path in model_tables:
            table = pd.read_csv(f"{tabledir}/{source}to{target}_{outcome}/" + table_path)
            # extract the rho, k value and nrounds from the file name
            rho_value = float(table_path.split("rho")[1].split("_")[0])
            k_value = int(table_path.split("k")[1].split("_")[0])
            nrounds_table = int(table_path.split("nrounds")[1].split(".csv")[0])
            table_list[(rho_value, k_value, nrounds_table)] = table
        # merge all tables into a single dataframe, using an additional column for rho and an additional column for k
        merged_table = pd.DataFrame()
        for rho in rhos:
            for k in ks:
                if (rho, k, n_rounds) not in table_list:
                    print(f"missing table for rho={rho}, k={k}, n_rounds={n_rounds}, target={target}")
                    continue
                table = table_list[(rho, k, n_rounds)]
                table["rho"] = rho  # add rho column
                table["k"] = k  # add k column
                merged_table = pd.concat([merged_table, table], ignore_index=True)
        # sort the merged table by rho
        merged_table.sort_values(by="rho", inplace=True)
        # in the column guide, remove everything in parentheses
        merged_table["guide"] = merged_table["guide"].str.replace(r" \(.*\)", "", regex=True).str.strip()
        # make column with target name
        merged_table["target"] = target
        target_tables.append(merged_table)
    # put all target tables together
    metric_table = pd.concat(target_tables, ignore_index=True)

    # order k and target
    k_order = sorted(metric_table["k"].unique())
    # convert to categorical with fixed ordering
    metric_table["k"] = pd.Categorical(metric_table["k"], categories=k_order, ordered=True)
    # filter population
    df = metric_table[metric_table["population"] == "Overall"].copy()
    # replace "renal" by "mortality" in the guidance column
    df["guide"] = df["guide"].str.replace("renal", "mortality")
    
    # convert names of datasets
    dataset_names = {
        "eicu": "eICU",
        "hirid": "HiRID",
        "miiv": "MIMIC-IV",
        "zigong": "ZFPH",
        "nwicu": "NWICU"
    }
    # convert names of guidance constraints
    guidance_names = {
        "LightGBM": "ERM (LightGBM)",
        "Simple DRO": "DRO, no guidance",
        "Demographic, marginal": "DRO, marginal demographics",
        "Demographic, conditional": "DRO, stratum mortality",
        "Elderly, prevalence": "DRO, subpopulation prevalence",
        "Elderly, mortality": "DRO, subpopulation mortality",
    }
    df["target"] = pd.Categorical(df["target"], categories=sorted(dataset_names.keys()), ordered=True)
    df["target"] = df["target"].cat.rename_categories(dataset_names)
    target_order = sorted(df["target"].unique())
    df["guide"] = df["guide"].map(guidance_names)
    # extract ERM baselines
    erm = df[df["guide"] == "ERM (LightGBM)"]
    baseline = erm.groupby("target", observed=False)[losstype].first().to_dict()
    data = df[df["guide"] != "ERM (LightGBM)"].copy()
    
    # set up the figure
    sns.set_context("talk", font_scale=1.5)
    g = sns.FacetGrid(
        data,
        col="target" if horizontal else "k",
        row="k" if horizontal else "target",
        margin_titles=True,
        sharex=True,
        sharey=False,
        height=height,
        aspect=aspect,
    )
    # create the panels
    axes = np.atleast_2d(g.axes)
    non_erm_guidance = [g for g in data["guide"].unique()]
    guide_order = [g for g in guidance_names.values() if g in non_erm_guidance]
    print(guide_order)
    markers_dict = {g: m for g, m in zip(guide_order, ["o", "s", "D", "^", "v", "P", "*"])}
    colors_dict = {g: c for g, c in zip(guide_order, sns.color_palette(n_colors=len(guide_order)))}
    for i, target_val in enumerate(target_order):
        for j, k_val in enumerate(k_order):
            ax = axes[j, i] if horizontal else axes[i, j]
            # plot ERM horizontal line
            yerm_overall = baseline.get(target_val, None)
            ax.axhline(y=yerm_overall, color="black", linestyle="--", linewidth=1.5)
            # plot other lines
            for guide in guide_order:
                subset = data[
                    (data["target"] == target_val) &
                    (data["k"] == k_val) &
                    (data["guide"] == guide)
                ]
                ax.plot(
                    subset["rho"],
                    subset[losstype],
                    label=guide,
                    marker=markers_dict[guide],
                    linestyle="-" if guide != "DRO, no guidance" else "--",
                    color=colors_dict[guide]
                )
        
    g.set(xscale="log")
    
    # make legend
    custom_handles = []
    custom_labels = []
    erm_handle = mlines.Line2D([], [], color="black", linestyle="--", label="ERM (LightGBM)")
    custom_handles.append(erm_handle)
    custom_labels.append("ERM (LightGBM)")
    
    for guide in guide_order:
        handle = mlines.Line2D([], [], color=colors_dict[guide],
                            marker=markers_dict[guide],
                            linestyle="-" if guide != "DRO, no guidance" else "--",
                            label=guide)
        custom_handles.append(handle)
        custom_labels.append(guide)
    g.fig.legend(
        handles=custom_handles,
        labels=custom_labels,
        bbox_to_anchor=(0.5, 0),
        loc="upper center",
        ncol=3,
        title=None,
        frameon=False,
    )
    if losstype == "loss":
        ylabel = "Binary log-loss" if outcome in ["mortality", "renal"] else "MSE"
    else:
        ylabel = "AUROC"
    g.set_axis_labels(r"$\rho$", ylabel)
    if horizontal:
        g.set_titles(row_template=r"$k = ${row_name}", col_template="{col_name}")
    else:
        g.set_titles(row_template="Target: {row_name}", col_template=r"$k$ = {col_name}")
    plt.tight_layout()
    plt.savefig(plotdir + f"{losstype}_vs_rho_panel_{source}toall_{outcome}_nrounds{n_rounds}.png", bbox_inches='tight', dpi=300)


# renal
plot_loss_rho(
    outcome="renal",
    source="miiv",
    targets=["eicu", "hirid", "miiv", "zigong", "nwicu"],
    tabledir="04_tables/",
    plotdir="05_icu_analyses/loss_vs_rho/",
    n_rounds=1000,
    ks=[1, 2],
    rhos=[0.01, 0.05, 0.1, 0.5, 1],
    aspect=1.2, height=6,
    horizontal=True,
)

# mortality
plot_loss_rho(
    outcome="mortality",
    source="miiv",
    targets=["eicu", "hirid", "miiv", "zigong", "nwicu"],
    n_rounds=1000,
    ks=[1, 2],
    rhos=[0.01, 0.05, 0.1, 0.5, 1],
    aspect=1.2, height=6,
)

# creatinine
plot_loss_rho(
    outcome="creatinine",
    source="miiv",
    targets=["eicu", "hirid", "miiv", "zigong", "nwicu"],
    n_rounds=1000,
    ks=[1, 2],
    rhos=[0.01, 0.05, 0.1, 0.5, 1],
    aspect=1.2, height=6,
)

# auroc
plot_loss_rho(
    outcome="renal",
    source="miiv",
    targets=["eicu", "hirid", "miiv", "zigong", "nwicu"],
    n_rounds=1000,
    ks=[1, 2],
    rhos=[0.01, 0.05, 0.1, 0.5, 1],
    aspect=1.2, height=6,
    horizontal=True,
    losstype="auroc"
)

plot_loss_rho(
    outcome="mortality",
    source="miiv",
    targets=["eicu", "hirid", "miiv", "zigong", "nwicu"],
    n_rounds=1000,
    ks=[1, 2],
    rhos=[0.01, 0.05, 0.1, 0.5, 1],
    aspect=1.2, height=6,
    losstype="auroc"
)


"""
    Performance on hard sub-population
"""

def plot_subgroup_loss_rho(
    source: str = "miiv",
    target: str = "miiv",
    outcome: str = "mortality",
    n_rounds: int = 1000,
    tabledir: str = REPO_ROOT / "results" / "tables" / "icu",
    plotdir: str = REPO_ROOT / "results" / "figures" / "icu" / "loss_vs_rho/",
    rhos: list[int] = [0.01, 0.1, 1],
    ks: list[int] = [1, 2],
    height: float = 6,
    aspect: float = 1.5,
    losstype: str = "loss"
    ):
    # get all model files
    model_tables = [
        f for f in os.listdir(tabledir / f"{source}to{target}_{outcome}") 
        if f.startswith(f"subgroupeval_{source}to{target}") 
        and f.endswith(".csv")
        and f"nrounds{n_rounds}" in f
        ] 
    # load all tables into a dictionary
    table_list = {}
    for table_path in model_tables:
        table = pd.read_csv(f"{tabledir}/{source}to{target}_{outcome}/" + table_path)
        # extract the rho, k value and nrounds from the file name
        rho_value = float(table_path.split("rho")[1].split("_")[0])
        k_value = int(table_path.split("k")[1].split("_")[0])
        nrounds_table = int(table_path.split("nrounds")[1].split(".csv")[0])
        table_list[(rho_value, k_value, nrounds_table)] = table

    # merge all tables into a single dataframe, using an additional column for rho and an additional column for k
    merged_table = pd.DataFrame()
    for rho in rhos:
        for k in ks:
            if (rho, k, n_rounds) not in table_list:
                print(f"missing table for rho={rho}, k={k}, n_rounds={n_rounds}")
                continue
            table = table_list[(rho, k, n_rounds)]
            table["rho"] = rho  # add rho column
            table["k"] = k  # add k column
            merged_table = pd.concat([merged_table, table], ignore_index=True)
    # sort the merged table by rho
    merged_table.sort_values(by="rho", inplace=True)
    # in the column guide, remove everything in parentheses
    merged_table["guide"] = merged_table["guide"].str.replace(r" \(.*\)", "", regex=True).str.strip()
    # in the column guide, replace every "renal" by "mortality"
    merged_table["guide"] = merged_table["guide"].str.replace("renal", "mortality")
    # order k
    k_order = sorted(merged_table["k"].unique())
    # convert to categorical with fixed ordering
    merged_table["k"] = pd.Categorical(merged_table["k"], categories=k_order, ordered=True)
    # convert names of guidance constraints
    guidance_names = {
        "LightGBM": "ERM (LightGBM)",
        "Simple DRO": "DRO, no guidance",
        "Demographic, marginal": "DRO, marginal demographics",
        "Demographic, conditional": "DRO, stratum mortality",
        "Elderly, prevalence": "DRO, subpopulation prevalence",
        "Elderly, mortality": "DRO, subpopulation mortality",
    }
    merged_table["guide"] = merged_table["guide"].map(guidance_names)
    # convert names of populations
    population_names = {
        "Overall": "All patients",
        "Subgroup": r"Patients aged $\geq 75$ years",
    }
    merged_table["population"] = merged_table["population"].map(population_names)
    
    erm = merged_table[merged_table["guide"] == "ERM (LightGBM)"]
    baseline = erm.groupby("population", observed=False)[losstype].first().to_dict()
    merged_table = merged_table[merged_table["guide"] != "ERM (LightGBM)"].copy()
    
    sns.set_context("talk", font_scale=1.5)
    g = sns.FacetGrid(
        merged_table, 
        row="k", col="population",
        margin_titles=True, 
        sharex=True, sharey="row",
        hue="guide",
        height=height, aspect=aspect
    )
    axes = np.atleast_2d(g.axes)
    guide_order = guidance_names.values()
    non_erm_guidance = [g for g in merged_table["guide"].unique()]
    non_erm_guidance = [g for g in guide_order if g in non_erm_guidance]
    markers_dict = {g: m for g, m in zip(non_erm_guidance, ["o", "s", "D", "^", "v", "P", "*"])}
    colors_dict = {g: c for g, c in zip(non_erm_guidance, sns.color_palette(n_colors=len(non_erm_guidance)))}
    population_order = [population_names["Overall"], population_names["Subgroup"]]
    for i, population_val in enumerate(population_order):
        for j, k_val in enumerate(k_order):
            ax = axes[j, i]
            # plot ERM horizontal line
            yerm_overall = baseline.get(population_val, None)
            ax.axhline(y=yerm_overall, color="black", linestyle="--", linewidth=1.5)
            # plot other lines
            for guide in non_erm_guidance:
                subset = merged_table[
                    (merged_table["population"] == population_val) &
                    (merged_table["k"] == k_val) &
                    (merged_table["guide"] == guide)
                ]
                ax.plot(
                    subset["rho"],
                    subset[losstype],
                    label=guide,
                    marker=markers_dict[guide],
                    linestyle="-" if guide != "DRO, no guidance" else "--",
                    color=colors_dict[guide]
                )
    
    g.set(xscale="log")
    
    # make legend
    custom_handles = []
    custom_labels = []
    erm_handle = mlines.Line2D([], [], color="black", linestyle="--", label="ERM (LightGBM)")
    custom_handles.append(erm_handle)
    custom_labels.append("ERM (LightGBM)")
    
    for guide in non_erm_guidance:
        handle = mlines.Line2D([], [], color=colors_dict[guide],
                               marker=markers_dict[guide],
                               linestyle="-" if guide != "DRO, no guidance" else "--",
                               label=guide)
        custom_handles.append(handle)
        custom_labels.append(guide)
    g.fig.legend(
        handles=custom_handles,
        labels=custom_labels,
        bbox_to_anchor=(0.5, 0),
        loc="upper center",
        ncol=3,
        title=None,
        frameon=False,
    )
    
    if losstype == "loss":
        ylabel = "Binary log-loss" if outcome in ["mortality", "renal"] else "MSE"
    else:
        ylabel = "AUROC"
    g.set_axis_labels(r"$\rho$", ylabel)
    g.set_titles(row_template=r"$k = ${row_name}", col_template="{col_name}")
    plt.tight_layout()
    plt.savefig(plotdir / f"{losstype}_vs_rho_panel_populations_{source}to{target}_{outcome}_nrounds{n_rounds}.png", bbox_inches='tight', dpi=300)

# losses
plot_subgroup_loss_rho(
    source = "miiv",
    target = "miiv",
    outcome = "mortality",
    n_rounds = 1000,
    rhos = [0.01, 0.05, 0.1, 0.5, 1],
    ks = [1, 2],
    losstype="loss"
    )

plot_subgroup_loss_rho(
    source = "miiv",
    target = "miiv",
    outcome = "renal",
    n_rounds = 1000,
    rhos = [0.01, 0.05, 0.1, 0.5, 1],
    ks = [1, 2],
    losstype="loss"
    )

plot_subgroup_loss_rho(
    source = "miiv",
    target = "miiv",
    outcome = "creatinine",
    n_rounds = 1000,
    rhos = [0.01, 0.05, 0.1, 0.5, 1],
    ks = [1, 2],
    losstype="loss"
    )

# auroc
plot_subgroup_loss_rho(
    source = "miiv",
    target = "miiv",
    outcome = "renal",
    n_rounds = 1000,
    rhos = [0.01, 0.05, 0.1, 0.5, 1],
    ks = [1, 2],
    losstype="auroc"
    )

plot_subgroup_loss_rho(
    source = "miiv",
    target = "miiv",
    outcome = "mortality",
    n_rounds = 1000,
    rhos = [0.01, 0.05, 0.1, 0.5, 1],
    ks = [1, 2],
    losstype="auroc"
    )
