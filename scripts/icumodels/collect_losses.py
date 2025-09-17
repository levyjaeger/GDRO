"""
    author: jaegerl
    created: 2025-08-30
    scope: collect target set losses from all models of interest
"""

import os
import polars as pl
import pickle as pk
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


""""
    Collect the target set losses and save to a csv file.
"""

source = "miiv"
n_rounds = 1000
rhos = [0.01, 0.1, 1]

for outcome in ["mortality", "creatinine", "renal"]:
    losses_outcome = {}
    print("Processing outcome:", outcome)
    for target in ["hirid", "miiv", "zigong"]:
        print("  Processing target:", target)
        model_files = [
            f for f in os.listdir(REPO_ROOT / "results" / "models") if os.path.isfile(os.path.join(REPO_ROOT / f"results" / "models", f)) and f"{source}to{target}_{outcome}" in f and f"nrounds{n_rounds}" in f and f.endswith(".pkl") and any(f"rho{rho}" in f for rho in rhos)
        ]
        # load the models into a dictionary
        models = {}
        for file in model_files:
            with open(REPO_ROOT / "results" / "models" /f"{file}", "rb") as f:
                model = pk.load(f)
                models[file] = model
        print("Model files loaded successfully. extracting losses.\n")
        # make dictionary of losses lists
        model_names = list(models.keys())
        losses_dict = {name: ([np.mean(ll) for ll in models[name].train_losses]) for name in model_names if hasattr(models[name], 'train_losses')}
        losses_df = pl.DataFrame({
            "dataset": [k for k, arr in losses_dict.items() for _ in arr],
            "loss": [v for arr in losses_dict.values() for v in arr]
        })
        losses_df = losses_df.with_columns([
            pl.col("dataset").str.extract(r"rho([0-9\.]+)_").alias("rho"),
            pl.col("dataset").str.extract(r"k(\d+)_").alias("k"),
            pl.col("dataset").str.extract(fr"{outcome}_(.*?)_").alias("guidance"),
        ])
        # remove everything in parentheses in the guidance column
        losses_df = losses_df.with_columns([
            pl.col("guidance").str.replace_all(r" \(.*?\)", "")
        ])
        # add colunn for outcome, source, and target
        losses_df = losses_df.with_columns([
            pl.lit(outcome).alias("outcome"),
            pl.lit(source).alias("source"),
            pl.lit(target).alias("target"),
        ])
        losses_outcome[target] = losses_df
    losses_outcome_df = pl.concat(list(losses_outcome.values()))
    print("  Writing losses for outcome:", outcome)
    losses_outcome_df.write_csv(REPO_ROOT / "results" / "tables" / "icu" / f"losses_{source}_nrounds{n_rounds}_{outcome}.csv")

print("All done.")