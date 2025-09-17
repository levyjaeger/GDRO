"""
    author: jaegerl
    created: 2025-08-30
    scope: collect weights from last iteration from all models of interest
"""

import os
import polars as pl
import pickle as pk
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

source = "miiv"
n_rounds = 1000
rhos = [0.01, 0.05, 0.1, 0.5, 1]

for outcome in ["mortality", "creatinine", "renal"]:
    weights_outcome = {}
    print("Processing outcome:", outcome)
    for target in ["eicu", "hirid", "miiv", "zigong", "nwicu"]:
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
        print("Model files loaded successfully. extracting weights.\n")
        # make dictionary of weights lists
        model_names = list(models.keys())
        weights_dict = {name: models[name].weights_list[-1] for name in model_names if hasattr(models[name], 'weights_list')}
        weights_df = pl.DataFrame({
            "dataset": [k for k, arr in weights_dict.items() for _ in arr],
            "weight": [v for arr in weights_dict.values() for v in arr]
        })
        weights_df = weights_df.with_columns([
            pl.col("dataset").str.extract(r"rho([0-9\.]+)_").alias("rho"),
            pl.col("dataset").str.extract(r"k(\d+)_").alias("k"),
            pl.col("dataset").str.extract(fr"{outcome}_(.*?)_").alias("guidance"),
        ])
        # remove everything in parentheses in the guidance column
        weights_df = weights_df.with_columns([
            pl.col("guidance").str.replace_all(r" \(.*?\)", "")
        ])
        # add colunn for outcome, source, and target
        weights_df = weights_df.with_columns([
            pl.lit(outcome).alias("outcome"),
            pl.lit(source).alias("source"),
            pl.lit(target).alias("target"),
        ])
        weights_outcome[target] = weights_df
    weights_outcome_df = pl.concat(list(weights_outcome.values()))
    print("  Writing weights for outcome:", outcome)
    weights_outcome_df.write_csv(REPO_ROOT / "results" / "tables" / "icu" / f"weights_{source}_nrounds{n_rounds}_{outcome}.csv")

print("All done.")