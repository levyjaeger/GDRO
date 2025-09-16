"""
    author: jaegerl
    created: 2025-08-08
    scope: simulations for concept shift
"""

import numpy as np
import pandas as pd
import polars as pl
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm
import lightgbm as lgb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / ""))
import gdro.ICUdata as gdrodata
import gdro.model as gdromodel
import gdro.deploy as gdrodeploy


""" 
    Simulation parameters
"""

# values of rho
rhos = [0.1, 1, 10, 100]
# values of k
ks = [1, 2, 4]
# number of boosting rounds
n_rounds = 350
xrounds = np.arange(1, n_rounds + 1)

# number of features
ncontfeatures = 2
ncatfeatures = 2
nfeatures = ncontfeatures + ncatfeatures

# simulate the beta coefficients for the linear model
np.random.seed(42)
betas = np.random.normal(size=nfeatures)
# normalize to make beta lie on the unit sphere
betas /= np.linalg.norm(betas)
# define a vector orthogonal to beta
v = np.random.normal(size=nfeatures)
v -= v.dot(betas) * betas
v /= np.linalg.norm(v)

phi_list = {
    "pi8": np.pi / 8,
    "pi4": np.pi / 4,
    "3pi8": 3 * np.pi / 8,
    "pi2": np.pi / 2,
    }
phi_titles = {
    "pi8": r"$\phi = \pi/8$",
    "pi4": r"$\phi = \pi/4$",
    "3pi8": r"$\phi = 3\pi/8$",
    "pi2": r"$\phi = \pi/2$",
    }

# define the effective number of training data points
ntrain = 2000
# define the effective number of test data points
ntest = 10000
# define the train-validate split ratio for source data
val_split_source = 0.5
# define the train-validate split ratio for target data
val_split_target = 0.5
# determine the number of observations after accounting for the splits
ntrain = int(ntrain / (1 - val_split_source))
ntest = int(ntest / (1 - val_split_target))

# lightgbm parameters
lgbm_params = {
    "objective": "binary",
    "max_depth": 3,
    "learning_rate": 0.01,
    "verbose": -1
    }

# graphic parameters
rhomarkers = ["o", "s", "D", "^", "v", "P", "*", "X", "<", ">"]


"""
    Generate source dataset
"""

# generate a training dataset
sourcedata = gdromodel.simulate_gdro_data(
    n_obs=ntrain, 
    n_cat_features=ncatfeatures,
    n_cont_features=ncontfeatures,
    betas=betas,
    outc_type="binary",
    random_seed=42,
    prop_shift=0,
    outc_shift=False,
    cont_shift=0,
    val_split=val_split_source,
    covar_shift=0,
    n_covar_shift=0,
    )
sourcedata = gdromodel.preprocess_lgbm(
    sourcedata["dataset"], outcome="outcome",
    )
trainingdata = sourcedata["training_data"]
validationdata = sourcedata["validation_data"]
yval = sourcedata["yval"]
ytrain = sourcedata["ytrain"]


"""
    Simulations for different angles phi
"""

losses_list_all = {}
for phi_name, phi in phi_list.items():
    print(f"\nRunning simulation for phi = {phi_name}")
    
    # define the shifted beta coefficients
    betas_shifted = betas * np.cos(phi) + v * np.sin(phi)
    # generate a test dataset
    targetdata = gdromodel.simulate_gdro_data(
        n_obs=ntest,
        n_cat_features=ncatfeatures,
        n_cont_features=ncontfeatures,
        betas=betas_shifted,
        outc_type="binary",
        random_seed=53,
        prop_shift=0,
        outc_shift=False,
        cont_shift=0,
        val_split=val_split_target,
        covar_shift=0,
        n_covar_shift=0,
        )
    targetdata = gdromodel.preprocess_lgbm(
        targetdata["dataset"], outcome="outcome",
        )
    # generate guidance set
    guidedata = targetdata["validation_data"]
    yguide = targetdata["yval"]
    # generate test set
    testdata = targetdata["training_data"]
    ytest = targetdata["ytrain"]

    # fit simple LightGBM model as benchmark
    print("\nFitting simple LightGBM model...")
    lgbm_train = lgb.Dataset(trainingdata, label=ytrain)
    lgbm_val = lgb.Dataset(validationdata, label=yval)
    lgbm_test = lgb.Dataset(testdata, label=ytest)
    evals_result = {}
    lgbm_model = lgb.train(
        lgbm_params,
        lgbm_train,
        num_boost_round=n_rounds,
        valid_sets=[lgbm_test, lgbm_val],
        callbacks=[lgb.record_evaluation(evals_result)],
        )
    # extract the test set loss
    test_loss_lgbm = evals_result["valid_0"]["l2"] if lgbm_params["objective"] == "regression" else evals_result["valid_0"]["binary_logloss"]
    val_loss_lgbm = evals_result["valid_1"]["l2"] if lgbm_params["objective"] == "regression" else evals_result["valid_1"]["binary_logloss"]
    # plot each validation loss over boosting rounds
    plt.plot(xrounds, test_loss_lgbm, label="Test loss")
    plt.plot(xrounds, val_loss_lgbm, label="Validation loss")
    plt.xlabel("Boosting round")
    plt.ylabel("MSE") if lgbm_params["objective"] == "regression" else plt.ylabel("Binary log-loss")
    plt.title("MSE of simple LightGBM model, " + phi_titles[phi_name]) if lgbm_params["objective"] == "regression" else plt.title("Binary log-loss of simple LightGBM model, " + phi_titles[phi_name])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # save the plot
    plt.savefig(REPO_ROOT / "exploration" / "simulations" / f"concept_shift_lgbm_loss_phi{phi_name}.png", bbox_inches="tight", dpi=300)
    plt.clf()

    # compute guidance matrices
    print("\nComputing guidance matrices...")
    trainingdata_grouped = trainingdata.with_columns(
        pl.when(pl.col("continuous__continuous_0") > 0)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("x0_group"),
        pl.when(pl.col("continuous__continuous_1") > 0)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("x1_group"),
        pl.Series(name="outcome", values=ytrain)
        )
    guidedata_grouped = guidedata.with_columns(
        pl.when(pl.col("continuous__continuous_0") > 0)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("x0_group"),
        pl.when(pl.col("continuous__continuous_1") > 0)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("x1_group"),
        pl.Series(name="outcome", values=yguide)
        )
    # based on the average of the outcome variable
    guidance_y = gdromodel.constr_avg(
        source=trainingdata_grouped,
        target=guidedata_grouped,
        variable="outcome",
        var_type="continuous",
        normalize=False
        )
    # based on the average of the features
    guidance_xcat0 = gdromodel.constr_avg(
        source=trainingdata_grouped,
        target=guidedata_grouped,
        variable="categorical__categorical_0",
        var_type="categorical",
        normalize=False
        )
    guidance_xcat1 = gdromodel.constr_avg(
        source=trainingdata_grouped,
        target=guidedata_grouped,
        variable="continuous__continuous_0",
        var_type="continuous",
        normalize=False
        )
    guidance_x = np.vstack((guidance_xcat0, guidance_xcat1))
    # outcome conditioned on features
    guidance_ycondcont0 = gdromodel.constr_avg_by_group(
        source=trainingdata_grouped,
        target=guidedata_grouped,
        variable="outcome",
        group="categorical__categorical_0",
        normalize=False,
        )
    guidance_ycondcont1 = gdromodel.constr_avg_by_group(
        source=trainingdata_grouped,
        target=guidedata_grouped,
        variable="outcome",
        group="x0_group",
        normalize=False,
        )
    guidance_ycond = np.vstack((guidance_ycondcont1, guidance_ycondcont0))

    # define list of guidance matrices
    guidance_matrices = {
        "None (simple DRO)": None,
        "Features": guidance_x,
        "Outcome, marginal": guidance_y,
        "Outcome, conditional": guidance_ycond,
        }

    # fit the GDRO models
    print("\nFitting GDRO models...")
    models_list = {}
    for k in ks:
        models_k = {}
        for rho in rhos:
            models_rho = {}
            for guidance_name, guidance_matrix in guidance_matrices.items():
                gdro_object = gdromodel.GuidedDRO(
                    num_boost_round=n_rounds,
                    rho=rho,
                    k=k,
                    lgbm_params=lgbm_params,
                    )
                # fit the model
                print(f"\nFitting GDRO model with rho={rho}, k={k}, guidance={guidance_name}, epsilon=0")
                gdro_object.fit(
                    X=trainingdata,
                    y=ytrain,
                    Xval=testdata,
                    yval=ytest,
                    guide_matrix=guidance_matrix,
                    epsilon=0,
                    )
                models_rho[f"{guidance_name}"] = gdro_object
            models_k[f"{rho}"] = models_rho
        models_list[f"{k}"] = models_k
    # extract the losses from each model
    losses_list = {}
    for k in ks:
        models_k = models_list[f"{k}"]
        losses_rho = {}
        for rho, models_rhok in models_k.items():
            losses_rhok = {}
            for guidance_name, model in models_rhok.items():
                losses_rhok[guidance_name] = np.mean(model.val_losses[-1])
            losses_rhok = pd.DataFrame.from_dict(losses_rhok, orient="index").reset_index()
            losses_rhok.rename(columns={"index": "guidance"}, inplace=True)
            # add rho and k to the dataframes
            losses_rhok["rho"] = rho
            losses_rhok["k"] = k
            # put the dataframes into the dictionaries  
            losses_rho[f"{rho}"] = losses_rhok
        # concatenate the dataframes for each k
        losses_list[f"{k}"] = pd.concat(losses_rho.values(), ignore_index=True)
    losses_df = pd.concat(
        [losses_list[k].assign(k=k) for k in losses_list.keys()],
        ignore_index=True
        )
    losses_df.rename(columns={0: "loss"}, inplace=True)
    # add a column for phi
    losses_df["phi"] = phi_name
    losses_df["phi_nice"] = phi_titles[phi_name]
    # add a row for the simple LightGBM model
    lgbm_row = pd.DataFrame({
        "guidance": ["ERM (LightGBM)"],
        "loss": [test_loss_lgbm[-1]],
        "rho": [np.nan],
        "k": [np.nan],
        "phi": [phi_name],
        "phi_nice": [phi_titles[phi_name]],
        })
    losses_df = pd.concat([losses_df, lgbm_row], ignore_index=True)
    losses_list_all[phi_name] = losses_df
    
# concatenate the dataframes
losses_df_all = pd.concat(losses_list_all.values(), ignore_index=True)
 
# save the losses dataframe
losses_df_all.to_csv(REPO_ROOT / "results" / "tables" / "simulations" / "losses_concept_shift.csv", index=False)


