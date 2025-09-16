#%%
"""
    author: jaegerl
    created: 2025-09-15
    scope: simulations for minority group
"""

import numpy as np
import pandas as pd
import polars as pl
import os
import pickle as pk
import sys
import importlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import lightgbm as lgb

sys.path.append("/Users/jaegerl/Documents/awesome_stuff/statistics_msc/fs25_master_thesis/analysis/01_code/")
import gdro.ICUdata as gdrodata
import gdro.model as gdromodel
import gdro.deploy as gdrodeploy

os.chdir("/Users/jaegerl/Documents/awesome_stuff/statistics_msc/fs25_master_thesis/analysis/")


#%%
"""
Hard minority group

"""

# reproducibility
np.random.seed(101)

# define values of rho and k to loop through
rhos = [1, 2, 5, 10, 20, 50, 100]
ks = [1, 2]
# minority group prevalence values to loop through
prop_list = [0.1, 0.2, 0.3]

# number of features
ncontfeatures = 2
ncatfeatures = 0
# number of boosting rounds
n_rounds = 100
# other lightgbm parameters
lgbm_params = {
    "objective": "regression",
    "max_depth": 3,
    "learning_rate": 0.05,
    "verbose": -1
    }
# number of training data points
ntrain = 1000
# number of guidance data points
nguide = 10000
# number of test data points
ntest = 10000
# standard deviation of noise term
sigma = .1

# sample the beta coefficients for the linear model
# betas = np.random.normal(size=ncontfeatures + ncatfeatures)
# betas = betas / np.linalg.norm(betas)
betas = np.array([1, 0.1])
betas_shifted = np.array([1, 2])


#%%
"""
Run simulations for different minority group prevalences
"""

importlib.reload(gdromodel)
losses_list = {}
weights_list = {}
for propshift in prop_list:
    # define train-validation split proportion
    val_split = 0.5
    # calculate the numbers of observations for the datasets before splitting
    ntraineff1 = int(ntrain * (1 - propshift) / (1 - val_split))
    ntraineff2 = int(ntrain * propshift / (1 - val_split))
    nguideeff1 = int(nguide * (1 - propshift) / (1 - val_split))
    nguideeff2 = int(nguide * propshift / (1 - val_split))
    ntesteff = int(ntest / (1 - val_split))

    # generate a source dataset
    sourcedata1 = gdromodel.simulate_gdro_data(
        n_obs=ntraineff1, 
        n_cat_features=ncatfeatures,
        n_cont_features=ncontfeatures,
        betas=betas,
        outc_type="continuous",
        random_seed=42,
        prop_shift=0,
        outc_shift=False,
        cont_shift=0,
        val_split=val_split,
        covar_shift=0,
        n_covar_shift=0,
        )
    # add column with subgroup indicator
    sourcedata1["dataset"] = sourcedata1["dataset"].with_columns(
        pl.Series(name="subgroup", values=np.zeros(sourcedata1["dataset"].height, dtype=int))
    )
    sourcedata2 = gdromodel.simulate_gdro_data(
        n_obs=ntraineff2, 
        n_cat_features=ncatfeatures,
        n_cont_features=ncontfeatures,
        betas=betas_shifted,
        outc_type="continuous",
        random_seed=53,
        prop_shift=0,
        outc_shift=False,
        val_split=val_split,
        covar_shift=0,
        n_covar_shift=0,
        )
    sourcedata2["dataset"] = sourcedata2["dataset"].with_columns(
        pl.Series(name="subgroup", values=np.ones(sourcedata2["dataset"].height, dtype=int))
    )
    sourcedata_all = pl.concat([sourcedata1["dataset"], sourcedata2["dataset"]], how="vertical")
    # extract training dataset
    sourcedata = gdromodel.preprocess_lgbm(
        source=sourcedata_all,
        outcome="outcome",
        )
    trainingdata = sourcedata["training_data"]
    print(f"Training data shape: {trainingdata.shape}\n\n")
    ytrain = sourcedata["ytrain"]
    # extract the subgroup column 
    trainingshifted = trainingdata["categorical__subgroup"].to_numpy().astype(bool)
    trainingdata = trainingdata.drop("categorical__subgroup")
    # extract validation dataset for lightgbm
    valdata = sourcedata["validation_data"]
    yval = sourcedata["yval"]
    valshifted = valdata["categorical__subgroup"].to_numpy().astype(bool)
    valdata = valdata.drop("categorical__subgroup")
    
    # simulate target dataset for guidance and test datasets
    targetdata1 = gdromodel.simulate_gdro_data(
        n_obs=nguideeff1, 
        n_cat_features=ncatfeatures,
        n_cont_features=ncontfeatures,
        betas=betas,
        outc_type="continuous",
        random_seed=101,
        prop_shift=propshift,
        outc_shift=False,
        val_split=val_split,
        covar_shift=0,
        n_covar_shift=0,
        )
    targetdata1["dataset"] = targetdata1["dataset"].with_columns(
        pl.Series(name="subgroup", values=np.zeros(targetdata1["dataset"].height, dtype=int))
    )
    targetdata2 = gdromodel.simulate_gdro_data(
        n_obs=nguideeff1, 
        n_cat_features=ncatfeatures,
        n_cont_features=ncontfeatures,
        betas=betas_shifted,
        outc_type="continuous",
        random_seed=123,
        prop_shift=propshift,
        outc_shift=False,
        val_split=val_split,
        covar_shift=0,
        n_covar_shift=0,
        )
    targetdata2["dataset"] = targetdata2["dataset"].with_columns(
        pl.Series(name="subgroup", values=np.ones(targetdata2["dataset"].height, dtype=int))
    )
    targetdata_all = pl.concat([targetdata1["dataset"], targetdata2["dataset"]], how="vertical")
    targetdata = gdromodel.preprocess_lgbm(
        source=targetdata_all,
        outcome="outcome",
        )
    # extract guidance dataset
    guidedata = targetdata["training_data"]
    print(f"Guidance data shape: {trainingdata.shape}\n\n")
    yguide = targetdata["ytrain"]
    guideshifted = guidedata["categorical__subgroup"].to_numpy().astype(bool)
    guidedata = guidedata.drop("categorical__subgroup")
    # extract test dataset
    testdata = targetdata["validation_data"]
    ytest = targetdata["yval"]
    testshifted = testdata["categorical__subgroup"].to_numpy().astype(bool)
    testdata = testdata.drop("categorical__subgroup")
    
    # define training dataset for lightgbm model
    lgbm_train = lgb.Dataset(
        trainingdata, 
        label=ytrain.astype(float))
    # define two validation sets: the overall validation set and the minority group set
    lgbm_val_overall = lgb.Dataset(valdata, label=yval)
    lgbm_val_shifted = lgb.Dataset(valdata.filter(valshifted), label=yval[valshifted])
    lgbm_test_overall = lgb.Dataset(testdata, label=ytest)
    lgbm_test_shifted = lgb.Dataset(testdata.filter(testshifted), label=ytest[testshifted])
    # train LightGBM model
    evals_result = {}
    lgbm_model = lgb.train(
        lgbm_params,
        lgbm_train,
        num_boost_round=n_rounds,
        valid_sets=[lgbm_test_overall, lgbm_test_shifted],
        callbacks=[lgb.record_evaluation(evals_result)],
        )
    # extract the test losses
    test_loss_overall = evals_result["valid_0"]["l2"] if "l2" in evals_result["valid_0"] else evals_result["valid_0"]["binary_logloss"]
    test_loss_shifted = evals_result["valid_1"]["l2"] if "l2" in evals_result["valid_1"] else evals_result["valid_1"]["binary_logloss"]
    # plot the losses
    plt.figure(figsize=(8, 5))
    plt.plot(test_loss_overall, label="Overall")
    plt.plot(test_loss_shifted, label="Minority group", linestyle="--")
    plt.xlabel("Boosting round")
    plt.ylabel("MSE")
    plt.title(f"LightGBM test losses, minority group prevalence {propshift}")
    plt.legend()
    plt.grid()
    plt.savefig(f"05_simulations/03_hard_subgroup/figures/lgbm_losses_prop{propshift}.png", bbox_inches="tight")
    
    # build guidance matrices
    trainingdata_grouped = trainingdata.with_columns(
        pl.Series(name="grouping_minority", values=trainingshifted.astype(int)),
        pl.Series(name="outcome", values=ytrain)
        )
    guidedata_grouped = guidedata.with_columns(
        pl.Series(name="grouping_minority", values=guideshifted.astype(int)),
        pl.Series(name="outcome", values=yguide)
        )
    # prevalence of minority group
    guidance_prev_subgroup = gdromodel.constr_avg(
        source=trainingdata_grouped,
        target=guidedata_grouped,
        variable="grouping_minority",
        var_type="categorical",
        normalize=False,
        )
    # outcome conditioned on minority group membership
    guidance_cond_subgroup = gdromodel.constr_avg_by_group(
        source=trainingdata_grouped,
        target=guidedata_grouped,
        variable="outcome",
        group="grouping_minority",
        normalize=False,
        )
    # stack guidance_cond_subgroup on top of guidance_prev_subgroup
    guidance_cond_subgroup = np.vstack([guidance_prev_subgroup, guidance_cond_subgroup])
    # define list of guidance matrices
    guidance_matrices = {
        "DRO, no guidance": None,
        "DRO, minority group constraints": guidance_cond_subgroup,
        }
    print("Fitting the GDRO models...\n")
    models_list = {}
    for k in ks:
        models_k = {}
        for rho in rhos:
            models_rho = {}
            for guidance_name, guidance_matrix in guidance_matrices.items():
                # create a GDRO object
                gdro_object = gdromodel.GuidedDRO(
                    num_boost_round=n_rounds,
                    rho=rho,
                    k=k,
                    lgbm_params=lgbm_params,
                    )
                # fit the model
                print(f"\nFitting GDRO model for minority group prevalence {propshift} with rho={rho}, k={k}, guidance={guidance_name}, epsilon=0")
                gdro_object.fit(
                    X=trainingdata,
                    y=ytrain,
                    Xval=testdata,
                    yval=ytest,
                    guide_matrix=guidance_matrix,
                    epsilon=0,
                    )
                # put the model into the dictionary
                models_rho[f"{guidance_name}"] = gdro_object
                # save the model
                print("Saving GDRO model...")
                with open(os.path.join("05_simulations/03_hard_subgroup/models", f"model_rho{rho}_k{k}_guidance{guidance_name}.pkl"), "wb") as f:
                    pk.dump(gdro_object, f)
            models_k[f"{rho}"] = models_rho
        models_list[f"{k}"] = models_k
    # extract overall and subgroup losses from the fitted models
    print("Extracting losses and weights...\n")
    loss_overall_list = {}
    loss_subgroup_list = {}
    loss_subgroup_list = {}
    last_weights_list = {}
    for k in ks:
        models_k = models_list[f"{k}"]
        loss_overall_rho = {}
        loss_subgroup_rho = {}
        last_weights_rho = {}
        for rho, models_rhok in models_k.items():
            loss_overall_rhok = {}
            loss_subgroup_rhok = {}
            last_weights_rhok = {}
            for guidance_name, model in models_rhok.items():
                loss_overall_rhok[guidance_name] = np.mean(model.val_losses[-1])
                loss_subgroup_rhok[guidance_name] = np.mean(model.val_losses[-1][testshifted])
                # make a dataframe with the last weights of the model as the column "weights" and a column "subgroup" that indicates whether the observation is in the subgroup or not
                last_weights_rhok[guidance_name] = pd.DataFrame({
                    "weights": model.weights_list[-1],
                    "subgroup": trainingshifted
                    })
                last_weights_rhok[guidance_name] = last_weights_rhok[guidance_name].sort_values(by="weights", ascending=True)
            # convert loss_overall_rhok and loss_subgroup_rhok to dataframes
            loss_overall_rhok = pd.DataFrame.from_dict(loss_overall_rhok, orient="index").reset_index()
            loss_overall_rhok.rename(columns={"index": "guidance"}, inplace=True)
            loss_subgroup_rhok = pd.DataFrame.from_dict(loss_subgroup_rhok, orient="index").reset_index()
            loss_subgroup_rhok.rename(columns={"index": "guidance"}, inplace=True)
            # convert last_weights_rhok to a dataframe with a column "guidance" indicating the guidance
            last_weights_rhok = pd.concat([
                last_weights_rhok[guidance].assign(guidance=guidance) for guidance in last_weights_rhok.keys()
                ], ignore_index=True)
            # add rho and k to the dataframes
            loss_overall_rhok["rho"] = rho
            loss_overall_rhok["k"] = k
            loss_subgroup_rhok["rho"] = rho
            loss_subgroup_rhok["k"] = k
            last_weights_rhok["rho"] = rho
            last_weights_rhok["k"] = k
            # put the dataframes into the dictionaries  
            loss_overall_rho[f"{rho}"] = loss_overall_rhok
            loss_subgroup_rho[f"{rho}"] = loss_subgroup_rhok
            last_weights_rho[f"{rho}"] = last_weights_rhok
        # concatenate the dataframes for each k
        loss_overall_list[f"{k}"] = pd.concat(loss_overall_rho.values(), ignore_index=True)
        loss_subgroup_list[f"{k}"] = pd.concat(loss_subgroup_rho.values(), ignore_index=True)
        last_weights_list[f"{k}"] = pd.concat(last_weights_rho.values(), ignore_index=True)
    # concatenate the dataframes for each k and rho
    losses_df = pd.concat(
        [loss_overall_list[k].assign(k=k) for k in loss_overall_list.keys()],
        ignore_index=True
        )
    losses_df.rename(columns={0: "loss_overall"}, inplace=True)
    losses_df = losses_df.merge(
        pd.concat(
            [loss_subgroup_list[k].assign(k=k) for k in loss_subgroup_list.keys()],
            ignore_index=True
            )
        .rename(columns={0: "loss_subgroup"}),
        on=["rho", "k", "guidance"],
        )
    # add a row with the lightgbm validation losses
    lgbm_losses_df = pd.DataFrame({
        "guidance": ["ERM (LightGBM)"],
        "loss_overall": test_loss_overall[-1],
        "loss_subgroup": test_loss_shifted[-1],
        "rho": np.nan,
        "k": np.nan
        }, index=[0])
    losses_df = pd.concat([losses_df, lgbm_losses_df], ignore_index=True)
    losses_list[f"{propshift}"] = losses_df
    # concatenate the last weights dataframes for each k and rho
    weights_df = pd.concat(
        [last_weights_list[k].assign(k=k) for k in last_weights_list.keys()],
        ignore_index=True
        )
    weights_list[f"{propshift}"] = weights_df

#%%
# turn losses_list into a dataframe by concatenating the dataframes and adding a column for the shifting proportion
losses_df_all = pd.concat(
    [losses_list[prop].assign(prop_shift=prop) for prop in losses_list.keys()],
    ignore_index=True
    )
# store the dataframe as a csv file
losses_df_all.to_csv("05_simulations/03_hard_subgroup/tables/losses_hard_subgroup.csv", index=False)


# turn weights_list into a dataframe
weights_df = pd.concat(
    [weights_list[prop].assign(prop_shift=prop) for prop in weights_list.keys()],
    ignore_index=True
    )
# store the dataframe as a csv file
weights_df.to_csv("05_simulations/03_hard_subgroup/tables/weights_hard_subgroup.csv", index=False)

# %%
