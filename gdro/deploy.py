"""
    author: jaegerl
    created: 2025-07-10
    description: Functions for deploying guided DRO models on ICU data.
"""

# -----------------------
# Imports
# -----------------------

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import contextlib
from sklearn.metrics import roc_auc_score
import io
import os
from . import ICUdata as gdrodata
from . import model as gdromodel


# -----------------------
# Define Standards
# -----------------------

# dictionary of nice outcome names for plotting
OUTCOME_NAMES = {
    "respiratory": "Respiratory failure within 24h",
    "circulatory": "Circulatory failure within 8h",
    "renal": "Acute kidney injury within 48h",
    "mortality": "Mortality within 24h",
    "lactate": "Log lactate after 4h",
    "creatinine": "Log creatinine after 24h",
    }
# dictionary of outcome types
OUTCOME_TYPES = {
    "respiratory": "categorical",
    "circulatory": "categorical",
    "renal": "categorical",
    "mortality": "categorical",
    "lactate": "continuous",
    "creatinine": "continuous",
    }
# dictionary of nice dataset names
DATASETS = {
    "eicu": "eICU",
    "miiv": "MIMIC-IV",
    "mimic": "MIMIC-III",
    "hirid": "HiRID",
    "nwicu": "NWICU",
    "sic": "SICdb",
    "zigong": "Zigong",
    "picdb": "PIC",
    }


# -----------------------
# Deploy GDRO on ICU data
# -----------------------

def fitgdroICU(
    source: str,
    target: str,
    outcome: str = "respiratory",
    other_columns: list[str] = [],
    rho: float = .1,
    k: int = 2,
    n_rounds: int = 500,
    epsilon: float = 0,
    guides: dict = {
        "quantile": [],
        "avg": [],
        "avg_by_group": [],
        },
    lgbm_params: dict = {
        "objective": "binary",
        "learning_rate": 0.05,
        "max_depth": 3,
        "verbose": -1,
        },
    codebook_path: str = gdrodata.CODEBOOK_PATH,
    datadir: str = "/Users/jaegerl/Documents/awesome_stuff/statistics_msc/fs25_master_thesis/analysis/02_data/02_data_input",
    **kwargs,
    ):
    """
    Run a GDRO experiment with different constraints on a source and a target dataset.
    
    This function fits a GDRO model on a training set using constraints derived from a specified guidance constraint set, and evaluates the model on a test set. It returns the fitted GDRO model object.
    
    Parameters
    ----------
    source : str
        The name of the source dataset to be used for training.
    target : str
        The name of the target dataset to be used for testing.
    outcome : str
        The outcome variable of interest. Must be either of `"circulatory"`, "`respiratory"`, `"renal"`, `"mortality"`, `"lactate"`, or `"creatinine"`. Defaults to `"respiratory"`.
    other_columns : str
        Additional columns to be included in the datasets, such as `sex` or `mortality_at_24h`. The variables provided in the guide matrix are automatically included, so they do not need to be specified here.
    rho : float
        The rho parameter for GDRO, by default 0.1.
    k : int
        The k parameter for GDRO, by default 2 (chi-square divergence).
    epsilon : float, optional
        The soft constraint parameter for GDRO, by default 0 (no soft constraints).
    n_rounds : int, optional
        The number of boosting rounds for LightGBM, by default 500.
    guides : dict, optional
        A dictionary specifying the guidance constraints to be used. Must contain keys `"quantile"`, `"avg"`, `"avg_cutoff"`. and `"avg_by_group"`, each mapping to a list of variable names for which the constraints should be computed. For `"avg_cutoff"`, the list must be one of 3-tuples, with the first entry denoting the variable whose cutoff is to be considered, the second entry indicating the cutoff value, and the third entry a Boolean indicating whether the constraining average refers to values above the cutoff. For `"avg_by_group"`, the list must be one of 2-tuples, with the first entry denoting the variable to be averaged and the second entry denoting the grouping variable. If not provided, no constraints are applied (default), and a standard DRO model is fitted.
    lgb_params : dict, optional
        The parameters for LightGBM, by default a dictionary with binary objective, `learning_rate` of 0.05, `max_depth` of 3, and `verbose` set to -1.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the GDRO fitting method.
    
    Returns
    -------
    None
    
    """
    
    # check if source and target are both available as keys of DATASETS
    if source not in DATASETS or target not in DATASETS:
        raise ValueError(f"Source or target dataset not recognized. Available datasets: {list(DATASETS.keys())}")
    if outcome not in gdrodata.OUTCOME_VARIABLES:
        raise ValueError(f"Unknown outcome. Available outcomes: {list(gdrodata.OUTCOME_VARIABLES.keys())}")
    # add to other_columns the entries of the guides dictionary
    other_columns = (
        other_columns +
        [gdrodata.OUTCOME_VARIABLES[outcome]] +
        [c for c in guides["quantile"]] +
        [c for c in guides["avg"]] +
        [g[0] for g in guides["avg_cutoff"]] +
        [g[0] for g in guides["avg_by_cutoff"]] +
        [g[1] for g in guides["avg_by_cutoff"]] +
        [g[0] for g in guides["avg_by_group"]] +
        [g[1] for g in guides["avg_by_group"]]
        )
    other_columns = list(set(other_columns))
    # check if sex_age is requested
    needs_sex_age = "sex_age_group" in other_columns
    needs_age_group = ("age_group" in other_columns) or ("sex_age_group" in other_columns)
    if needs_sex_age:
        other_columns += ["sex", "age"]
        other_columns.remove("sex_age_group")
    if "age_group" in other_columns:
        other_columns += ["age"]
        other_columns.remove("age_group")
    other_columns = list(set(other_columns))  # remove duplicates
    # columns that must not be empty
    nonempty_columns = other_columns + [gdrodata.OUTCOME_VARIABLES[outcome]]
    nonempty_columns = list(set(nonempty_columns))  # remove duplicates
    # load and preprocess the datasets
    print(f"Loading training dataset from {DATASETS[source]}...")
    trainingset = gdrodata.load_icudata(
        datadir=datadir,
        dataset=source,
        outcome=outcome,
        subset=["train"],
        nonempty_columns=nonempty_columns,
        other_columns=other_columns,
        codebook_path=codebook_path,
        )
    print(f"Loading guidance dataset from {DATASETS[target]}...")
    guideset = gdrodata.load_icudata(
        datadir=datadir,
        dataset=target,
        outcome=outcome,
        subset=["train"],
        nonempty_columns=nonempty_columns,
        other_columns=other_columns,
        codebook_path=codebook_path,
        )
    print(f"Loading test dataset from {DATASETS[target]}...\n")
    testset = gdrodata.load_icudata(
        datadir=datadir,
        dataset=target,
        outcome=outcome,
        subset=["test"],
        nonempty_columns=nonempty_columns,
        other_columns=other_columns,
        codebook_path=codebook_path,
        )
    # convert the keys of the guides dictionary so that they match the names
    # in the preprocessed datasets
    print("Constructing guide matrices for GDRO...\n")
    outcome_for_guides = f"{OUTCOME_TYPES[outcome]}__{gdrodata.OUTCOME_VARIABLES[outcome]}"
    allcolumns = trainingset["X"].columns + [outcome_for_guides]
    if needs_sex_age:
        allcolumns.append("categorical__sex_age_group")
    if needs_age_group:
        allcolumns.append("categorical__age_group")
    guides = {
        "quantile": [f"continuous__{var}" for var in guides["quantile"]],
        "avg": [f"continuous__{var}" for var in guides["avg"] if f"continuous__{var}" in allcolumns] +
        [f"categorical__{var}" for var in guides["avg"] if f"categorical__{var}" in allcolumns],
        "avg_cutoff": [(f"continuous__{var[0]}", var[1], var[2]) for var in guides["avg_cutoff"]],
        "avg_by_cutoff": [(f"continuous__{var[0]}", f"continuous__{var[1]}", var[2], var[3]) for var in guides["avg_by_cutoff"] if f"continuous__{var[0]}" in allcolumns] +
        [(f"categorical__{var[0]}", f"continuous__{var[1]}", var[2], var[3]) for var in guides["avg_by_cutoff"] if f"categorical__{var[0]}" in allcolumns],
        "avg_by_group": [(f"continuous__{var[0]}", f"categorical__{var[1]}") for var in guides["avg_by_group"] if f"continuous__{var[0]}" in allcolumns] +
        [(f"categorical__{var[0]}", f"categorical__{var[1]}") for var in guides["avg_by_group"] if f"categorical__{var[0]}" in allcolumns],
    }
    # create an variables based on age groups if requested
    if needs_sex_age or needs_age_group:
        trainingset["X"] = trainingset["X"].with_columns(
            (pl.col("continuous__age") // 10 * 10).alias("categorical__age_group")
            ).with_columns(pl.Series(name=outcome_for_guides, values=trainingset["y"]))
        guideset["X"] = guideset["X"].with_columns(
            (pl.col("continuous__age") // 10 * 10).alias("categorical__age_group")
            ).with_columns(pl.Series(name=outcome_for_guides, values=guideset["y"]))
        # trim age categories in the two datasets to the same range
        min_age_group = max(guideset["X"]["categorical__age_group"].min(),
                            trainingset["X"]["categorical__age_group"].min())
        trainingset["X"] = trainingset["X"].with_columns(
            pl.when(pl.col("categorical__age_group") < min_age_group)
            .then(min_age_group)
            .otherwise(pl.col("categorical__age_group"))
            .alias("categorical__age_group")
            )
        guideset["X"] = guideset["X"].with_columns(
            pl.when(pl.col("categorical__age_group") < min_age_group)
            .then(min_age_group)
            .otherwise(pl.col("categorical__age_group"))
            .alias("categorical__age_group")
            )
        max_age_group = min(guideset["X"]["categorical__age_group"].max(),
                            trainingset["X"]["categorical__age_group"].max())
        trainingset["X"] = trainingset["X"].with_columns(
            pl.when(pl.col("categorical__age_group") > max_age_group)
            .then(max_age_group)
            .otherwise(pl.col("categorical__age_group"))
            .alias("categorical__age_group")
            )
        guideset["X"] = guideset["X"].with_columns(
            pl.when(pl.col("categorical__age_group") > max_age_group)
            .then(max_age_group)
            .otherwise(pl.col("categorical__age_group"))
            .alias("categorical__age_group")
            )
        # create sex-age groups if requested
        if needs_sex_age:
            trainingset["X"] = trainingset["X"].with_columns(
                (pl.col("categorical__sex").cast(str) + "_" + pl.col("categorical__age_group").cast(str)).alias("categorical__sex_age_group"))
            # create the same for the guideset
            guideset["X"] = guideset["X"].with_columns(
                (pl.col("categorical__sex").cast(str) + "_" + pl.col("categorical__age_group").cast(str)).alias("categorical__sex_age_group"))
    # go through the guides dictionary and construct the corresponding matrices
    guide_matrices = []
    for variable in guides["quantile"]:
        guide_matrices.append(
            gdromodel.constr_quantile(
                source=trainingset["X"].
                with_columns(pl.Series(name=outcome_for_guides, values=trainingset["y"])),
                target=guideset["X"].
                with_columns(pl.Series(name=outcome_for_guides, values=guideset["y"])),
                variable=variable,
                quantile=[0.25, 0.5, 0.75],
                )
            )
    for variable in guides["avg"]:
        guide_matrices.append(
        gdromodel.constr_avg(
            source=trainingset["X"]
            .with_columns(pl.Series(name=outcome_for_guides, values=trainingset["y"])),
            target=guideset["X"]
            .with_columns(pl.Series(name=outcome_for_guides, values=guideset["y"])),
            variable=variable,
            )
        )
    for cutvariable, cutoff, above in guides["avg_cutoff"]:
        guide_matrices.append(
            gdromodel.constr_avg_cutoff(
                source=trainingset["X"],
                target=guideset["X"],
                cutvariable=cutvariable,
                cutoff=cutoff,
                above=above,
            )
        )
    for variable, cutvariable, cutoff, above in guides["avg_by_cutoff"]:
        guide_matrices.append(
            gdromodel.constr_avg_by_cutoff(
                source=trainingset["X"],
                target=guideset["X"],
                variable=variable,
                cutvariable=cutvariable,
                cutoff=cutoff,
                above=above,
            )
        )
    for variable, group in guides["avg_by_group"]:
        guide_matrices.append(
            gdromodel.constr_avg_by_group(
                source=trainingset["X"],
                target=guideset["X"],
                variable=variable,
                group=group,
            )
        )
    # stack all matrices in guide_matrices vertically into a single matrix
    if len(guide_matrices) != 0:
        guide_matrix = np.vstack(guide_matrices)
    else:
        guide_matrix = None
    # print warning if any entries of gudie_matrix are empty
    if guide_matrix is not None and np.any(np.isnan(guide_matrix)):
        print("Warning: Some entries of the guide matrix are NaN values. This may cause trouble in the optimization routine.\n")
    
    # only keep the variables relevant for the current model
    outcomefeatures = gdrodata.get_features(
        outcome=outcome,
        codebook_path=codebook_path,
        )
    outcomefeatures = ["continuous__" + var for var in outcomefeatures if "continuous__" + var in trainingset["X"].columns] + ["categorical__" + var for var in outcomefeatures if "categorical__" + var in trainingset["X"].columns]

    trainingset["X"] = trainingset["X"].select(
        pl.col(outcomefeatures)
        )
    testset["X"] = testset["X"].select(
        pl.col(outcomefeatures)
        )
    
    # initialize the gdro object
    gdro_object = gdromodel.GuidedDRO(
        num_boost_round=n_rounds,
        rho=rho,
        k=k,
        lgbm_params=lgbm_params,
        )
    
    print(f"Fitting GDRO model with the following parameters:\n" +
          f"  Source dataset: {DATASETS[source]}\n" +
          f"  Target dataset: {DATASETS[target]}\n" +
          f"  Outcome variable: {gdrodata.OUTCOME_VARIABLES[outcome]}\n" +
          f"  Model parameters: rho = {rho}; k = {k}; epsilon = {epsilon}\n" +
          f"  Number of boosting rounds: {n_rounds}\n" +
          f"  Learning rate: {lgbm_params['learning_rate']}\n" +
          f"  Max depth: {lgbm_params['max_depth']}\n" +
          f"  Guide matrix shape: {guide_matrix.shape if guide_matrix is not None else 'None'}\n")
    
    # fit the model
    gdro_object.fit(
            X=trainingset["X"],
            y=trainingset["y"],
            Xval=testset["X"],
            yval=testset["y"],
            guide_matrix=guide_matrix,
            eval_metrics=True,
            epsilon=epsilon,
            **kwargs,
            )
    
    print("\nThe model has been fitted successfully.\n")
    
    # return the fitted gdro object with different attributes
    gdroICU = {
        "model": gdro_object,
        "source": source,
        "target": target,
        "outcome": outcome,
        "guides": guides,
        "epsilon": epsilon,
        "trainingset": trainingset,
        "guideset": guideset,
        "testset": testset,
        "training_data": trainingset["X"],
        "test_data": testset["X"],
        "nonempty_columns": nonempty_columns,
        "other_columns": other_columns,
        }
    return gdroICU


def find_epsilon(
    epsilonmax: float = 1.0,
    **kwargs,
    ):
    """
    Find an appropriate epsilon value for soft constraining in GDRO by trying different values (powers of ten) until the model fits successfully. The smallest power of ten that allows a successful fit will be returned.
    
    Parameters
    ----------
    epsilonmax: float
        The maximum epsilon value to try. The function will start with epsilon = 0 and increase it until a successful fit is achieved or the maximum value is reached.
    **kwargs : dict, optional
        Keyword arguments to be passed to the fitgdroICU function.
        
    Returns
    -------
    None
    
    """
    # if kwargs contains "n_rounds", remove it to avoid conflicts
    if "n_rounds" in kwargs:
        del kwargs["n_rounds"]
    # same for "epsilon"
    if "epsilon" in kwargs:
        del kwargs["epsilon"]
    # set n_rounds to 1 in kwargs, since we only need to check the first boosting round
    kwargs["n_rounds"] = 2
    
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fitgdroICU(epsilon=0,
                       **kwargs)
        print("Optimization with hard constraints is tractable, no need for soft constraints.\n")
    except TypeError as e:
        if "unsupported operand type(s) for *: 'int' and 'NoneType'" in str(e):
            print("Initial fit with hard constraints failed due to intractability.\nTrying larger epsilon values...")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    fitgdroICU(epsilon=epsilonmax,
                               **kwargs)
                epsilontry = epsilonmax / 10
                while True:
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            fitgdroICU(epsilon=epsilontry,
                                    **kwargs)
                        print(f"Successfully fitted GDRO model with epsilon = {epsilontry}.\nTrying smaller epsilon values...")
                        epsilontry = epsilontry / 10
                    except TypeError as e2:
                        if "unsupported operand type(s) for *: 'int' and 'NoneType'" in str(e2):
                            print(f"Failed to fit GDRO model with epsilon = {epsilontry}.\nThe largest still tractable value is epsilon = {epsilontry * 10}.\n")
                            return epsilontry * 10
                        else:
                            print("Unexpected error during fitting:", e2)
                            return None
            except TypeError as e2:
                if "unsupported operand type(s) for *: 'int' and 'NoneType'" in str(e2):
                    print(f"Failed to fit GDRO model even with maximum epsilon value of {epsilonmax}.\nPlease provide a larger value for epsilonmax.")
                    return None
                else:
                    print("Fit failed with an unexpected error:", e2)
                    return None
        else:
            print("Fit failed with an unexpected error:", e)
            return None
    except Exception as e3:
        print("Fit failed with an unexpected error:", e3)
    
    return 0


# -----------------------
# Subgroup evaluation
# -----------------------


def eval_subgroup(
    model: gdromodel.GuidedDRO,
    dataset: str = "miiv",
    outcome: str = "respiratory",
    subgroup: tuple = None,
    n_rounds: int = None,
    datadir: str = "/Users/jaegerl/Documents/awesome_stuff/statistics_msc/fs25_master_thesis/analysis/02_data/02_data_input",
    codebook_path: str = gdrodata.CODEBOOK_PATH,
    other_columns: list[str] = None,
    ):
    """
    Evaluate the GDRO model on a specific subgroup defined by a cutoff value.
    
    The subgroup is defined by a tuple containing the variable name, the cutoff value, and a Boolean indicating whether the subgroup is defined by values above or below the cutoff.
    
    Parameters
    ----------
    model : gdromodel.GuidedDRO
        The GDRO model to be evaluated.
    dataset : str
        The name of the dataset to be used for evaluation. Default is "miiv".
    outcome : str
        The outcome variable to be used for evaluation. Default is "respiratory".
    subgroup : tuple
        A tuple containing the variable name (str), the cutoff value (float), and a Boolean indicating whether the subgroup is defined by values above (True) or below (False) the cutoff value. If `None` (default), the model is evaluated on the overall population.
    n_rounds : int, optional
        The number of boosting rounds for LightGBM. If none, the number of boosting rounds of the model is used.
    datadir : str, optional
        The directory where the datasets are stored.
    codebook_path : str, optional
        The path to the codebook for the datasets. Default is `gdrodata.CODEBOOK_PATH`.
    other_columns : list[str], optional
        A list of additional columns to be included in the evaluation dataset. Default is `None`, which means no additional columns are considered.
    
    Returns
    -------
    None
        
    """

    if subgroup is None:
        raise ValueError("Subgroup must be defined as a tuple of (variable name, cutoff value, above).")
    
    if n_rounds is None:
        n_rounds = model.num_boost_round
    other_columns = other_columns + [subgroup[0]]
    other_columns = list(set(other_columns))
    # define the columns that must not be empty in the evaluation dataset
    nonempty_columns = other_columns + [gdrodata.OUTCOME_VARIABLES[outcome]]
    nonempty_columns = list(set(nonempty_columns))
    # load the evaluation dataset
    target = gdrodata.load_icudata(
        datadir=datadir,
        dataset=dataset,
        outcome=outcome,
        subset=["test"],
        codebook_path=codebook_path,
        other_columns=other_columns,
        nonempty_columns=nonempty_columns,
        )
    # extract dataset for prediction
    Xall = target["X"]
    # extract the subgroup
    subgroupvar = "categorical__" + subgroup[0] if "categorical__" + subgroup[0] in target["X"].columns else "continuous__" + subgroup[0]
    if subgroup[2]:  # above cutoff
        subgroup_idx = Xall[subgroupvar] >= subgroup[1]
    else:  # below cutoff
        subgroup_idx = Xall[subgroupvar] < subgroup[1]
    Xsub = Xall.filter(subgroup_idx)
    Ysub = target["y"][subgroup_idx]
    # only keep the columns of the model, and in the same order (important!!)
    Xall = Xall.select(model.dro_model.feature_name())
    Xsub = Xsub.select(model.dro_model.feature_name())
    print(f"Evaluating GDRO model on subgroup of size {Xsub.shape[0]} (original dataset of size {Xall.shape[0]}).\n")
    # predict the model on the target dataset for each boosting round
    pred_all = [
        model.dro_model.predict(
            data=Xall,
            num_iteration=_ + 1,
            raw_score=True,
            ) for _ in range(model.num_boost_round)
        ]
    # predict on the subgroup
    pred_sub = [
        model.dro_model.predict(
            data=Xsub,
            num_iteration=_ + 1,
            raw_score=True,
            ) for _ in range(n_rounds)
        ]
    # calculate losses (and AUROC) for the overall population and the subgroup
    if model.lgbm_params["objective"] == "binary":
        prob_all = [1 / (1 + np.exp(-pred)) for pred in pred_all]
        prob_sub = [1 / (1 + np.exp(-pred)) for pred in pred_sub]
        loss_all = [np.mean(-target["y"] * np.log(prob) - (1 - target["y"]) * np.log(1 - prob)) for prob in prob_all]
        loss_sub = [np.mean(-Ysub * np.log(prob) - (1 - Ysub) * np.log(1 - prob)) for prob in prob_sub]
        auroc_all = [roc_auc_score(target["y"], prob) for prob in prob_all]
        auroc_sub = [roc_auc_score(Ysub, prob) for prob in prob_sub]
    elif model.lgbm_params["objective"] == "regression":
        loss_all = [np.mean((target["y"] - pred) ** 2) for pred in pred_all]
        loss_sub = [np.mean((Ysub - pred) ** 2) for pred in pred_sub]
    
    evaluations = {
        "overall": {
            "loss": loss_all,
            "auroc": auroc_all if model.lgbm_params["objective"] == "binary" else None,
            "predictions": pred_all,
            "X": Xall,
            "y": target["y"],
        },
        "subgroup": {
            "loss": loss_sub,
            "auroc": auroc_sub if model.lgbm_params["objective"] == "binary" else None,
            "predictions": pred_sub,
            "X": Xsub,
            "y": Ysub,
        },
    }
    return evaluations


# -----------------------
# Save and plot GDRO models
# -----------------------

def plotgdroICU(
    gdromodels: list[gdromodel.GuidedDRO],
    guidenames: list[str],
    plotname: str,
    modeldir: str = "03_models/",
    plotdir: str = "04_figures/",
    datadir: str = "/Users/jaegerl/Documents/awesome_stuff/statistics_msc/fs25_master_thesis/analysis/02_data/02_data_input",
    gdrocolor: str = "black",
    simpledrocolor: str = "orange",
    simpledrolty: str = "-",
    lightgbmcolor: str = "blue",
    lightgbmlty: str = "--",
    codebook_path: str = gdrodata.CODEBOOK_PATH,
    nonempty_columns: list[str] = ["mortality_at_24h", "sex", "age"],
    ):
    """
    Plot the test losses and AUROCs of a list of GDRO models.
    
    Parameters
    ----------
    gdromodels : list[dict]
        A list of GDRO models as returned from `fitgdroICU`. Each model should be a dictionary containing the keys `"model"`, `"source"`, `"target"`, and `"outcome"`.
    guidenames : list[str]
        A list of names for the guidance constraints corresponding to the GDRO models. Must be provided!
    plotname : str
        The name of the set of models, usually reflecting the type of guidance constraints used, such as `"Demographic"`.
    modeldir : str, optional
        The directory where the models are saved, by default `"02_data/02_data_dump/"`.
    plotdir : str, optional
        The directory where the plots are saved, by default `"05_gdro/"`.
    gdrocolor : str, optional
        The color used for the GDRO model lines in the plots, by default `"black"`.
    simpledrocolor : str, optional
        The color used for the simple GDRO model (without guidance constraints) lines in the plots, by default `"orange"`.
    lightgbmcolor : str, optional
        The color used for the LightGBM model lines in the plots, by default `"blue"`.
    lightgbmlty : str, optional
        The line style used for the LightGBM model lines in the plots, by default `"--"`.
    codebook_path : str, optional
        The path to the codebook for the datasets, by default `CODEBOOK_PATH` from the `ICUdata` module.
    subgroup: tuple, optional
        A tuple specifying a subgroup for which the models should be plotted. The first element is the name of the variable that defines the subgroup (str), the second element is the cutoff value (float), and the third element denotes whether the subgroup is defined by values above (True) or below (False) the cutoff value. A panel of two plots is made, one for the overall population and one for the subpopulation. Default is `None`, which means no subgroup is considered.
    
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If the models do not have the same source, target, outcome, hyperparameters (rho, k, number of boosting iterations, LightGBM hyperparameters) or if the number of guidenames does not match the number of models.
    
    """
    
    # check if all models in the list have the same source
    if not all(model["source"] == gdromodels[0]["source"] for model in gdromodels):
        raise ValueError("All models must have the same source dataset.")
    # same for target
    if not all(model["target"] == gdromodels[0]["target"] for model in gdromodels):
        raise ValueError("All models must have the same target dataset.")
    # same for outcome
    if not all(model["outcome"] == gdromodels[0]["outcome"] for model in gdromodels):
        raise ValueError("All models must have the same outcome variable.")
    if not all(model["model"].dro_model.feature_name() == gdromodels[0]["model"].dro_model.feature_name() for model in gdromodels):
        raise ValueError("All models must have the same feature names.")
    # check if the number of guidenames matches the number of models    
    if len(guidenames) != len(gdromodels):
        raise ValueError("The number of guidenames must match the number of GDRO models.")
    # check if all models have the same k values
    if not all(model["model"].k == gdromodels[0]["model"].k for model in gdromodels):
        raise ValueError("All models must have the same k value.")
    # check if all models have the same rho values
    if not all(model["model"].rho == gdromodels[0]["model"].rho for model in gdromodels):
        raise ValueError("All models must have the same rho value.")
    # check if all models have the same number of boosting rounds
    if not all(model["model"].num_boost_round == gdromodels[0]["model"].num_boost_round for model in gdromodels):
        raise ValueError("All models must have the same number of boosting rounds.")
    # check if all models have the same lgbm_params
    if not all(model["model"].lgbm_params == gdromodels[0]["model"].lgbm_params for model in gdromodels):
        raise ValueError("All models must have the same LightGBM parameters.")
    
    # extract all relevant parameters from the first gdromodel
    source = gdromodels[0]["source"]
    target = gdromodels[0]["target"]
    outcome = gdromodels[0]["outcome"]
    n_rounds = gdromodels[0]["model"].num_boost_round
    lgbm_params = gdromodels[0]["model"].lgbm_params.copy() # avoid in-place modification!
    if lgbm_params["objective"] == "binary":
        lgbm_params["metric"] = ["auc", "binary_logloss"]
    # extract the epsilon values from the corresponding models
    guidenames = [
        f"{guidenames[i]} (" + r"$\epsilon = $" + f"{gdromodels[i]['epsilon']})"
        if gdromodels[i]['epsilon'] > 0
        else f"{guidenames[i]}"
        for i in range(len(gdromodels))
    ]
    
    # initialize list of losses
    test_losses = []
    # go through the gdromodels and save the results into appropriate files
    for i, gdromodel in enumerate(gdromodels):
        # save the model
        gdromodel["model"].save(
            file_name=f"gdro_{gdromodel['source']}to{gdromodel['target']}_{outcome}_{guidenames[i]}_rho{gdromodel['model'].rho}_k{gdromodel['model'].k}_nrounds{n_rounds}_learnrate{lgbm_params['learning_rate']}_maxdepth{lgbm_params['max_depth']}",
            dir_name=modeldir,
            )
        # calculate average test losses for each model
        test_losses.append(np.mean(gdromodel["model"].val_losses, axis=1))
    
    # fit and evaluate a simple LightGBM model with the same hyperparameters
    print("Loading training dataset for LightGBM...")
    trainingset_lgbm = gdrodata.load_icudata(
        datadir=datadir,
        dataset=source,
        outcome=outcome,
        subset=["train"],
        codebook_path=codebook_path,
        other_columns=nonempty_columns,
        nonempty_columns=nonempty_columns,
        )
    print("Loading test dataset for LightGBM...")
    testset_lgbm = gdrodata.load_icudata(
        datadir=datadir,
        dataset=target,
        outcome=outcome,
        subset=["test"],
        codebook_path=codebook_path,
        other_columns=nonempty_columns,
        nonempty_columns=nonempty_columns,
        )
    print("Fitting simple LightGBM model...")
    # make sure that only the features that are present in the gdromodels are used
    train_columns = gdromodels[0]["model"].dro_model.feature_name()
    trainingset_lgbm["X"] = trainingset_lgbm["X"].select(pl.col(train_columns))
    testset_lgbm["X"] = testset_lgbm["X"].select(pl.col(train_columns))
    lgbm_cat_feat = [c for c in train_columns if "categorical" in c]
    lgb_trainingdata = lgb.Dataset(
        data=trainingset_lgbm["X"].to_arrow(),
        label=trainingset_lgbm["y"],
        categorical_feature=lgbm_cat_feat,
        feature_name=train_columns,
        free_raw_data=False,
        )
    lgb_testdata = lgb.Dataset(
        data=testset_lgbm["X"].to_arrow(),
        label=testset_lgbm["y"],
        categorical_feature=lgbm_cat_feat,
        feature_name=testset_lgbm["X"].columns,
        free_raw_data=False,
        )
    evals_result = {}
    lgbm_model = lgb.train(
        params=lgbm_params,
        train_set=lgb_trainingdata,
        num_boost_round=n_rounds,
        valid_sets=lgb_testdata,
        callbacks=[lgb.record_evaluation(evals_result)]
        )
    # plot test losses over iterations in the same diagram
    x_axis = np.arange(1, n_rounds + 1)
    linestyles = [':', '-.', (0, (5, 1)), (0, (3,5,1,5,1,5))]
    plt.figure(figsize=(10, 5))
    plt.title(f"Outcome: {OUTCOME_NAMES[outcome]}\nTraining Set: {DATASETS[source]}, Test Set: {DATASETS[target]}\nParameters: " + r"$\rho = $" + f"{gdromodels[0]['model'].rho}, " + r"$k = $" + f"{gdromodels[0]['model'].k}\nGuidance: {plotname}")
    for i, losses in enumerate(test_losses):
        linestylei = linestyles[i % len(linestyles)] if len(gdromodels[i]["guides"]["quantile"]) + len(gdromodels[i]["guides"]["avg"]) + len(gdromodels[i]["guides"]["avg_by_group"]) + len(gdromodels[i]["guides"]["avg_cutoff"]) + len(gdromodels[i]["guides"]["avg_by_cutoff"]) > 0 else simpledrolty
        colori = gdrocolor if len(gdromodels[i]["guides"]["quantile"]) + len(gdromodels[i]["guides"]["avg"]) + len(gdromodels[i]["guides"]["avg_by_group"]) + len(gdromodels[i]["guides"]["avg_cutoff"]) + len(gdromodels[i]["guides"]["avg_by_cutoff"]) > 0 else simpledrocolor
        plt.plot(x_axis, losses,
                 label= f"{guidenames[i]}",
                 color=colori,
                 linestyle=linestylei)
    # also plot the simple LightGBM model
    if lgbm_params["objective"] == "binary":
        val_loss = evals_result["valid_0"]["binary_logloss"]
    elif lgbm_params["objective"] == "regression":
        val_loss = evals_result["valid_0"]["l2"]
    plt.plot(x_axis, val_loss, label="LightGBM", linestyle = "--", color=lightgbmcolor)
    plt.xlabel("Boosting Round")
    if lgbm_params["objective"] == "binary":
        plt.ylabel("Binary Log Loss")
    elif lgbm_params["objective"] == "regression":
        plt.ylabel("MSE")
    plt.legend()
    plt.grid()
    plt.savefig(plotdir + f"{trainingset_lgbm['dataset']}to{testset_lgbm['dataset']}_{outcome}_{plotname}_rho{gdromodels[0]['model'].rho}_k{gdromodels[0]['model'].k}_nrounds{n_rounds}_learnrate{lgbm_params['learning_rate']}_maxdepth{lgbm_params['max_depth']}.png")
    
    # for binary classification, also plot test AUROCs over iterations in the same diagram
    if lgbm_params["objective"] == "binary":
        plt.figure(figsize=(10, 5))
        plt.title(f"Outcome: {OUTCOME_NAMES[outcome]}\nTraining Set: {DATASETS[source]}, Test Set: {DATASETS[target]}\nParameters: " + r"$\rho = $" + f"{gdromodels[0]['model'].rho}, " + r"$k = $" + f"{gdromodels[0]['model'].k}\nGuidance: {plotname}")
        for i, gdromodel in enumerate(gdromodels):
            linestylei = linestyles[i % len(linestyles)] if len(gdromodels[i]["guides"]["quantile"]) + len(gdromodels[i]["guides"]["avg"]) + len(gdromodels[i]["guides"]["avg_by_group"]) + len(gdromodels[i]["guides"]["avg_cutoff"]) + len(gdromodels[i]["guides"]["avg_by_cutoff"]) > 0 else simpledrolty
            colori = gdrocolor if len(gdromodels[i]["guides"]["quantile"]) + len(gdromodels[i]["guides"]["avg"]) + len(gdromodels[i]["guides"]["avg_by_group"]) + len(gdromodels[i]["guides"]["avg_cutoff"]) + len(gdromodels[i]["guides"]["avg_by_cutoff"]) > 0 else simpledrocolor
            plt.plot(x_axis, gdromodel["model"].val_auroc, label= f"{guidenames[i]}", linestyle=linestylei, color=colori)
        # also plot the simple LightGBM model
        plt.plot(x_axis, evals_result["valid_0"]["auc"], label="LightGBM", linestyle=lightgbmlty, color=lightgbmcolor)
        plt.xlabel("Boosting Round")
        plt.ylabel("AUROC")
        plt.legend()
        plt.grid()
        plt.savefig(plotdir + f"{trainingset_lgbm['dataset']}to{testset_lgbm['dataset']}_{outcome}_{plotname}_rho{gdromodels[0]['model'].rho}_k{gdromodels[0]['model'].k}_nrounds{n_rounds}_learnrate{lgbm_params['learning_rate']}_maxdepth{lgbm_params['max_depth']}_auroc.png")


def plotgdroICUsubgroup(
    gdromodels: list[gdromodel.GuidedDRO],
    guidenames: list[str],
    subgroupname: str,
    n_rounds: int = None,
    subgroup: tuple = None,
    tabledir: str = "05_gdro/",
    plotdir: str = "05_gdro/",
    datadir: str = "/Users/jaegerl/Documents/awesome_stuff/statistics_msc/fs25_master_thesis/analysis/02_data/02_data_input",
    codebook_path: str = gdrodata.CODEBOOK_PATH,
    nonempty_columns: list[str] = None,
    ):
    
    # check if all models in the list have the same source
    if not all(model["source"] == gdromodels[0]["source"] for model in gdromodels):
        raise ValueError("All models must have the same source dataset.")
    # same for target
    if not all(model["target"] == gdromodels[0]["target"] for model in gdromodels):
        raise ValueError("All models must have the same target dataset.")
    # same for outcome
    if not all(model["outcome"] == gdromodels[0]["outcome"] for model in gdromodels):
        raise ValueError("All models must have the same outcome variable.")
    if not all(model["model"].dro_model.feature_name() == gdromodels[0]["model"].dro_model.feature_name() for model in gdromodels):
        raise ValueError("All models must have the same feature names.")
    # check if the number of guidenames matches the number of models    
    if len(guidenames) != len(gdromodels):
        raise ValueError("The number of guidenames must match the number of GDRO models.")
    # check if all models have the same k values
    if not all(model["model"].k == gdromodels[0]["model"].k for model in gdromodels):
        raise ValueError("All models must have the same k value.")
    # check if all models have the same rho values
    if not all(model["model"].rho == gdromodels[0]["model"].rho for model in gdromodels):
        raise ValueError("All models must have the same rho value.")
    # check if all models have the same number of boosting rounds
    if not all(model["model"].num_boost_round == gdromodels[0]["model"].num_boost_round for model in gdromodels):
        raise ValueError("All models must have the same number of boosting rounds.")
    # check if all models have the same lgbm_params
    if not all(model["model"].lgbm_params == gdromodels[0]["model"].lgbm_params for model in gdromodels):
        raise ValueError("All models must have the same LightGBM parameters.")
    
    # extract nonempty columns from the models
    if nonempty_columns is None:
        nonempty_columns = gdromodels[0]["nonempty_columns"]
    # extract feature names from the models
    featurenames = gdromodels[0]["model"].dro_model.feature_name()  
    # if n_rounds is none, extract from the first model
    if n_rounds is None:
        n_rounds = gdromodels[0]["model"].num_boost_round
    # extract parameters for filenames
    source = gdromodels[0]["source"]
    target = gdromodels[0]["target"]
    outcome = gdromodels[0]["outcome"]
    rho = gdromodels[0]["model"].rho
    k = gdromodels[0]["model"].k
    losstype = gdromodels[0]["model"].lgbm_params["objective"]
    # extract parameters for LightGBM model
    lgbm_params = gdromodels[0]["model"].lgbm_params.copy()
    if lgbm_params["objective"] == "binary":
        lgbm_params["metric"] = ["auc", "binary_logloss"]
    # extract the epsilon values from the corresponding models
    guidenames = [
        f"{guidenames[i]} (" + r"$\epsilon = $" + f"{gdromodels[i]['epsilon']})"
        if gdromodels[i]['epsilon'] > 0
        else f"{guidenames[i]}"
        for i in range(len(guidenames))
        ]
    
    # fit a simple LightGBM model with the same parameters
    print("Loading training dataset for LightGBM...")
    trainingset_lgbm = gdrodata.load_icudata(
        datadir=datadir,
        dataset=source,
        outcome=outcome,
        subset=["train"],
        codebook_path=codebook_path,
        other_columns=nonempty_columns,
        nonempty_columns=nonempty_columns,
        )
    print("Loading test dataset for LightGBM...")
    testset_lgbm = gdrodata.load_icudata(
        datadir=datadir,
        dataset=target,
        outcome=outcome,
        subset=["test"],
        codebook_path=codebook_path,
        other_columns=nonempty_columns,
        nonempty_columns=nonempty_columns,
        )
    print("Fitting simple LightGBM model...")
    # define dataset for subgroup evaluation
    subgroupvar = "categorical__" + subgroup[0] if "categorical__" + subgroup[0] in testset_lgbm["X"].columns else "continuous__" + subgroup[0]
    # extract the row indices of the data subgroup
    if subgroup[2]:  # if subgroup is greater than or equal to the threshold
        subgroup_idx = testset_lgbm["X"][subgroupvar] >= subgroup[1] 
    else:
        subgroup_idx = testset_lgbm["X"][subgroupvar] < subgroup[1]
    Xsubgroup = testset_lgbm["X"].filter(subgroup_idx)
    # reorder the columns of the subgroup dataset to match the training set
    Xsubgroup = Xsubgroup.select(featurenames)
    Ysubgroup = testset_lgbm["y"][subgroup_idx]
    # keep the relevant features, and in the correct order
    trainingset_lgbm["X"] = trainingset_lgbm["X"].select(featurenames)
    testset_lgbm["X"] = testset_lgbm["X"].select(featurenames)
    lgbm_cat_feat = [c for c in featurenames if "categorical" in c]
    lgb_trainingdata = lgb.Dataset(
        data=trainingset_lgbm["X"].to_arrow(),
        label=trainingset_lgbm["y"],
        categorical_feature=lgbm_cat_feat,
        feature_name=featurenames,
        free_raw_data=False,
        )
    lgb_testdata_overall = lgb.Dataset(
        data=testset_lgbm["X"].to_arrow(),
        label=testset_lgbm["y"],
        categorical_feature=lgbm_cat_feat,
        feature_name=featurenames,
        free_raw_data=False,
        )
    lgb_testdata_subgroup = lgb.Dataset(
        data=Xsubgroup.to_arrow(),
        label=Ysubgroup,
        categorical_feature=lgbm_cat_feat,
        feature_name=featurenames,
        free_raw_data=False,
        )
    evals_result = {}
    lgbm_model = lgb.train(
        params=lgbm_params,
        train_set=lgb_trainingdata,
        num_boost_round=n_rounds,
        valid_sets=[lgb_testdata_overall, lgb_testdata_subgroup],
        callbacks=[lgb.record_evaluation(evals_result)]
        )
    
    # evaluate the models at n_rounds using eval_subgroup
    evaluations = []
    for i, gdromodel in enumerate(gdromodels):
        print(f"Evaluating model {i + 1} of {len(gdromodels)}: {guidenames[i]}")
        eval = eval_subgroup(
            model=gdromodel["model"],
            dataset=gdromodel["target"],
            outcome=gdromodel["outcome"],
            subgroup=subgroup,
            n_rounds=n_rounds,
            datadir=datadir,
            codebook_path=codebook_path,
            other_columns=nonempty_columns,
            )
        evaluations.append(
            {
                "loss": eval["overall"]["loss"][n_rounds - 1],
                "auroc": eval["overall"]["auroc"][n_rounds - 1] if eval["overall"]["auroc"] is not None else None,
                "guide": guidenames[i],
                "population": "Overall",
            })
        evaluations.append(
            {
                "loss": eval["subgroup"]["loss"][n_rounds - 1],
                "auroc": eval["subgroup"]["auroc"][n_rounds - 1] if eval["subgroup"]["auroc"] is not None else None,
                "guide": guidenames[i],
                "population": "Subgroup",
            })
    # evaluate LightGBM model
    evaluations.append(
        {
            "loss": evals_result["valid_0"]["binary_logloss"][n_rounds - 1] if lgbm_params["objective"] == "binary" else evals_result["valid_0"]["l2"][n_rounds - 1],
            "auroc": evals_result["valid_0"]["auc"][n_rounds - 1] if lgbm_params["objective"] == "binary" else None,
            "guide": "LightGBM",
            "population": "Overall",
        })
    evaluations.append(
        {
            "loss": evals_result["valid_1"]["binary_logloss"][n_rounds - 1] if lgbm_params["objective"] == "binary" else evals_result["valid_1"]["l2"][n_rounds - 1],
            "auroc": evals_result["valid_1"]["auc"][n_rounds - 1] if lgbm_params["objective"] == "binary" else None,
            "guide": "LightGBM",
            "population": "Subgroup",
        })
    # convert this list of dictionaries to a dictionary of lists
    evaluations = pd.DataFrame(evaluations).to_dict(orient="list")
    # convert this dictionary to a DataFrame
    evaluations = pd.DataFrame.from_dict(evaluations, orient="columns")
    # control order
    evaluations["guide"] = pd.Categorical(evaluations["guide"], categories=["LightGBM"] + guidenames, ordered=True)
    evaluations["population"] = pd.Categorical(evaluations["population"], categories=["Overall", "Subgroup"], ordered=True)
    # round and format values in loss and auroc column to 3 digits
    evaluations_table = evaluations.copy()
    evaluations_table["loss"]=evaluations_table["loss"].round(3).apply(lambda x: f"{x:.3f}" if x is not None else None)
    evaluations_table["auroc"]= evaluations_table["auroc"].round(3).apply(lambda x: f"{x:.3f}" if x is not None else None)
    # export this table as an Excel file
    evaluations_table.to_excel(f"{tabledir}subgroupeval_{source}to{target}_{subgroupname}_rho{rho}_k{k}_nrounds{n_rounds}.xlsx", index=False)
    # export this table as a CSV file
    evaluations_table.to_csv(f"{tabledir}subgroupeval_{source}to{target}_{subgroupname}_rho{rho}_k{k}_nrounds{n_rounds}.csv", index=False)
    
    # barplot of the losses
    pivoted = evaluations.pivot(columns="guide", index="population", values="loss").loc[evaluations["population"].unique()]
    ax = pivoted.plot(kind="barh", width=0.6)
    ax.set_title(f"Subgroup: {subgroupname}\nOutcome: {OUTCOME_NAMES[outcome]}\nSource: {DATASETS[source]}, Target: {DATASETS[target]}\nParameters: " + r"$\rho = $" + f"{rho}, " + r"$k = $" + f"{k}, boosting rounds = {n_rounds}")
    plt.xticks(rotation=0)
    plt.legend(title="Guidance Constraints", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.ylabel(None)
    if losstype == "binary":
        plt.xlabel("Binary Log Loss")
    elif losstype == "regression":
        plt.xlabel("MSE")
    plt.grid(axis='x')
    plt.savefig(f"{plotdir}subgroupeval_{source}to{target}_{subgroupname}_rho{rho}_k{k}_nrounds{n_rounds}_losses.png")
    
    # barplot of the AUROCs if binary
    if losstype == "binary":
        pivoted = evaluations.pivot(columns="guide", index="population", values="auroc").loc[evaluations["population"].unique()]
        ax = pivoted.plot(kind="barh", width=0.6)
        ax.set_title(f"Subgroup: {subgroupname}\nOutcome: {OUTCOME_NAMES[outcome]}\nSource: {DATASETS[source]}, Target: {DATASETS[target]}\nParameters: " + r"$\rho = $" + f"{rho}, " + r"$k = $" + f"{k}, boosting rounds = {n_rounds}")
        plt.xticks(rotation=0)
        plt.legend(title="Guidance Constraints", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.ylabel(None)
        plt.xlabel("AUROC")
        plt.grid(axis="x")
        plt.savefig(f"{plotdir}subgroupeval_{source}to{target}_{subgroupname}_rho{rho}_k{k}_nrounds{n_rounds}_aurocs.png")
    
    return evaluations
    
def plotgdroICUlossrho(
    source: str = "miiv", 
    target: str = "eicu", 
    outcome: str = "lactate",
    k: int = 2, 
    nrounds: int = 500,
    figsize: tuple = (15, 10),
    tabledir: str = "04_tables/",
    savefig: bool = False,
    plotdir: str = "05_icu_analyses/loss_vs_rho/",
    ):
    """
    Create line plots for the model loss (or AUROC for binary classification tasks) versus log10(rho) for a given source, target, outcome, k and nrounds, with one line for each guidance constraint, and with simple LightGBM as a benchmark.
    
    Parameters
    ----------
    source : str
        The source dataset.
    target : str
        The target dataset.
    outcome : str
        The outcome variable.
    k : int 
        The number of guidance constraints.
    nrounds : int
        The number of rounds for the LightGBM model.
    figsize : tuple
        The size of the figure as (width, height) in inches.
    savefig : bool
        Whether to save the figure.
    tabledir : str
        The path to the directory containing the model evaluation tables.
    plotdir : str
        The path to save the figure.
        
    Returns
    -------
    None
        Displays the plots and optionally saves them to a file.
    """
    
    # get the names of the model files
    model_tables = [
        f for f in os.listdir(f"{tabledir}/{source}to{target}_{outcome}") 
        if f.startswith(f"subgroupeval_{source}to{target}") 
        and f.endswith(".csv")
        and "nrounds2" not in f
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
    # extract all tables with the relevant k and nrounds values
    table_list_subset = {
        rho: table for (rho, k_value, nrounds_table), table in table_list.items() 
        if k_value == k and nrounds_table == nrounds
        }
    # merge all tables into a single dataframe, using an additional column for rho
    merged_table = pd.DataFrame()
    for rho, table in table_list_subset.items():
        table["rho"] = rho  # add rho column
        merged_table = pd.concat([merged_table, table], ignore_index=True) 
    # sort the merged table by rho
    merged_table.sort_values(by="rho", inplace=True)
    # in the column guide, remove everything in parentheses
    merged_table["guide"] = merged_table["guide"].str.replace(r" \(.*\)", "", regex=True).str.strip()
    # define markers for the line plot
    marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X', 'h']
    # get the unique guides excluding LightGBM and Simple DRO
    guides = [g for g in merged_table["guide"].unique() if g != "LightGBM" and g != "Simple DRO"]
    # mapping from guide to marker
    marker_map = {g: marker_styles[i % len(marker_styles)] for i, g in enumerate(guides)}
    
    # make the two plots, one for Overall and one for the Subgroup
    for population in ["Overall", "Subgroup"]:
        plt.figure(figsize=figsize)
        # lines for LightGBM and Simple DRO
        lightgbm_subset = merged_table[
            (merged_table["guide"] == "LightGBM") & 
            (merged_table["population"] == population)]
        simpledro_subset = merged_table[
            (merged_table["guide"] == "Simple DRO") & 
            (merged_table["population"] == population)]
        plt.axhline(
            y=lightgbm_subset["loss"].mean(), 
            color='darkgray', label='ERM (LightGBM)', linestyle = "-.")
        plt.plot(
            np.log10(simpledro_subset["rho"]),
            simpledro_subset["loss"],
            label="No guidance", color='black', linestyle='--')   
        for guide in guides:
            subset = merged_table[
                (merged_table["guide"] == guide) & 
                (merged_table["population"] == population)]
            plt.plot(
                np.log10(subset["rho"]), 
                subset["loss"], label=guide, 
                marker=marker_map[guide])
        plt.xlabel(r"$\rho$" + r" ($\log_{10}$ scale)")
        plt.xticks(
            np.log10(merged_table["rho"].unique()), 
            [f"{rho:.2f}" for rho in merged_table["rho"].unique()])
        if outcome in ["creatinine", "lactate"]:
            plt.ylabel("MSE" + f" ({population})")
        else:
            plt.ylabel("Binary Log Loss" + f" ({population})")
        plt.title(f"Source: {DATASETS[source]}, Target: {DATASETS[target]},\nOutcome: {OUTCOME_NAMES[outcome]}\n" + r"$k =$" + f"{k_value}, nrounds={nrounds}")
        # put legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()
        plt.tight_layout()
        if savefig:
            plt.savefig(f"{plotdir}/lossvsrho_{source}to{target}_{outcome}_{population}_k{k_value}_nrounds{nrounds}.png")
    # do the same for auroc values if any of the values is not NaN
    if any(merged_table["auroc"].notna()):
        for population in ["Overall", "Subgroup"]:
            plt.figure(figsize=figsize)
            # lines for LightGBM and Simple DRO
            lightgbm_subset = merged_table[
                (merged_table["guide"] == "LightGBM") & 
                (merged_table["population"] == population)]
            simpledro_subset = merged_table[
                (merged_table["guide"] == "No guidance") & 
                (merged_table["population"] == population)]
            plt.axhline(
                y=lightgbm_subset["auroc"].mean(), 
                color='darkgray', label='ERM (LightGBM)', linestyle = "-.")
            plt.plot(
                np.log10(simpledro_subset["rho"]),
                simpledro_subset["auroc"],
                label="Simple DRO", color='black', linestyle='--')   
            for guide in guides:
                subset = merged_table[
                    (merged_table["guide"] == guide) & 
                    (merged_table["population"] == population)]
                plt.plot(
                    np.log10(subset["rho"]), 
                    subset["auroc"], label=guide, 
                    marker=marker_map[guide])
            plt.xlabel(r"$\rho$" + r" ($\log_{10}$ scale)")
            plt.xticks(
                np.log10(merged_table["rho"].unique()), 
                [f"{rho:.2f}" for rho in merged_table["rho"].unique()])
            plt.ylabel(f"AUROC ({population})")
            plt.title(f"Source: {DATASETS[source]}, Target: {DATASETS[target]},\nOutcome: {OUTCOME_NAMES[outcome]}\n" + r"$k =$" + f"{k_value}, nrounds={nrounds}")
            # put the legend outside the plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid()
            plt.tight_layout()
            if savefig:
                plt.savefig(f"{plotdir}/aurocvsrho_{source}to{target}_{outcome}_{population}_k{k_value}_nrounds{nrounds}.png")