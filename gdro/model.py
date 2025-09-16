"""
    author: jaegerl
    created: 2025-07-10
    description: GuidedDRO modeling class and methods for fitting and prediction of guided distributionally robust optimization models using LightGBM. Also includes functions for data preprocessing, simulation, and utility functions.
    modified: 2025-07-12
    - modified constr_avg to handle categorical variables with more than two levels
    - modified constr_avg_by_group to only consider grouping levels that are also present in the source dataset
"""

# -----------------------
# Imports
# -----------------------

import numpy as np
import polars as pl
import lightgbm as lgb
import cvxpy as cp
import pickle as pk
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer


# -----------------------
# Data preprocessing and utility functions
# -----------------------

# Cressie-Read divergence family
def f_k(
    u: np.ndarray, 
    k: int = 2,
    ):
    """ Evaluate the convex function underlying the k-th Cressie-Read divergence family element-wise at the input u. Tailored for use in cvxpy. MAY NOT BE RELEVANT - SEE MORE STABLE IMPLEMENTATION OF KL DIVERGENCE BELOW.
    
    Parameters
    ----------
    u : array-like
        Input values.
    k : int
        The order of the Cressie-Read divergence. Must be greater than or equal to 1. The default is 2, which corresponds to the chi-squared divergence. k=1 corresponds to the Kullback-Leibler divergence.
    
    Returns
    -------
    array-like
        The evaluated function values (element-wise).
        
    """
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")
    elif k == 1:
        return cp.kl_div(u, 1) # == u * cp.log(u) - u + 1, see https://www.cvxpy.org/tutorial/functions/index.html
    else:
        return (cp.power(u, k) - u * k + k - 1) / (k * (k - 1))

# function to preprocess a dataset for LightGBM fitting
def preprocess_lgbm(
    source: pl.DataFrame, 
    outcome: str = "mortality_at_24h",
    split_col: str = "split", 
    split_values: list[str] = [
        "train", 
        "val"
        ],
    ):
    """ Preprocess a polars.DataFrame source dataset for use with LightGBM or GDRO.
    
    Parameters
    ----------
    source : polars.DataFrame
        Source dataset containing the features and outcome. Must contain a column to indicate the train/validation split.
    outcome : str
        Name of the outcome column in the source dataset. Default is "mortality_at_24h".
    split_col : str
        Name of the column indicating the train/validation split. Default is "split".
    split_values : list
        List of length two of values in the split column indicating the training and validation datasets, respectively. Default is ["train", "val"].
    
    Returns
    -------
    dict
        Dictionary containing the training and validation data,
        along with the corresponding labels.   
    """
    # split into training and validation data
    training_data = source.filter(pl.col(split_col) == split_values[0])
    validation_data = source.filter(pl.col(split_col) == split_values[1])
    # extract outcome variable
    ytrain = training_data[outcome].to_numpy()
    yval = validation_data[outcome].to_numpy()
    # extract features
    features = [col 
                for col in source.columns 
                if col not in [outcome, "split"]]
    features_cat = [col
                     for col, coltype in source.select(features).schema.items()
                     if coltype.is_integer()]
    features_cont = [c for c in features if c not in features_cat]
    # define the preprocessor
    preprocessor = ColumnTransformer(
    transformers=[
        ("continuous", "passthrough", features_cont),
        ("categorical",
            OrdinalEncoder(
                handle_unknown = "use_encoded_value",
                unknown_value = 99),
            features_cat)]).set_output(transform = "polars")
    # apply preprocessor
    training_data = preprocessor.fit_transform(training_data[features])
    validation_data = preprocessor.fit_transform(validation_data[features])
    
    return {
        "training_data": training_data,
        "validation_data": validation_data,
        "ytrain": ytrain,
        "yval": yval,
        }


def constr_avg(
    source: pl.DataFrame,
    target: pl.DataFrame,
    variable: str = "mortality_at_24h",
    var_type: str = "categorical",
    normalize: bool = True,
    ):
    """ Construct a constraint vector for the expected value of a variable in the target dataset. If this variable is continuous, the constraint involves the average value of the variable. If it is categorical, the constraint involves the average rates of dummy variables for each level of the categorical variable (after making sure that only the levels present in the source dataset are involved).
    
    Parameters
    ----------
    source : polars.DataFrame
        Source dataset containing the outcome feature.
    target : polars.DataFrame
        Target dataset containing the outcome feature.
    variable : str
        Name of the variable column in the source and target datasets to derive the constraint from.
    var_type : str
        Type of the variable in the source and target datasets. Can be either `"categorical"` or `"continuous"`. Default is `"categorical"`.
    normalize : bool, optional
        If `True`, the constraint vector (each row of the constraint matrix for categorical variables with more than two levels) is normalized to have a standard deviation of 1. Default is `True`.
    
    Returns
    -------
    numpy.ndarray
        Array containing the constraint for the average value of a variable in the target dataset.
    
    """
    
    # check if the variable contains any nulls
    if source[variable].is_null().any() or target[variable].is_null().any():
        raise ValueError(f"The variable '{variable}' contains null values in either the source or target dataset. Please handle them before proceeding.")
    
    if var_type == "categorical":
        # extract the variable of interest
        xsource = source.select(variable).to_numpy().astype(str)
        xtarget = target.select(variable).to_numpy().astype(str)
        # get all unique levels in the target dataset but one
        levels_target = np.unique(xtarget)[:-1]
        # define dummy variables. careful: same order in source and target datasets!
        dummies_source = np.array([[1 if level in row else 0 for level in levels_target] for row in xsource])
        dummies_target = np.array([[1 if level in row else 0 for level in levels_target] for row in xtarget])
        # calculate average rates in target dataset
        avg_rates_target = dummies_target.mean(axis=0)
        # subtract this target average from the source values
        z = dummies_source - avg_rates_target
        # normalize if requested: divide each column of z by its standard deviation
        if normalize:
            z = z / np.std(z, axis=0, keepdims=True)
    elif var_type == "continuous":
        # extract the variable of interest
        xsource = source.select(variable).to_numpy().astype(float)
        xtarget = target.select(variable).to_numpy().astype(float)
        # calculate average rate in target dataset
        avg_target = xtarget.mean()
        # subtract this target average from the source values
        z = xsource - avg_target
        # normalize if requested
        if normalize:
            z = z / np.std(z)
    
    return z.T

def constr_avg_cutoff(
    source: pl.DataFrame,
    target: pl.DataFrame,
    cutvariable: str = "age",
    cutoff: float = 75,
    above: bool = True,
    normalize: bool = True,
    ):
    """ Construct a constraint vector for the value of a variable in the target dataset, where the value is either above or below a specified cutoff.
    
    Parameters
    ----------
    source : polars.DataFrame
        Source dataset containing the variable.
    target : polars.DataFrame
        Target dataset containing the variable.
    cutvariable : str
        Name of the variable column in the source and target datasets to derive the constraint from.
    cutoff : float
        The cutoff value for the variable.
    above : bool, optional
        If `True`, the constraint is for values above the cutoff. If `False`, the constraint is for values below the cutoff. Default is `True`.
    normalized : bool, optional
        If `True`, the constraint vector is normalized to have a standard deviation of 1. Default is `True`.
    
    Returns
    -------
    numpy.ndarray
        Array containing the constraint for `variable` in the target dataset, indicating whether each observation is above or below `cutoff`.
    
    Raises
    ------
    ValueError
        If the variable contains null values in either the source or target dataset.
    """
    # check if the variable contains any nulls
    if source[cutvariable].is_null().any() or target[cutvariable].is_null().any():
        raise ValueError(f"The variable '{cutvariable}' contains null values in either the source or target dataset. Please handle them before proceeding.")
    # extract the variable of interest
    xsource = source.select(cutvariable).to_numpy()
    xtarget = target.select(cutvariable).to_numpy()
    # create a constraint vector based on the cutoff
    if above:
        constr_vector = (xsource >= cutoff).astype(float)
        avg_target = (xtarget >= cutoff).mean()
    else:
        constr_vector = (xsource < cutoff).astype(float)
        avg_target = (xtarget < cutoff).mean()
    # subtract the target value from the constraint vector
    z = constr_vector - avg_target
    # normalize if requested
    if normalize:
            z = z / np.std(z)
    # return z as a 2D array with one row
    return z.T


def constr_avg_by_cutoff(
    source: pl.DataFrame,
    target: pl.DataFrame,
    variable: str = "mortality_at_24h",
    cutvariable: str = "age",
    cutoff: float = 75,
    above: bool = True,
    normalize: bool = True,
    ):
    """ Construct a constraint matrix for the average values of a variable by groups of another variable in the target dataset. The grouping variable is continuous and the groups are defined by whether the corresponding value is above or below a specific cutoff (e.g., mortality rates among patients aged 70 years or older or among patients with lactate levels of 2 mmol/L or above).
    
    Parameters
    ----------
    source : polars.DataFrame
        Source dataset containing the grouping variable and outcome.
    target : polars.DataFrame
        Target dataset containing the grouping variable and outcome.
    variable : str
        Name of the outcome column in the source and target datasets. Default is "mortality_at_24h".
    cutvariable : str
        Name of the grouping variable column in the source and target datasets. Default is "age".
    cutoff : float
        The cutoff value for the grouping variable. Default is 75.
    above : bool, optional
        If `True`, the constraint is for the group defined by values of `cutvariable` greater than or equal to the cutoff. If `False`, the constraint is for values smaller than the cutoff. Default is `True`.
    normalize : bool, optional
        If `True`, the constraint matrix is normalized so that each of its rows has a standard deviation of 1. Default is `True`.
    
    Returns
    -------
    numpy.ndarray
        Array containing the average rate of `variable` by groups of the grouping variable `cutvariable` in the target dataset.
    
    Raises
    ------
    ValueError
        If the variable contains null values in either the source or target dataset.
        If the grouping variable does not have values above or below the cutoff, according to the value of `above`, in either dataset.
    
    """
    
    # check if the variables of interest contain any nulls
    if source[variable].is_null().any() or target[variable].is_null().any():
        raise ValueError(f"The variable '{variable}' contains null values in either the source or target dataset. Please handle them before proceeding.")
    if source[cutvariable].is_null().any() or target[cutvariable].is_null().any():
        raise ValueError(f"The variable '{cutvariable}' contains null values in either the source or target dataset. Please handle them before proceeding.")
    # check if the datasets are compatible with the cutoff
    if above:
        if source[cutvariable].max() < cutoff or target[cutvariable].max() < cutoff:
            raise ValueError(f"The variable '{cutvariable}' in the source or target dataset does not have values above the cutoff {cutoff}.")
    else:
        if source[cutvariable].min() >= cutoff or target[cutvariable].min() >= cutoff:
            raise ValueError(f"The variable '{cutvariable}' in the source or target dataset does not have values below the cutoff {cutoff}.")
        
    # extract the variable of interest (so as to avoid in-place modification of the source and target datasets)
    source = source.select([variable, cutvariable])
    target = target.select([variable, cutvariable])
    # extract the indices of interest in the source and target dataset
    if above:
        source_indices = source[cutvariable] >= cutoff
        target_indices = target[cutvariable] >= cutoff
    else:
        source_indices = source[cutvariable] < cutoff
        target_indices = target[cutvariable] < cutoff
    # calculate the average rate of variable in the entries indexed by target_indices
    avg_target = target.filter(target_indices).select(variable).mean()
    # make an np.array as follows: where source_indices is True, subtract avg_target from variable. where source_indices is False, set the value to 0.
    z = np.where(source_indices, source[variable].to_numpy() - avg_target, 0.0)
    # normalize if requested: divide each row of z by its standard deviation
    if normalize:
            z = z / np.std(z, axis=1, keepdims=True)
    
    return z
    

def constr_avg_by_group(
    source: pl.DataFrame, 
    target: pl.DataFrame, 
    variable: str = "mortality_at_24h",
    group: str = "age_group",
    normalize: bool = True,
    ):
    """ Construct a constraint matrix for the average values of a variable in the target dataset, grouped by the levels of a categorical variable (e.g., mortality rates by age-sex strata). Note that only the groups present in the source dataset are considered.
    
    Parameters
    ----------
    source : polars.DataFrame
        Source dataset containing the group feature and outcome.
    target : polars.DataFrame
        Target dataset containing the group feature and outcome. 
    outcome : str
        Name of the outcome column in the source and target datasets.
    group : str
        Name of the group column in the source and target datasets.
    normalize : bool, optional
        If `True`, the constraint matrix is normalized so that each of its rows has a standard deviation of 1. Default is `True`.
        
    Returns
    -------
    numpy.ndarray
        Array containing the average rate of `variable` by groups of the categorical variable `group`.
        
    """
    
    # check if the variable contains any nulls
    if source[variable].is_null().any() or target[variable].is_null().any():
        raise ValueError(f"The variable '{variable}' contains null values in either the source or target dataset. Please handle them before proceeding.")
    
    # extract the variable of interest (so as to avoid in-place modification of the source and target datasets)
    source = source.select([variable, group])
    target = target.select([variable, group])
    # extract the groups in the source dataset
    groups_source = source[group].unique().to_numpy()
    # only keep groups in the target dataset that are also present in the source dataset
    target = target.filter(pl.col(group).is_in(groups_source))
    # calculate average rate by grouping variable in target dataset
    rates_target = target.group_by(group).agg(pl.col(variable).mean().alias("mean_rate"))
    # join source target average rates rates into observations from source dataset
    rates_source = source.join(rates_target, on=group, how="left").with_columns((variable - pl.col("mean_rate")).alias("yr"))
    # define dummy variables for each level of group
    rates_source = rates_source.with_columns(
        [pl.when(pl.col(group) == g).then(1).otherwise(0).alias(f"{group}_{g}")
         for g in rates_source[group].unique()])
    rates_source = rates_source.with_columns(
        [pl.col(f"{group}_{g}") * pl.col("yr") for g in rates_source[group].unique()])
    rates_source_columns = [col for col in rates_source.columns if col.startswith(f"{group}_")]
    rates_constraints = rates_source.select(rates_source_columns).transpose(include_header=True)
    rates_constraints = rates_constraints[:, 1:]  # remove the first row with the group names
    rates_constraints = rates_constraints.to_numpy()
    # take care of the case when source and target datasets have different groups:
    # - source has a group that target does not have: observations of this group will
    #   give rise to columns containing only NaN values, which can safely be replaced with zeros.
    # - target has a group that source does not have: observations of this group will
    #   give rise to rows containing only NaN values, which can safely be removed.
    # replace all NaN values with zeros
    rates_constraints = np.nan_to_num(rates_constraints, nan=0.0)
    # remove rows of constr_matrix that only contain zeros
    rates_constraints = rates_constraints[~np.all(rates_constraints == 0, axis=1)]
    # normalize if requested
    if normalize:
        rates_constraints = rates_constraints / np.std(rates_constraints, axis=1, keepdims=True)
    
    return rates_constraints


def constr_quantile(
    source: pl.DataFrame,
    target: pl.DataFrame,
    variable: str = "age",
    quantile: float = [.25, .5, .75],
    normalize: bool = True,
    ):
    """ Construct a constraint matrix for the quantiles of a variable in the target dataset.
    
    Parameters
    ----------
    source : polars.DataFrame
        Source dataset containing the outcome feature.
    target : polars.DataFrame
        Target dataset containing the outcome feature.
    variable : str
        Name of the variable column in the source and target datasets to derive the constraint from.
    quantile : list of float
        List of quantiles to calculate the constraints for. Default is [0.25, 0.5, 0.75] (quartiles).
    normalize : bool, optional
        If `True`, the constraint matrix is normalized so that each of its rows has a standard deviation of 1. Default is `True`.
        
    Returns
    -------
    numpy.ndarray
        Array containing the constraints for the quantiles of a variable in the target dataset.
    
    """
    # calculate matrix of quantiles in target dataset
    quantiles_target = target.select(variable).to_numpy().astype(float)
    quantiles_target = np.quantile(quantiles_target, 
                                   quantile)
    # extract the variable of interest
    xsource = source.select(variable).to_numpy().astype(float)
    # construct a matrix: each row corresponds to a quantile, each column to an observation in the source dataset. the value in the i-th row for the j-th observation is the difference between an indicator of whether the j-th observation is below the i-th quantile and the value of the i-th quantile in the target dataset.
    constr_matrix = np.zeros((len(quantile), len(xsource)))
    for i, q in enumerate(quantile):
        constr_matrix[i, :] = (xsource < quantiles_target[i]).flatten() - q
    # normalize if requested
    if normalize:
        constr_matrix = constr_matrix / np.std(constr_matrix, axis=1, keepdims=True)
    
    return constr_matrix


def critical_rho(
    k: int = 2,
    guidance_matrix: np.ndarray = None,
    solver: str = "CLARABEL",
    verbose: bool = False,
    ):
    """ Calculate critical value of the f-divergence radius under which the DRO problem becomes infeasible, based on the guidance matrix and the order of the Cressie-Read divergence family.
    
    Parameters
    ----------
    k : int
        The order of the Cressie-Read divergence. Must be greater than or equal to 1. The default is 2, which corresponds to the chi-squared divergence. k=1 corresponds to the Kullback-Leibler divergence.
    guidance_matrix : numpy.ndarray
        The guidance matrix used in the DRO problem.
    solver : str
        The solver to use for the cvxpy optimization problem. Options are "SCS", "ECOS", "CVXOPT", and "CLARABEL". Default is "CLARABEL".
    verbose : bool
        Whether to print verbose output from the cvxpy solver. Default is `False`.
    
    Returns
    -------
    float
        The critical value of the f-divergence radius.
    
    Raises
    ------
    ValueError
        If k is less than 1.
    ValueError
        If guidance_matrix is None.
    ValueError
        If an unknown solver is specified.
        
    """
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")
    if guidance_matrix is None:
        raise ValueError("guidance_matrix must be provided")
    solver_mapping = {
        "SCS": cp.SCS,
        "ECOS": cp.ECOS,
        "CVXOPT": cp.CVXOPT,
        "CLARABEL": cp.CLARABEL,
        }
    if solver not in solver_mapping:
        raise ValueError(f"Unknown solver '{solver}'. Available solvers: {list(solver_mapping.keys())}")
    solver_constant = solver_mapping[solver]
    
    n = guidance_matrix.shape[1]
    q = cp.Variable(n, nonneg=True)
    Gmat = cp.Constant(guidance_matrix)
    objective = cp.Minimize(cp.mean(f_k(n * q, k=k)))
    constraints = [cp.sum(q) == 1,
                   cp.matmul(Gmat, q) == 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver_constant, verbose=verbose)
    return prob.value
    


# -----------------------
# GuidedDRO class and fitting definitions
# -----------------------

class GuidedDRO:
    def __init__(
        self, 
        num_boost_round: int = 100,
        rho: float = .1, 
        k: int = 2,
        upd_period: int = 1,
        lgbm_params: dict = {
            "objective": "binary",
            "first_metric_only": True,
        },
        solver: str = "CLARABEL",
        solver_params: dict = {
            # see https://clarabel.org/stable/api_settings/
            "max_iter": 5000,
            "tol_gap_abs": 1e-8,
            "tol_gap_rel": 1e-8,
            "tol_feas": 1e-8,
            "tol_infeas_abs": 1e-8,
            "tol_infeas_rel": 1e-8,
            },
        verbose_weights: bool = True,
        verbose_cvxpy: bool = False,
        trace_weights: bool = True,
        trace_losses: bool = True,
        trace_auroc: bool = True,
        ):
        """ Initialize a GuidedDRO instance.
        
        Parameters
        ----------
        num_boost_round : int
            Number of boosting rounds for LightGBM. Default is 100.
        upd_period : int
            Period of updating the weights in the optimization problem. Default is 1, meaning weights are updated every iteration.
        rho : float
            Regularization parameter for the DRO optimization problem. Default is 0.1.
        k : int
            The order of the Cressie-Read divergence family to use in the optimization problem. Default is 2, which corresponds to the chi-squared divergence.
        lgbm_params : dict
            Parameters for LightGBM training. Default is a dictionary with "objective" set to "binary".
        solver : str
            The solver to use for the cvxpy optimization problem. Options are "SCS", "ECOS", "CVXOPT", and "CLARABEL". Default is "CLARABEL".
        solver_params : dict
            Parameters for the cvxpy solver. Default is a dictionary with parameters suitable for CLARABEL.
        verbose_weights : bool
            Whether to print a summary of the weights after each boosting round. Default is True.
        verbose_cvxpy : bool
            Whether to print verbose output from the cvxpy solver. Default is False.
        trace_weights : bool
            Whether to trace and store the weights at each boosting round. Default is True.
        trace_losses : bool
            Whether to trace and store the training losses at each boosting round. Default is True.
        trace_auroc : bool
            Whether to trace and store the AUROC scores at each boosting round. Default is True.
        
        Raises
        ------
        ValueError
            If an unknown solver is specified.
        ValueError
            If k is less than 1.
        ValueError
            If rho is not a non-negative float.
        ValueError
            If objective in lgbm_params is not "binary" or "regression".
        """
        
        if solver not in ["SCS", "ECOS", "CVXOPT", "CLARABEL"]:
            raise ValueError(f"Unknown solver '{solver}'. Available solvers: ['SCS', 'ECOS', 'CVXOPT', 'CLARABEL']")
        if k < 1:
            raise ValueError("k must be greater than or equal to 1")
        if rho < 0:
            raise ValueError("rho must be a non-negative float")
        if lgbm_params["objective"] not in ["binary", "regression"]:
            raise ValueError("objective in lgbm_params must be either 'binary' or 'regression'")
        
        # make sure AUROC is not evaluated in regression tasks
        if lgbm_params["objective"] == "regression":
            trace_auroc = False
        
        self.num_boost_round = num_boost_round
        self.upd_period = upd_period
        self.rho = rho
        self.k = k
        self.solver = solver
        self.lgbm_params = lgbm_params
        self.solver_params = solver_params
        self.verbose_weights = verbose_weights
        self.verbose_cvxpy = verbose_cvxpy
        self.trace_weights = trace_weights
        self.trace_losses = trace_losses
        self.trace_auroc = trace_auroc
    
    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        guide_matrix: np.ndarray = None,
        epsilon: float = 0,
        eval_metrics: bool = True,
        Xval: pl.DataFrame = None,
        yval: np.ndarray = None,
        early_stopping_round: int = 0,
        rhosmooth: bool = False,
        rho_nsmooth: int = 100,
        **kwargs,
        ):
        
        """ Fits a guided DRO model to a GuidedDRO instance. 
        
        Parameters
        ----------
        X : pl.DataFrame
            Training dataset features.
        y : np.ndarray
            Training dataset labels.
        eval_metrics : bool
            Whether to evaluate metrics on a validation dataset. If True, Xval and yval must be provided.
        Xval : pl.DataFrame, optional
            Validation dataset features. Required if eval_metrics is True.
        yval : np.ndarray, optional
            Validation dataset labels. Required if eval_metrics is True.
        guide_matrix : np.ndarray, optional
            A matrix defining the constraints for the DRO optimization problem. If None, no constraints are applied.
        early_stopping_round : int, optional
            Number of rounds for early stopping based on validation AUROC. If 0, no early stopping is applied.
        epsilon : float, optional
            Soft constraint parameter for the guidance matrix. Must be non-negative. Defaults to 0, meaning no soft constraints are applied. Ignored if guide_matrix is None.
        rhosmooth : bool, optional
            Whether to linearly increase the rho parameter from 0 to the value specified in `rho` within the first `rho_nsmooth` iterations. Defaults to False.
        rho_nsmooth : int, optional
            Number of iterations to linearly increase the rho parameter if `rhosmooth` is True. Defaults to 100. Is ignored if `rhosmooth` is False.
        
        **kwargs : dict
            Additional parameters for LightGBM training steps.
        
        Returns
        -------
        self : GuidedDRO
            The fitted GuidedDRO instance. Updated attributes include:
            - `dro_model`: The fitted LightGBM model.
            - `weights_list`: List of weights at each boosting round if `trace_weights` is True.
            - `train_losses`: List of losses at each boosting round if `trace_losses` is True.
            - `train_auroc`: List of AUROC scores at each boosting round if `trace_auroc` is True.
            - `val_losses`: List of validation losses if `eval_metrics` is `True` and `Xval` and `yval` are provided.
            - `val_auroc`: List of validation AUROC scores if `eval_metrics` is True and `Xval` and `yval` are provided.
            - `best_iteration`: dict containing the the best validation AUROC score and its boosting round number if `eval_metrics` is `True` and `Xval` and `yval` are provided.
            - `early_stopping_round`: The boosting round number where early stopping occurred, if applicable.
        
        Raises
        ------
        ValueError
            If an unknown solver is specified or if `eval_metrics` is `True` but `Xval` and `yval` are not provided.
        ValueError
            If early_stopping_round is specified but `eval_metrics` is False or `Xval` or `yval` are not provided.
        """
        
        if eval_metrics and (Xval is None or yval is None):
            raise ValueError("If eval_metrics is True, both Xval and yval must be provided.")
        if early_stopping_round > 0 and (not eval_metrics or Xval is None or yval is None):
            raise ValueError("If early_stopping_round is specified, eval_metrics must be True and a validation dataset (Xval and yval) must be provided.")
        if guide_matrix is not None and epsilon < 0:
            raise ValueError("If guide_matrix is provided, epsilon must be a non-negative float.")
        
        # check guidance matrix
        if guide_matrix is not None and np.linalg.matrix_rank(guide_matrix) < guide_matrix.shape[0]:
            print("Warning: The guide matrix is not full rank. This may lead to numerical issues during fitting.")
        
        # compute critical rho if guidance matrix is provided and stop execution if rho is too small
        if guide_matrix is not None:
            rho_crit = critical_rho(k=self.k, guidance_matrix=guide_matrix, solver=self.solver, verbose=self.verbose_cvxpy)
            if self.rho < rho_crit:
                print(f"Warning: The specified rho={self.rho} is smaller than the critical value rho_crit={rho_crit:.4f}. Please increase rho to at least rho_crit to ensure feasibility of the optimization problem.")
            if self.rho < 2 * rho_crit and rhosmooth:
                print(f"Warning: The specified rho={self.rho} is smaller than twice the critical value rho_crit={rho_crit:4f}. When using rhosmooth, rho is required to be at least twice the critical value rho_crit to ensure that the optimization problem is feasible at the beginning of the smoothing period. Please increase rho accordingly.")
        elif rhosmooth:
            rho_crit = 0
        
        # define the appropriate solver
        solver_mapping = {
        "SCS": cp.SCS,
        "ECOS": cp.ECOS,
        "CVXOPT": cp.CVXOPT,
        "CLARABEL": cp.CLARABEL,
        }
        if self.solver not in solver_mapping:
            raise ValueError(f"Unknown solver '{self.solver}'. Available solvers: {list(solver_mapping.keys())}")
        solver_constant = solver_mapping[self.solver]
        
        # make dataset Arrow-compatible
        X_arrow = X.to_arrow()
        X_columns = X.columns
        if Xval is not None:
            Xval_arrow = Xval.to_arrow()
            Xval_arrow = Xval_arrow.select(X_columns)  # ensure same column order as training set to avoid issues when predicting
        else:
            Xval_arrow = None
        
        print("Initializing optimization problem...")
        n = len(y)
        q = cp.Variable(n, nonneg=True)
        losses = cp.Parameter(n, nonneg=True)
        rhocurrent = cp.Parameter(nonneg=True)
        objective = cp.Maximize(cp.sum(cp.multiply(losses, q)))
        constraints = [cp.sum(q) == 1]
        rhocurrent.value = 2 * rho_crit if rhosmooth else self.rho
        if self.k == 1:
            constraints.append(-cp.sum(cp.entr(q)) <= rhocurrent - np.log(n))
        else:
            constraints.append(cp.mean(f_k(n * q, k=self.k)) <= rhocurrent)
        if guide_matrix is not None:
            Gmat = cp.Constant(guide_matrix)
            if epsilon > 0:
                epsilon = epsilon / np.sqrt(len(y)) * np.linalg.norm(guide_matrix, ord="fro")
                constraints.append(cp.norm(cp.matmul(Gmat, q), 2) <= epsilon)
            else:
                constraints.append(cp.matmul(Gmat, q) == 0)   
        prob = cp.Problem(objective, constraints)
        
        # initialize weights, training dataset, model
        weights = np.ones(len(y))
        if self.trace_weights:
            weights_list = [weights]
        if self.trace_losses:
            train_losses = []
        if self.trace_auroc:
            train_auroc = []
        dro_dataset = lgb.Dataset(
            data=X_arrow,
            label=y,
            weight=weights,
            categorical_feature=[c for c in X_columns if "categorical" in c],
            feature_name=X_columns,
            free_raw_data=False,
            )
        dro_model = lgb.train(
            params=self.lgbm_params,
            train_set=dro_dataset,
            num_boost_round=1,
            **kwargs,
            )
        # initialize validation metrics if requested
        if eval_metrics:
            val_losses = []
            if self.lgbm_params["objective"] == "binary":
                val_auroc = []
        failed_iterations = [] # counter for rounds where weight update failed
        val_metric_previous = float("-inf") # for early stopping
        patience_counter = 0 # for early stopping
        # evaluate first boosting round on validation dataset
        if eval_metrics and Xval is not None:
            val_prediction = dro_model.predict(Xval_arrow, raw_score=True)
            if self.lgbm_params["objective"] == "binary":
                val_pscores = 1 / (1 + np.exp(-val_prediction))
                #clip probabilities to avoid numerical issues
                val_pscores = np.clip(val_pscores, 1e-15, 1 - 1e-15)
                val_losses.append(-yval.astype(int) * np.log(val_pscores) - (1 - yval.astype(int)) * np.log(1 - val_pscores))
                val_metric_current = metrics.roc_auc_score(yval, val_prediction)
                val_auroc.append(val_metric_current)
            else:
                val_losses.append(-(val_prediction - yval) ** 2)
                val_metric_current = np.mean(val_losses[-1])
        print("Optimization problem has been initialized successfully.\nStart boosting rounds:")
        for i in range(self.num_boost_round - 1):
            # update value of rho if necessary
            if rhosmooth and i <= rho_nsmooth - 2:
                rhocurrent.value = (self.rho - 2 * rho_crit) / (rho_nsmooth - 2) + 2 * rho_crit
                # rhocurrent.value = (i + 1) / (rho_nsmooth - 1) * self.rho
            # get predictions from current round and calculate losses
            current_prediction = dro_model.predict(X_arrow, raw_score=True)
            # binary logloss for classification
            if self.lgbm_params["objective"] == "binary":
                pscores = 1 / (1 + np.exp(-current_prediction))
                # clip probabilities to avoid numerical issues
                pscores = np.clip(pscores, 1e-15, 1 - 1e-15)
                yint = y.astype(int)
                current_losses = -yint * np.log(pscores) - (1 - yint) * np.log(1 - pscores)
                losses.value = current_losses / np.sum(current_losses)
            # mean squared error for regression
            else:
                current_losses = (current_prediction - y) ** 2
                losses.value = current_losses / np.sum(current_losses)
            if self.trace_losses:
                train_losses.append(current_losses)
            if self.trace_auroc:
                train_auroc.append(metrics.roc_auc_score(y, current_prediction))
            # optimize to get weights from losses, but only all upd_period rounds
            if i % self.upd_period == 0:
                try:
                    print(f"- Boosting round {i+2}: Solving inner optimization problem...")
                    if rhosmooth:
                        print(f"  Current value of rho: {rhocurrent.value:.5g}")
                    prob.solve(
                        solver=solver_constant,
                        verbose=self.verbose_cvxpy,
                        ignore_dpp=True,
                        **self.solver_params,
                    )
                except Exception as e:
                    try:
                        print(f"  Optimization failed with error: {e}\n  Retrying with losses multiplied by n for numerical stability...")
                        losses.value = n * losses.value
                        prob.solve(
                            solver=solver_constant,
                            verbose=self.verbose_cvxpy,
                            ignore_dpp=True,
                            **self.solver_params,
                        )
                        print("  Optimization succeeded with rescaled losses.")
                    except Exception as e:
                        failed_iterations.append(i)
                        if self.verbose_weights:
                            print(f"  Optimization failed with error: {e}\n  Skipping this boosting round.")
                        if i == 0:
                            weights = np.ones(n) / n
            weights = n * q.value # multiply by n for numerical stability
            if self.trace_weights:
                weights_list.append(weights)
            if(self.verbose_weights):
                print(f"  Weights after boosting round {i+2}: minimum {min(weights):.5g}, maximum {max(weights):.5g}")
            # update the weights in the dataset
            dro_dataset = lgb.Dataset(
                data=X_arrow,
                label=y,
                weight=weights,
                categorical_feature=[c for c in X_columns if "categorical" in c],
                feature_name=X_columns,
                free_raw_data=False,
                )
            # perform the next boosting round
            dro_model = lgb.train(
                params=self.lgbm_params,
                train_set=dro_dataset,
                num_boost_round=1,
                init_model=dro_model,
                **kwargs,
                )
            # evaluate on validation dataset
            if eval_metrics and Xval is not None:
                val_prediction = dro_model.predict(Xval_arrow, raw_score=True)
                if self.lgbm_params["objective"] == "binary":
                    val_pscores = 1 / (1 + np.exp(-val_prediction))
                    # clip probabilities to avoid numerical issues
                    val_pscores = np.clip(val_pscores, 1e-15, 1 - 1e-15)
                    val_losses.append(-yval.astype(int) * np.log(val_pscores) - (1 - yval.astype(int)) * np.log(1 - val_pscores))
                    val_metric_current = metrics.roc_auc_score(yval, val_prediction)
                    val_auroc.append(val_metric_current)
                else:
                    val_losses.append(-(val_prediction - yval) ** 2) # careful: -MSE is supposed to increase with better predictions
                    val_metric_current = np.mean(val_losses[-1])
            # early stopping if requested
            if early_stopping_round > 0:
                if val_metric_current <= val_metric_previous:
                    patience_counter += 1
                else:
                    patience_counter = 0
                if patience_counter >= early_stopping_round:
                    print(f"Early stopping at boosting round {i+1} due to no improvement in validation metric for {early_stopping_round} rounds.")
                    self.early_stopping_round = i + 1
                    break
                val_metric_previous = val_metric_current
        self.dro_model = dro_model
        # calculate losses from the last boosting round
        if self.lgbm_params["objective"] == "binary":
                pscores = 1 / (1 + np.exp(-current_prediction))
                # clip probabilities to avoid numerical issues
                pscores = np.clip(pscores, 1e-15, 1 - 1e-15)
                yint = y.astype(int)
                current_losses = -yint * np.log(pscores) - (1 - yint) * np.log(1 - pscores)
                losses.value = current_losses / np.sum(current_losses)
        else:
            current_losses = (current_prediction - y) ** 2
            losses.value = current_losses / np.sum(current_losses)
        if self.trace_losses:
            train_losses.append(current_losses)
        if self.trace_auroc:
            train_auroc.append(metrics.roc_auc_score(y, current_prediction))
        # assign weights, losses, and auroc lists if tracing is enabled
        if self.trace_weights:
            self.weights_list = weights_list
        if self.trace_losses:
            self.train_losses = train_losses
        if self.trace_auroc:
            self.train_auroc = train_auroc
        # assign evaluation metrics if calculation is enabled
        if eval_metrics:
            if self.lgbm_params["objective"] == "binary":
                self.val_losses = val_losses
                self.val_auroc = val_auroc
                self.best_iteration = {
                    "best_iter": np.argmax(self.val_auroc) + 1,
                    "best_auroc": max(self.val_auroc),
                }
            else:
                val_losses = [-loss for loss in val_losses]
                self.val_losses = val_losses
                self.val_mse = [np.mean(loss) for loss in val_losses]
                self.best_iteration = {
                    "best_iter": np.argmin(self.val_mse) + 1,
                    "best_mse": min(self.val_mse),
                }
        return self
    
    def predict(
        self,
        Xpred: pl.DataFrame,
        num_iteration: int = None,
        raw_score: bool = True,
        ):
        """ Predict using the fitted guided DRO model.
        
        Parameters
        ----------
        Xpred : pl.DataFrame
            Input features for prediction.
        num_iteration : int, optional
            Number of boosting rounds to use for prediction. If None, uses the number of boosting rounds from the fitted model.
        raw_score : bool, optional
            If True, returns the raw scores from the model. If False, returns the predicted probabilities. Defaults to True.
            
        Returns
        -------
        np.ndarray
            Predicted values. If raw_score is True, returns raw scores; otherwise, returns predicted probabilities.
        
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        
        if not hasattr(self, 'dro_model'):
            raise ValueError("The model has not been fitted yet. Please call fit() before predict().")
        
        Xpred_arrow = Xpred.to_arrow()
        if num_iteration is None:
            num_iteration = self.num_boost_round
        return self.dro_model.predict(
            Xpred_arrow,
            num_iteration=num_iteration, 
            raw_score=raw_score)
    
    def save(
        self,
        file_name: str,
        dir_name: str = "02_data/02_data_dump/",
        ):
        """ Save a fitted guided DRO model to a specified directory.
        
        This method saves the model in two formats: a text file and a pickle file.
        
        Parameters
        ----------
        file_name : str
            The name of the file to save the model. The ".txt" suffix is added automatically, so it should not be included.
        dir_name : str, optional
            The directory to save the model. Defaults to "02_data/02_data_dump".
        
        Returns
        -------
        None
        """
        self.dro_model.save_model(dir_name + file_name + ".txt")
        with open(f"{dir_name}{file_name}.pkl", "wb") as f:
            pk.dump(self, f)
    
    def eff_sample_size(
        self,
        ):
        """ Calculate the effective sample size at each boosting round of the fitted guided DRO model.
        
        This method computes the effective sample size based on the weights assigned to each observation during the fitting process.
        
        Returns
        -------
        list
            A list containing the effective sample size at each boosting round.
        """
        if not hasattr(self, 'weights_list'):
            if not self.trace_weights:
                raise ValueError("Weights tracing is not enabled. Set trace_weights=True when initializing the GuidedDRO instance.")
            else:
                raise ValueError("The model has not been fitted yet. Please call fit() before calculating effective sample size.")
        
        eff_sample_sizes = [np.sum(weights)**2 / np.sum(np.square(weights)) for weights in self.weights_list]
        
        return eff_sample_sizes


# -----------------------
# Tuning
# -----------------------


# -----------------------
# Simulation
# -----------------------

# function to simulate a dataset for guided DRO fitting
def simulate_gdro_data(
    n_obs: int = 1000, 
    n_cat_features: int = 5,
    n_cont_features: int = 5,
    betas: np.ndarray = None,
    beta_range: tuple = (-5, 5),
    outc_type: str = "binary",
    cont_sd: float = 1.0,
    cat_prob: float = 0.5,
    random_seed: int = None,
    prop_shift: float = 0.1,
    outc_shift: bool = True,
    cont_shift: float = 1.0,
    val_split: float = 0.2,
    covar_shift: float = 0,
    n_covar_shift: int = 0,
    ):
    """ Simulate a dataset for testing GuidedDRO modelling.
    
    Parameters
    ----------
    n_obs : int
        Number of observations to simulate.
    n_cat_features : int
        Number of categorical features to simulate.
    n_cont_features : int
        Number of continuous features to simulate.
    betas : np.ndarray, optional
        Regression coefficients for the features. If None, coefficients are sampled uniformly from beta_range.
    beta_range : tuple of float
        Range of regression coefficients to simulate. This is used to create a linear relationship between features and outcome.
    outc_type : str
        Type of outcome variable to simulate. Options are `"binary"` or `"continuous"`.
    cont_sd : float
        Standard deviation for the continuous outcome variable noise. Only used if outc_type is `"continuous"` and is ignored otherwise.
    cat_prob : float
        Probability of success for the binary categorical features. Only used if n_cat_features > 0 and is ignored otherwise.
    random_seed : int, optional
        Random seed for reproducibility. If None, no seed is set.
    prop_shift : float
        Proportion of observations to be shifted in the outcome variable. This is used to create a distributional shift in the data.
    outc_shift : bool
        If `True`, applies a shift to the outcome variable in a proportion `prop_shift` of the observations.  If `outc_type`is `"continuous"`, then `cont_shift` is added to the outcome variable. If `outc_tye` is `"binary"`, then the observations are simply flipped. Defaults to `True`.
    cont_shift: float
        Shift applied to the continuous outcome variable. Only used if outc_type is `"continuous"` and is ignored otherwise.
    val_split : float
        Proportion of the dataset to use for validation. This is not used in the simulation but can be useful for later splitting.
    covar_shift : float
        Shift applied to continuous features. Defaults to 0, meaning no shift is applied.
    n_covar_shift : int
        Number of continuous features to apply the shift to. Defaults to 0, meaning no shift is applied. If > 0, the shift `covar_shift` is applied to the first `n_covar_shift` continuous features.
        
    Returns
    -------
    X : pl.DataFrame
        Simulated dataset as a Polars DataFrame with features and outcome variable.
        
    Raises
    ------
    ValueError
        If outc_type is not "binary" or "continuous".
    """
    
    if outc_type not in ["binary", "continuous"]:
        raise ValueError("outc_type must be either 'binary' or 'continuous'.")
    if betas is not None and len(betas) != n_cat_features + n_cont_features:
        raise ValueError(f"betas must have length {n_cat_features + n_cont_features}, but got {len(betas)}.")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # simulate regression coefficients uniformly in the specified range
    if betas is None:
        betas = np.random.uniform(beta_range[0], beta_range[1], size=n_cat_features + n_cont_features)
    # generate features: categorical from Bernoulli disdtribution, continuous from standard normal
    X_cat = np.random.binomial(1, cat_prob, size=(n_obs, n_cat_features))
    # if cat_prob != 0.5:
    #     X_cat = np.random.binomial(1, cat_prob, size=(n_obs, n_cat_features))
    # else:
    #     X_cat = np.random.randint(0, 2, size=(n_obs, n_cat_features))
    X_cont = np.random.normal(0, 1, size=(n_obs, n_cont_features))
    X = np.hstack((X_cat, X_cont))
    # make sure X is a Polars DataFrame
    X = pl.DataFrame(X, schema={f"categorical_{i}": pl.Int8 for i in range(n_cat_features)} | {f"continuous_{i}": pl.Float64 for i in range(n_cont_features)})
    # calculate linear predictor
    linear_combination = X @ betas[:n_cat_features + n_cont_features]
    # calculate number of observations to flip or shift
    n_shifted = int(n_obs * prop_shift)
    # calculate outcome variable based on the specified type and apply flip/shift
    if outc_type == "binary":
        y = np.random.binomial(1, 1 / (1 + np.exp(-linear_combination)))
        if outc_shift:
            y[:n_shifted] = 1 - y[:n_shifted]
    elif outc_type == "continuous":
        y = linear_combination + np.random.normal(0, cont_sd, size=n_obs)
        if outc_shift:
            y[:n_shifted] += cont_shift
    # apply shift to first n_shifted observations of continuous features if specified
    if n_covar_shift > 0:
        covar_shift_vec = np.zeros(n_obs)
        covar_shift_vec[:n_shifted] = covar_shift
        for i in range(n_covar_shift):
            # add covar_shift to the column "continuous_i"
            X = X.with_columns(pl.col(f"continuous_{i}") + pl.lit(covar_shift_vec).alias(f"continuous_{i}"))
    
    # bind y as a column to X
    X = X.with_columns(pl.Series("outcome", y))
    
    # make train and validation splits: use first val_split proportion of observations in each the shifted and non-shifted parts of the dataset
    split = np.array(["train"] * n_obs)
    val_indices = list(range(int(n_shifted * val_split))) + list(range(n_shifted, n_shifted + int((n_obs - n_shifted) * val_split)))
    split[val_indices] = "val"
    X = X.with_columns(pl.Series("split", split))
    
    # generate output
    sim_output = {
        "dataset": X,
        "betas": betas,
    }
    
    return sim_output


# -----------------------
# Testing and evaluation
# -----------------------

# function for testing via simulation
def test_gdro_simulation(
    rho: float = .1, 
    k: int = 2,
    n_iter: int = 1000,
    lgbm_params: dict = {
        "objective": "regression",
        "learning_rate": 0.05,
        "max_depth": 2,
        "verbose": -1,
    },
    solver: str = "CLARABEL",
    verbose_weights: bool = True,
    trace_weights: bool = True,
    trace_losses: bool = True,
    early_stopping_round: int = 50,
    n_weightsamples: int = 20,
    dirname = "03_tests/00_gdro_tests",
    **kwargs
    ):
    """ Test the GuidedDRO implementation on a simulated dataset.
    
    Creates a simulated dataset using the `simulate_gdro_data` function from the `guided_dro` module, fits both a simple LightGBM model and a GuidedDRO model to the data (using the same LightGBM hyperparameters for both), and plots the training losses and weights over row indices of the simulated dataset and a sample of weight traces over iterations.
    
    Parameters
    ----------
    rho : float
        Radius of the f-divergence ball of the GuidedDRO model.
    k : int
        The order of the Cressie-Read divergence family to use in the GuidedDRO model.
    n_iter : int
        Number of iterations for the GuidedDRO model.
    lgbm_params : dict
        Parameters for the LightGBM fitting routine.
    solver : str
        The solver to use for the cvxpy optimization problem in the GuidedDRO model. Options are "SCS", "ECOS", "CVXOPT", and "CLARABEL". Default is "CLARABEL".
    verbose_weights : bool
        Whether to print a summary of the weights after each iteration.
    early_stopping_round : int
        Number of rounds for early stopping based on validation AUROC. If 0, no early stopping is applied.
    n_weightsamples : int
        Number of random rows to sample for plotting weight traces.
    dirname : str
        Directory to save the plots and model files. Default is "03_tests/00_gdro_tests".
    **kwargs : dict
        Additional keyword arguments for the `simulate_gdro_data` function.
        
    Returns
    -------
    Dictionary containing the fitted LightGBM model and the fitted GuidedDRO model as well as the training and validation losses (and AUROC scores).
    """
    
    # simulate a dataset 
    gdro_source = simulate_gdro_data(
        **kwargs,
        )
    gdro_source = gdro_source["dataset"]
    # preprocess dataset for lightgbm model fitting
    source_prc = preprocess_lgbm(gdro_source, outcome="outcome")
    X_columns = source_prc["training_data"].columns
    lgb_dataset = lgb.Dataset(
        data=source_prc["training_data"],
        label=source_prc["ytrain"],
        categorical_feature=[c for c in X_columns if "categorical" in c],
        feature_name=X_columns,
        free_raw_data=False,
        )
    lgb_validation_set = lgb.Dataset(
        data=source_prc["validation_data"],
        label=source_prc["yval"],
        categorical_feature=[c for c in X_columns if "categorical" in c],
        feature_name=X_columns,
        free_raw_data=False,
        )
    # fit simple lightgbm model
    print("Fitting simple LightGBM model...")
    # if objective is binary, use binary_logloss and auc as metrics
    if lgbm_params["objective"] == "binary":
        lgbm_params["metric"] = ["binary_logloss", "auc"]
        lgbm_params["first_metric_only"] = True
    evals_result = {}
    lgb_model = lgb.train(
        params=lgbm_params,
        train_set=lgb_dataset,
        num_boost_round=n_iter,
        valid_sets=[lgb_validation_set,
                    lgb_dataset],
        valid_names=["validation",
                    "training"],
        callbacks=[lgb.record_evaluation(evals_result)]
        )
    # fit simple gdro model
    print(f"Fitting GuidedDRO model...\nrho = {rho}, k = {k}, n_iter = {n_iter}")
    gdro_model = GuidedDRO(
        num_boost_round=n_iter,
        rho=rho,
        k=k,
        lgbm_params=lgbm_params,
        trace_weights=trace_weights,
        trace_losses=trace_losses,
        verbose_weights=verbose_weights,
        solver=solver,
        )
    gdro_model.fit(
        X=source_prc["training_data"],
        y=source_prc["ytrain"],
        guide_matrix=None,
        eval_metrics=True,
        Xval=source_prc["validation_data"],
        yval=source_prc["yval"],
        early_stopping_round=early_stopping_round,
        )
    print("GuidedDRO model fitted successfully.")
    # information for plot names
    plotinfo = f"rho{rho}_k{k}_nobs{kwargs['n_obs']}"
    plotinfo = "binary_" + plotinfo if lgbm_params["objective"] == "binary" else "regression_" + plotinfo
    # plot training loss over iterations for both models
    if lgbm_params["objective"] == "regression":
        mean_train_loss = [np.mean(loss) for loss in gdro_model.train_losses]
        plt.figure(figsize=(10, 6))
        plt.plot(evals_result["training"]["l2"], label="Training MSE (LightGBM)", color="blue")
        plt.plot(mean_train_loss, label="Training MSE (GuidedDRO)", color="orange", linestyle='--', linewidth=2)
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        plt.title(f"Training MSE over iterations for LightGBM and GuidedDRO models\nrho = {rho}, k = {k}, n_obs = {kwargs['n_obs']}")
        plt.tight_layout()
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f"{dirname}/sim_{plotinfo}_comparetrain.png", bbox_inches="tight")
    elif lgbm_params["objective"] == "binary":
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        # binary logloss
        axes[0].plot(evals_result["training"]["binary_logloss"], label="Training Logloss (LightGBM)", color="blue")
        axes[0].plot([np.mean(loss) for loss in gdro_model.train_losses], label="Training Logloss (GuidedDRO)", color="orange", linestyle='--', linewidth=2)
        axes[0].set_xlabel("Iterations")
        axes[0].set_ylabel("Binary Logloss")
        axes[0].set_title(f"Training Logloss over iterations\nrho = {rho}, k = {k}, n_obs = {kwargs['n_obs']}")
        axes[0].grid()
        axes[0].legend(loc='upper right')
        # auroc
        axes[1].plot(evals_result["training"]["auc"], label="Training AUROC (LightGBM)", color="blue")
        axes[1].plot(gdro_model.train_auroc, label="Training AUROC (GuidedDRO)", color="orange", linestyle='--', linewidth=2)
        axes[1].set_xlabel("Iterations")
        axes[1].set_ylabel("AUROC")
        axes[1].set_title(f"Training AUROC over iterations\nrho = {rho}, k = {k}, n_obs = {kwargs['n_obs']}")
        axes[1].grid()
        axes[1].legend(loc='lower right')
        plt.tight_layout()
        # save plot
        plt.savefig(f"{dirname}/sim_{plotinfo}_comparetrain.png", bbox_inches="tight")
    
    # plot validation loss over iterations for both models
    if lgbm_params["objective"] == "regression":
        mean_val_loss = [np.mean(loss) for loss in gdro_model.val_losses]
        plt.figure(figsize=(10, 6))
        plt.plot(mean_val_loss, label="Validation MSE (GuidedDRO)", color="orange", linestyle='--', linewidth=2)
        plt.plot(evals_result["validation"]["l2"], label="Validation MSE (LightGBM)", color="blue")
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        plt.title(f"Validation MSE over iterations for LightGBM and GuidedDRO models\nrho = {rho}, k = {k}, n_obs = {kwargs['n_obs']}")
        plt.tight_layout()
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f"{dirname}/sim_{plotinfo}_compareval.png", bbox_inches="tight")
    elif lgbm_params["objective"] == "binary":
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        # binary logloss
        axes[0].plot(evals_result["validation"]["binary_logloss"], label="Validation Logloss (LightGBM)", color="blue")
        axes[0].plot([np.mean(loss) for loss in gdro_model.val_losses], label="Validation Logloss (GuidedDRO)", color="orange", linestyle='--', linewidth=2)
        axes[0].set_xlabel("Iterations")
        axes[0].set_ylabel("Binary Logloss")
        axes[0].set_title(f"Validation Logloss over iterations\nrho = {rho}, k = {k}, n_obs = {kwargs['n_obs']}")
        axes[0].grid()
        axes[0].legend(loc='upper right')
        # auroc
        axes[1].plot(evals_result["validation"]["auc"], label="Validation AUROC (LightGBM)", color="blue")
        axes[1].plot(gdro_model.val_auroc, label="Validation AUROC (GuidedDRO)", color="orange", linestyle='--', linewidth=2)
        axes[1].set_xlabel("Iterations")
        axes[1].set_ylabel("AUROC")
        axes[1].set_title(f"Validation AUROC over iterations\nrho = {rho}, k = {k}, n_obs = {kwargs['n_obs']}")
        axes[1].grid()
        axes[1].legend(loc='lower right')
        plt.tight_layout()
        # save plot
        plt.savefig(f"{dirname}/sim_{plotinfo}_compareval.png", bbox_inches="tight")

    
    # plot weights at the last iteration over the row indices of the data
    plt.figure(figsize=(10, 6))
    plt.plot(gdro_model.weights_list[-1])
    plt.xlabel("Row Index")
    plt.ylabel("Weight")
    plt.title(f"Weights at last iteration for GuidedDRO model\nrho = {rho}, k = {k}, n_obs = {kwargs['n_obs']}")
    plt.tight_layout()
    plt.axvline(x=source_prc["training_data"].shape[0] * kwargs.get("prop_shift", 0), color='red', linestyle='--')
    plt.savefig(f"{dirname}/sim_{plotinfo}_finalweights.png", bbox_inches="tight")
    
    # extract a few observations randomly from the training data and plot their weight traces
    sample_indices = np.random.choice(
        range(source_prc["training_data"].shape[0]), 
        size=n_weightsamples, 
        replace=False)
    plt.figure(figsize=(10, 6))
    for i in sample_indices:
        plt.plot([weights[i] for weights in gdro_model.weights_list], label=f"Row {i}")
    plt.xlabel("Iterations")
    plt.ylabel("Weight")
    plt.title(f"Weight traces for {n_weightsamples} random rows\nrho = {rho}, k = {k}, n_obs = {kwargs['n_obs']}")
    plt.tight_layout()
    # plt.legend()
    plt.savefig(f"{dirname}/sim_{plotinfo}_weighttraces.png", bbox_inches="tight")
    
    print(f"GuidedDRO model fitted and diagnostic plots saved to {dirname}.")
    
    results = {
        "gdro_model": gdro_model,
        "lgbm_model": lgb_model,
        "evals_result": evals_result,
    }
    
    return results
