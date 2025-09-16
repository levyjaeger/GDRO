# GDRO: Guided Distributionally Robust Optimization for Early Outcome Prediction in the Intensive Care Unit

This repository contains the code that generated the results of a master thesis conducted at the [Seminar for Statistics (SfS)](https://math.ethz.ch/sfs) of the Department of Mathematics (D-MATH) at ETH Zurich. The aim of this project is to explore extensions of distributionally robust optimization (DRO) modelling based on $f$-divergences in form of __guidance constraints__ that involve available summary statistics of target datasets. The idea is that such guidance constraints may improve model performance by  

This README is divided in two sections. Section 1 gives an explanation of the repository, including how the files must be run in order to reproduce the results presented in the thesis. Section 2 comments on several issues encountered during implementation of the guided DRO routine. The terminology and mathematical notation follows the exposition of the thesis Jäger (2025). For more details, you may ask the author directly at jaegerl@student.ethz.ch.

## 2 Implementation

### 2.1 Code structure

The guided DRO routine was written in Python, and the analyses were run on the high-performance computing platform [Euler](https://docs.hpc.ethz.ch/) at ETH Zurich. The core of the code is contained in the three `gdro` modules: `model.py`, `ICUdata.py`, and `deploy.py`. 

The core of the implementation is the `model.py`module, which contains the implementation of the guided DRO estimation algorithm corresponding to Algorithm 1 in Jäger (2025). This algorithm involves a modification of the gradient boosting machine LightGBM (Microsoft, 2024). In particular, required packages are `lightgbm` (version 4.6.0) and `cvxpy` (version 1.5.3). In this module, we defined a class `GuidedDRO` whose objects are initialized with the following main components:
- Number of boosting rounds
- Radius $\rho > 0$ for the $f$-divergence constraint
- Index $k \geq 1$ of the chosen $f$-divergence of the Cressie-Read family
- The solver for the CVXPY optimization routine
- Precision parameters to be passed to the CVXPY optimization routine
- Additional parameters to be passed to `lightgbm.fit()`, which may involve the type of objective (`"binary"` versus `"regression"`), the maximum tree depth, or the learning rate
The central method of the `GuidedDRO` class is a `fit()`method that takes training and evaluation datasets as inputs and implements the model fitting routine as described in Algorithm 1 of Jäger (2025). The method takes a training dataset and, optionally, an evaluation dataset as well as further parameters, such as a softening parameter $\epsilon > 0$ for softened guidance constraints, as inputs. The algorithm involves alternating between single `lightgbm.fit()` gradient boosting steps using a $n \times 1$ vector of training weights $\boldsymbol{q}$ obtained by solving, at each round, a convex optimization problem involving a $n \times 1$ loss vector $\boldsymbol{\ell}$ determined from the respective previous boosting round. This 'inner optimization problem' maximizes the linear objective $\boldsymbol{\ell}^T \boldsymbol{q}$ under specific constraints that include guidance constraints and is formulated and solved using the CVXPY language (Agrawal, 2018; Diamond, 2016).

A guided DRO model is fitted as a `lightgbm.Booster` object and stored as an attribute of the `GuidedDRO` object. Optionally, further outputs from modeling can be stored in the `GuidedDRO` object, such as training weights or evaluation metrics for each boosting round.

Guidance constraints for $n \times 1$ probability vectors $\boldsymbol{q}$  in the inner optimization problem are formulated in the form $\boldsymbol{Z} \boldsymbol{q} = \mathbf{0}$, where $\boldsymbol{Z}$ is a $r \times n$ guidance matrix and $\mathbf{0}$ is the $n \times 1$ vector of zeros. The `model.py` module includes functions for generation of different forms of guidance constraints:
- average constraints (`constr_avg()`) for the expected value of continuous variables or the expected marginal frequencies of classes of categorical variables (examples: average age in years, prevalence of female patients, mortality rate),
- conditional average constraints (`constr_avg_by_group()`) for the conditional expectation of a continuous or binary 0-1 variable within the different categories of a categorical variable (example: mortality rate by age-sex strata),
- conditional average constraints (`constr_avg_by_cutoff()`) for the conditional expectation of a continuous or binary 0-1 variable within two groups of a continuous variable defined by whether this continuous variable lies above or below a specific cutoff (example: mortality rate by age groups $\geq75$ years or $<75$ years)
- quantile constraints (`constr_quantile()`) for quantiles of a continuous variable (example: $0.25$-, $0.5$-, and $0.75$-quantiles of age in years).

Another section of the `model.py` module contains functions for the generation of simulated datasets that can be used for testing of the guided DRO modeling routine and for exploration of model behavior. These functions underlie our simulation analyses. The simulations focus on simple datasets with either linear models for regression and logistic models for binary classification tasks. In particular, it is possible to generate datasets with distributional shifts in the outcome or feature variables, controlling the frequency and the extent of the shifts.

The loading and preprocessing of ICU datasets, including the choice of appropriate model features, is governed by the functions in the module `ICUdata.py`. The code in this module is largely adapted from the code used by Londschien (2025), available on https://github.com/eth-mds/icu-features.

The module `deploy.py` contains routines that were used to fit the models on ICU datasets, enabling pipelines that start from data loading, over data preprocessing to the generation and storage of fitted models and evaluation results.

### 2.2 Issues
Here is a list of issues encountered during optimization and how we approached them in our fitting routine.
- The `lightgbm.fit()` function may encounter underflow issues with weights normalized to sum $1$ for large dataset sizes $n$. The weights $\boldsymbol{q}$ determined by the CVXPY routine are therefore rescaled by multiplication with $n$ before being passed to `lightgbm.fit()` for the corresponding gradient boosting round.
- Especially for large values of $\rho$, meaning that the worst-case scenario is searched in a large space of probability distributions, the weights $\boldsymbol{q}$ can become highly imbalanced, which reflects in a high imbalance of the losses $\boldsymbol{\ell}$ computed after the respective boosting round. The highly imbalanced losses may pose numerical problems to the respective following CVXPY optimization step. When this is the case and CVXPY fails to solve the inner optimization problem even though it would be feasible, a second attempt is performed using a rescaled loss vector $n \cdot \boldsymbol{\ell}$. If CVXPY still fails, the loss update is simply skipped in this boosting round.
- For large values of $\rho$, the CVXPY routine may slow down considerably. We implemented two options for the algorithm that can be used to speed up the algorithm can therefore be useful for testing purposes:
	- Skipping weight updates: The inner optimization for updating $\boldsymbol{q}$ is performed only at every $m$th round, where $m \leq M$.
	- Smoothing $\rho$: The value of $\rho$ used for the inner optimization increases linearly with each round until it reaches its final, intended value at a specified round $\leq M$.

## 2 Code Utilization
### 2.1 Setup
The analyses were run using Python, version 3.10.9 to ensure compatibility with the `cvxpy` package. See the `requirements.txt` and `requirements.yml` files for a more comprehensive list of dependencies.

### 2.1 Utilization
To reproduce our results, follow these steps:
	1) After cloning the repository, set up the Python environment using the `environment.yml` file.
	2) Add the ICU datasets as .parquet files to the `data` folder. Use the names `miiv`, `eicu`, `hirid`, `nwicu`, and `zigong` for MIMIC-IV, eICU, HiRID, and ZFPH, respectively.
	3) The simulation and the ICU dataset analyses can be run independently of each other. For the ICU analyses, navigate to folder `scripts/icumodels`. The codes used to fit the individual models are in the modeling files of the form `[outcome]_[source]to[target].py`, where `outcome` is either of `mortality`(for mortality within 24h), `renal`(for acute kidney injury within 48h), or `creatinine`(for serum creatinine within 24h), `source` is either of `miiv` or `eicu`, and `target` can denote any of the datasets. Each individual modeling file provides the code to fit models over all possible combinations of $k \in \lbrace 1, 2 \rbrace$ and $\rho \in \lbrace 0.01, 0.05, 0.1, 0.5, 1 \rbrace$ with three different guidance constellations (marginal demographic constraints, stratum-wise mortality constraints, and simple DRO without guidance constraints), saving the fitted model in the  `results/models` folder and tables containing model evaluations in the `results/tables`folder. Since for each outcome-source-target combination, a total of $2\cdot5\cdot3=30$ models needs to be fitted, the scripts are designed to be run on Euler with parallelization over the $2\cdot 5 =10$  combinations of $k$ and $\rho$ values. The parallel processing can be initiated by running the corresponding SLURM files named according to `[outcome]_[source]to[target].slurm`. The latter also automatically store log and error files in corresponding folders generated in the `results/logs` folder. After the desired models files have been run, the corresponding figures can be generated by running the file `rhofigures.py` and `weightfigures.py`. The generated figures are saved in the `results/figures` folder.
	4) For the simulation analyses, the code files `minority_group.py` and `concept_shift.py` in the `scripts/simulations` can be run independently of each other. The corresponding models are stored together with tables for the figures in the `results/models` folder. After running these files, run `minority_group_figures.py` and `concept_shift_figures.py` to generate the corresponding figures, which are then stored in `results/figures`.


## References

Agrawal, A., R. Verschueren, S. Diamond, and S. Boyd (2018). A rewriting system for convex optimization problems. J Control Decis 5 (1), 42–60.

Diamond, S. and S. Boyd (2016). CVXPY: A Python-embedded modeling language for convex optimization. J Mach Learn Res 17 (83), 1–5.

Jäger, L. (2025). Guided distributionally robust optimization for early outcome prediction in the intensive care unit. Master thesis in statistics. Seminar for Statistics (SfS), Department of Mathematics (D-MATH), ETH Zurich.

Microsoft (2024). LightGBM Documentation, Release 4.6.0.99. Microsoft.
