import os
import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / ""))
import gdro.deploy as gdrodeploy


# define hyperparameters for this run
outcome = "mortality"
objective = "binary"
source = "miiv"
target = "zigong"
n_rounds = 1000

# define setting variables
codebook_path = REPO_ROOT / "data" / "variables.tsv"
plotdir = REPO_ROOT / "results" / "figures" / "icu" / "models_inspect" / f"{source}to{target}_{outcome}/"
modeldir =  REPO_ROOT / "results" / "models" / f"/{source}to{target}_{outcome}/"
tabledir = REPO_ROOT / "results" / "tables" / "icu" / f"{source}to{target}_{outcome}/"
# IMPORTANT: change username!
datadir = os.environ.get("TMPDIR", "/cluster/work/math/username/data")

# define grid for parallel execution
rhos = [1e-12, 0.01, 0.1]
ks = [1, 2]
guides_list = {
    "Simple DRO": {
        "quantile": [],
        "avg": [],
        "avg_cutoff": [],
        "avg_by_cutoff": [],
        "avg_by_group": [],
        },
    "Demographic, marginal": {
        "quantile": ["age"],
        "avg": ["sex", "mortality_at_24h"],
        "avg_cutoff": [],
        "avg_by_cutoff": [],
        "avg_by_group": [],
        },
    }

# Cartesian product of hyperparameters to generate parallel arrays
grid = list(itertools.product(ks, rhos))

# get SLURM task ID
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
k, rho = grid[task_id]

deploy_params = {
    "outcome": outcome,
    "n_rounds": n_rounds,
    "rho": rho,
    "k": k,
    "source": source,
    "target": target,
    "lgbm_params": {
        "objective": objective,
        "learning_rate": 0.05,
        "max_depth": 3,
        "verbose": -1,
        },
    "other_columns": ["mortality_at_24h", "sex", "age"],
    }

epsilon_list = []
for name, guide in guides_list.items():
    print(f"Finding epsilon for {name}...") 
    epsilon = gdrodeploy.find_epsilon(
        **deploy_params,
        datadir=datadir,
        guides=guide,
        epsilonmax=1,
        codebook_path=codebook_path,
    )
    epsilon_list.append(epsilon)

# fit the models
gdromodels_list = [
    gdrodeploy.fitgdroICU(
        **deploy_params,
        guides=guide,
        epsilon=epsilon,
        codebook_path=codebook_path,
        datadir=datadir,
        ) for guide, epsilon in zip(guides_list.values(), epsilon_list)
    ]

# export figures and tables
gdrodeploy.plotgdroICU(
    gdromodels=gdromodels_list,
    guidenames=list(guides_list.keys()),
    plotname="Demographics",
    nonempty_columns=["mortality_at_24h", "sex", "age"],
    codebook_path=codebook_path,
    modeldir=modeldir,
    plotdir=plotdir,
    datadir=datadir,
    )

# evaluate model overall and in subgroups
gdrodeploy.plotgdroICUsubgroup(
    gdromodels=gdromodels_list,
    guidenames=list(guides_list.keys()),
    subgroupname="Elderly (age " + r"$\geq 75$" + " years)",
    subgroup=("age", 75, True),
    n_rounds=n_rounds,
    plotdir=plotdir,
    datadir=datadir,
    tabledir=tabledir,
    codebook_path=codebook_path,
)