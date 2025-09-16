"""
    author: jaegerl
    created: 2025-07-10
    description: Functions for loading and preprocessing ICU data.
"""

# -----------------------
# Imports
# -----------------------

import polars as pl
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

# -----------------------
# Variable Definitions
# -----------------------

# Top variables according to fig 8a of Lyu et al 2024: An empirical study on
# KDIGO-defined acute kidney injury prediction in the intensive care unit.
KIDNEY_VARIABLES = [
    "time_hours",  # Time in hours since ICU admission
    "ufilt",  # Ultrafiltration on cont. RRT
    "ufilt_ind",  # Ultrafiltration on cont. RRT
    "rel_urine_rate",  # Urine rate per weight (ml/kg/h)
    "weight",
    "crea",  # Creatinine
    "etco2",  # End-tidal CO2
    "crp",  # C-reactive protein
    "anti_coag_ind",  # Indicator for antocoagulants treatment
    "hep",  # Heparin
    "hep_ind",  # Heparin
    "loop_diur",  # Loop diuretics
    "loop_diur_ind",  # Loop diuretics
    "resp",  # Respiratory rate
    "fluid_ind",  # Fluids
    "airway",  # Ventilation type
    "vent_ind",  # Indicator for any ventilation
    "bili",  # Bilirubin
    "anti_delir_ind",  # Indicator for antidelirium treatment
    "mg",  # Magnesium
    "op_pain_ind",  # Opioid pain medication
    "abx_ind",  # Antibiotics indicator
    "k",  # Potassium
]

# "preliminary selected variables" according to
# https://www.medrxiv.org/content/10.1101/2024.01.23.24301516v1 supp table 3
RESP_VARIABLES = [
    "fio2",
    "norepi",  # Norepinephrine
    "norepi_ind",  # Norepinephrine
    "dobu",  # Dobutamine
    "dobu_ind",  # Dobutamine
    "loop_diur",  # Loop diuretics
    "loop_diur_ind",  # Loop diuretics
    "benzdia",  # Benzodiazepines
    "benzdia_ind",  # Benzodiazepines
    "prop",  # Propofol
    "prop_ind",  # Propofol
    "ins_ind",  # Insulin
    "hep",  # Heparin
    "hep_ind",  # Heparin
    "cf_treat_ind",  # circulatory failure treatments incl. dobu, norepi.
    "sed_ind",  # sedation medication indicator incl. benzdia, prop.
    "age",
    # no emergency admission
    "vent_ind",  # Indicator for any ventilation
    "airway",  # Ventilation type
    "pco2",  # Partial pressure of carbon dioxide PaCO2
    "po2",  # Partial pressure of oxygen PaO2
    "sao2",  # Oxygen saturation (lab value) SaO2
    "spo2",  # Oxygen saturation (finger) SpO2
    "ps",  # Pressure support
    # No MV exp / MV spont. These are available in HiRID only
    "resp",  # Respiratory rate
    "supp_o2_vent",  # Oxygen supplementation
    "tgcs",  # Total Glasgow Coma Scale (Response)
    "mgcs",  # Motor Glasgow Coma Scale
    "peep",  # Positive end-expiratory pressure
    "map",  # Mean arterial pressure. ABPm is window-mean of map
    "peak",  # Peak airway pressure
    "ph",  # Used to determine po2 from sao2 according to the serveringhaus equation
    "temp",  # Temperature, used to determine po2 from sao2 according to serveringhaus
    "pf_ratio",  # ratio of po2 to fio2
]

# Top 20 variables of Hyland et al.: Early prediction of circulatory failure in the
# intensive care unit using machine learning. Table 1.
CIRC_VARIABLES = [
    "lact",  # Lactate
    "map",  # mean arterial pressure
    "time_hours",  # Time in hours since ICU admission
    "age",
    "hr",  # Heart rate
    "dobu",  # Dobutamine
    "dobu_ind",  # Dobutamine
    "milrin",  # Milrinone
    "milrin_ind",  # Milrinone
    "levo",  # Levosimendan
    "levo_ind",  # Levosimendan
    "teophyllin",  # Theophylline
    "teophyllin_ind",  # Theophylline
    "cf_treat_ind",  # circ. failure treatments incl. dobu, norepi, milrin, theo, levo
    "cout",  # Cardiac output
    "rass",  # Richmond Agitation Sedation Scale
    "inr_pt",  # Prothrombin
    "glu",  # Serum glucose
    "crp",  # C-reactive protein
    "dbp",  # Diastolic blood pressure
    "sbp",  # Systolic blood pressure
    "peak",  # Peak airway pressure
    "spo2",  # Oxygen saturation (finger) SpO2
    "nonop_pain_ind",  # Non-opioid pain medication
    "supp_o2_vent",  # Oxygen supplementation
]

# Variables used to determine apache II
APACHE_II_VARIABLES = [
    "age",
    "map",
    "crea",
    "fio2",
    "hct",
    "hr",
    "k",
    "na",
    "pco2",
    "po2",
    "resp",
    "temp",
    "tgcs",
    "wbc",
    "ph",
]

OUTCOME_FEATURES = {
    "circulatory": CIRC_VARIABLES,
    "respiratory": RESP_VARIABLES,
    "renal": KIDNEY_VARIABLES,
    "mortality": APACHE_II_VARIABLES,
    "lactate": CIRC_VARIABLES,
    "creatinine": KIDNEY_VARIABLES,
    }

OUTCOME_VARIABLES = {
    "respiratory": "respiratory_failure_at_24h",
    "circulatory": "circulatory_failure_at_8h",
    "renal": "kidney_failure_at_48h",
    "mortality": "mortality_at_24h",
    "lactate": "log_lactate_in_4h",
    "creatinine": "log_creatinine_in_24h",
    }

OUTCOME_HORIZONS = {
    "circulatory": 8,
    "respiratory": 24,
    "renal": 24,
    "mortality": 24,
    "lactate": 8,
    "creatinine": 24,
    }

CONTINUOUS_FEATURES = [
    "mean",
    "sq_mean",
    "std",
    "slope",
    "fraction_nonnull",
    "all_missing",
    "min",
    "max",
    ]
CATEGORICAL_FEATURES = ["mode", "num_nonmissing"]
TREATMENT_INDICATOR_FEATURES = ["num", "any"]
TREATMENT_CONTINUOUS_FEATURES = ["rate"]
CAT_MISSING_NAME = "(MISSING)"
CODEBOOK_PATH =  "/Users/jaegerl/Documents/awesome_stuff/statistics_msc/fs25_master_thesis/analysis/02_data/02_data_input/variables.tsv"


# -----------------------
# Dataset Loading and Preprocessing
# -----------------------

def get_features(
    outcome: str = "respiratory",
    codebook_path: str = CODEBOOK_PATH,
    ):
    """
    Get the features for the specified outcome from the codebook.
    
    Parameters
    ----------
    outcome : str
        The type of outcome to be predicted. Must be either of `"circulatory"`, "`respiratory"`, `"renal"`, `"mortality"`, `"lactate"`, or `"creatinine"`. Defaults to `"respiratory"`.
    codebook_path : str
        The path to the codebook file. Defaults to `CODEBOOK_PATH`.
    apacheii : bool, optional
        If `True`, the Apache II score at admission will be included in the features. Defaults to `False`.
        
    Returns
    -------
    features : list[str]
        A list of feature names for the specified outcome.
    """
    
    # load the codebook
    codebook = pl.read_csv(
        codebook_path, separator="\t", null_values=["None"]
    )
    # filter the appropriate variables for the outcome of interest
    variables = OUTCOME_FEATURES[outcome]
    horizon = OUTCOME_HORIZONS[outcome]
    
    codebook_outcome = codebook.filter(
        pl.col("VariableTag").is_in(variables)
    )
    # initialize list of features
    features = []    
    for row in codebook_outcome.rows(named=True):
        variable = row["VariableTag"]

        if row["LogTransform"] is True:
            variable = f"log_{variable}"

        if row["VariableType"] == "static":
            features += [variable]
        elif row["DataType"] == "continuous":
            features += [  # These variables are not based on any horizon.
                f"{variable}_ffilled",
                f"{variable}_missing",
                f"{variable}_sq_ffilled",
            ]
            features += [ # These are variables based on horizons.
                f"{variable}_{feature}_h{horizon}"
                for feature in CONTINUOUS_FEATURES
            ]
        elif row["DataType"] == "categorical":
            features += [variable]
            features += [
                f"{variable}_{feature}_h{horizon}"
                for feature in CATEGORICAL_FEATURES
            ]
        elif row["DataType"] == "treatment_ind":
            features += [variable]
            features += [
                f"{variable}_{feature}_h{horizon}"
                for feature in TREATMENT_INDICATOR_FEATURES
            ]
        elif row["DataType"] == "treatment_cont":
            features += [
                f"{variable}_{feature}_h{horizon}"
                for feature in TREATMENT_CONTINUOUS_FEATURES
            ]
        else:
            raise ValueError(f"Unknown DataType: {row['DataType']}")
    
    return features


def load_icudata(
    datadir: str = "/Users/jaegerl/Documents/awesome_stuff/statistics_msc/fs25_master_thesis/analysis/02_data/02_data_input",
    dataset: str = "eicu",
    outcome: str = "respiratory",
    subset: list[str] = ["train"],
    other_columns: list[str] | None = None,
    nonempty_columns: list[str] | None = None,
    codebook_path: str = CODEBOOK_PATH,
    apacheii: bool = False,
    ):
    """
    Load ICU dataset from Parquet file with the appropriate features, extract the appropriate subset, and return the features and labels.
    
    Parameters
    ----------
    datadir : str
        The directory containing the Parquet file of the desired dataset.
    dataset : str, optional
        The name of the dataset. Must be either of `"eicu"`, `"hirid"`, `"miiv"`, `"mimic"`, `"nwicu"`, `"picdb"`, `"sic"`, `"zigong"`.  Defaults to `"eicu"`.
    outcome : str, optional
        The type of outcome to be predicted. Must be either of `"circulatory"`, "`respiratory"`, `"renal"`, `"mortality"`, `"lactate"`, or `"creatinine"`. Defaults to `"respiratory"`.
    subset : str, optional
        The subset of the dataset to be loaded. Must be either of `"train"`, `"val"`, `"test"`. Defaults to `"train"`.
    other_columns : list[str], optional
        Additional columns to include in the returned DataFrame. Defaults to `None`.
    nonempty_columns : list[str], optional
        If specified, all row with `None` in any of these columns will be dropped. Defaults to `None`.
    codebook_path : str, optional
        The path to the codebook file. Defaults to `CODEBOOK_PATH`.
    apacheii : bool, optional
        If `True`, the APACHE II score will be included in the features. Defaults to `False`.
    
    Returns
    -------
    X : pl.DataFrame
        The features of the dataset, including the specified outcome variable.
    y : np.ndarray
        The labels of the dataset.
    """
    
    # get the appropriate outcome variable
    outcome_var = OUTCOME_VARIABLES[outcome]
    # get the features
    features = get_features(
        outcome=outcome, 
        codebook_path=codebook_path,
        )
    # load the dataset
    if other_columns is None:
        other_columns = []
    loaded_columns = features + other_columns + [outcome_var]
    if apacheii:
        apacheii_columns = [var + "_min_h24" for var in APACHE_II_VARIABLES if var != "age"] + [var + "_max_h24" for var in APACHE_II_VARIABLES if var != "age"]
        apacheii_columns = [
            "log_" + col if col in ["po2_min_h24", "po2_max_h24", "crea_min_h24", "crea_max_h24", "wbc_min_h24", "wbc_max_h24", "resp_min_h24", "resp_max_h24"]
            else col for col in apacheii_columns
        ]
        loaded_columns += apacheii_columns
    loaded_columns = list(set(loaded_columns))  # remove duplicates
    df = (
        pl.scan_parquet(f"{datadir}/{dataset}.parquet")
        .filter(pl.col("split").is_in(subset) & pl.col(outcome_var).is_not_null())
        .select(loaded_columns)
        .collect()
        )
    # remove all rows where any of the columns in nonempty_columns is None
    if nonempty_columns is not None:
        nonempty_columns = list(set(nonempty_columns))
        df = df.drop_nulls(subset=nonempty_columns)
    
    # if required, calculate the APACHE II score
    if apacheii:
        # replace missing values of fio2 with 21% (ambient air)
        df = df.with_columns([
            pl.col("fio2_min_h24").fill_null(21),
            pl.col("fio2_max_h24").fill_null(21),
        ])
        # calculate the alveolar-arterial gradient
        # formula on https://en.wikipedia.org/wiki/Alveolar%E2%80%93arterial_gradient
        df = df.with_columns(
            (np.exp(pl.col("log_po2_min_h24"))).alias("po2_min_h24"),
            (np.exp(pl.col("log_po2_max_h24"))).alias("po2_max_h24"),
        )
        df = df.with_columns(
            (pl.col("fio2_max_h24") * 7.13 - pl.col("pco2_min_h24") - np.exp(pl.col("po2_min_h24"))).alias("aao2_max_h24"),
        )
        # replace missing values of GCS with 15
        df = df.with_columns([
            pl.col("tgcs_min_h24").fill_null(15),
            pl.col("tgcs_max_h24").fill_null(15),
            ])
        # calculate the individual APACHE II subscores
        df = df.with_columns([
            # temperature
            pl.when(pl.col("temp_min_h24").is_null() | pl.col("temp_max_h24").is_null()).then(0)
            .when(pl.col("temp_min_h24") <= 29.9).then(4)
            .when(pl.col("temp_min_h24") < 32).then(3)
            .when(pl.col("temp_min_h24") < 34).then(2)
            .when(pl.col("temp_min_h24") < 36).then(1)
            .when(pl.col("temp_max_h24") < 38.5).then(0)
            .when(pl.col("temp_max_h24") < 39.0).then(1)
            .when(pl.col("temp_max_h24") < 41.0).then(3)
            .otherwise(4)
            .alias("score_temp"),
            # mean arterial pressure
            pl.when(pl.col("map_min_h24").is_null() | pl.col("map_max_h24").is_null()).then(0)
            .when(pl.col("map_min_h24") <= 49).then(4)
            .when(pl.col("map_min_h24") < 70).then(2)
            .when(pl.col("map_max_h24") < 109).then(0)
            .when(pl.col("map_max_h24") < 129).then(2)
            .when(pl.col("map_max_h24") < 159).then(3)
            .otherwise(4)
            .alias("score_map"),
            # heart rate
            pl.when(pl.col("hr_min_h24").is_null() | pl.col("hr_max_h24").is_null()).then(0)
            .when(pl.col("hr_min_h24") <= 39).then(4)
            .when(pl.col("hr_min_h24") < 55).then(3)
            .when(pl.col("hr_min_h24") < 70).then(2)
            .when(pl.col("hr_max_h24") < 110).then(0)
            .when(pl.col("hr_max_h24") < 140).then(2)
            .when(pl.col("hr_max_h24") < 180).then(3)
            .otherwise(4)
            .alias("score_hr"),
            # respiratory rate
            pl.when(pl.col("log_resp_min_h24").is_null() | pl.col("log_resp_max_h24").is_null()).then(0)
            .when(pl.col("log_resp_min_h24") <= np.log(5)).then(4)
            .when(pl.col("log_resp_min_h24") < np.log(10)).then(2)
            .when(pl.col("log_resp_min_h24") < np.log(12)).then(1)
            .when(pl.col("log_resp_max_h24") < np.log(25)).then(0)
            .when(pl.col("log_resp_max_h24") < np.log(35)).then(1)
            .when(pl.col("log_resp_max_h24") < np.log(50)).then(3)
            .otherwise(4)
            .alias("score_resp"), 
            # oxygenation: if fio2_max_h24 is >= 0.5, then we use the alveolar-arterial gradient calculated above. if fio2_max_h24 is < 0.5, we use the pao2_min_24h.
            pl.when((pl.col("fio2_max_h24") >= 50) & (pl.col("aao2_max_h24") < 200)).then(0)
            .when((pl.col("fio2_max_h24") >= 50) & (pl.col("aao2_max_h24") < 350)).then(2)
            .when((pl.col("fio2_max_h24") >= 50) & (pl.col("aao2_max_h24") < 500)).then(3)
            .when((pl.col("fio2_max_h24") >= 50) & (pl.col("aao2_max_h24") >= 500)).then(4)
            .when((pl.col("fio2_max_h24") < 50) & (pl.col("po2_min_h24") < 55)).then(4)
            .when((pl.col("fio2_max_h24") < 50) & (pl.col("po2_min_h24") < 60)).then(3)
            .when((pl.col("fio2_max_h24") < 50) & (pl.col("po2_min_h24") < 70)).then(1)
            .when((pl.col("fio2_max_h24") < 50) & (pl.col("po2_min_h24") >= 70)).then(0)
            .otherwise(0)
            .alias("score_fio2"),
            # arterial pH
            pl.when(pl.col("ph_min_h24").is_null() | pl.col("ph_max_h24").is_null()).then(0)
            .when(pl.col("ph_min_h24") < 7.15).then(4)
            .when(pl.col("ph_min_h24") < 7.25).then(3)
            .when(pl.col("ph_min_h24") < 7.33).then(2)
            .when(pl.col("ph_max_h24") < 7.50).then(0)
            .when(pl.col("ph_max_h24") < 7.60).then(1)
            .when(pl.col("ph_max_h24") < 7.70).then(3)
            .otherwise(4)
            .alias("score_ph"),
            # serum sodium
            pl.when(pl.col("na_min_h24").is_null() | pl.col("na_max_h24").is_null()).then(0)
            .when(pl.col("na_min_h24") <= 110).then(4)
            .when(pl.col("na_min_h24") < 120).then(3)
            .when(pl.col("na_min_h24") < 130).then(2)
            .when(pl.col("na_max_h24") < 150).then(0)
            .when(pl.col("na_max_h24") < 155).then(1)
            .when(pl.col("na_max_h24") < 160).then(2)
            .when(pl.col("na_max_h24") < 180).then(3)                
            .otherwise(4)
            .alias("score_na"),
            # serum potassium
            pl.when(pl.col("k_min_h24").is_null() | pl.col("k_max_h24").is_null()).then(0)
            .when(pl.col("k_min_h24") < 2.5).then(4)
            .when(pl.col("k_min_h24") < 3).then(2)
            .when(pl.col("k_min_h24") < 3.5).then(1)
            .when(pl.col("k_max_h24") < 5.5).then(0)
            .when(pl.col("k_max_h24") < 6.0).then(1)
            .when(pl.col("k_max_h24") < 7.0).then(3)
            .otherwise(4)
            .alias("score_k"),
            # serum creatinine
            pl.when(pl.col("log_crea_min_h24").is_null() | pl.col("log_crea_max_h24").is_null()).then(0)
            .when(pl.col("log_crea_min_h24") + np.log(88.42) < np.log(53)).then(2)
            .when(pl.col("log_crea_max_h24") + np.log(88.42) < np.log(134)).then(0)
            .when(pl.col("log_crea_max_h24") + np.log(88.42) < np.log(177)).then(2)
            .when(pl.col("log_crea_max_h24") + np.log(88.42) < np.log(309)).then(3)
            .otherwise(4)
            .alias("score_crea"),
            # GCS
            (15 - pl.col("tgcs_min_h24"))
            .alias("score_tgcs"),
            # hematocrit
            pl.when(pl.col("hct_min_h24").is_null() | pl.col("hct_max_h24").is_null()).then(0)
            .when(pl.col("hct_min_h24") < 20).then(4)
            .when(pl.col("hct_min_h24") < 30).then(2)
            .when(pl.col("hct_max_h24") < 46).then(0)
            .when(pl.col("hct_max_h24") < 50).then(1)
            .when(pl.col("hct_max_h24") < 60).then(2)
            .otherwise(4)
            .alias("score_hct"),
            # white blood cell count
            pl.when(pl.col("log_wbc_min_h24").is_null() | pl.col("log_wbc_max_h24").is_null()).then(0)
            .when(pl.col("log_wbc_min_h24") < np.log(1.0)).then(4)
            .when(pl.col("log_wbc_min_h24") < np.log(3.0)).then(2)
            .when(pl.col("log_wbc_max_h24") < np.log(15.0)).then(0)
            .when(pl.col("log_wbc_max_h24") < np.log(20.0)).then(1)
            .when(pl.col("log_wbc_max_h24") < np.log(40.0)).then(2)
            .otherwise(4)
            .alias("score_wbc"),
            # age
            pl.when(pl.col("age").is_null()).then(0)
            .when(pl.col("age") <= 44).then(0)
            .when(pl.col("age") < 55).then(2)
            .when(pl.col("age") < 65).then(3)
            .when(pl.col("age") < 75).then(5)
            .otherwise(6)
            .alias("score_age"),
            ])
        # replace empty values with 0 (there shouldn't be any, but still)
        df = df.with_columns(
            pl.col("score_temp").fill_null(0),
            pl.col("score_map").fill_null(0),
            pl.col("score_hr").fill_null(0),
            pl.col("score_resp").fill_null(0),
            pl.col("score_fio2").fill_null(0),
            pl.col("score_ph").fill_null(0),
            pl.col("score_na").fill_null(0),
            pl.col("score_k").fill_null(0),
            pl.col("score_crea").fill_null(0),
            pl.col("score_tgcs").fill_null(0),
            pl.col("score_hct").fill_null(0),
            pl.col("score_wbc").fill_null(0),
            pl.col("score_age").fill_null(0),
        )
        # calculate the APACHE II score as the sum of these subscores
        df = df.with_columns([
            (pl.sum_horizontal([
                pl.col("score_temp"),
                pl.col("score_map"),
                pl.col("score_hr"),
                pl.col("score_resp"),
                pl.col("score_fio2"),
                pl.col("score_ph"),
                pl.col("score_na"),
                pl.col("score_k"),
                pl.col("score_crea"),
                pl.col("score_tgcs"),
                pl.col("score_hct"),
                pl.col("score_wbc"),
                pl.col("score_age"),
            ])).alias("apache_ii_calculated")
        ])
        # remove all unnecessary columns
        columns_apacheii = features + other_columns + [outcome_var] + ["apache_ii_calculated"]
        columns_apacheii = list(set(columns_apacheii))
        df = df.select([pl.col(col) for col in columns_apacheii])
    
    # extract outcome variable, convert it to 1-dimensional array (otherwise the loss calculation will crash)
    y = df.select(outcome_var).to_numpy().flatten()
    # if y is Boolean, convert it to 0/1
    if y.dtype == "bool":
        y = y.astype(int)
    # remove the outcome variable from the DataFrame if not mentioned in other_columns
    if outcome_var not in other_columns:
        df = df.drop(outcome_var)
    
    # extract APACHE II score if it was calculated
    if apacheii:
        apacheii_calculated = df.select("apache_ii_calculated").to_numpy().flatten()
        df = df.drop("apache_ii_calculated")
    
    # preprocess the features
    continuous_variables = [
        col
        for col, dtype in df.schema.items()
        if dtype.is_float() or dtype.is_integer()
    ]
    other = [c for c in df.columns if c not in continuous_variables]
    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", "passthrough", continuous_variables),
            ("categorical",
             OrdinalEncoder(
                 handle_unknown="use_encoded_value",
                 # LGBM from pyarrow allows only int, bool, float types. So we
                 # have to transform `airway` from str to int. Unknown value must
                 # be an int. 99 works since we should never have so many
                 # categories.
                 unknown_value=99,
                 ),
                 other,
                ),
            ]
        ).set_output(transform="polars")
    df = preprocessor.fit_transform(df)
    
    # put together the output data
    output_data = {
        "X": df,
        "y": y,
        "dataset": dataset,
        "outcome": outcome_var,
        }
    if apacheii:
        output_data["apache_ii"] = apacheii_calculated
    
    return output_data


# -----------------------
# Plotting
# -----------------------

