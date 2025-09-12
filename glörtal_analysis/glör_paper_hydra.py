"""
This script performs linear modeling and forecasting for ps time series data using in-situ measurements:
A number of models that include and exclude various terms are optimized and the best scoring models are saved to the results folder.
- Data loading and preprocessing from multiple Excel sources.
- Construction of exogenous variables with lagged and interaction terms.
- Linear regression modeling with autoregressive components.
- Model selection based on mean absolute residuals.
- Saving fitted models and intermediate results for further analysis.
Key functions:
- select_daily_exog: Constructs lagged exogenous variables for modeling.
- prep_modeling_linear: Prepares the design matrix for linear regression, including autoregressive and trend terms.
- linear_pipeline: Fits a linear regression model to the target variable using specified exogenous variables and options for interaction and differencing.
- find_aic_model: Searches for the best model configuration based on residuals and retrains the optimal model.
- load_raw: Loads and merges raw data from Excel files, interpolates missing values, and prepares in-situ and satellite datasets.
- main: Orchestrates the modeling workflow using Hydra for configuration management, fits models for different periods, and saves results.
Configuration:
- Uses Hydra and OmegaConf for flexible experiment configuration via YAML files.
Usage:
- Run as a standalone script. Requires configuration YAML and appropriate data files in the specified directories.
"""
import warnings
from datetime import datetime
import pandas as pd
import pickle
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
import statsmodels.api as sm
import os
from functools import reduce
import numpy as np
import pickle
import hydra
from omegaconf import DictConfig
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", ValueWarning)
warnings.simplefilter("ignore", ConvergenceWarning)




def select_daily_exog(
    d,
    external_timesteps,
    timestamps,
    step_factor=1,
    time_name="date",
    variable_name="W",
    replacement="_",
):
    stack = []
    for step in external_timesteps:
        move = d.copy()
        move["date"] = move[time_name].shift(-1 * step_factor * step)
        if time_name != "date":
            move.drop(columns=time_name, inplace=True)
        move = move[move["date"].isin(timestamps)]
        move = move[["date", variable_name]]
        move.rename(
            {variable_name: variable_name + replacement + str(step)},
            axis=1,
            inplace=True,
        )
        stack.append(move)
    exog = reduce(
        lambda left, right: pd.merge(left, right, on=["date"], how="outer"), stack
    )
    return exog


def prep_modeling_linear(endog, exog, autoregressive):
    temporary = exog.copy()
    # add autoregressive components as additional variables
    if autoregressive > 0:
        for value in np.arange(1, autoregressive + 1):
            temporary[str(value)] = endog.shift(value).values
    temporary["const"] = 1.0
    temporary["trend"] = np.arange(0, 1, 1 / len(temporary["const"]))
    return temporary


def linear_pipeline(
    target,
    in_situ,
    cols=["T"],
    past_days=10,
    autoregressive=1,
    interaction=None,
    difference= False,
    save_table=False,
):
    stack = []
    for col in cols:
        stack.append(
            select_daily_exog(
                in_situ.reset_index(),
                np.arange(past_days + 1),
                target.index,
                variable_name=col,
            ).set_index("date")
        )

    if interaction != -1:
        W = select_daily_exog(
            in_situ.reset_index(),
            np.arange(interaction + 1),
            target.index,
            variable_name="W",
        ).set_index("date")
        T = select_daily_exog(
            in_situ.reset_index(),
            np.arange(interaction + 1),
            target.index,
            variable_name="T",
        ).set_index("date")
        W = (W - W.min()) / (W.max() - W.min())
        T = (T - T.min()) / (T.max() - T.min())
        W[W.columns] = W.values * T.values
        W.columns = W.columns.str.replace("W_", "inter_")
        stack.append(W)

    if difference:
        W = select_daily_exog(
            in_situ.reset_index(),
            np.arange(2),
            target.index,
            variable_name="W",
        ).set_index("date")
        T = select_daily_exog(
            in_situ.reset_index(),
            np.arange(2),
            target.index,
            variable_name="T",
        ).set_index("date")

        W["W_0"] = W["W_1"] - W["W_0"]
        T["T_0"] = T["T_1"] - T["T_0"]
        W.columns = W.columns.str.replace("W_0", "W_diff")
        T.columns = T.columns.str.replace("T_0", "T_diff")
        stack.append(W[["W_diff"]])
        stack.append(T[["T_diff"]])

    exog = pd.concat(stack, axis=1)
    assert exog.isnull().sum().max() < 2, "Nans in exogenous variables."
    exog = (exog - exog.min()) / (exog.max() - exog.min())
    target = (target - target.min()) / (target.max() - target.min())
    exog.fillna(value=exog.mean(), inplace=True)
    temporary = prep_modeling_linear(target, exog, autoregressive=autoregressive)
    if save_table: # TODO this overwrites previous tables, should be fixed if needed.
        if not os.path.exists("data_saves/"):
            os.makedirs("data_saves/")
        temporary.to_csv("data_saves/temporary" + save_table + ".csv")
        target.to_csv("data_saves/target" + save_table + ".csv")
    olsmod = sm.OLS(
        exog=temporary.fillna(method="backfill"),
        endog=target,
    )
    olsres = olsmod.fit()
    return olsres


def find_aic_model(target, in_situ, orders, cols, past_days, interaction,difference=False, naming=None):
    full_stack = []
    for order in orders:
        stack = []
        if interaction != -1:
            for x in range(0, past_days):
                inter_stack = []
                for y in range(0, interaction):
                    m = linear_pipeline(
                        target,
                        in_situ,
                        past_days=x,
                        autoregressive=order,
                        cols=cols,
                        difference=difference,
                        interaction=y,
                    )
                    inter_stack.append(m.resid.abs().mean())
                stack.append(inter_stack)
            full_stack.append(stack)

        else:
            for x in range(0, past_days):
                m = linear_pipeline(
                    target,
                    in_situ,
                    past_days=x,
                    autoregressive=order,
                    cols=cols,
                    difference=difference,
                    interaction=-1,
                    save_table=False,
                )
                stack.append(m.resid.abs().mean())
            full_stack.append(stack)
            
    print("Number of models evaluated: ", len(np.array(full_stack).flatten()))

    best = np.where(np.array(full_stack) == np.array(full_stack).min())
    # retrain best
    m = linear_pipeline(
        target,
        in_situ,
        past_days=best[1][0],
        autoregressive=orders[int(best[0][0])],
        cols=cols,
        difference=difference,
        interaction=best[2][0] if interaction != -1 else -1,
        save_table=naming
    )

    return m


# Forecasting the left out year for now. (Other option can be added in the future.)

def load_raw(which_ps):
    a = pd.read_excel(
        "new_data/alleStauanlagen_temperatur_mitGloerEnnepe.xlsx", header=1
    )
    b = pd.read_excel("new_data/Gloer_Stau(3).xlsx")

    if which_ps == "t+s":
        c = pd.read_excel("new_data/Gloer_TSXS1_200507_Seg2_RAW_RAD_mm.xlsx")
    elif which_ps == "tsx":
        c = pd.read_excel("new_data/Gloer_TSX_200507_Seg2_RAW_RAD_mm.xlsx")
    elif which_ps == "S1":
        c = pd.read_excel("new_data/Gloer_S1_200507_Seg2_RAW_RAD_mm.xlsx")
    else:
        raise ValueError("Unknown PS type. Use 'merge', 'tsx' or 'S1'.")

    b = b[["Datum/Uhrzeit", "Wasserstand (NHN) [m NHN]"]]
    b["Datum/Uhrzeit"] = b["Datum/Uhrzeit"].dt.round("D")
    a = a[["Datum:", "Glör L2:"]]
    c = c[["date", "deform"]]
    a.columns = ["date", "T"]
    b.columns = ["date", "W"]
    a["date"] = a["date"].round("D")
    b["date"] = b["date"].round("D")
    a = a[a["date"].dt.year > 2016]
    b = b[b["date"].dt.year > 2016]
    a = a[a["date"].dt.year < 2023]
    b = b[b["date"].dt.year < 2023]
    # 2 duplicated dates.
    b.drop_duplicates(subset=["date"], inplace=True)
    in_situ = a.merge(b, on="date", how="outer")
    in_situ.set_index("date", inplace=True)

    ps = c
    ps.set_index("date", inplace=True)
    in_situ["W"].interpolate(method="linear", inplace=True)

    return ps, in_situ


@hydra.main(version_base=None, config_path="config", config_name="reg.yaml")
def main(cfg: DictConfig):
    print(cfg)
    start = datetime.now()
    print("Starting at ", start)
    ps, in_situ = load_raw(which_ps=cfg.which_ps)

    # periods:
    start = in_situ[in_situ["W"] == 284.04].index.min()
    end = "2019-01-14"
    refill_with_full_data_start = "2019-09-09"
    fill_end = "2021-02-08"

    drain = ps[(ps.index > start) & (ps.index < end)]
    refill = ps[(ps.index > refill_with_full_data_start) & (ps.index < fill_end)]
    full = ps[(ps.index > fill_end)]

    if not os.path.exists("results/" + cfg.which_ps):
        os.makedirs("results/" + cfg.which_ps)

    m = find_aic_model(
        drain, in_situ, cfg.orders, cols=["T"], past_days=cfg.past_days, interaction=-1, naming="drain"
    )
    pickle.dump(m, open("results/" + cfg.which_ps + "/drain.p", "wb"))
    print(np.abs(m.resid).mean())


    m = find_aic_model(
        refill, in_situ, cfg.orders, cols=["T"], past_days=cfg.past_days, interaction=-1, naming="refill_T"
    )
    pickle.dump(m, open("results/" + cfg.which_ps + "/refill_T.p", "wb"))
    print(np.abs(m.resid).mean())

    m = find_aic_model(
        refill, in_situ, cfg.orders, cols=["W"], past_days=cfg.past_days, interaction=-1, naming="refill_W"
    )
    pickle.dump(m, open("results/" + cfg.which_ps + "/refill_W.p", "wb"))
    print(np.abs(m.resid).mean())

    m = find_aic_model(
        full, in_situ, cfg.orders, cols=["T"], past_days=cfg.past_days, interaction=-1, naming="full_T"
    )
    pickle.dump(m, open("results/" + cfg.which_ps + "/full_T.p", "wb"))
    print(np.abs(m.resid).mean())

    m = find_aic_model(
        full, in_situ, cfg.orders, cols=["W"], past_days=cfg.past_days, interaction=-1, naming="full_W"
    )
    pickle.dump(m, open("results/" + cfg.which_ps + "/full_W.p", "wb"))
    print(np.abs(m.resid).mean())

    m = find_aic_model(
        refill,
        in_situ,
        cfg.orders,
        cols=["T", "W"],
        past_days=cfg.past_days,
        interaction=-1,
        naming="refill"
    )
    pickle.dump(m, open("results/" + cfg.which_ps + "/refill.p", "wb"))
    print(np.abs(m.resid).mean())


    m = find_aic_model(
        full,
        in_situ,
        cfg.orders,
        cols=["T", "W"],
        past_days=cfg.past_days,
        interaction=-1,
        naming="full"       
    )
    pickle.dump(m, open("results/" + cfg.which_ps + "/full.p", "wb"))
    print(np.abs(m.resid).mean())

    m = find_aic_model(
        refill,
        in_situ,
        cfg.orders,
        cols=["T", "W"],
        past_days=cfg.past_days,
        interaction=cfg.interaction,
        difference=cfg.difference,
        naming="refill_inter"
    )
    pickle.dump(m, open("results/" + cfg.which_ps + "/refill_inter.p", "wb"))
    print(np.abs(m.resid).mean())
    
    m = find_aic_model(
        full,
        in_situ,
        cfg.orders,
        cols=["T", "W"],
        past_days=cfg.past_days,
        interaction=cfg.interaction,
        difference=cfg.difference,
        naming="full_inter"
    )
    pickle.dump(m, open("results/" + cfg.which_ps + "/full_inter.p", "wb"))
    print(np.abs(m.resid).mean())

    print("Done.")


if __name__ == "__main__":
    main()
