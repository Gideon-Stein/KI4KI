import os
import pickle
import warnings
import pandas as pd
import pickle
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from utils.data_parsing_tools import load_data_bases
from utils.general_tools import results_to_pd
from utils.regression_tools import forecast_with_best

import hydra
from omegaconf import DictConfig

warnings.simplefilter("ignore", ConvergenceWarning)


@hydra.main(version_base=None, config_path="conf", config_name="forecast_config.yaml")
def main(cfg: DictConfig):

    # Load the complete data (So we only need to load the data once)
    data, exogs = load_data_bases(cfg)
    locations = cfg.all_locs if cfg.loc == "X" else [cfg.loc]
    dirs = cfg.all_dirs if cfg.dir == "X" else [cfg.dir]
    model_types = cfg.all_model_types if cfg.model_type == "X" else [cfg.model_type]


    print("EXECUTE: ")
    print((locations, dirs))

    isExist = os.path.exists(cfg.forecast_path)
    if not isExist:
        os.makedirs(cfg.forecast_path)
        print("A new directory is created!")

    # For all combinations of loc dir
    for location in locations:
        for direction in dirs:
            for model_t in model_types:
                # Loading grid result

                # Quick fix so we include the univariate models also in the full search. 
                result_stack = pickle.load(
                    open(
                        cfg.save_path
                        + "/"
                        + location
                        + "_"
                        + direction
                        + "_"
                        + model_t
                        + "_results_stack.p",
                        "rb",
                    )
                )

                print("Process_saves")
                scores = results_to_pd(result_stack)
                print(len(scores))


                # construct the scoring table for all runs.
                for additional in ["univariate","remove_T","remove_W","remove_W_remove_T"]:
                    try:
                        print("Add: " + additional)
                        result_stack = pickle.load(
                            open(
                                cfg.save_path 
                                + "/"
                                + location
                                + "_"
                                + direction
                                + "_"
                                + model_t 
                                + "_"
                                + additional
                                + "_results_stack.p",
                                "rb",
                            )
                        )
                        print("Process_saves")
                        scores2 = results_to_pd(result_stack)
                        print(len(scores2))
                        scores = pd.concat([scores, scores2]).reset_index(drop=True)
                        print(len(scores))
                    except:
                            print("No additional runs found for:" + additional +  " is this correct?")
                print("Filtering:")
                if cfg.search == "0_full_search.yaml" or cfg.search == "4_igarss_search.yaml": 
                    print("No filters applied for full search space.")
                    pass

                else:
                    # Filtering step based on config.
                    scores = scores.loc[scores["M_Stau"].isin(cfg.search.mean_orders)]
                    print(len(scores))
                    scores = scores.loc[scores["M_T"].isin(cfg.search.mean_orders)]
                    print(len(scores))

                    if len(cfg.search.exog_steps) == 0: 
                        scores = scores.loc[[len(x) == 0  for x in scores["STAU"]]]
                        scores = scores.loc[[len(x) == 0 for x in scores["T"]]]
                    else:
                        # kinda hacky to keep empty exog specification
                        for variable in ["STAU", "T"]:
                            select = []
                            for x in scores[variable]:
                                if len(x) == 0:
                                    if cfg.remove_everything:
                                        select.append(False)
                                    else:
                                        select.append(True)
                                else:
                                    if max(x) in cfg.search.exog_steps:
                                        select.append(True)
                                    else:
                                        select.append(False)
                            scores = scores.loc[select]
                    print(len(scores))


                    scores = scores.loc[scores["F"].isin(cfg.search.fourier_orders)]
                    scores = scores.loc[scores["Decompose"].isin(cfg.search.decompose)]
                    if "Interaction" in scores.columns:
                        scores = scores.loc[scores["Interaction"].isin(cfg.search.interaction)]
                    scores = scores.loc[(scores["P"] < (cfg.search.order_high[0])) & (scores["P"] >= (cfg.search.order_low[0]))]
                    print(len(scores))

                    if model_t in ["sarimax", "var"]:
                        scores = scores.loc[(scores["Q"] < cfg.search.order_high[2]) & (scores["Q"] >= cfg.search.order_low[2])]
                    if model_t == "sarimax":
                        scores = scores.loc[(scores["D"] < (cfg.search.order_high[1])) & (scores["D"] >= (cfg.search.order_low[1]))]

                print("Items after applying filter:")
                print(len(scores))

                stack = []
                # for each unique ps point, select the best scoring model.
                for x in scores["Index"].unique():
                    stack.append(
                        scores[scores["Index"] == x]
                        .sort_values(cfg.criteria)
                        .reset_index(drop=True)
                        .loc[:0]
                    )
                best_params = pd.concat(stack)
                # save the best models for eac hps point as table.
                best_params.to_csv(
                    cfg.forecast_path
                    + "/"
                    + location
                    + "_"
                    + direction
                    + "_"
                    + model_t
                    + "_selected_models.csv"
                )
                

                # print the best params for quick overview.
                print(best_params[best_params.columns[:7]])
                print(best_params[best_params.columns[7:14]]) # all exog params
                output, model_summaries = forecast_with_best(
                    best_params,
                    data[location][direction],
                    exogs[location],
                    model_t,
                    long_range=False# "igarss" in cfg.save_path
                )

                output.to_csv(
                    cfg.forecast_path
                    + "/"
                    + location
                    + "_"
                    + direction
                    + "_"
                    + model_t
                    + "_forecast_stack.csv"
                )
                pickle.dump(
                    model_summaries,
                    open(
                        cfg.forecast_path
                        + "/"
                        + location
                        + "_"
                        + direction
                        + "_"
                        + model_t
                        + "_model_stack.p",
                        "wb",
                    ),
                )
    print("Done.")


if __name__ == "__main__":
    main()
