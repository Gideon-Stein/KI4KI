import os
import pickle
import warnings
from datetime import datetime

import sys

sys.path.append("..")
import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

from utils.general_tools import make_model_hp_list

from utils.data_parsing_tools import (
    load_data_bases,
)
from utils.regression_tools import find_best_order
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

# seed!
import numpy as np

np.random.seed(42)

import warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", ValueWarning)
warnings.simplefilter("ignore", ConvergenceWarning)




@hydra.main(version_base=None, config_path="conf", config_name="grid_config.yaml")
def main(cfg: DictConfig):

    # Track runtime
    start = datetime.now()
    print(OmegaConf.to_yaml(cfg))

    # Loads the full datasets. data holds the target variables for different locations
    # and directions while exogs holds the corresponding exogs (Both dicts.)
    data, exogs = load_data_bases(cfg)


    # Replace full loc and dir if specified
    locations = cfg.all_locs if cfg.loc == "X" else [cfg.loc]
    dirs = cfg.all_dirs if cfg.dir == "X" else [cfg.dir]
    model_types = cfg.all_model_types if cfg.model_type == "X" else [cfg.model_type]

    print("EXECUTE: ")
    print((locations, dirs))
    for location in locations:
        for direction in dirs:
            print("Evaluating for loc: " + location + " dir: " + direction)
            # Expand all model combinations and build the list of combos to test
            for model_t in model_types:
                P = np.arange(int(cfg.search.order_low[0]), int(cfg.search.order_high[0]))
                D = np.arange(int(cfg.search.order_low[1]), int(cfg.search.order_high[1]))
                Q = np.arange(int(cfg.search.order_low[2]), int(cfg.search.order_high[2]))
                E = [np.arange(x + 1) for x in cfg.search.exog_steps]
                if len(E) == 0: 
                    E = [[]]
                combos = make_model_hp_list(
                    P=P,
                    D=(D if model_t == "sarimax" else [None]),
                    Q=(Q if model_t in ["sarimax", "var"] else [None]),
                    F=cfg.search.fourier_orders,
                    M=cfg.search.mean_orders,
                    E=E,
                    decompose=cfg.search.decompose,
                    interaction=cfg.search.interaction,
                    estimators=(cfg.search.estimator if model_t == "ada" else [None]),
                    n_estimators=(
                        cfg.search.n_estimators if (model_t in ["forest", "ada"]) else [None]
                    ),
                    max_depth=(cfg.search.max_depth if model_t == "forest" else [None]),
                    learning_rate=(cfg.search.learning_rate if model_t == "ada" else [None]),
                    remove_W =cfg.remove_W,
                    remove_T = cfg.remove_T
                )

                print("Combos to test: " + str(len(combos)))
                # Evaluate all the models that were specified on all locations.
                results = find_best_order(
                    combos,
                    data[location][direction],
                    exogs[location],
                    model_t,
                    model_search=True,
                    parallel=cfg.parallel,
                )
                # Saves all the scorings to specified location for later model selection.
                if not os.path.exists(cfg.save_path):
                    os.makedirs(cfg.save_path)
                    print("New directory is created.")
                
                save_name = f"{location}_{direction}_{model_t}"
                pickle.dump(
                    results,
                    open(
                        cfg.save_path
                        + "/" + save_name + "_results_stack.p","wb"),
                )

    print(datetime.now() - start)
    print("Done.")


if __name__ == "__main__":
    main()
