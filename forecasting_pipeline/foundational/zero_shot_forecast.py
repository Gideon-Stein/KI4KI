import os
from datetime import datetime
import pandas as pd

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

import sys

sys.path.append("..")

from utils.data_parsing_tools import load_data_bases
from utils.general_tools import param_holder

from utils.data_processing_tools import (
    bring_to_original_range,
    full_preprocessing_pipeline,
)
import hydra
from omegaconf import DictConfig
import timesfm


def tfm(
    train,
    test,
    parallel,
    max_data,
    min_data,
    test_trend,
    test_seasonality,
    trend,
    seasonality,
    endog,
):

    # Frequency 0 : daily resolution Frequency 1: Weekly (This is absolutery randomly stated by google research. Was trained that way.)
    out_stack = []

    prediction_length = 1
    train_temporary = train[0].copy()
    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    boundary = test[1].index.values
    boundary = [train[1].index.values[-1]] + list(boundary)

    tfm = timesfm.TimesFm(
        context_len=64,
        horizon_len=1,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="gpu",
    )
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

    if parallel:
        frequency_input = (
            [0] if train[0].columns[0] == "Radial" else np.ones((len(train[0].columns)))
        )

        # Careful with ram.
        step_pred = []
        for n in range(len(boundary)):
            current_window = pd.concat([train_temporary, test[0].loc[: boundary[n]]])
            context = current_window.values.T
            point_forecast, experimental_quantile_forecast = tfm.forecast(
                context,
                freq=frequency_input,
            )
            step_pred.append(point_forecast)
        pred = np.concatenate(step_pred, axis=1)
        for n, p in enumerate(test[0].columns):
            # Transform to original range and appends results
            Y, Y_hat = bring_to_original_range(
                test[0][p],
                pred[n],
                max_data[p],
                min_data[p],
                (test_trend[p] if isinstance(trend, pd.DataFrame) else None),
                (
                    test_seasonality[p]
                    if isinstance(seasonality, pd.DataFrame)
                    else None
                ),
                endog,
                p,
            )
            out_stack.append(Y_hat)

    else:
        frequency_input = [0] if train[0].columns[0] == "Radial" else [1]
        for m, p in enumerate(test[0].columns):
            step_pred = []
            for n in range(len(boundary)):
                current_window = pd.concat(
                    [train_temporary, test[0].loc[: boundary[n], p]]
                )
                context = np.expand_dims(current_window[p].values, axis=0)
                point_forecast, experimental_quantile_forecast = tfm.forecast(
                    context,
                    freq=frequency_input,
                )
                step_pred.append(point_forecast)
            pred = np.array(step_pred)[:, 0]
            Y, Y_hat = bring_to_original_range(
                test[0][p],
                pred[:, 0],
                max_data[p],
                min_data[p],
                (test_trend[p] if isinstance(trend, pd.DataFrame) else None),
                (
                    test_seasonality[p]
                    if isinstance(seasonality, pd.DataFrame)
                    else None
                ),
                endog,
                p,
            )
            out_stack.append(Y_hat)

    return out_stack


def chronos(
    train,
    test,
    parallel,
    max_data,
    min_data,
    test_trend,
    test_seasonality,
    trend,
    seasonality,
    endog,
):
    out_stack = []

    prediction_length = 1
    train_temporary = train[0].copy()
    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    boundary = test[1].index.values
    boundary = [train[1].index.values[-1]] + list(boundary)

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    if parallel:
        # Careful with ram.
        step_pred = []
        for n in range(len(boundary)):
            current_window = pd.concat([train_temporary, test[0].loc[: boundary[n]]])
            context = torch.tensor(current_window.values.T)
            step_pred.append(
                pipeline.predict(context, prediction_length)
            )  # shape [num_series, num_samples, prediction_length]

        pred = torch.concat(step_pred, dim=2)

        for n, p in enumerate(test[0].columns):
            low, median, high = np.quantile(pred[n].numpy(), [0.1, 0.5, 0.9], axis=0)
            # Transform to original range and appends results
            Y, Y_hat = bring_to_original_range(
                test[0][p],
                median,
                max_data[p],
                min_data[p],
                (test_trend[p] if isinstance(trend, pd.DataFrame) else None),
                (
                    test_seasonality[p]
                    if isinstance(seasonality, pd.DataFrame)
                    else None
                ),
                endog,
                p,
            )
            out_stack.append(Y_hat)
    else:
        for m, p in enumerate(test[0].columns):
            step_pred = []
            for n in range(len(boundary)):
                current_window = pd.concat(
                    [train_temporary[p], test[0].loc[: boundary[n], p]]
                )
                context = torch.tensor(current_window.values.T)
                step_pred.append(pipeline.predict(context, prediction_length))
            pred = torch.concat(step_pred, dim=2)
            print(pred.shape)
            low, median, high = np.quantile(pred[0].numpy(), [0.1, 0.5, 0.9], axis=0)
            Y, Y_hat = bring_to_original_range(
                test[0][p],
                median,
                max_data[p],
                min_data[p],
                (test_trend[p] if isinstance(trend, pd.DataFrame) else None),
                (
                    test_seasonality[p]
                    if isinstance(seasonality, pd.DataFrame)
                    else None
                ),
                endog,
                p,
            )
            out_stack.append(Y_hat)
    return out_stack


def zero_shot(endog, exog_base, model_type, parallel=False):

    # We keep the format but want no exogenous since we cant use them.
    combo = param_holder(
        p=0,
        d=0,
        q=0,
        f=0,
        ex=[[], []],
        mean=[0, 0],
        decompose=False,
        interaction=False,
        estimator=None,
        n_estimators=None,
        max_depth=None,
        learning_rate=None,
    )

    (
        train,
        test,
        max_data,
        min_data,
        trend,
        seasonality,
        test_trend,
        test_seasonality,
    ) = full_preprocessing_pipeline(endog, exog_base, combo, model_search=False)

    if model_type == "chronos":
        out_stack = chronos(
            train,
            test,
            parallel,
            max_data,
            min_data,
            test_trend,
            test_seasonality,
            trend,
            seasonality,
            endog,
        )

    elif model_type == "tfm":
        out_stack = tfm(
            train,
            test,
            parallel,
            max_data,
            min_data,
            test_trend,
            test_seasonality,
            trend,
            seasonality,
            endog,
        )

    else:
        print("Model type unknown!")
        exit()

    final = pd.DataFrame(np.array(out_stack).T)
    final.columns = endog.columns
    print(final)
    return final


@hydra.main(version_base=None, config_path="../conf", config_name="forecast_config.yaml")
def main(cfg: DictConfig):

    start = datetime.now()
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

                output = zero_shot(
                    data[location][direction],
                    exogs[location],
                    model_type=cfg.model_type,
                    parallel=cfg.parallel,
                )
                output.to_csv(
                    cfg.save_path
                    + "/"
                    + location
                    + "_"
                    + direction
                    + "_"
                    + model_t
                    + "_zero_shot_chronos.csv"
                )
    print("Done.")


if __name__ == "__main__":
    main()
