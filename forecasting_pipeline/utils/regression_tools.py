import sys

sys.path.append("..")

from utils.general_tools import pd_to_param_holder
from utils.data_processing_tools import (
    bring_to_original_range,
    full_preprocessing_pipeline,
)
from utils.modeling_tools import (
    perform_modeling_forest,
    perform_modeling_linear,
    perform_modeling_varmax,
    perform_modeling_sarimax,
    perform_modeling_ada,
)
from utils.forecast_tools import (
    forecast_test_forest,
    forecast_test_linear,
    forecast_test_ada,
)

import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.metrics import mean_absolute_percentage_error


import statsmodels.api as sm
import pandas as pd


def forecast_with_best(best_params, endog, exog, model_type, long_range=False):
    out_stack = []
    model_stack = []
    assert list(best_params["Index"].unique()) == list(
        endog.columns.values
    ), "Ordering broken!"
    for m, p in enumerate(best_params["Index"].unique()):
        exog_base = exog.copy()
        # gets the best combo as param holder
        combo = pd_to_param_holder(best_params, p)
        # peforms preprocessing pipeline according to combo

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
        # DIFFERENT MODEL CLASSES
        if model_type == "linear":
            model, trend_step = perform_modeling_linear(train[0][p], train[1], combo)
            pred = forecast_test_linear(test[0][p],test[1], combo, model, trend_step)

        if model_type == "sarimax":
            model = perform_modeling_sarimax(
                train[0][p],
                train[1],
                combo,
            )
            if long_range: 
                print("Forecast in one step")
                if len(test[1].columns)  == 0:
                    pred_one_step = model.forecast(steps=len(test[0]))
                else:
                    pred_one_step = model.forecast(steps=len(test[0]), exog=test[1])
            else:
                print("Forecast step by step...")
                if len(test[1].columns)  == 0:
                    append_model = model.append(test[0][p], refit=False)
                else:
                    append_model = model.append(test[0][p], refit=False, exog=test[1])
                pred_one_step = append_model.predict(start=test[0].index[0])
            pred_one_step = pd.DataFrame(pred_one_step)
            pred_one_step.columns = [0]
            pred = pred_one_step

        if model_type == "var":
            model = perform_modeling_varmax(train[0], train[1], combo)
            if long_range: 
                pred = model.forecast(steps=len(test[0]), exog=test[1])
            else:
                if len(test[1].columns) != 0:
                    append_res = model.append(test[0], refit=False, exog=test[1])
                else:
                    append_res = model.append(test[0], refit=False)
                pred = append_res.predict(start=test[0].index[0])
            pred.columns = test[0].columns
            pred = pd.DataFrame(pred[p])
            pred.columns = [0]
    
        if model_type == "forest":
            model = perform_modeling_forest(train[0][p], train[1], combo)
            pred = forecast_test_forest(test[0][p], test[1], combo, model)
        if model_type == "ada":
            model = perform_modeling_ada(train[0][p], train[1], combo)
            pred = forecast_test_ada(test[0][p], test[1], combo, model)

        # Transform to original range and appends results
        Y, Y_hat = bring_to_original_range(
            test[0][p],
            pred[0],
            max_data[p],
            min_data[p],
            (test_trend[p] if isinstance(trend, pd.DataFrame) else None),
            (test_seasonality[p] if isinstance(seasonality, pd.DataFrame) else None),
            endog,
            p,
        )

        # Check if the recreated Y is actually the same values as the original data.
        out_stack.append(Y_hat)
        model_stack.append(model)

    final = pd.concat(out_stack, axis=1)
    final.columns = [p for p in best_params["Index"].unique()]
    return final, model_stack


def score_on_original_range_and_save(
    res_dict,
    test,
    pred,
    max_data,
    min_data,
    trend,
    seasonality,
    endog,
    model,
    long_pred=None
):
    for point in pred.columns:
        # Transform to original range and appends results
        Y, Y_hat = bring_to_original_range(
            test[0][point],
            pred[point],
            max_data[point],
            min_data[point],
            (trend[point] if isinstance(trend, pd.DataFrame) else None),
            (seasonality[point] if isinstance(seasonality, pd.DataFrame) else None),
            endog,
            point,
        )
        mse = np.nanmean((Y - Y_hat) ** 2)
        mae = np.nanmean(np.abs(Y - Y_hat))
        mape = mean_absolute_percentage_error(Y, Y_hat)
        res_dict[point] = [mse, mae, mape]

        if isinstance(long_pred, pd.DataFrame):
            Y_2, Y_hat_2 = bring_to_original_range(
                test[0][point],
                long_pred[point],
                max_data[point],
                min_data[point],
                (trend[point] if isinstance(trend, pd.DataFrame) else None),
                seasonality[point] if isinstance(seasonality, pd.DataFrame) else None,
                endog,
                point,
            )
            mse2 = np.nanmean((Y_2 - Y_hat_2) ** 2)
            mae2 = np.nanmean(np.abs(Y - Y_hat_2))
            mape2 = mean_absolute_percentage_error(Y_2, Y_hat_2)

            res_dict[point] = [mse, mae, mse2, mae2, mape, mape2, model.aic]


def score_single_combo(
    combo, train, test, max_data, min_data, trend, seasonality, model_type, endog
):

    res_dict = {}
    # Models joint so we can run a single model per combo
    if model_type == "var":
        if combo.p == 0 and combo.q == 0:
            # VAR breaks if we do not use at least a single model order
            print("Skipping empty VAR")
            return res_dict
        else:
            model = perform_modeling_varmax(train[0], train[1], combo)
            if not model:
                return res_dict
            long_pred_table = model.forecast(steps=len(test[0]), exog=test[1])
            long_pred_table.columns = test[0].columns

            if len(test[1].columns) != 0:
                append_res = model.append(test[0], refit=False, exog=test[1])
            else:
                append_res = model.append(test[0], refit=False)
            pred_one_step_table = append_res.predict(start=test[0].index[0])
            pred_one_step_table.columns = test[0].columns

    
    else:
        pred_one_step_table =  []
        long_pred_table =  []

        for n, point in enumerate(train[0].columns):
            if model_type == "linear":
                model, trend_step = perform_modeling_linear(
                    train[0][point], train[1], combo
                )
                pred_one_step = forecast_test_linear(test[0][point],test[1], combo, model, trend_step)
                pred_one_step.columns = [point]
                pred_one_step_table.append(pred_one_step)

            if model_type == "forest":
                model = perform_modeling_forest(train[0][point], train[1], combo)
                pred_one_step = forecast_test_forest(test[0][point], test[1], combo, model)
                pred_one_step.columns = [point]
                pred_one_step_table.append(pred_one_step)

            if model_type == "ada":
                model = perform_modeling_ada(train[0][point], train[1], combo)
                pred_one_step = forecast_test_ada(test[0][point], test[1], combo, model)
                pred_one_step.columns = [point]               
                pred_one_step_table.append(pred_one_step)

            if model_type == "sarimax":
                model = perform_modeling_sarimax(
                    train[0][point],
                    train[1],
                    combo,
                )
                if not model:
                    res_dict[point] = None
                    continue
                if len(test[1].columns) != 0:
                    long_pred = model.forecast(len(test[0][point]), exog=test[1])
                else:
                    long_pred = model.forecast(len(test[0][point]))
                long_pred = pd.DataFrame(long_pred)
                long_pred.columns = [point]
                long_pred_table.append(long_pred)

                if len(test[1].columns) != 0:
                    append_model = model.append(test[0][point], refit=False, exog=test[1])
                else:
                    append_model = model.append(test[0][point], refit=False)

                pred_one_step = append_model.predict(start=test[0].index[0])
                pred_one_step = pd.DataFrame(pred_one_step)
                pred_one_step.columns = [point]
                pred_one_step_table.append(pred_one_step)
      
        if model_type =="sarimax":
            long_pred_table = pd.concat(long_pred_table,axis=1)
        else:
            long_pred_table = None
        pred_one_step_table = pd.concat(pred_one_step_table,axis=1)

    # Transforms the predictions to the original data space and score the performance. 
    # By doing this, the metrics actually correspond to mm.
    score_on_original_range_and_save(
        res_dict,
        test,
        pred_one_step_table,
        max_data,
        min_data,
        trend,
        seasonality,
        endog,
        model,
        long_pred_table,
    )
    return res_dict



def score_model(combo, endog, exog, model_type, model_search, counter=None):
    # Loads the correct external terms and the PCI dataframe
    if counter: 
        print("Run:", counter)
    (
        train,
        test,
        max_data,
        min_data,
        trend,
        seasonality,
        test_trend,
        test_seasonality,
    ) = full_preprocessing_pipeline(endog, exog, combo, model_search)
    param_stamp = combo.export()[:-1]
    # run model fit and return metricd
    result = score_single_combo(
        combo,
        train,
        test,
        max_data,
        min_data,
        test_trend if combo.decompose else None,
        test_seasonality if combo.decompose else None,
        model_type,
        endog,
    )

    return [param_stamp, result]


def find_best_order(
    combos,
    endog,
    exog,
    model_type="sarimax",
    model_search=False,
    parallel=False,
):
    # Runs some combos in parallel if needed
    if parallel:
        executor = Parallel(n_jobs=cpu_count() - 1, backend="multiprocessing")
        tasks = (
            delayed(score_model)(combo, endog, exog, model_type, model_search, n)
            for n,combo in enumerate(combos)
        )
        return executor(tasks)
    else:
        return [
            score_model(combo, endog, exog, model_type, model_search,n)
            for n,combo in enumerate(combos)
        ]
