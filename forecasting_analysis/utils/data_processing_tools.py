import pandas as pd
from functools import reduce
import numpy as np
import sys

sys.path.append("..")

from utils.filter_tools import (
    apply_filter,
    fit_sin,
    custom_fourier_terms,
    detrend_via_model,
    simple_boxplot_filter,
)


def full_preprocessing_pipeline(endog, exog_base, combo, model_search, outlier_filtering=True, interpolate=True, normalization=True ):

    # 1. Load all variables
    # 2. Split
    # 3. Filter
    # 4. Decompose #r
    # 1. Interpolate
    # 2. Normalize   #r

    exog = exog_base.copy()
    if endog.columns[0] == "Radial":  # Lot measurements are daily.
        endog = endog.asfreq("d")

    # 1. ' THIS IS THE ONLY THING THAT HAS TO BE CHANGED HERE IF OTHER DATA IS USED.
    exog_prep = build_specific_ds(
        endog,
        exog,
        future=False,
        fourier_order=combo.f,
        external_timesteps=combo.ex,
        mean_period=combo.mean,
        interaction=combo.interaction
    )
    # 2.
    train, test = train_test_split(endog, exog_prep, model_search=model_search)
    # 3.
    if outlier_filtering:
        # clean the data via simple trend removing and boxplot filtering
        step1, _ = detrend_via_model(train[0])
        step2 = simple_boxplot_filter(step1, iqr=2.25)
        train[0] = apply_filter(train[0], step2)
    # 4
    if combo.decompose:
        (
            deseasoned,
            seasonality,
            trend,
            test_deseasoned,
            test_seasonality,
            test_trend,
        ) = decompose_data(
            train[0],
            test[0],
            fixed_freq=365.2422 if endog.columns[0] == "Radial" else 61,
        )
        # decompose the temperature as well.
        if len(train[1].columns) != 0:
            (
                exog_prep,
                exog_seasonality,
                exog_trend,
                test_exog_prep,
                exog_test_seasonality,
                exog_test_trend,
            ) = decompose_data(
                train[1],
                test[1],
                fixed_freq=365.2422 if endog.columns[0] == "Radial" else 61,
                ignore_columns=[x for x in train[1].columns if (("order" in x) or ("W_" in x))],
            )
        else:
            exog_prep, test_exog_prep = train[1], test[1]


        train = [deseasoned, exog_prep]
        test = [test_deseasoned, test_exog_prep]
    else:
        trend = None
        test_trend = None
        test_seasonality = None
        seasonality = None

    # 5.
    if interpolate:
        train, test = interpolate_simple(train, test)



    if normalization:
        train, test, max_data, min_data = normalize(train, test)
    else:
        max_data, min_data = None, None

    return (
        train,
        test,
        max_data,
        min_data,
        trend,
        seasonality,
        test_trend,
        test_seasonality,
    )


def decompose_data(
    train, test=None, error_threshold=1e-12, fixed_freq=61, ignore_columns=[]
):
    """
    Removes trend and seasonal cycle from the original data.
    """


    def sinfunc(t, A, p, c):
        w = 2 * np.pi / fixed_freq
        return A * np.sin(w * t + p) + c

    detrended, trend, test_detrended, test_trend = detrend_via_model(
        train, test, ignore_columns=ignore_columns
    )
    seasonality = detrended.copy()
    seasonality.loc[:] = 0
    test_seasonality = test_detrended.copy()
    test_seasonality.loc[:] = 0

    for key in [x for x in seasonality.columns if x not in ignore_columns]:
        A, w, p, c = fit_sin(
            detrended.loc[:, key].values, sinfunc=sinfunc, fixed_freq=61
        )["rawres"]
        seasonality[key] = [sinfunc(y, A, p, c) for y in np.arange(len(detrended))]
        test_seasonality[key] = [
            sinfunc(y, A, p, c)
            for y in np.arange(len(detrended), len(detrended) + len(test_detrended))
        ]

    deseasoned = detrended - seasonality
    test_deseasoned = test_detrended - test_seasonality

    distance = np.nanmax(((deseasoned + trend + seasonality) - train).values)
    assert error_threshold > distance, "Too much error" + str(distance)

    return deseasoned, seasonality, trend, test_deseasoned, test_seasonality, test_trend


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


def build_specific_ds(
    endog,
    exog,
    future=False,
    external_timesteps=[[0], [0]],
    fourier_order=0,
    mean_period=[0, 0],
    interaction=False
):
    # Loads the PCI points from files and constructs all relevant external variables
    # Also does some reformating to be consistent
    # Possible Loads: data, Temp/Water, Fourier terms, Mean Temp/Water

    endog.sort_index(inplace=True)
    # # TODO FUTURE OPTION CURRENTLY OUT OF ORDER
    # if future:
    #     # We make an artificial datetime ts for future data
    #     # we use some more past data to look sufficiently into the past (prevent nans)
    #     data = pd.DataFrame(
    #         index=pd.date_range(data.index.values[-5], periods=67, freq="6D")
    #     )

    exogs = []
    # Yearly cycle fourier
    # TODO Adapt to fixed cycle. Not used currently.
    fourier = custom_fourier_terms(
        data=exog, order=fourier_order, date_col="datetime", value_col="T"
    )
    fourier = fourier[fourier["date"].isin(endog.index)]
    fourier.sort_values("date", inplace=True)
    exogs.append(fourier)

    # Mean periods calculation
    if mean_period[0] == 0:
        pass
    else:
        exog_Weather = exog.rolling(mean_period[0], closed="left").mean()
        exog_Weather["date"] = exog["datetime"]

        exogs.append(
            select_daily_exog(
                exog_Weather,
                [0],
                endog.index,
                variable_name="W",
                replacement="_mean_" + str(mean_period[0]) + "_",
            )
        )
    if mean_period[1] == 0:
        pass
    else:
        exog_stau = exog.rolling(mean_period[1], closed="left").mean()
        exog_stau["date"] = exog["datetime"]

        exogs.append(
            select_daily_exog(
                exog_stau,
                [0],
                endog.index,
                variable_name="T",
                replacement="_mean_" + str(mean_period[1]) + "_",
            )
        )

    if len(external_timesteps[0]) > 0:
        exogs.append(
            select_daily_exog(
                exog,
                external_timesteps[0],
                endog.index,
                time_name="datetime",
                variable_name="W",
                step_factor=1,  # has 6 day resolution.
            )
        )

    if len(external_timesteps[1]) > 0:
        exogs.append(
            select_daily_exog(
                exog,
                external_timesteps[1],
                endog.index,
                time_name="datetime",
                variable_name="T",
                step_factor=1,  # has 6 day resolution.
            )
        )

    if interaction:
        W = select_daily_exog(
                exog,
                [0],
                endog.index,
                time_name="datetime",
                variable_name="W",
                step_factor=1,  # has 6 day resolution.
            )
        T = select_daily_exog(
                exog,
                [0],
                endog.index,
                time_name="datetime",
                variable_name="T",
                step_factor=1,  # has 6 day resolution.
            )
        # Option 1: We use the raw values to multiply
        if interaction == 1:
            W["W_0"] = W["W_0"] * T["T_0"].values
        # Option 2: We use the scales values to multiply (prevention + * - )
        # THIS IS A SLIGHT VIOLATION OF TRAIN/TEST SPLITTING. But I guess its not too impactful.
        if interaction == 2:
            W["W_0"] = W["W_0"] - W["W_0"].min() / (W["W_0"].max() - W["W_0"].min())
            T["T_0"] = T["T_0"] - T["T_0"].min() / (T["T_0"].max() - T["T_0"].min())
            W["W_0"] = W["W_0"] * T["T_0"].values

        W.rename(columns={"W_0":"INTER"}, inplace=True)
        exogs.append(W)



    if len(exogs) > 1:
        exogs = reduce(
            lambda left, right: pd.merge(left, right, on=["date"], how="outer"), exogs
        )
    else:
        exogs = exogs[0]

    exogs.index = exogs["date"]
    exogs.drop(columns="date", inplace=True)

    return exogs


def bring_to_original_range(Y, Y_hat, max_data, min_data, trend, seasonality, endog,point):
    # TODO FIX FOR NON RADIAL RESCALING

    if max_data:
        Y = Y * (max_data - min_data) + min_data
        Y_hat = Y_hat * (max_data - min_data) + min_data
    if isinstance(seasonality, pd.Series):
        Y_hat += seasonality[Y_hat.index]
        Y += seasonality[Y_hat.index]
    if isinstance(trend, pd.Series):
        Y += trend[Y_hat.index]
        Y_hat = Y_hat + trend[Y_hat.index]
    assert (
        (endog.loc[Y.index, point] - Y).abs().max()
    ) < 0.001, "Inconsistent data!"
    return Y, Y_hat


def normalize(train, test):

    # normalize (we need to save for renormalization later)
    max_exog = train[1].max()
    min_exog = train[1].min()
    max_data = train[0].max()
    min_data = train[0].min()

    train[0] = (train[0] - min_data) / (max_data - min_data)
    train[1] = (train[1] - min_exog) / (max_exog - min_exog)
    test[1] = (test[1] - min_exog) / (max_exog - min_exog)
    test[0] = (test[0] - min_data) / (max_data - min_data)

    return train, test, max_data, min_data


def interpolate_simple(train, test):

    train[0] = train[0].interpolate()
    train[0] = train[0].fillna(method="backfill")
    test[0] = test[0].interpolate()
    test[0] = test[0].fillna(method="backfill")
    # interpolate missing values and fill them at the start
    train[1] = train[1].fillna(method="backfill")
    train[1] = train[1].fillna(method="ffill")
    # interpolate missing values and fill them at the start
    test[1] = test[1].fillna(method="backfill")
    test[1] = test[1].fillna(method="ffill")

    return train, test


def train_test_split(endog, exog, model_search, training_split=0.9):

    data = endog.copy()
    # remove one year completely for evaluation later
    if model_search:
        # some magic splits and interpolation of externals
        # Okay I guess I need to update this to full years...
        train = [data.loc[data.index.year < 2019], exog.loc[exog.index.year < 2019]]
        test = [data.loc[data.index.year == 2019], exog.loc[exog.index.year == 2019]]
    else:
        exog_a = exog[exog.index.year < 2020]
        exog_b = exog[exog.index.year == 2020]
        train = [data[data.index.year < 2020], exog_a]
        test = [data[data.index.year == 2020], exog_b]
    return train, test
