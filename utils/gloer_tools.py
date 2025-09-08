import pickle
from datetime import date, datetime, timedelta
from os import listdir
from os.path import isfile, join
from pathlib import Path

import numpy as np
import pandas as pd
from dwdweather import DwdWeather
from joblib import Parallel, delayed
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from functools import reduce
from statsmodels.tsa.deterministic import Fourier
import statsmodels.api as sm

from utils.filters import (
    custom_fourier_terms,
    cosfunc,
    fit_sin,
    decompose_data,
)

from utils.baseline_tools import (
    apply_filter,
    detrend_via_model,
    dl_to_raw_ts_new,
    simple_boxplot_filter,
    param_holder,
    score_arimax,
    select_daily_exog,
)

# TODO Check external variables processing. can I add t0 steps as it is currently?


def prep_for_analysis(data, exog, simple_split=False):
    # normalize
    data = (data - data.min()) / (data.max() - data.min())
    exog = (exog - exog.min()) / (exog.max() - exog.min())
    # We skip some indices to remove the transition periods since they probably behave different.
    phases = []
    phases.append([0, 160])
    phases.append([161, 230])
    phases.append([233, 346])
    phases.append([355, -1])
    out = {}

    if simple_split:
        ph = phases[0]
        d = data[ph[0] : ph[1]]
        ex = exog[ph[0] : ph[1]]
        l = int(len(d) * 0.85)
        tr = (d[:l], ex[:l])
        te = (d[l:], ex[l:])

        ph = phases[1]
        d = data[ph[0] : ph[1]]
        ex = exog[ph[0] : ph[1]]
        l = int(len(d) * 0.85)
        tr2 = (d[:l], ex[:l])
        te2 = (d[l:], ex[l:])

        return {"p1": (tr, te), "p2": (tr2, te2)}

    else:
        for n, ph in enumerate(phases):
            d = data[ph[0] : ph[1]]
            ex = exog[ph[0] : ph[1]]
            l = int(len(d) * 0.85)
            tr = (d[:l], ex[:l])
            te = (d[l:], ex[l:])
            out["p_" + str(n)] = (tr, te)
        return out


def build_gloer_exog(
    data,  # table of direction
    stau,
    weather,
    weather_variables=["temperature"],
    fourier_order=1,
    external_timesteps=[[0, 1], [0]],
    mean_period=[3, 0],  # first for stau second for weather.
):
    exogs = []

    #  THIS IS ALIGNED WITH THE SEASONAL CYCLE BUT NOT PERFECT. NEED TO FIND A BETTER WAY.
    # direct exog for last two points:
    # fourier_gen = Fourier(365.2422, order=fourier_order)
    # just use weather now since it start directily at 1.1.
    # fourier = fourier_gen.in_sample(weather.index)
    # fourier["date"] = weather.date
    # fourier = fourier[fourier["date"].isin(data.index)]
    # fourier.sort_values("date", inplace=True)

    # custom version that considers offset and amplitude. # We take temperature to infer the cycle.
    #
    fourier = custom_fourier_terms(weather, order=fourier_order)
    fourier = fourier[fourier["date"].isin(data.index)]
    fourier.sort_values("date", inplace=True)
    exogs.append(fourier)

    descriptor = "_mean_" + str(mean_period) + "_"

    # Weather
    exog_Weather = weather.rolling(mean_period[1]).mean()
    exog_Weather["date"] = weather.date

    for weathe_var in weather_variables:
        if len(external_timesteps[1]) > 0:
            exogs.append(
                select_daily_exog(
                    weather,
                    external_timesteps[1],
                    data.index,
                    variable_name=weathe_var,
                    step_factor=6,  # has 6 day resolution.
                )
            )
        # 0 specifices already the mean of last x days
        if mean_period[1] != 0:
            exogs.append(
                select_daily_exog(
                    exog_Weather,
                    [0],
                    data.index,
                    variable_name=weathe_var,
                    replacement=descriptor,
                )
            )

    if len(external_timesteps[0]) > 0:
        # Water level
        exogs.append(
            select_daily_exog(
                stau,
                external_timesteps[0],
                data.index,
                variable_name="W",
                step_factor=6,
            )
        )
    if mean_period[0] != 0:  # 0 denotes we do not use this exog variables.
        ex_W = stau.rolling(mean_period[0]).mean()
        ex_W["date"] = stau.date
        # 0 specifices already the mean of last x days
        exogs.append(
            select_daily_exog(
                ex_W,
                [0],
                data.index,
                replacement=descriptor,
            )
        )
    # only reduce when multiple options are available.
    if len(exogs) > 1:
        exog = reduce(
            lambda left, right: pd.merge(left, right, on=["date"], how="outer"), exogs
        )
    else:
        exog = exogs[0]

    exog.index = exog["date"]
    exog.drop(columns="date", inplace=True)

    # other than last time I should return a dict with all directions for data.
    return exog


def find_best_order_gloer(
    combos, endog, weather, stau, parallel=True, simple_split=False
):
    # Runs some combos in parallel
    if parallel:
        executor = Parallel(n_jobs=cpu_count() - 1, backend="multiprocessing")
        tasks = (
            delayed(score_gloer)(combo, endog, weather, stau, simple_split)
            for combo in combos
        )
        return executor(tasks)
    else:
        tasks = [
            score_gloer(combo, endog, weather, stau, simple_split) for combo in combos
        ]
        return tasks


def score_gloer(combo, endog, weather, stau, simple_split):
    deseasoned, seasonality, trend = decompose_data(endog)

    exog = build_gloer_exog(
        deseasoned,
        stau,
        weather,
        weather_variables=["temperature"],
        fourier_order=0,
        external_timesteps=combo.ex,
        mean_period=combo.mean,
    )
    phases = prep_for_analysis(deseasoned, exog, simple_split=simple_split)
    # lets just run normal and abstau phases for now (1 and 2):
    param_stamp = combo.export()[:-1]  # we dont need the result placeholder here.

    if simple_split:
        p1_score = score_single_combo(combo, phases["p1"][0], phases["p1"][1])
        p2_score = score_single_combo(combo, phases["p2"][0], phases["p2"][1])
        return [param_stamp, p1_score, p2_score]

    else:
        p1_score = score_single_combo(combo, phases["p_0"][0], phases["p_0"][1])
        p2_score = score_single_combo(combo, phases["p_1"][0], phases["p_1"][1])

        return [param_stamp, p1_score, p2_score]


def score_single_combo(combo, tr, te):
    endog, exog = tr
    res_dict = {}
    for n, point in enumerate(endog.columns):
        # there are models with no exog params. fit Arima instead.
        if len(exog.columns) != 0:
            mod = sm.tsa.statespace.SARIMAX(
                endog[point].values,
                exog,
                order=(combo.p, combo.d, combo.q),
                trend="c",
                use_exact_diffuse=True,
            )
        else:
            mod = sm.tsa.statespace.SARIMAX(
                endog[point].values,
                order=(combo.p, combo.d, combo.q),
                trend="c",
                use_exact_diffuse=True,
            )
        # sometimes we need to catch because fitting crashes
        try:
            res = mod.fit(disp=False)
            endog_test, exog_test = te
            scorings = score_arimax(res, (endog_test.values, exog_test), n)
        except:
            print("FITTING FAILED")
            scorings = np.nan
        res_dict[point] = scorings
    return res_dict


def make_model_hp_list(P=[1], D=[0], Q=[1], F=[0, 1], E=[[0]], M=[6], decompose=[0]):
    """
    Constructs a data grid for model hps.
    """
    combos = []
    for p in P:
        for d in D:
            for q in Q:
                for f in F:
                    for e1 in E:
                        for e2 in E:
                            for m1 in M:
                                for m2 in M:
                                    for dec in decompose:
                                        combos.append(
                                            param_holder(
                                                p=p,
                                                d=d,
                                                q=q,
                                                f=f,
                                                ex=[e1, e2],
                                                mean=[m1, m2],
                                                decompose=dec,
                                            )
                                        )
    return combos


def weather_gloer():
    weather = get_dwd_data_from_location(
        lon=7.500833,  # loations of the dam
        lat=51.242778,
        resolution="daily",
        start=datetime(2015, 1, 1),
        end=datetime(2021, 12, 31),
    )
    cols = [
        x for x in weather.columns if weather.isnull().sum()[x] < 1600
    ]  # snow_depth
    weather = weather[cols]
    weather.rename(columns={"datetime": "date"}, inplace=True)
    weather = weather[list(weather.columns[:2][::-1]) + list(weather.columns[2:])]
    weather["snow_depth"] = weather["snow_depth"].fillna(0)
    weather[weather.columns[2:]] = weather[weather.columns[2:]].interpolate()
    return weather


def exog_gloer(
    st_p="/home/datasets4/stein/gloertalsperre/Gloer_Stau.xlsx",
    si_p="/home/datasets4/stein/gloertalsperre/GLOsick.xlsx",
):
    """
    Returns the exogenous variables for the Gloer Talsperre dataset.
    """
    stau = pd.read_excel(st_p)
    sicker = pd.read_excel(si_p)
    stau["date"] = stau["Datum/Uhrzeit"].dt.round("D")
    stau["W"] = stau["Wasserstand (NHN) [m NHN]"]
    stau = stau[["date", "W"]]
    stau = fill_pd_missing_time(stau)
    stau["W"] = stau["W"].interpolate()
    stau["W"] = stau["W"].fillna(method="backfill")

    sicker["date"] = sicker["Datum/Uhrzeit"].dt.round("D")
    sicker["S"] = sicker["Flussmenge mittel [l/s]"]
    sicker = sicker[["date", "S"]]
    sicker = fill_pd_missing_time(sicker)
    sicker["S"] = sicker["S"].interpolate()
    sicker["S"] = sicker["S"].fillna(method="backfill")
    return stau, sicker


def fill_pd_missing_time(df, freq="1D", date_col="date"):
    df = df.sort_values(date_col).copy()
    spread = pd.DataFrame(
        pd.date_range(
            start=df[date_col].values[0], end=df[date_col].values[-1], freq=freq
        ),
        columns=[date_col],
    )
    assert np.all(
        df[date_col].isin(spread.values[:, 0])
    ), "time spread is not matching time stamps!"

    out = spread.merge(df, on=date_col, how="outer")
    return out


def preprocessing_gloer(
    data_dict, remove_last_year, interpolate_endog=False, filter_range=2.25
):
    # full preprocessing and splitting to train on
    # This follows the strategy from the original dam data
    # remove one year completely for evaluation later

    for x in data_dict:
        data = data_dict[x].copy()
        if remove_last_year:
            data = data.loc[data.index.year < 2021]

        # clean the data via simple trend removing and boxplot filtering
        step1 = detrend_via_model(data)
        step2 = simple_boxplot_filter(step1, iqr=filter_range)
        data = apply_filter(data, step2)

        if interpolate_endog:
            data = data.interpolate()
            data = data.fillna(method="backfill")
        data_dict[x] = data
    return data_dict


def training_prep_gloer(
    data,
    training_split=0.9,
):
    max_data = data.max()
    min_data = data.min()
    data = (data - data.min()) / (data.max() - data.min())

    # some magic splits and interpolation of externals
    split = int(len(data) * training_split)
    train = data.values[:split]
    test = data.values[split:]

    return train, test, max_data, min_data


def load_gloer(
    gloer_path="/home/datasets4/stein/gloertalsperre/FME_157A7D12_1686040966463_5100(2)/GENERIC_1/output/",
    remove_last_year=True,
    interpolate_endog=True,
    filter_range=2.25,
):
    onlyfiles = [
        gloer_path + f for f in listdir(gloer_path) if isfile(join(gloer_path, f))
    ]
    data = {}
    meta = {}
    for f in onlyfiles:
        d, m = dl_to_raw_ts_new(
            pd.read_csv(f), keyname="ID", date_identifier="20", rename_id_columns={}
        )
        data[f.split("/")[-1][9:-4]] = d
        meta[f.split("/")[-1][9:-4]] = m

    data = preprocessing_gloer(
        data,
        remove_last_year=remove_last_year,
        interpolate_endog=interpolate_endog,
        filter_range=filter_range,
    )
    return data, meta


def get_dwd_data_from_location(
    lon=7.500833,
    lat=51.242778,
    resolution="daily",
    start=date(2013, 1, 1),
    end=date(2013, 1, 30),
):
    # iterate through time stamps
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    # Create client object.
    dwd = DwdWeather(resolution=resolution)
    # Find closest station to position.
    closest = dwd.nearest_station(lon=lon, lat=lat)
    out = []
    for single_date in daterange(start, end):
        out.append(dwd.query(station_id=closest["station_id"], timestamp=single_date))

    out = pd.DataFrame(out)
    out["datetime"] = pd.to_datetime(out["datetime"], format="%Y%m%d")
    return pd.DataFrame(out)


# Residuals:
def load_models(location, direction):
    path = "grid_results_short"
    p2 = Path(path)

    # load grid search results
    p = p2 / (path + "_" + location + "_" + direction + ".p")
    grid = pickle.load(open(p, "rb"))

    # check for correct length of the grid
    # (depends on the grid search and should be asjusted):
    assert len(grid) == 810, "From length: " + str(len(grid))

    # Get all available PCI points from the position
    points = list(grid[0][-1].keys())

    output = {}
    # Perform for all points in this loc_dir
    for n, point in enumerate(points):
        # order based on MSE results

        # Repair diverged results
        for x in grid:
            if not isinstance(x[-1][point], (list, tuple)):
                x[-1][point] = (np.nan, np.nan)

        order = np.argsort(([x[-1][point][0] for x in grid]))
        # best index
        ind = order[0]
        # best model order
        m = grid[ind][:-1]
        print(m)
        print("Consistency testing")
        # get proper ds
        out = build_sarimax_ds(
            location=location,
            direction=direction,
            fourier_order=m[3],
            external_timesteps=m[4],
            mean_period=m[5],
        )

        # process data. Use normalization values for renormalization later
        (
            data,
            exog,
            train,
            test,
            max_exog,
            min_exog,
            max_data,
            min_data,
        ) = preprocessing_pipeline(out, remove_last_year=True)
        print(data.index.year.max())
        # Train model again exactly like in the grid
        if len(out[1].columns) > 0:
            mod = sm.tsa.statespace.SARIMAX(
                train[0][:, n], train[1], order=(m[0], m[1], m[2]), trend="c"
            )
        else:
            mod = sm.tsa.statespace.SARIMAX(
                train[0][:, n], order=(m[0], m[1], m[2]), trend="c"
            )
        result = mod.fit(disp=False)

        # Score it the same way as we did in the grid search
        check = score_arimax(result, test, n)

        # Check for model difference. Difference should be sufficiently small for now
        diff = np.abs(grid[ind][-1][point][1] - check[1])
        rank_diff = np.abs(grid[order[1]][-1][point][1] - grid[order[0]][-1][point][1])
        print(diff)
        print(rank_diff)
        if diff > rank_diff:
            print("AIC diff too big!:" + str(diff))
            # vague.append((location, direction, point, diff, rank_diff))
        diff = np.abs(check[0] - grid[ind][-1][point][0])
        rank_diff = np.abs(grid[order[1]][-1][point][0] - grid[order[0]][-1][point][0])
        print(diff)
        print(rank_diff)
        if diff > rank_diff:
            print("MSE diff too big!:" + str(diff))

        # Now we retrain with the full dataset
        print("Full training")
        if len(exog.columns) > 0:
            mod = sm.tsa.statespace.SARIMAX(
                data.values[:, n], exog.values, order=(m[0], m[1], m[2]), trend="c"
            )
        else:
            mod = sm.tsa.statespace.SARIMAX(
                data.values[:, n], order=(m[0], m[1], m[2]), trend="c"
            )
        result = mod.fit(disp=False)

        output[point] = (result, m)
    return output


def results_to_pd(r):
    stack = []
    params = []
    for x in r:
        try:
            stack.append(pd.concat([pd.DataFrame(x[1]), pd.DataFrame(x[2])]).T)
            params.append(x[0][:4] + x[0][4] + x[0][5])
            params.append(x[0][:4] + x[0][4] + x[0][5])
        except:
            print(x[0])
    stack = pd.concat(stack)
    stack.reset_index(inplace=True)
    out = pd.concat([stack, pd.DataFrame(params)], axis=1)
    return out


def transform_res_to_param_holder(scores, ind):
    p = scores.iloc[ind].values[5:]
    return param_holder(p=p[0], d=p[1], q=p[2], f=p[3], ex=p[4:6], mean=p[6:])
