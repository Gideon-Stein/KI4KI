import copy

import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.linear_model import LinearRegression

# Filters that return a boolean Outlier markers
# Input: pandas Table
# Output: equally sized pandas table with boolean outlier flags
# Not all of them are currently used for this pipeline but are included for consistency


def running_average_filter(table, threshold=4, convSize=3):
    # Filter datapoints based on mean of surrounding points
    out = []
    table = (table - table.min()) / (table.max() - table.min())

    for x in table.columns:
        assert (convSize % 2) != 0, "Odd Kernel size please!"
        convF = [1 / convSize] * convSize
        # drop nans temporarely to filter properly
        sel = np.where(~(np.isnan(table[x]).values))[0]
        no_fill = table[x].dropna().values

        a = np.convolve(no_fill, convF, mode="same")
        offset = -int(convSize / 2)
        a = np.roll(a, offset)  # Middle of the convolve is the target point
        a[offset:] = np.nan  # replace the rolled elements
        # reasign the values with nan
        rebuild = np.zeros(len(table))
        rebuild[sel] = a
        score = np.abs(table[x].values - rebuild)
        score[np.isnan(score)] = 0
        out.append(score > threshold)
    return pd.DataFrame(np.array(out).T, columns=table.columns, index=table.index)


def seasonal_mean(table):
    # calculates seasonal mean (1 year)
    df = copy.deepcopy(table)
    df["wy"] = pd.Series(df.index).dt.weekofyear.values
    df = df.groupby("wy").mean().reset_index().drop(columns=["wy"])
    return df


def substract_seasonal_mean(df, mean):
    # spread to df timestamps
    tr = copy.deepcopy(df)
    tr["wy"] = pd.Series(tr.index).dt.weekofyear.values
    a = tr.loc[:, ["wy"]].merge(mean, how="left", left_on="wy", right_index=True)
    tr.drop(columns="wy", inplace=True)
    tr
    # substract the mean from all relevant columns.
    tr = tr - a
    return tr


def detrend_via_model(df, return_trend=False, return_complete_component=False):
    # Calculates a linear fit to the data to substract it.
    params = []
    trend_table = df.copy()
    table = copy.deepcopy(df)
    for x in table.columns:
        X = [i for i in range(0, len(table[x]))]
        X = np.reshape(X, (len(X), 1))
        y = table[x].values
        valid = np.where(~np.isnan(y))
        model = LinearRegression()
        model.fit(X[valid], y[valid])
        if not return_trend:
            # calculate trend
            trend = model.predict(X)
            trend_table[x] = trend
            table[x] = table[x].values - trend
        else:
            params.append((model.coef_, model.intercept_))
    if return_trend:
        return params
    elif return_complete_component:
        return table, trend_table
    else:
        return table


def simple_boxplot_filter(df, iqr=1.5, diff=False):
    # boxplotfilter for each column
    table = copy.deepcopy(df)
    if diff:
        table = table.diff(1)
    a = table.mean()
    b = table.std()
    upper_bound = a + iqr * b
    lower_bound = a - iqr * b
    c = (
        ((table > upper_bound).astype(int) + (table < lower_bound).astype(int)) > 0
    ).astype(bool)
    return c


def change_filter(table, threshold=20):
    # IDEA find the points with the highest consecutive change
    change = (table - table.shift(-1)).abs()
    change = (change + change.shift(1)).shift(-1)
    return change > threshold


def apply_filter(table, filter_t):
    # assumes one data one bool table same shape
    table[filter_t] = np.nan
    return table

#TODO make this more flexible.
#def sinfunc(t, A, p, c):
#    w = 2* np.pi /61
#    return A * np.sin(w * t + p) + c


def cosfunc(t, A, w, p, c):
    return A * np.cos(w * t + p) + c

def sinfunc(t, A, w, p, c):
    return A * np.sin(w * t + p) + c


def fit_sin(yy, sinfunc, fixed_freq=61):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
   
    tt = np.arange(len(yy))
    yy = np.array(yy)
    filter_nans = ~ np.isnan(yy)
    tt = tt[filter_nans]
    yy = yy[filter_nans]

    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(
        ff[np.argmax(Fyy[1:]) + 1]
    )  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.0**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 0.0, guess_offset])
    filter_nans = ~ np.isnan(yy)

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, p, c = popt
    f = 2* np.pi /fixed_freq / (2.0 * np.pi)
    fitfunc = lambda t: A * np.sin(2* np.pi /fixed_freq * t + p) + c
    return {
        "amp": A,
        "omega": 2* np.pi /fixed_freq,
        "phase": p,
        "offset": c,
        "freq": f,
        "period": 1.0 / f,
        "fitfunc": fitfunc,
        "maxcov": np.max(pcov),
        "rawres": (A, fixed_freq, p, c),
    }


def custom_fourier_terms(
    data, date_col="date", value_col="temperature", fixed_freq=365.24219, order=1
):
    """
    #TODO there is some issue here concerning the daytime stamps and the frequency which makes it not smooth. Fix it at some point.
    input: df with date and value column
    Fits custom fourier for daily values and yearly cycle.
    extends on https://www.statsmodels.org/dev/generated/statsmodels.tsa.deterministic.Fourier.html
    by fitting the first Sin completely to the data also considering amplitude.
    Additional terms are added in the same way
    returns a pd table holding the single components (normalized).
    """

    # get step index from the data column

    def sinfunc(t, A, p, c, ff=fixed_freq):
        w = 2* np.pi /ff
        return A * np.sin(w * t + p) + c



    df = data.copy()
    df[date_col] = df[date_col].dt.dayofyear
    df[date_col] = df[date_col] - 1

    # step one: calculate seasonal mean of dat
    raw = df.groupby(date_col).mean()[value_col].values

    raw = (raw - raw.min()) / (raw.max() - raw.min())
    # step two: get the proper sin fit
    A, w, p, c = fit_sin(raw, sinfunc=sinfunc, fixed_freq=fixed_freq)["rawres"]
    out = pd.DataFrame(index=df[date_col])
    for x in range(1, order + 1):
        a = [sinfunc(y, A, p, c,w * (x)) for y in df[date_col]]
        b = [cosfunc(y, A, p, c, w * (x)) for y in df[date_col]]
        out["sin_order_" + str(x)] = a
        out["cos_order_" + str(x)] = b
    out.reset_index(drop=True, inplace=True)
    out["date"] = data[date_col].values
    return out



def decompose_data(data, error_threshold=1e-12, fixed_freq=61):
    """ 
    Removes trend and seasonal cycle from the original data.
    """

    def sinfunc(t, A, p, c):
        w = 2* np.pi /fixed_freq
        return A * np.sin(w * t + p) + c


    detrended, trend = detrend_via_model(data, return_complete_component=True)

    seasonality = detrended.copy()
    for key in seasonality.columns:
        A, w,p, c = fit_sin(detrended.loc[:,key].values, sinfunc=sinfunc, fixed_freq=61)["rawres"]
        seasonality[key] = [sinfunc(y, A, p, c) for y in np.arange(len(detrended))]


    deseasoned = (detrended - seasonality)

    distance =  np.nanmax(((deseasoned + trend + seasonality) - data).values)
    assert error_threshold > distance, "Too much error" + str(distance)

    return deseasoned, seasonality, trend
