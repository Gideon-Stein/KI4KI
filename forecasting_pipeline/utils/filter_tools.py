import numpy as np
import copy
from sklearn.linear_model import LinearRegression
import pandas as pd
import scipy.optimize

def apply_filter(table, filter_t):
    # assumes one data one bool table same shape

    table.copy()[filter_t] = np.nan
    return table

def cosfunc(t, A, w, p, c):
    return A * np.cos(w * t + p) + c


def sinfunc(t, A, w, p, c):
    return A * np.sin(w * t + p) + c


def detrend_via_model(train, test=None, ignore_columns=[]):
    # Calculates a linear fit to the data to substract it.
    trend_table = train.copy()
    trend_table.loc[:] = 0
    table = train.copy()
    if isinstance(test, pd.DataFrame): 
        trend_table_test = test.copy()
        trend_table_test.loc[:] = 0
        table_test = test.copy()

    for x in [w for w in table.columns if w not in ignore_columns]:
        X = np.expand_dims(np.arange(0,len(train)),1)
        y = train.loc[:,x].values
        valid = np.where(~np.isnan(y))

        model = LinearRegression()
        model.fit(X[valid], y[valid])
        # calculate trend
        trend = model.predict(X)
        trend_table[x] = trend
        table[x] = table[x].values - trend

        if isinstance(test, pd.DataFrame): 
            X_2 = np.expand_dims(np.arange(len(train),len(train)+ len(test)),1)
            trend = model.predict(X_2)
            trend_table_test[x] = trend
            table_test[x] = table_test[x].values - trend
    if isinstance(test, pd.DataFrame): 
        return table, trend_table, table_test, trend_table_test
    else:
        return table, trend_table


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


def fit_sin(yy, sinfunc, fixed_freq=61):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''

    tt = np.arange(len(yy))
    yy = np.array(yy)
    filter_nans = ~np.isnan(yy)
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
    filter_nans = ~np.isnan(yy)

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, p, c = popt
    f = 2 * np.pi / fixed_freq / (2.0 * np.pi)
    fitfunc = lambda t: A * np.sin(2 * np.pi / fixed_freq * t + p) + c
    return {
        "amp": A,
        "omega": 2 * np.pi / fixed_freq,
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

    # TODO THIS BRICKS COMPLETELY WHEN USING SECOND ORDER TERMS. FIND OUT WHY IN THE FUTURE.
    input: df with date and value column
    Fits custom fourier for daily values and yearly cycle.
    extends on https://www.statsmodels.org/dev/generated/statsmodels.tsa.deterministic.Fourier.html
    by fitting the first Sin completely to the data also considering amplitude.
    Additional terms are added in the same way
    returns a pd table holding the single components (normalized).
    """

    # get step index from the data column

    def sinfunc(t, A, p, c, ff=fixed_freq):
        w = 2 * np.pi / ff
        return A * np.sin(w * t + p) + c

    def cosfunc(t, A, p, c, ff=fixed_freq):
        w = 2 * np.pi / ff
        return A * np.cos(w * t + p) + c

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
        a = [sinfunc(y, A, p, c, w * (x)) for y in df[date_col]]
        b = [cosfunc(y, A, p, c, w * (x)) for y in df[date_col]]
        out["sin_order_" + str(x)] = a
        out["cos_order_" + str(x)] = b
    out.reset_index(drop=True, inplace=True)
    out["date"] = data[date_col].values
    return out
