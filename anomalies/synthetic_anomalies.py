
import numpy as np
import scipy.optimize
import scipy
import pandas as pd
import numpy as np
import copy
from sklearn.linear_model import LinearRegression
from functools import reduce
import statsmodels.api as sm
from scipy.stats import ranksums
import matplotlib.pyplot as plt


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


def prep_modeling_linear(
    endog, exog, autoregressive
):
    temporary = exog.copy()
    # add autoregressive components as additional variables
    if autoregressive > 0:
        for value in np.arange(1, autoregressive + 1):
            temporary[str(value)] = endog.shift(value).values
    temporary["const"] = 1.0
    temporary["trend"] = np.arange(0, 1, 1 / len(temporary["const"]))
    
    return temporary
    
    
def linear_pipeline(target, in_situ, cols=["T"], past_days=10, autoregressive=1, return_temporary=False, min_max= None):
    min_max_values = None
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
    exog = pd.concat(stack, axis=1)
    assert exog.isnull().sum().max() < 2, "Nans in exogenous variables."
    if min_max is not None:
        # Use provided min_max for scaling
        exog = (exog - min_max[0]) / (min_max[1] - min_max[0])
        target = (target - min_max[2]) / (min_max[3] - min_max[2])
    else:
        # return the min and max values later
        min_max_values = (exog.min(), exog.max(), target.min(), target.max())
        exog = (exog - exog.min()) / (exog.max() - exog.min())
        target = (target - target.min()) / (target.max() - target.min())
        
    exog.fillna(value=exog.mean(), inplace=True)
    temporary = prep_modeling_linear(target, exog, autoregressive=autoregressive)
    if return_temporary: 
        return target, temporary 
    olsmod = sm.OLS(
        exog=temporary.fillna(method="backfill"),
        endog=target,
    )    
    olsres = olsmod.fit()
    return olsres, min_max_values



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

def add_trend_increase(
        df,
        increase_start_index=150,
        increase_strength= 1,
        nonlinear_trend =False, 
        nl_trend_strength = 2

):
        # We take the original trend of the time series and increase it from start_index onwards.
        # We scale the strength of the increase with increase_strength. 
        # increase strength of 1 denotes adding 100% of the original trend on top.
        # as an alternative we can add an exponential increase on top.
        new_df  = df.copy()
        original_trends = detrend_via_model(df, return_trend=True)       
        for n, x in enumerate(df.columns):
                # trend is a scaling of the original trend.
                a = np.arange(0,
                              original_trends[n][0] *len(df),
                                original_trends[n][0]
                                ) + original_trends[n][1]
                a = a[:len(df)]
                a[:increase_start_index] = 0 
                a = a * increase_strength
                a = a - a[increase_start_index]
                a[:increase_start_index] = 0 
                new_df[x] += a
                if nonlinear_trend: 
                        new_df[x] += a**nl_trend_strength
        return new_df


def sinfunc(t,A, w, p, c): 
    return A * np.sin(w*t + p) + c

def cosfunc(t,A, w, p, c): 
    return A * np.cos(w*t + p) + c

def fit_sin(yy, fixed_freq=61):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.arange(len(yy))
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (A,w,p,c)}


def add_seasonal_cycle_increase(
        df,
        increase_start_index=150,
        increase_strength= 0.002,
):
        # We extract the yearly cycle of original ts, and increase the amplitude with x from a certain index onwards.
        new_df  = df.copy()
        # remove trend first to get the cycle
        detrend = detrend_via_model(df.interpolate())
        new_df = df.copy()    
        for n, x in enumerate(df.columns):
            # fit sin to the data with a fixed period
            A,w,p,c = fit_sin(detrend[x].values)["rawres"]
            # calculate a new seasonal cycle where only the amplitude increases
            a = [sinfunc(x,A-((x-increase_start_index)*increase_strength), w, p, c) for x in range(len(df))]
            b = fit_sin(detrend[x].values)["fitfunc"](np.arange(len(df)))
            # substract from original cycle to see what we have to add aditionally
            add = a-b
            add[:increase_start_index] = 0
            # add on the original data.
            new_df[x] += add
        return new_df


def add_jump(
        df,
        increase_start_index=150,
        increase_strength = 1,
):
        # simply add constant on top from index onwards
        new_df  = df.copy()
        add = np.zeros(len(df))
        add[increase_start_index:] = increase_strength

        for n, x in enumerate(df.columns):
            # add on the original data.
            new_df[x] += add
        return new_df

def add_outlier_group(
        df,
        increase_start_index=150,
        n_consecutive_outliers = 4,
        up= True, 
        increase_strength = 1 #specifies the std of the gaussian noise addition
):
        # simply add noise on top from index onwards (positive or negative)
        new_df  = df.copy()
        add = np.zeros(len(df))

        for n, x in enumerate(df.columns):

            add = np.zeros(len(df))
            noise = np.abs(np.random.normal(scale = increase_strength, size = n_consecutive_outliers))
            if not up: 
                  noise = noise*-1
            add[increase_start_index:increase_start_index+n_consecutive_outliers] = noise
            # add on the original data.
            new_df[x] += add
        return new_df


def add_event_group(
        df,
        event,
        increase_start_index=150
):
        # agnostic function to replace a certain part of the original ts with a set of events
        new_df  = df.copy()
        for n, x in enumerate(df.columns):
            add = new_df[x]
            add[increase_start_index:increase_start_index + len(event[n])] = event[n]
            # add on the original data.
            new_df[x] = add
        return new_df

def simple_noise_event(
        df,
        length, 
        std = 0.05,
        increase_start_index=150
        ):
        out = []
        for x in df: 
                mean = df[x][increase_start_index: increase_start_index+length].mean()
                out.append(np.random.normal(loc = mean, size= (length), scale= std))
        return out

def reverse_event(
        df,
        length, 
        increase_start_index=150
        ):
        out = []
        for x in df: 
                replace = np.flip(df[x][increase_start_index: increase_start_index+length].values)
                out.append(replace)
        return out



def model(
    train,
    test,
    exogs,
    ax,
    past_days=0,
    autoregressive=0,
    cols=["W"],
    plot_bases=True,
    naming="Bad model",
    text_pos=0.8,
    color = "orange"
):
    targetT, temporaryT = linear_pipeline(
        train,
        exogs,
        cols=cols,
        past_days=past_days,
        autoregressive=autoregressive,
        return_temporary=True,
    )
    m, min_max = linear_pipeline(
        train,
        exogs,
        cols=cols,
        past_days=past_days,
        autoregressive=autoregressive,
        return_temporary=False,
    )
    target, temporary = linear_pipeline(
        test,
        exogs,
        cols=cols,
        past_days=past_days,
        autoregressive=autoregressive,
        return_temporary=True,
        min_max=min_max,
    )
    pred = m.predict(temporary)
    predT = m.predict(temporaryT)
    if plot_bases:
        ax[2].hlines(0.05, 0, 50, color="red", linestyle="--", label="p=0.05 threshold")

        targetT.reset_index().plot.scatter(
            x="date", y=targetT.columns[0], ax=ax[0], label="Train", color="violet"
        )
        target[:40].reset_index().plot.scatter(
            x="date",
            y=target.columns[0],
            ax=ax[0],
            label="Normal Test",
            color="blue",
        )

        target[40:].reset_index().plot.scatter(
            x="date",
            y=target.columns[0],
            ax=ax[0],
            label="Anomalous Test",
            color="red",
        )

    pred.plot(ax=ax[0], color=color, linewidth=2)
    predT.plot(ax=ax[0], color=color, linewidth=2)

    train_resids = (predT - targetT.values[:, 0]).abs()
    test_resids = (pred - target.values[:, 0]).abs()

    ax[1].hist(test_resids[:40], label=naming, alpha=0.5, bins=10, color=color)

    # Plot train and test residuals as text
    ax[0].text(
        -1.5,
        text_pos,
        naming
        + f"\nTrain MAE: {train_resids.mean():.3f}\nTest MAE: {test_resids[:40].mean():.3f}",
        transform=ax[1].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8),
    )
    stack = []
    stack2 = []
    for x in range(50):
        # shuffle test
        p_value = ranksums(
            test_resids[:40].dropna(), test_resids.dropna()[40 + x : 40 + x + 50]
        ).pvalue
        stack.append(p_value)
        # TODO Other tests might be worth exploring here. Make this an option this strategy will be explored further.
        # bins = np.histogram_bin_edges(
        #     np.concatenate(
        #         [test_resids[:40].dropna(), test_resids.dropna()[40 + x : 40 + x + 50]]
        #     ),
        #     bins=10,
        # )
        # hist_p, _ = np.histogram(test_resids[:40].dropna(), bins=bins, density=True)
        # hist_q, _ = np.histogram(
        #     test_resids.dropna()[40 + x : 40 + x + 50], bins=bins, density=True
        # )
        # # Add a small value to avoid division by zero
        # hist_p += 1e-10
        # hist_q += 1e-10
        # stack2.append(entropy(hist_p, hist_q))

    #mean = test_resids[:40].mean()
    #std = test_resids[:40].std()
    #Z = (test_resids[40:] - mean) / std
    # Z.abs().cumsum().plot(ax=ax[1,1], label=naming + "Accumulated Z-Score")

    # Z.abs().plot(ax=ax[1,1], label=naming + "Accumulated Z-Score")
    #print(stack2)
    #ax[2].plot(stack2, label=naming, color=color)

    ax[2].plot(stack, label=naming, color=color)
    ax[2].set_title("Shuffle test p-values", fontsize=12)
    ax[0].set_title("Residuals for non-anomalous test data")
    ax[1].set_title("Distribution over absolute residuals \n during normal test window")




def display_anomaly(test_case, data):
    
    train = test_case[:200]
    test = test_case[200:]
    fig, ax = plt.subplots(figsize=(6, 3))
    data.rename(columns={"720593": "Train"}).interpolate()[:250].plot(ax=ax,linewidth=2, color="darkblue")
    data.rename(columns={"720593": "Original TS"}).interpolate()[250:].plot(ax=ax,linewidth=2, color="darkblue", alpha=0.4)

    test.rename(columns={"720593": "Anomalous TS"}).iloc[50:].plot(ax=ax, color="red", linewidth=2)
    test.rename(columns={"720593": "Calibration interval"}).iloc[:50].plot(ax=ax, color="green", linewidth=2)

    plt.title("Test Case: Lister, asc0, 720593")


    plt.vlines(
        x=train.index[-1],
        ymin=test_case.min().values[0],
        ymax=test_case.max().values[0],
        color="red",
        linestyle="--",
        label="Train/Test Split",
    )

    plt.vlines(
        x=test.index[50],
        ymin=test_case.min().values[0],
        ymax=test_case.max().values[0],
        color="red",
        linestyle="--",
        label="Train/Test Split",
    )
    plt.show()