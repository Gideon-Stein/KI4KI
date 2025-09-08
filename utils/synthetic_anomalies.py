from utils.filters import detrend_via_model
import numpy as np
import scipy.optimize
import scipy



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
