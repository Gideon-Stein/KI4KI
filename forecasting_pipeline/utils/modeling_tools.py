import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

import numpy as np

# Keeps track of the modeling of each evaluated modelclass


def perform_modeling_varmax(
    endog,
    exog,
    combo,
):
    if len(exog.columns) != 0:
        mod = sm.tsa.VARMAX(
            endog.values, exog, trend="ct", order=(combo.p, combo.q)
        )  # ct mi
    else:
        mod = sm.tsa.VARMAX(endog.values, trend="ct", order=(combo.p, combo.q))
    # sometimes we need to catch because fitting crashes
    try:
        return mod.fit(disp=False)
    except:
        return None


def perform_modeling_sarimax(
    endog,
    exog,
    combo,
):
    if len(exog.columns) != 0:
        mod = sm.tsa.statespace.SARIMAX(
            endog,
            exog,
            order=(combo.p, combo.d, combo.q),
            trend="ct",
            use_exact_diffuse=True,
        )
    else:
        mod = sm.tsa.statespace.SARIMAX(
            endog,
            order=(combo.p, combo.d, combo.q),
            trend="ct",
            use_exact_diffuse=True,
        )
    # sometimes we need to catch because fitting crashes
    try:
        return mod.fit(disp=False)
    except:
        return None


def perform_modeling_forest(endog, exog, combo):
    temporary = exog.copy()
    for value in np.arange(1, combo.p + 1):
        temporary[str(value)] = endog.shift(value)
    regr = RandomForestRegressor(
        n_estimators=combo.n_estimators,
        max_depth=int(combo.max_depth) if isinstance(combo.max_depth, int) else None,
        random_state=0,
    )
    regr.fit(temporary.fillna(method="backfill").values, endog.values)
    return regr


def perform_modeling_ada(endog, exog, combo):
    temporary = exog.copy()
    for value in np.arange(1, combo.p + 1):
        temporary[str(value)] = endog.shift(value)

    if combo.estimator == "dt":
        base_est = DecisionTreeRegressor(random_state=0)
    elif combo.estimator == "linear":
        base_est = LinearRegression()
    else:
        print("Estimator not found.")

    regr = AdaBoostRegressor(
        estimator=base_est,
        n_estimators=combo.n_estimators,
        learning_rate=combo.learning_rate,
        loss="linear",
        random_state=0,
    )
    regr.fit(temporary.fillna(method="backfill").values, endog.values)
    return regr


def perform_modeling_linear(
    endog, exog, combo, return_temporary=False  # For data double check
):
    temporary = exog.copy()
    # add autoregressive components as additional variables
    for value in np.arange(1, combo.p + 1):
        temporary[str(value)] = endog.shift(value).values
    temporary["const"] = 1.0
    temporary["trend"] = np.arange(0, 1, 1 / len(temporary["const"]))
    
    if return_temporary:
        return temporary
    else:
        olsmod = sm.OLS(
            exog=temporary.fillna(method="backfill").values,
            endog=endog.values,
        )
        olsres = olsmod.fit()
        return olsres, 1 / len(temporary["const"])