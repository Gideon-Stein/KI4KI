import numpy as np
import pandas as pd
import statsmodels.api as sm


def forecast_test_linear(endog_test, exog_test, combo, model, trend_step, return_temporary=False):
    temporary_test = exog_test.copy()
    for value in np.arange(1, combo.p + 1):
        temporary_test[str(value)] = endog_test.shift(value)
    temporary_test["const"] = 1.0
    # trend has to continue from train onwards
    # there are some numerical inonsistencies here
    # so we add more trend than we need and remove the not used indices
    temporary_test["trend"] = np.arange(
        1 + trend_step,
        1 + ((len(temporary_test["const"]) + 2) * trend_step),
        trend_step,
    )[: len(temporary_test)]

    if return_temporary:
        return temporary_test
    else:
        pred = model.predict(temporary_test.fillna(method="backfill"))
        pred = pd.DataFrame(pred, index=endog_test.index)
        return pred


def forecast_test_forest(endog_test, exog_test, combo, model):
    temporary_test = exog_test.copy()
    for value in np.arange(1, combo.p + 1):
        temporary_test[str(value)] = endog_test.shift(value)
    pred = model.predict(temporary_test.fillna(method="backfill"))
    pred = pd.DataFrame(pred, index=endog_test.index)
    return pred


def forecast_test_ada(endog_test, exog_test, combo, model):
    temporary_test = exog_test.copy()
    for value in np.arange(1, combo.p + 1):
        temporary_test[str(value)] = endog_test.shift(value)
    pred = model.predict(temporary_test.fillna(method="backfill"))
    pred = pd.DataFrame(pred, index=endog_test.index)
    return pred


# # Autoregressive effects. # TODO Finish in necessary.

# endog_test, exog_test = test
# temporary_test = exog_test.copy()
# temporary_test["const"] = 1.0
# temporary_test["trend"] = np.arange(1,1 + (len(temporary_test["const"])* trend),trend)

# for value in np.arange(1, combo.p + 1):
#     temporary_test[str(value)] = endog_test.shift(value)
# temporary_test= temporary_test.fillna(method="backfill")


# prediction = []
# for x in range(1,4):
#     res = model.predict(temporary_test[x-1:x].fillna(method="backfill")).values
#     print(res)
#     temporary_test.loc[x:x+1, "1"] = res[0]
#     prediction.append(res[0])
