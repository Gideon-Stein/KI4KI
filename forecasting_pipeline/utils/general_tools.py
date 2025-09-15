import pandas as pd
import numpy as np


class param_holder:
    # Holder for model order
    def __init__(
        self,
        p=None,
        d=None,
        q=None,
        f=None,
        ex=None,
        mean=None,
        decompose=None,
        interaction=None,
        estimator=None,
        n_estimators=None,
        max_depth=None,
        learning_rate=None,
    ):
        self.p = p
        self.d = d
        self.q = q
        self.f = f
        self.ex = ex
        self.mean = mean
        self.decompose = decompose
        self.interaction = interaction
        self.scoring = {}
        self.model = None
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def print(self):
        return (
            (self.p, self.d, self.q),
            (self.f, self.ex, self.mean),
            (self.decompose, self.interaction),
            (self.estimator, self.n_estimators, self.learning_rate, self.max_depth),
        )

    def export(self):
        return [
            self.p,
            self.d,
            self.q,
            self.f,
            self.ex,
            self.mean,
            self.decompose,
            self.interaction,
            self.estimator,
            self.n_estimators,
            self.max_depth,
            self.learning_rate,
            self.scoring,
        ]  

def make_model_hp_list(
    P=[1],
    D=[0],
    Q=[1],
    F=[0],
    E=[[0]],
    M=[0],
    decompose=[0],
    interaction = [0],
    max_depth=[None],
    estimators=[None],
    learning_rate=[None],
    n_estimators=[None],
    remove_T = False,
    remove_W = False
):
    """
    Constructs a data grid for model hps.
    """
    combos = []
    for p in P:
        for d in D:
            for q in Q:
                for f in F:
                    for e1 in (E if not remove_W else [[]]):
                        for e2 in (E if not remove_T else [[]]):
                            for m1 in M:
                                for m2 in M:
                                    for dec in decompose:
                                        for inter in interaction:
                                            for est in estimators:
                                                for md in max_depth:
                                                    for lr in learning_rate:
                                                        for nes in n_estimators:
                                                            combos.append(
                                                                param_holder(
                                                                    p=p,
                                                                    d=d,
                                                                    q=q,
                                                                    f=f,
                                                                    ex=[e1, e2],
                                                                    mean=[m1, m2],
                                                                    decompose=dec,
                                                                    interaction = inter,
                                                                    max_depth=int(md) if ((md != "None") and (md)) else None,
                                                                    n_estimators=int(nes) if nes else None,
                                                                    learning_rate=float(lr) if lr else None,
                                                                    estimator=est if est else None,
                                                                )
                                                            )

    return combos


def pd_to_param_holder(best_params, p):
    combo = param_holder(
        p=best_params[best_params["Index"] == p].P.values[0],
        d=best_params[best_params["Index"] == p].D.values[0],
        q=best_params[best_params["Index"] == p].Q.values[0],
        f=best_params[best_params["Index"] == p]["F"].values[0],
        ex=[
            best_params[best_params["Index"] == p]["STAU"].values[0],
            best_params[best_params["Index"] == p]["T"].values[0],
        ],
        mean=[
            best_params[best_params["Index"] == p]["M_Stau"].values[0],
            best_params[best_params["Index"] == p]["M_T"].values[0],
        ],
        decompose=best_params[best_params["Index"] == p]["Decompose"].values[0],
        interaction=best_params[best_params["Index"] == p]["Interaction"].values[0] if "Interaction" in best_params.columns else 0,
        estimator=best_params[best_params["Index"] == p]["Estimator"].values[0] if "Estimator" in best_params.columns else 0,
        n_estimators=best_params[best_params["Index"] == p]["n_estimator"].values[0] if "n_estimator" in best_params.columns else 0,
        max_depth=best_params[best_params["Index"] == p]["max_depth"].values[0] if "max_depth" in best_params.columns else 0,
        learning_rate=best_params[best_params["Index"] == p]["learning_rate"].values[0] if "learning_rate" in best_params.columns else 0,
    )
    return combo


def txt_to_pandas(path):
    # parsing helper
    meta = open(path, "r+")
    meta = meta.read()
    data = [x.split(";") for x in np.array(meta.split("\n"))]
    return pd.DataFrame(data[1:], columns=data[0])


def results_to_pd(r):
    stack = []
    params = []
    for x in r:
        if isinstance(x[-1],list):
            continue
        else: # old formatting
            if "ALL" in x[1].keys():
                continue
            else:
                stack.append(pd.DataFrame(x[1]).T)
                for l in range(len(pd.DataFrame(x[1]).T)):
                    params.append(x[0][:4] + x[0][4] + x[0][5] + x[0][6:])
    stack = pd.concat(stack)
    stack.reset_index(inplace=True)
    out = pd.concat([stack, pd.DataFrame(params)], axis=1)

    # old version
    if len(out.columns) == 14:
        out.columns = [
            "Index",
            "MSE",
            "MAE",
            "AIC",
            "MAPE",
            "P",
            "D",
            "Q",
            "F",
            "STAU",
            "T",
            "M_Stau",
            "M_T",
            "Decompose"]




    elif len(out.columns) != 18:
        out.columns = [
            "Index",
            "MSE",
            "MAE",
            "MSE_1step",
            "MAE_1step",
            "AIC",
            "MAPE",
            "MAPE_1step",
            "P",
            "D",
            "Q",
            "F",
            "STAU",
            "T",
            "M_Stau",
            "M_T",
            "Decompose",
            "Interaction",
            "Estimator",
            "n_estimator",
            "max_depth",
            "learning_rate"]

    else:
        out.columns = [
            "Index",
            "MSE",
            "MAE",
            "MAPE",
            "P",
            "D",
            "Q",
            "F",
            "STAU",
            "T",
            "M_Stau",
            "M_T",
            "Decompose",
            "Interaction",
            "Estimator",
            "n_estimator",
            "max_depth",
            "learning_rate"
        ]

    return out
