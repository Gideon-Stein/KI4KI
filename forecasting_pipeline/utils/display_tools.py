import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from numpy import linspace
from matplotlib.pyplot import cm
from os import listdir
from os.path import isfile, join
from sklearn.metrics import r2_score


def display_points(holder, row_length=8, header="", maxiTick=2021, plot_length=15):
    # takes in table and tries to display all ts
    cols = holder.columns
    subG = len(cols)
    rows = math.ceil(subG / row_length)
    print(subG)

    # calc which indices are the start of the year
    tickPos = np.where(pd.to_datetime(holder.index.values).weekofyear == 26)[0]
    tickPos = [x for x in tickPos if x - 1 not in tickPos]
    tickLabels = np.arange(2015, maxiTick)

    if subG > row_length:
        fig, axs = plt.subplots(
            rows, row_length, figsize=(plot_length * row_length, 5 * rows)
        )
        count = 0
        for n, key in enumerate(cols):
            if (n % row_length == 0) and (n != 0):
                count += 1
            axs[count, n % row_length].plot(holder[key].interpolate().values)
            axs[count, n % row_length].scatter(
                np.arange(len(holder)), holder[key].values
            )
            axs[count, n % row_length].set_title(key, fontsize=25)
            axs[count, n % row_length].tick_params(
                axis="both", which="major", labelsize=20
            )

            axs[count, n % row_length].set_xticks(
                ticks=tickPos, labels=tickLabels, fontsize=20
            )
    else:
        fig, axs = plt.subplots(1, subG, figsize=(30, 5 * rows))
        if subG == 1:
            axs = np.array([axs])

        for n, key in enumerate(cols):
            axs[n].plot(holder[key].interpolate().values)
            axs[n].scatter(np.arange(len(holder)), holder[key].values)
            axs[n].set_title(key)
    for n in range(rows):
        if rows < 2:
            axs[n].set_ylabel("Distance from start in mm", fontsize=20)
        else:
            axs[n, 0].set_ylabel("Distance from start in mm", fontsize=20)
    for n in range(row_length):
        if rows < 2:
            axs[min(n, subG - 1)].set_xlabel("Year", fontsize=20)
        else:
            axs[rows - 1, n].set_xlabel("Year", fontsize=20)
    fig.suptitle("PCI points (" + header + ")", fontsize=25)
    return fig, axs


def remove_lines(axs):
    # remove line objects from figure
    if axs.ndim == 1:
        # wrap in list to account for single row
        axs = [axs]
    for x in axs:
        for y in x:
            if len(y.lines):
                y.lines[0].set_visible(False)


def display_predictions(
    d, forecasts, rmse, key="Radial", x_lim=[1200, 2140], alphas=[1, 1, 1, 1]
):

    if key != "Radial":
        tickPos = np.where(d.index.weekofyear == 1)[0]
        tickLabels = np.arange(2016, 2021)
    else:
        tickPos = np.where(d.index.dayofyear == 1)[0]
        tickLabels = np.arange(2016, 2022)

    cm_subsection = linspace(0, 1, 7)
    colors = iter([cm.Accent(x) for x in cm_subsection])
    colors = list(colors)

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    train = d.loc[d.index.year < 2020, key].interpolate()
    test = d.loc[d.index.year == 2020, key].interpolate()

    axs.scatter(
        np.arange(len(train)), train.values, label="Train Data", color="lightblue"
    )
    axs.scatter(
        np.arange(len(train), len(train) + len(test)),
        test,
        label="Test Data",
        color="lightgrey",
    )
    axs.set_xticks(tickPos, tickLabels)
    axs.set_title("Dammverformung (" + key + ")", fontsize=20)
    axs.tick_params(axis="both", which="major", labelsize=14)

    axs.set_xlim(x_lim)

    if len(forecasts.columns) > 0:
        axs.text(axs.get_xlim()[0] + 20, axs.get_ylim()[1] - 1, "MAE: ", weight="bold")

    for n, x in enumerate(forecasts.columns):

        axs.plot(
            np.arange(len(train), len(train) + len(test)),
            forecasts[x],
            color=colors[n],
            alpha=alphas[n],
            linewidth=3,
            label=x,
        )
        res = rmse.loc[x].values[0]
        axs.text(
            axs.get_xlim()[0] + 20,
            axs.get_ylim()[1] - (1.75 + 0.6 * n),
            x + ": " + str(round(res, 3)) + " mm",
        )

    axs.legend(loc=4)
    axs.set_ylabel("Deformation in mm", fontsize=12)
    axs.set_xlabel("Time", fontsize=12)

    # fig.savefig("img_1", dpi=300)
    plt.show()


def load_best_forecasts(origin, direction="asc0"):
    onlyfiles = [origin + "/" + f for f in listdir(origin) if isfile(join(origin, f))]
    onlyfiles = [x for x in onlyfiles if direction in x]
    forecasts = [x for x in onlyfiles if "forecast" in x]
    forecasts = {
        x.split("/")[-1].split("_")[2]: pd.read_csv(x, index_col=0) for x in forecasts
    }
    model_spec = [x for x in onlyfiles if "selected_model" in x]

    model_spec = {
        x.split("/")[-1].split("_")[2]: pd.read_csv(x, index_col=0) for x in model_spec
    }
    model_spec = pd.concat(model_spec)
    if direction == "lot":
        forecasts = pd.concat(forecasts, axis=1)
        forecasts.index = pd.to_datetime(forecasts.index)
        forecasts.columns = [x[0] for x in forecasts.columns.values]

    model_spec.index = [x[0] for x in model_spec.index.values]
    model_spec["Index_2"] = model_spec.index
    return model_spec, forecasts


def display_predictions_2(
    fig,
    axs,
    d,
    forecasts,
    y_,
    rmse,
    colors=None,
    residual_plot=True,
    key="Radial",
    x_lim=[1200, 2140],
    y_lim=[0, 5],
    r_2_round = 2,
    fontsize=20,
    spacing=0.6,
    add_errors=True,
    add_stats_to_residual_plot=False,
    error_coords=[10, 1.75, 1],
    provided_axes=None,
    shortening=True,
    ax2_inside=True,
    only_best_error=False,
    alphas=[1, 1, 1, 1],
    loc_2=[],
    rs_position=[5,-2],
    rs_fontsize=12,
    marker="o",
):

    if key != "Radial":
        tickPos = list(np.where(d.index.isocalendar().week == 1)[0])
        tickPos.append(tickPos[-1] + 66)
        tickLabels = np.arange(2016, 2022)
    else:
        tickPos = list(np.where(d.index.dayofyear == 1)[0])
        tickPos.append(tickPos[-1] + 365)

        tickLabels = np.arange(2016, 2022)


    train = d.loc[d.index.year < 2020, key].interpolate()
    test = d.loc[d.index.year == 2020, key].interpolate()

    axs.scatter(
        np.arange(len(train)),
        train.values,
        label="Train Data",
        color="lightblue",
        s=100,
        alpha=0.8,
        marker=marker,
    )
    axs.scatter(
        np.arange(len(train), len(train) + len(test)),
        test,
        label="Test Data",
        color="lightgrey",
        marker=marker,
        s=100,
    )
    axs.set_xticks(tickPos, tickLabels)
    axs.tick_params(axis="both", which="major", labelsize=20)

    axs.set_xlim(x_lim)
    if len(y_lim) > 0:
        if y_lim == "Auto":
            axs.set_ylim(test.min()-4,test.max()+4)
        else:
            axs.set_ylim(y_lim)

    if add_stats_to_residual_plot:
        to_add = provided_axes
    else:
        to_add = axs
   
    best = rmse.sort_values("MAE").index[0]

    r2 = r2_score(y_.values, forecasts[best].values)

    if residual_plot:

        left, bottom, width, height = loc_2
        if ax2_inside:
            ax2 = fig.add_axes([left, bottom, width, height])
        else:
            ax2 = provided_axes

        ax2.text(
            rs_position[0],
            rs_position[1],
            "R²: " + str(round(r2, r_2_round)),
            fontsize=rs_fontsize,
        )
        ax2.plot(
            np.arange(y_.values.min(), y_.values.max()),
            np.arange(y_.values.min(), y_.values.max()),
            color="black",
            linestyle="dashed",
        )
        ax2.scatter(
            y_.values, forecasts[best].values, color=colors[best], alpha=alphas[1]
        )
        ax2.set_xlabel("True")
        ax2.set_ylabel("Predicted")

    if (len(forecasts.columns) > 0) and add_errors:
            to_add.text(
                to_add.get_xlim()[0] + error_coords[0],
                to_add.get_ylim()[1] - ((to_add.get_ylim()[1] - to_add.get_ylim()[0]) / error_coords[2]) ,
                "MAE (mm): ",
                weight="bold",
                fontsize=fontsize,
            )


    for n, x in enumerate(forecasts.columns):
        name = x
        res = rmse.loc[x].values[0]
        if add_errors:
            if only_best_error:
                if (x==best):
                    to_add.text(
                    to_add.get_xlim()[0] + error_coords[0],
                    to_add.get_ylim()[1] - ((to_add.get_ylim()[1] - to_add.get_ylim()[0]) / error_coords[1]) - (spacing * 1),
                    (
                        name + ": " + str(round(res, 1))[:]
                        if not shortening
                        else name + ": " + str(round(res, 2))[1:]
                    ),
                    fontsize=fontsize,
                )


            else:
                to_add.text(
                    to_add.get_xlim()[0] + error_coords[0],
                    to_add.get_ylim()[1] - ((to_add.get_ylim()[1] - to_add.get_ylim()[0]) / error_coords[1]) - (spacing * n),
                    (
                        name + ": " + str(round(res, 2))[:]
                        if not shortening
                        else name + ": " + str(round(res, 2))[1:]
                    ),
                    fontsize=fontsize,
                )
        if x == best:
            axs.plot(
                np.arange(len(train), len(train) + len(test)),
                forecasts[x],
                color=colors[x],
                alpha=alphas[0] if (x != best) else alphas[1],
                linewidth=5,
                label=x,
            )