import pathlib
import pandas as pd
from os import listdir
import numpy as np
import copy
from pathlib import Path
import pickle




def get_baseline_external(where, path):
    # Loads the weather external variables and formats them
    rv = dl_RV(path)
    w = rv[where]["Stau"][["datetime", "W"]]
    t = rv["alleStauanlagen"]["temperatur"][["datetime", where]]
    w["datetime"] = w["datetime"].dt.round("D")
    out = w.merge(t, on="datetime", how="outer").sort_values("datetime")
    out.reset_index(drop=True, inplace=True)
    out.rename(columns={where: "T"}, inplace=True)
    out.drop_duplicates(inplace=True)
    return out


def txt_to_pandas(path):
    # parsing helper
    meta = open(path, "r+")
    meta = meta.read()
    data = [x.split(";") for x in np.array(meta.split("\n"))]
    return pd.DataFrame(data[1:], columns=data[0])


def load_location(path, loc):
    # location put and load all 4 provided variations
    data = {}
    path = pathlib.Path() / path
    data["ew"] = txt_to_pandas(path / loc / (loc + "_EastWest.txt"))
    data["ver"] = txt_to_pandas(path / loc / (loc + "_Vertical.txt"))

    name = [
        x for x in listdir(path / loc / "Descending_LineOfSight") if x[-4:] == ".txt"
    ]
    assert len(name) != 0, "No files detected!"
    for n, x in enumerate(name):
        data["desc" + str(n)] = txt_to_pandas(path / loc / "Descending_LineOfSight" / x)

    name = [
        x for x in listdir(path / loc / "Ascending_LineOfSight") if x[-4:] == ".txt"
    ]
    assert len(name) != 0, "No files detected!"
    for n, x in enumerate(name):
        data["asc" + str(n)] = txt_to_pandas(path / loc / "Ascending_LineOfSight" / x)
    return data


def load_all_locations(path):
    # Load all PCI points from data
    locs = listdir(path)
    meta_complete = {}
    data_complete = {}
    for x in locs:
        raw = load_location(path, x)
        meta_info = {}
        location = {}
        for variant in raw.keys():
            data, meta = dl_to_raw_ts_new(raw[variant])
            location[variant] = data
            meta_info[variant] = meta
        data_complete[x] = location
        meta_complete[x] = meta_info
    return data_complete, meta_complete


def dl_RV(path):
    # Load and parse the original external variables
    data_stack = {}
    dp = pathlib.Path() / path 
    files = pathlib.Path(dp).glob("**/*")
    files = [x for x in files if x.is_file()]
    for x in files:
        if x.suffix == ".zip":
            continue
        n = x.stem.split("_")
        if n[0] not in data_stack.keys():
            data_stack[n[0]] = {n[1]: pd.read_excel(x, engine="openpyxl")}
        else:
            data_stack[n[0]][n[1]] = pd.read_excel(x, engine="openpyxl")
    # Some cleaning
    for x in data_stack:
        for y in data_stack[x]:
            if x == "alleStauanlagen":
                data_stack[x][y].columns = data_stack[x][y].iloc[0].values
                data_stack[x][y] = data_stack[x][y].iloc[1:]
                data_stack[x][y]["datetime"] = pd.to_datetime(
                    data_stack[x][y]["Datum:"]
                ).values
                data_stack[x][y].drop(columns=["Datum:"], inplace=True)
                data_stack[x][y].replace(" ---", np.nan, inplace=True)
                for column in data_stack[x][y].columns:
                    if column == "datetime":
                        pass
                    else:
                        data_stack[x][y][column]
                        data_stack[x][y][column] = data_stack[x][y][column].astype(
                            float
                        )
                        data_stack[x][y].rename(
                            columns={column: column[:-1]}, inplace=True
                        )

                data_stack[x][y].rename(
                    columns={"Möhne": "Moehne", "Fürwigge": "Fuerwigge"}, inplace=True
                )
            else:
                data_stack[x][y].drop(
                    columns=["Kurzbezeichnung", "Objektname", "Messpunktname"],
                    inplace=True,
                )
                data_stack[x][y].rename(
                    columns={
                        "Datum/Uhrzeit": "datetime",
                        "Wasserstand (NHN) [m NHN]": "W",
                    },
                    inplace=True,
                )
                if "W" in data_stack[x][y].columns:
                    data_stack[x][y]["W"] = data_stack[x][y]["W"].astype(float)
    return data_stack


def get_lot(
    p="../raw_data/Ruhrverband/Daten/Moehne/Moehne_Lotanlage.xlsx",
):
    lot = pd.read_excel(p, engine="openpyxl")
    lot.columns = ["a", "b", "c", "d", "e", "f"]
    lot = lot[["a", "e", "f"]]
    lot["a"] = pd.to_datetime(lot["a"].astype(str).str[:-9])
    print(lot["a"])
    lot = lot[lot["a"] >= "2015-04-01"]
    lot = lot[lot["a"] < "2021-01-01"]

    lot.index = lot["a"]
    lot.drop(columns=["a", "f"], inplace=True)
    lot.index.name = "datetime"
    lot.columns = ["Radial"]
    lot = lot[~lot.index.duplicated(keep="first")]
    return {"Moehne": {"lot": lot}}


def load_all_external(locations, path):
    return {x: get_baseline_external(x, path) for x in locations}


def dl_to_raw_ts_new(
    table,
    keyname="OBJECTID",
    date_identifier="date_",
    remove_from_date="date_",
    # removes some preliminary string before date information if necessary
    rename_id_columns={"PS_ID": "ID"},
):
    # Parses the original data to Dataframes
    # takes in table from dl_bbd
    # replace tokens for parsing
    table.replace(",", ".", regex=True, inplace=True)
    table.replace("", None, regex=True, inplace=True)
    # Seperate meta from data
    a = [x for x in table.columns if date_identifier not in x]
    meta = copy.deepcopy(table[a])
    meta.loc[:, "X"] = pd.to_numeric(meta.loc[:, "X"]).values
    meta.loc[:, "Y"] = pd.to_numeric(meta.loc[:, "Y"]).values
    meta.loc[:, "Z"] = pd.to_numeric(meta.loc[:, "Z"]).values
    meta.rename(columns=rename_id_columns, inplace=True)

    key = table[keyname].values
    # get all dates
    b = [x for x in table.columns if date_identifier in x]
    # build data range to have no missing timestamps
    index = pd.to_datetime([x.split(remove_from_date)[-1] for x in b])
    spread = pd.DataFrame(
        pd.date_range(start=index.values[0], end=index.values[-1], freq="6D"),
        columns=["datetime"],
    )
    assert np.all(
        index.isin(spread.values[:, 0])
    ), "time spread is not matching time stamps!"
    # merge and return
    table = pd.DataFrame(
        table[b].astype(float).values.T.astype(float), index=index, columns=key
    )
    table = spread.merge(table, left_on="datetime", right_index=True, how="outer")
    table.index = table.datetime
    table.drop(columns=["datetime"], inplace=True)
    return table, meta


def load_data_bases(cfg):
    # Load the complete data (So we only need to load the data once)
    if cfg.dir == "lot":
        if Path("../saves_and_results/cache_lot.p").is_file():
            data = pickle.load(open("../saves_and_results/cache_lot.p", "rb"))
            print("Load cached data")
        else:
            data = get_lot()
            pickle.dump(data, open("../saves_and_results/cache_lot.p", "wb"))
    else:
        if Path("../saves_and_results/cache_endog.p").is_file():
            data = pickle.load(open("../saves_and_results/cache_endog.p", "rb"))
            print("Load cached data")
        else:
            data, meta = load_all_locations(path=cfg.source_path + "/Datenpaket_BBD")
            pickle.dump(data, open("../saves_and_results/cache_endog.p", "wb"))
    if Path("../saves_and_results/cache_exogs.p").is_file():
        exogs = pickle.load(open("../saves_and_results/cache_exogs.p", "rb"))
        print("Load cached data")
    else:
        exogs = load_all_external(locations=cfg.all_locs, path=cfg.source_path + "/Ruhrverband")
        pickle.dump(exogs, open("../saves_and_results/cache_exogs.p", "wb"))
    return data, exogs