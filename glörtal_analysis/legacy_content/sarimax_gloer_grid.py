import argparse
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
from pandas import datetime
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import sys
sys.path.append("..")
from glörtal_analysis.legacy_content.gloer_tools import load_gloer, exog_gloer, weather_gloer, make_model_hp_list, find_best_order_gloer
from forecasting_pipeline.utils.filters import decompose_data

warnings.simplefilter('ignore', ConvergenceWarning)

# seed!
import numpy as np

np.random.seed(42)


def main():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument("--orderHigh", default=(3,2,3), type=int,nargs="+") #excluded
    parser.add_argument("--orderLow", default=(0,0,0), type=int,nargs="+")
    parser.add_argument("--fourierOrders", default=(0,), type=int,nargs="+")
    parser.add_argument("--meanOrders", default=(0,3), type=int,nargs="+")
    parser.add_argument("--exogSteps", default=(0,1,2), type=int,nargs="+")
    parser.add_argument("--savePath", default="/home/stein/project_repos/dam/full_baseline/saves_and_results/gloer_grid", type=str)
    parser.add_argument("--key", default="Full", type=str)
    parser.add_argument("--simple_split", action='store_true')
    parser.add_argument("--parallel", action='store_true')


    args = parser.parse_args()    
    start = datetime.now()


    isExist = os.path.exists(args.savePath)
    if not isExist:
        os.makedirs(args.savePath)
        print('New directory is created.')


    # Genereller Dataload. Nur einmal.

    data, meta = load_gloer(remove_last_year=False, interpolate_endog=True)
    stau, sicker = exog_gloer()
    weather = weather_gloer()



    P = np.arange(args.orderLow[0],args.orderHigh[0])
    D = np.arange(args.orderLow[1],args.orderHigh[1])
    Q = np.arange(args.orderLow[2],args.orderHigh[2])
    F = args.fourierOrders
    M = args.meanOrders
    E = [np.arange(x) for x in args.exogSteps + (args.exogSteps[-1]+1,)]
    print(P,D,Q,F,M,E)


    # generate all model combos to test each time
    combos = make_model_hp_list(P=P,D=D,Q=Q, F=F, M=M, E=E)
    print(len(combos))
                                    
    #go through all directions available
    if args.key == "Full":
        for x in data.keys():
            print("Evaluation: " +  str(x))
                #iterate through every point.
            results = find_best_order_gloer(combos=combos, endog=data[x], weather= weather, stau= stau, parallel=args.parallel, simple_split=args.simple_split)
            pickle.dump(results, open(args.savePath + "/" +  x + "_" +  "results_stack.p", "wb"))
    else:
        results = find_best_order_gloer(combos=combos, endog=data[args.key], weather= weather, stau= stau, parallel=args.parallel, simple_split=args.simple_split)
        pickle.dump(results, open(args.savePath + "/" +  args.key + "_" +  "results_stack.p", "wb"))    

    #print time    
    logs = [args, start, datetime.now()]
    print(logs[2] - logs[1])
    print("saving")

    print("Done.")
if __name__ == "__main__":
    main()
