import copy
import math
import pathlib
import warnings
from functools import reduce
from multiprocessing import cpu_count
from os import listdir

import matplotlib.pyplot as plt
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_percentage_error

warnings.simplefilter(action="ignore", category=FutureWarning)
import warnings

import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
# This collection might contain duplicates with tools.
# seed!
import numpy as np
from statsmodels.tsa.deterministic import Fourier
from sklearn.ensemble import RandomForestRegressor
from utils.filters import (
    apply_filter,
    custom_fourier_terms,
    decompose_data,
    detrend_via_model,
    simple_boxplot_filter,
)






