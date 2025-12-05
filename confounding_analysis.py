import warnings

import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

from test import read_and_return_file
from scipy.stats import norm


def confounding_analysis(df, X='smo_rev', Y='SGHS1', confounders={}):
    # Make data numeric
    df[X] = pd.to_numeric(df[X], errors='coerce')
    df[Y] = pd.to_numeric(df[Y], errors='coerce')
    for key in confounders:
        df[key] = pd.to_numeric(df[key], errors='coerce')
        if not confounders[key]: # Center continous confounders
            df[key] = df[key] - df[key].mean()

    # Remove rows with missing values
    variables = [key for key in confounders]
    variables.extend([X, Y])
    df = df.dropna(subset=variables)

    # Construct regression string
    regression_string = f"{Y} ~ C({X})"
    for key in confounders:
        if confounders[key]:
            regression_string += f" + C({key})"
        else:
            regression_string += f" + {key}"
    # model_a = smf.ols(f'{Y} ~ C({X})', data=df).fit()
    # print(model_a.summary())
    model_b = smf.ols(regression_string, data=df).fit()
    print(model_b.summary())
    keys = str(confounders.keys())[10:-1]
    print(f"Regression of {Y} on {X} adjusting for {keys}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    dataframe = read_and_return_file()
    # Model 1
    confounding_analysis(dataframe, confounders={'age':0, 'sex':1})
    # Model 2
    # confounding_analysis(dataframe, confounders={'age':0, 'sex':1, 'AntiHT':1, 'Antilipid':1})
    # Model 3
    # confounding_analysis(dataframe, confounders={'age':0, 'sex':1, 'AntiHT':1, 'Antilipid':1, \
    #                                              'BMI':0, 'SBP':0, 'DBP':0, 'TG':0, 'Glucose':0, 'HbA1c':0, 'LDL':0})

