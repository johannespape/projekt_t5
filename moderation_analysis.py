import warnings

import statsmodels.formula.api as smf
import pandas as pd
# import numpy as np

from test import read_and_return_file
# from scipy.stats import norm


def moderation_analysis(df, X='smo_rev', Y='SGHS1', M='HbA1c', M_categorical=False, confounders={}):
    # Make data numeric
    df[X] = pd.to_numeric(df[X], errors='coerce')
    df[Y] = pd.to_numeric(df[Y], errors='coerce')
    df[M] = pd.to_numeric(df[M], errors='coerce')
    for key in confounders:
        df[key] = pd.to_numeric(df[key], errors='coerce')
        if not confounders[key]: # Center continous confounders
            df[key] = df[key] - df[key].mean()

    # Remove rows with missing values
    variables = [key for key in confounders]
    variables.extend([X, Y])
    df = df.dropna(subset=variables)

    if M_categorical:
        # Construct regression string
        regression_string = f'{Y} ~ C({X}) * C({M})'
        for key in confounders:
            if confounders[key]:
                regression_string += f" + C({key})"
            else:
                regression_string += f" + {key}"
        # Regress Y on X, M and M* X
        model_a = smf.ols(regression_string, data=df).fit()
        print(model_a.summary())
        print(f'C({X})[T.1.0]:C({M})[T.1.0]', model_a.params[f'C({X})[T.1.0]:C({M})[T.1.0]'])
        print(f'C({X})[T.1.0]:C({M})[T.2.0]', model_a.params[f'C({X})[T.1.0]:C({M})[T.2.0]'])
        print(f'C({X})[T.2.0]:C({M})[T.1.0]', model_a.params[f'C({X})[T.2.0]:C({M})[T.1.0]'])
        print(f'C({X})[T.2.0]:C({M})[T.2.0]', model_a.params[f'C({X})[T.2.0]:C({M})[T.2.0]'])
    else:
        # Step 1: Center moderator M
        df[M] = df[M] - df[M].mean()

        # Construct regression string
        regression_string = f'{Y} ~ C({X}) * {M}'
        for key in confounders:
            if confounders[key]:
                regression_string += f" + C({key})"
            else:
                regression_string += f" + {key}"

        # Step 2: Regress Y on X, M and M_centered * X
        model_a = smf.ols(regression_string, data=df).fit()
        print(model_a.summary())
        print(f'C({X})[T.1.0]:{M}', model_a.params[f'C({X})[T.1.0]:{M}'])
        print(f'C({X})[T.2.0]:{M}', model_a.params[f'C({X})[T.2.0]:{M}'])

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    dataframe = read_and_return_file()
    moderation_analysis(dataframe, Y='SGHS1', M='BMI', M_categorical=1, confounders={'age':0, 'sex':1, 'AntiHT':1, 'Antilipid':1})

