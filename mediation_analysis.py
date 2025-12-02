import warnings

import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import pingouin as pg

from test import read_and_return_file
from scipy.stats import norm


def mediation_analysis(df, X='smo_rev', Y='SGHS1', M='HbA1c'):
    # Make data numeric
    df[X] = pd.to_numeric(df[X], errors='coerce')
    df[Y] = pd.to_numeric(df[Y], errors='coerce')
    df[M] = pd.to_numeric(df[M], errors='coerce')

    # Step 1: Regress M on X
    model_a = smf.ols(f'{M} ~ C({X})', data=df).fit()
    print(model_a.summary())
    print("\n"*2)

    # Step 2: Regress Y on X (total effect)
    model_c = smf.ols(f'{Y} ~ C({X})', data=df).fit()
    print(model_c.summary())
    print("\n"*2)

    # Step 3: Regress Y on X and M (direct effect)
    model_c_prime = smf.ols(f'{Y} ~ C({X}) + {M}', data=df).fit()
    print(model_c_prime.summary())
    print("\n"*2)

    # Indirect effect = effect of X on M * effect of M on Y
    a1 = model_a.params[f'C({X})[T.1.0]']  # effect of former smoker on M
    a2 = model_a.params[f'C({X})[T.2.0]']  # effect of smoker on M
    b = model_c_prime.params[f'{M}']     # effect of M on Y

    # Standard errors
    SE_a1 = model_a.bse[f'C({X})[T.1.0]']
    SE_a2 = model_a.bse[f'C({X})[T.2.0]']
    SE_b = model_c_prime.bse[f'{M}']

    # Sobel test
    SE_a1b = np.sqrt(b**2 * SE_a1**2 + a1**2 * SE_b**2)
    SE_a2b = np.sqrt(b**2 * SE_a2**2 + a2**2 * SE_b**2)
    z_value_a1 = (a1 * b) / SE_a1b
    z_value_a2 = (a2 * b) / SE_a2b
    p_value_a1 = 2 * (1 - norm.cdf(abs(z_value_a1)))
    p_value_a2 = 2 * (1 - norm.cdf(abs(z_value_a2)))

    print(f"Indirect effect of former smoker on {Y} through {M}: {a1*b} ({a1*b-SE_a1b*1.96}, {a1*b+SE_a1b*1.96})")
    print(f"Indirect effect of current smoker on {Y} through {M}: {a2*b} ({a2*b-SE_a2b*1.96}, {a2*b+SE_a2b*1.96})")
    print(f"Effect of {M} on {Y}:", b)

    print("\n")
    print("Sobel test z for former smoker:", z_value_a1)
    print("p-value for former smoker:", p_value_a1)
    print("\n")

    print("Sobel test z for current smoker:", z_value_a2)
    print("p-value for current smoker:", p_value_a2)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    dataframe = read_and_return_file()
    mediation_analysis(dataframe, Y='SGHS1', M='TG')

