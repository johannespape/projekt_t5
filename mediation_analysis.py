import warnings

import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

from test import read_and_return_file
from scipy.stats import norm
from tqdm import tqdm


def mediation_analysis_bootstrap(df, B=10, X='smo_rev', Y='SGHS1', M='HbA1c', confounders={}):
    # Make data numeric
    df[X] = pd.to_numeric(df[X], errors='coerce')
    df[Y] = pd.to_numeric(df[Y], errors='coerce')
    df[M] = pd.to_numeric(df[M], errors='coerce') # M is required to be metric for mediation analysis 
    df[M] = df[M] - df[M].mean() # Center M
    for key in confounders:
        df[key] = pd.to_numeric(df[key], errors='coerce')
        if not confounders[key]: # Center continous confounders
            df[key] = df[key] - df[key].mean()

    # Remove rows with missing values
    variables = [key for key in confounders]
    variables.extend([X, Y])
    df = df.dropna(subset=variables)

    # Method using bootstrapped CI
    B = 100  # number of bootstrap samples
    boot_indirect1 = []
    boot_indirect2 = []

    M_regression_string = f"{M} ~ C({X})"
    Y_regression_string = f"{Y} ~ C({X}) + {M}"
    for key in confounders:
        if confounders[key]:
            M_regression_string += f" + C({key})"
            Y_regression_string += f" + C({key})"
        else:
            M_regression_string += f" + {key}"
            Y_regression_string += f" + {key}"

    for _ in tqdm(range(B)):
        # Resample with replacement
        sample = df.sample(n=len(df), replace=True)

        # Mediator model
        med_model = smf.ols(M_regression_string, data=sample).fit()
        a1 = med_model.params[f"C({X})[T.1.0]"]  # example for former smoker vs non-smoker
        a2 = med_model.params[f"C({X})[T.2.0]"]  # example for former smoker vs non-smoker
        # (you can loop over all levels if categorical has >2)

        # Outcome model
        out_model = smf.ols(Y_regression_string, data=sample).fit()
        b = out_model.params[f"{M}"]

        # Indirect effect
        boot_indirect1.append(a1 * b) # Former smoker
        boot_indirect2.append(a2 * b) # Current smoker

    # Convert to numpy array
    boot_indirect1 = np.array(boot_indirect1)
    boot_indirect2 = np.array(boot_indirect2)

    # 95% percentile CI
    lower1 = np.percentile(boot_indirect1, 2.5)
    upper1 = np.percentile(boot_indirect1, 97.5)
    lower2 = np.percentile(boot_indirect2, 2.5)
    upper2 = np.percentile(boot_indirect2, 97.5)
    mean_indirect1 = np.mean(boot_indirect1)
    mean_indirect2 = np.mean(boot_indirect2)

    print(f"Indirect effect of former smoker on {Y}: {mean_indirect1:.5f}")
    print(f"95% CI: [{lower1:.5f}, {upper1:.5f}]")
    print(f"Indirect effect of current smoker on {Y}: {mean_indirect2:.5f}")
    print(f"95% CI: [{lower2:.5f}, {upper2:.5f}]")


def mediation_analysis_sobel_test(df, X='smo_rev', Y='SGHS1', M='HbA1c', confounders={}):
    # Make data numeric
    df[X] = pd.to_numeric(df[X], errors='coerce')
    df[Y] = pd.to_numeric(df[Y], errors='coerce')
    df[M] = pd.to_numeric(df[M], errors='coerce') # M is required to be metric for mediation analysis 
    df[M] = df[M] - df[M].mean() # Center M
    for key in confounders:
        df[key] = pd.to_numeric(df[key], errors='coerce')
        if not confounders[key]: # Center continous confounders
            df[key] = df[key] - df[key].mean()

    # Remove rows with missing values
    variables = [key for key in confounders]
    variables.extend([X, Y])
    df = df.dropna(subset=variables)

    # Step 1: Regress M on X
    regression_string = f"{M} ~ C({X})"
    for key in confounders:
        if confounders[key]:
            regression_string += f" + C({key})"
        else:
            regression_string += f" + {key}"
    model_a = smf.ols(regression_string, data=df).fit()
    print(model_a.summary())
    print("\n"*2)

    # Step 2: Regress Y on X (total effect)
    regression_string = f"{Y} ~ C({X})"
    for key in confounders:
        if confounders[key]:
            regression_string += f" + C({key})"
        else:
            regression_string += f" + {key}"
    model_c = smf.ols(regression_string, data=df).fit()
    print(model_c.summary())
    print("\n"*2)

    # Step 3: Regress Y on X and M (direct effect)
    regression_string = f"{Y} ~ C({X}) + {M}"
    for key in confounders:
        if confounders[key]:
            regression_string += f" + C({key})"
        else:
            regression_string += f" + {key}"
    model_c_prime = smf.ols(regression_string, data=df).fit()
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

    keys = str(confounders.keys())[10:-1]
    print(f"Mediation analysis adjusted for {keys}")
    print(f"Indirect effect of former smoker on {Y} through {M}: {a1*b} ({a1*b-SE_a1b*1.96}, {a1*b+SE_a1b*1.96})")
    print(f"Indirect effect of current smoker on {Y} through {M}: {a2*b} ({a2*b-SE_a2b*1.96}, {a2*b+SE_a2b*1.96})")

    print("\n")
    print("Sobel test z for former smoker:", z_value_a1)
    print("p-value for former smoker:", p_value_a1)
    print("\n")

    print("Sobel test z for current smoker:", z_value_a2)
    print("p-value for current smoker:", p_value_a2)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    dataframe = read_and_return_file()
    mediation_analysis_sobel_test(dataframe, Y='SGHS1', M='BMI', confounders={'age':0, 'sex':1, 'AntiHT':1, 'Antilipid':1})
    # mediation_analysis_bootstrap(dataframe, B=100, Y='SGHS1', M='HbA1c', confounders={'age':0, 'sex':1, 'AntiHT':1, 'Antilipid':1})

