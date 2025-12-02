import math
import warnings
warnings.filterwarnings('ignore')

import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

# Read data
def read_and_return_file():
    filename = "data.csv"

    df = pd.read_csv(filename)
    return df

# Print things about data
def print_data(dataframe, col="age"):
    print("Total number of rows: %d" % len(dataframe))

    print("Field names are: " + ', '.join(dataframe.columns))

    print("\nFirst 5 rows are:\n")

    print(dataframe.head())

    # Counting the number of confirmed smokers
    dataframe['smo_rev'] = pd.to_numeric(dataframe['smo_rev'], errors='coerce')
    print("The number of confirmed smokers is %d" % dataframe['smo_rev'].sum())

    # Calculating the mean of the requested column
    dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    print("%s mean for sample is: %.1f" % (col, dataframe[col].mean()))

# Function to plot two columns against each other
def lin_reg_plot_data(dataframe, col1, col2):
    dataframe[col1] = pd.to_numeric(dataframe[col1], errors='coerce')
    dataframe[col2] = pd.to_numeric(dataframe[col2], errors='coerce')
    dataframe_clean = dataframe.dropna(subset=[col1, col2])

    slope, intercept, r_value, p_value, std_err = \
        scipy.stats.linregress(dataframe_clean[col1], dataframe_clean[col2])
    X = np.linspace(dataframe_clean[col1].min(), dataframe_clean[col1].max())
    Y = X*slope + intercept
    print(f"Linear regression for {col1} as a function of {col2} resulted in a slope " + \
          f"of {slope:.2f}, R value of {r_value:.2f}, a standard error of {std_err:.5f} " + \
          f"and a P-value of {p_value:.5f} ")
    print(f"Plotting {col2} as a function of {col1}")
    plt.plot(X,Y, color="red")
    plt.scatter(dataframe_clean[col1], dataframe_clean[col2], s=1)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()

# Function to plot smoking
def plot_smoking(dataframe, col1, col2='smo_rev'):
    # Format data
    dataframe[col1] = pd.to_numeric(dataframe[col1], errors='coerce')
    dataframe[col2] = pd.to_numeric(dataframe[col2], errors='coerce')
    dataframe_clean = dataframe.dropna(subset=[col1, col2])
    values = [0,2]
    df = dataframe_clean[dataframe_clean[col2].isin(values)]
    df[col2] = df[col2].replace(2, 1)

    # Fit the model to data
    logreg = LogisticRegression(random_state=16)
    x_val = np.array(df[col1]).reshape(-1,1)
    y_val = np.array(df[col2])
    logreg.fit(x_val, y_val)
    X = np.linspace(x_val.min(), x_val.max(), 300).reshape(-1, 1)
    Y = logreg.predict(X)
    print(f"Logistic regression for {col1} as a function of {col2}." )
    print(f"Plotting {col2} as a function of {col1}")
    plt.plot(X,Y, color="red")
    plt.scatter(df[col1], df[col2], s=1)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()

# Function to perform ANOVA and t-test
def anova_and_ttest(dataframe, cohort=0):
    corresponding_cohort = {1: 'MKC', 2: 'SCAPIS', 3: 'LifeGene'}
    # corresponding_cohort = ['', 'MKC', 'SCAPIS', 'LifeGene']
    dataframe["Cohort"] = pd.to_numeric(dataframe["Cohort"], errors='coerce')
    if cohort != 0: # 0 for all cohorts
        dataframe = dataframe[dataframe["Cohort"] == cohort]
    col1 = 'SGHS1'
    col2 = 'SGHS2'
    IV = "smo_rev"
    dataframe[IV] = pd.to_numeric(dataframe[IV], errors='coerce')
    dataframe[col1] = pd.to_numeric(dataframe[col1], errors='coerce')
    dataframe[col2] = pd.to_numeric(dataframe[col2], errors='coerce')

    dataframe = dataframe.dropna(subset=[col1, col2])
    assert (len(pd.DataFrame(dataframe[col1]).index) == len(pd.DataFrame(dataframe[col2]).index))

    # Filter data and print nr of individuals in each group
    df_no_smoke = dataframe[dataframe[IV] == 0] # 0 corresponds to not a smoker
    df_former_smoke = dataframe[dataframe[IV] == 1] # 1 corresponds to a former smoker
    df_smoke = dataframe[dataframe[IV] == 2] # 1 corresponds to a smoker
    if cohort != 0:
        print(f"Analyzing cohort: {corresponding_cohort.get(cohort)}")
    else:
        print(f"Analyzing MKC, SCAPIS and LifeGene")
    print(f"Number of non-smokers: {len(df_no_smoke.index)}")
    print(f"Number of smokers: {len(df_smoke.index)}")
    print(f"Number of former smokers: {len(df_former_smoke.index)}")

    # Calaculate and print results of ANOVA and t-test
    anova_results1 = scipy.stats.f_oneway(df_former_smoke[col1], df_smoke[col1], df_no_smoke[col1])
    anova_results2 = scipy.stats.f_oneway(df_former_smoke[col2], df_smoke[col2], df_no_smoke[col2])
    results1 = scipy.stats.ttest_ind(df_smoke[col1], df_no_smoke[col1], equal_var=True)
    results2 = scipy.stats.ttest_ind(df_smoke[col2], df_no_smoke[col2], equal_var=True)
    print(f"Result of ANOVA for SGHS1: {anova_results1}")
    print(f"Result of ANOVA for SGHS2: {anova_results2}")
    print(f"Result of t-test for SGHS1 between smokers and non smokers: {results1}")
    print(f"Result of t-test for SGHS2 between smokers and non smokers: {results2}")
    print(f"SGHS1 means are: Former smokers {df_former_smoke[col1].mean():.3f}, smokers {df_smoke[col1].mean():.3f}, "+ \
          f"non smokers {df_no_smoke[col1].mean():.3f}")
    print(f"SGHS1 SEMs are: Former smokers {df_former_smoke[col1].sem():.3f}, smokers {df_smoke[col1].sem():.3f}, "+ \
          f"non smokers {df_no_smoke[col1].sem():.3f}")
    print(f"SGHS2 means are: Former smokers {df_former_smoke[col2].mean():.3f}, smokers {df_smoke[col2].mean():.3f}, "+ \
          f"non smokers {df_no_smoke[col2].mean():.3f}")
    print(f"SGHS2 SEMs are: Former smokers {df_former_smoke[col2].sem():.3f}, smokers {df_smoke[col2].sem():.3f}, "+ \
          f"non smokers {df_no_smoke[col2].sem():.3f}")

    # Plot stuff
    x = [f"F ({df_former_smoke[col1].count()})", f"C ({df_smoke[col1].count()})", f"N ({df_no_smoke[col1].count()})"]
    y = [df_former_smoke[col1].mean(), df_smoke[col1].mean(), df_no_smoke[col1].mean()]
    c = [df_former_smoke[col1].sem(), df_smoke[col1].sem(), df_no_smoke[col1].sem()]
    if cohort == 0:
        plt.bar(x, y, width=.3, facecolor="none", edgecolor='black', label="All")
    else:
        plt.bar(x, y, width=.3, facecolor="none", edgecolor='black', label=corresponding_cohort.get(cohort))
    plt.errorbar(x, y, yerr=c, fmt='o', color="black")
    plt.ylabel(col1, fontsize=14)

    color = {1:'r', 2:'g', 3:'b'}
    ytot = []
    ctot = []
    if cohort == 0:
        for i in [1,2,3]:
            x = []
            y = []
            c = []
            former = df_former_smoke[df_former_smoke['Cohort'] == i]
            current = df_smoke[df_smoke['Cohort'] == i]
            non = df_no_smoke[df_no_smoke['Cohort'] == i]

            x.extend([f"F ({former[col1].count()})", \
                      f"C ({current[col1].count()})", \
                      f"N ({non[col1].count()})"])
            y.extend([former[col1].mean(), current[col1].mean(), non[col1].mean()])
            ytot.extend([former[col1].mean(), current[col1].mean(), non[col1].mean()])
            c.extend([former[col1].sem(), current[col1].sem(), non[col1].sem()])
            ctot.extend([former[col1].sem(), current[col1].sem(), non[col1].sem()])
            c = np.array(c) * 1.96
            plt.bar(x, y, width=.3, facecolor="none", edgecolor=color[i], label=f"{corresponding_cohort.get(i)}")
            plt.errorbar(x, y, yerr=c, fmt='o', color="black")
        plt.ylim(min(ytot) - max(ctot), max(ytot) + max(ctot))

        plt.title(f"Mean of {col1} for current smokers (C), former smokers (F) and non-smokers (N) for all cohorts pooled together, and each respective cohort", fontweight="bold")
    else:
        plt.title(f"Mean of {col1} for  smokers (C), former smokers (F) and non-smokers (N) in {corresponding_cohort.get(cohort)}", fontweight="bold")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    dataframe = read_and_return_file()
    mediation_analysis(dataframe, )
    # print_data(dataframe, "HbA1c")
    # lin_reg_plot_data(dataframe, 'age', 'BMI')
    # plot_smoking(dataframe, 'age')
    # anova_and_ttest(dataframe, 3)

