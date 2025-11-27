import math
import scipy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
def plot_data(dataframe, col1, col2):
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

if __name__ == "__main__":
    dataframe = read_and_return_file()
    print_data(dataframe, "HbA1c")
    plot_data(dataframe, 'age', 'SGHS1')

