# Import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from typing import List

def time_price_plot(data: List[pd.DataFrame]):
    # Initialize figure
    plt.figure()
    # Iterate through the elements of the input-data-list.
    # Assuming data is a pd.Dataframe with one column being a string-date and the other being the value to plot
    for elem in data: 
        # Extract x as datetime and y as is
        x = [datetime.strptime(x, '%Y-%m-%d') for x in elem.iloc[:,0].values]
        y = elem.iloc[:,1].values

        # Generate line
        plt.plot(x,y, label = elem.columns.tolist()[1])
    
    # Initialize legends and show figure
    plt.legend()
    plt.show()

def scatterplot(data1: pd.DataFrame, data2: pd.DataFrame):
    # Initializing figure
    plt.figure()

    # Getting the two data-sequences
    y1 = data1.iloc[:, 1].values
    y2 = data2.iloc[:, 1].values

    # Matching dimensions that dont match by removing excess items through a merge
    if y1.shape[0] != y2.shape[0]:
        merged_array = pd.merge(data1,data2, left_index=True, right_index=True)
        y1 = merged_array.iloc[:,1]
        y2 = merged_array.iloc[:, 3]

    # Plotting the scatterplot
    plt.scatter(y1, y2)
    plt.xlabel(data1.columns.tolist()[1])
    plt.ylabel(data2.columns.tolist()[1])
    plt.show()

def main():
    # import data
    raw_data_brent = pd.read_csv("DCOILBRENTEUv2.csv" , decimal='.', sep=',')
    raw_data_wit = pd.read_csv("DCOILWTICOv2.csv", decimal='.', sep=',')

    # Plot data against eachother
    time_price_plot([raw_data_brent, raw_data_wit])

    # Create a scatterplot
    scatterplot(raw_data_brent, raw_data_wit)

    return

main()