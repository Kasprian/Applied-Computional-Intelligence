#%%
# Import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# FUNCTIONS TO DETECT OUTLIERS
#=============================
def plot_data_boxplots(df: pd.DataFrame, normalize: bool = False):
    # Copy data and remove string-columns
    data = df.copy()
    data.drop(columns=['Time (UTC)', 'Datetime'], inplace=True)

    # Normalize data if specified
    if normalize:
        data=(data-data.min())/(data.max()-data.min())

    # Plot boxplots
    plt.figure()
    data.boxplot()
    plt.show()

# Function which takes in a multiplier k and returns datapoints that are k*std away on any column of the 
# dataframe. k = 1 removes all but the outlier. k = 0.3 removes a decent amount, but not all. 
def detect_outliers(df: pd.DataFrame, k):
    # Copy data 
    data = df.copy()
    data.drop(columns=['Time (UTC)', 'Datetime'], inplace=True)

    # Calculate the standard deviation of each column and mean of each column
    df_mean = data.mean(axis=0)
    df_std = data.std(axis=0)
    
    # Return outliers that are k*std away
    condition_1 = data < df_mean-df_std*k
    condition_2 = data > df_mean+df_std*k
    data.where(condition_1 | condition_2, inplace=True)
    data.dropna(inplace=True)
    
    # Return a dataframe of the outliers
    return data

# FUNCTIONS TO REMOVE OUTLIERS
#=============================
def delete_outliers(df: pd.DataFrame, outliers: pd.DataFrame, inplace = False):
    if inplace:
        # Delete the line from the dataframe
        df.drop(outliers.index, inplace=True)
        return
    
    # Copy dataframe
    data = df.copy()

    # Delete the line from the copy of the dataframe
    data.drop(outliers.index, inplace=True)

    return data

def replace_outlier(df: pd.DataFrame, outliers: pd.DataFrame, inplace = False): 
    if inplace:
        # Assign non-time cells to NaN. Columns hardcoded because its easier for now
        df.iloc[outliers.index, 1:6] = np.NaN
        
        # Fill all NaN-values with the previous valid observation
        df.fillna(method='ffill', inplace=True)
        return

    # copy dataframe
    data = df.copy()

    # Assign non-time cells to NaN. Columns hardcoded because its easier for now
    data.iloc[outliers.index, 1:6] = np.NaN
    
    # Fill all NaN-values with the previous valid observation
    data.fillna(method='ffill', inplace=True)
    return data
        
def interpolate_outlier(df:pd.DataFrame, outliers: pd.DataFrame, inplace = False):
    if inplace:
        # Assign non-time cells to NaN. Columns hardcoded because its easier for now
        df.iloc[outliers.index, 1:6] = np.NaN
        
        # Fill all NaN-values with the previous valid observation
        df.iloc[:,1:6] = df.iloc[:,1:6].interpolate(axis=0)
        return
    
    # copy dataframe
    data = df.copy()

    # Assign non-time cells to NaN. Columns hardcoded because its easier for now
    data.iloc[outliers.index, 1:6] = np.NaN
    
    # Fill all NaN-values with the previous valid observation
    data.iloc[:,1:6] = data.iloc[:,1:6].interpolate(axis=0)
    return data

def main():
    # Import data
    raw_data = pd.read_csv("EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv", sep = ";", decimal=",")
    raw_data["Datetime"] = pd.to_datetime(raw_data["Time (UTC)"])
    
    # Detect outliers
    plot_data_boxplots(raw_data, normalize=True)
    outliers = detect_outliers(raw_data, k=0.3)
    print("=============================\nThe outliers are: \n", outliers, "\n=============================\n")

    # Delete outliers
    deleted_outliers = delete_outliers(raw_data, outliers, False)
    deleted_outliers.to_csv("results/deleted_outliers.csv")

    replaced_outliers = replace_outlier(raw_data, outliers, False)
    replaced_outliers.to_csv("results/replaced_outliers.csv")

    interpolated_outliers = interpolate_outlier(raw_data, outliers, False)
    interpolated_outliers.to_csv("results/interpolated_outliers.csv")

    # Plot the transformed data sets:
    plot_data_boxplots(deleted_outliers, normalize=True)
    plot_data_boxplots(replaced_outliers, normalize=True)
    plot_data_boxplots(interpolated_outliers, normalize=True)


#%%
main()    
# %%
