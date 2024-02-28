import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import os
from scipy.stats import anderson
import numpy as np
import seaborn as sns


# Read the CSV file into a DataFrame
def read_csv(file_path):
    return pd.read_csv(file_path)

# Clean the DataFrame
def clean_data(data):
    # Drop rows with missing values
    data_cleaned = data.dropna()
    
    # Convert the 'Date' column to a datetime object
    data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], format='%d-%m-%Y')

    # Remove outliers
    numeric_columns = data_cleaned.select_dtypes(include='number').columns
    
    for column in numeric_columns:
        # Calculate the first quartile (Q1) and third quartile (Q3)
        Q1 = data_cleaned[column].quantile(0.25)
        Q3 = data_cleaned[column].quantile(0.75)
        
        # Calculate the interquartile range (IQR)
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers from the column
        data_cleaned = data_cleaned[(data_cleaned[column] >= lower_bound) & (data_cleaned[column] <= upper_bound)]
    
    return data_cleaned


def clean_out_features(data, features):
    # Check if the features exist in the DataFrame before dropping them
    existing_features = [col for col in features if col in data.columns]
    
    # Drop only the existing features
    data_cleaned = data.drop(existing_features, axis=1)
    return data_cleaned

    
import matplotlib.pyplot as plt

def create_and_save_box_plot(data, column='Weekly_Sales'):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data[column])
    ax.set_title(f'Box Plot of {column}')
    ax.set_ylabel(column)
    return fig


import matplotlib.pyplot as plt

def create_box_plot_no_outliers(data, column='Weekly_Sales', show_outliers=False):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data[column], showfliers=show_outliers)
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)

    # Return the figure
    return plt.gcf()

from scipy.stats import kstest

from scipy.stats import shapiro

def normal_test(data):
    if data.empty:
        return ["DataFrame is empty. Unable to perform normality test."]
    
    # Convert columns to numeric type
    numeric_data = data.apply(pd.to_numeric, errors='coerce')
    numeric_columns = numeric_data.columns
    
    if numeric_columns.empty:
        return ["No numeric columns found in the DataFrame. Unable to perform normality test."]
    
    # Check for normal distribution using the Anderson-Darling test
    alpha = 0.05
    results = []
    for column in numeric_columns:
        result = anderson(numeric_data[column].dropna())
        statistic = result.statistic
        critical_values = result.critical_values
        p_value = result.significance_level / 100
        if all(statistic < critical_values):
            results.append(f"Data in column '{column}' is normally distributed (p = {p_value[0]:.4f})")
        else:
            results.append(f"Data in column '{column}' is not normally distributed (p = {p_value[0]:.4f})")
    return results

def visualize_sales_histogram(data, save_path='sales_histogram.png'):
    # Convert 'Date' column to datetime type
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Group data by 3-month intervals and sum weekly sales
    grouped_data = data.resample('3M', on='Date')['Weekly_Sales'].sum()
    
    # Plot the histogram
    plt.bar(grouped_data.index.astype(str), grouped_data.values)
    
    plt.title('Distribution of Weekly Sales Over Time (3-Month Intervals)')
    plt.xlabel('Time Interval')
    plt.ylabel('Total Weekly Sales')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def create_correlation_heatmap(data):
    # Calculate correlation matrix
    correlation_matrix = data.corr()

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

    # Return the heatmap plot
    return plt.gcf()

import os

def create_sales_by_stores_plot(data):
    # Create the bar plot
    plt.figure(figsize=(15, 5))
    sns.barplot(x='Store', y='Weekly_Sales', color='#FFC220', data=data)

    plt.xlabel('Store Number')
    plt.ylabel('Weekly Sales')
    plt.title('Weekly Sales by Stores')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Return the plot
    return plt.gcf()

