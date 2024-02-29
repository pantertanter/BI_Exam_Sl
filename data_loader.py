import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import anderson, norm
import numpy as np
import seaborn as sns

# ------------------------------------------------Loading the data------------------------------------------------


# Read the CSV file into a DataFrame
def read_csv(file_path):
    return pd.read_csv(file_path)

# ------------------------------------------------Cleaning the data------------------------------------------------

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

# ------------------------------------------------Clean out features------------------------------------------------

def clean_out_features(data, features):
    # Check if the features exist in the DataFrame before dropping them
    existing_features = [col for col in features if col in data.columns]
    
    # Drop only the existing features
    data_cleaned = data.drop(existing_features, axis=1)
    return data_cleaned

# ------------------------------------------------Normality test------------------------------------------------

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
        p_value = result.significance_level[0]  # Extracting the p-value
        if all(statistic < critical_values):
            results.append(f"Data in column '{column}' is normally distributed (p = {p_value:.4f})")
        else:
            results.append(f"Data in column '{column}' is not normally distributed (p = {p_value:.4f})")
    return results

# ------------------------------------------------Visualizing the normal distribution interactive------------------------------------------------

def plot_normal_distribution(data, column_name):
    # Extract the column data
    column_data = data[column_name].dropna()
    
    # Fit a normal distribution to the data
    mu, std = norm.fit(column_data)

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the histogram of the data
    ax.hist(column_data, bins=30, density=True, alpha=0.6, color='g')

    # Plot the PDF (Probability Density Function) of the fitted normal distribution
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)

    ax.set_title(f"Histogram and Normal Distribution Fit for {column_name}")
    ax.set_xlabel(column_name)
    ax.set_ylabel("Frequency")
    ax.grid(True)

    # Display the figure in Streamlit
    st.pyplot(fig)

# ------------------------------------------------Creating an interactive box plot------------------------------------------------

def create_box_plot(data):
    # Generate a unique key for the selectbox
    selectbox_key = "select_column_box_plot"
    
    # Create a dropdown menu for column selection
    selected_column = st.selectbox("Select a column:", data.columns, key=selectbox_key)
    
    # Create the box plot based on the selected column
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data[selected_column])
    ax.set_title(f'Box Plot of {selected_column}')
    ax.set_ylabel(selected_column)
    return fig

# ------------------------------------------------Weekly Sales by Stores------------------------------------------------

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

# ------------------------------------------------Correlation Heatmap------------------------------------------------
    
def create_correlation_heatmap(data):
    # Calculate correlation matrix
    correlation_matrix = data.corr()

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

    # Return the heatmap plot
    return plt.gcf()

