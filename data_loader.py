import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
from scipy.stats import anderson, norm
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

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

# ------------------------------------------------Correlation Heatmap------------------------------------------------
    
def create_correlation_heatmap(data):
    # Calculate correlation matrix
    correlation_matrix = data.corr()

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

    # Return the heatmap plot
    return plt.gcf()

# ------------------------------------------------Silhouette Score------------------------------------------

# Define calculate_silhouette_scores function
def calculate_silhouette_scores(data, max_clusters=10):
    numeric_data = data.select_dtypes(include=['number'])
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(numeric_data)
        silhouette_avg = silhouette_score(numeric_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores

# ------------------------------------------------Clustering------------------------------------------------

# Define calculate_wcss and plot_elbow functions
def calculate_wcss(data, max_clusters=10):
    numeric_data = data.select_dtypes(include=['number'])
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(numeric_data)
        wcss.append(kmeans.inertia_)
    return wcss

def plot_elbow(wcss, max_clusters=10):
    fig, ax = plt.subplots()
    ax.plot(range(1, max_clusters + 1), wcss, marker='o')
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
    return fig

def apply_kmeans_clustering(data, num_clusters=2, sample_size=None, random_state=None):
    if sample_size:
        data = data.sample(n=sample_size, random_state=random_state)
    
    numeric_data = data.drop(columns=['Store', 'Date', 'Holiday_Flag'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(reduced_data)

    fig, ax = plt.subplots()
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('K-Means Clustering')
    
    return fig

# ------------------------------------------------random forest regression------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_random_forest_regression(data, target_column, test_size=0.2, random_state=42):
    """
    Train a Random Forest Regression model on the given DataFrame.

    Parameters:
    - data (DataFrame): The DataFrame containing features and target variable.
    - target_column (str): The name of the target variable column.
    - test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
    - random_state (int, optional): Random state for reproducibility. Default is 42.

    Returns:
    - dict: A dictionary containing evaluation metrics.
    """

    # Splitting the data into features (X) and target variable (y)
    X = data.drop(columns=[target_column, 'Date'])
    y = data[target_column]

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the Random Forest Regression model
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)  # You can adjust hyperparameters as needed

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    # Return evaluation metrics
    return {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'R-squared': r2
    }

# ------------------------------------------------random forest regression visualization------------------------------------------------

def train_random_forest_regression(data, target_column, test_size=0.2, random_state=42):
    """
    Train a Random Forest Regression model on the given DataFrame.

    Parameters:
    - data (DataFrame): The DataFrame containing features and target variable.
    - target_column (str): The name of the target variable column.
    - test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
    - random_state (int, optional): Random state for reproducibility. Default is 42.

    Returns:
    - fig: A matplotlib figure containing the scatter plot of actual vs. predicted values.
    """

    # Splitting the data into features (X) and target variable (y)
    X = data.drop(columns=[target_column, 'Date'])
    y = data[target_column]

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the Random Forest Regression model
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)  # You can adjust hyperparameters as needed

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Visualize the actual vs. predicted values
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, predictions, color='blue', alpha=0.5)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Actual vs. Predicted Values')
    ax.grid(True)

    # Return the figure
    return fig