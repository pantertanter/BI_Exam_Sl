import streamlit as st
from data_loader import (read_csv, normal_test, clean_data,
                        create_box_plot, create_correlation_heatmap,
                        clean_out_features, plot_normal_distribution,
                        apply_kmeans_clustering, calculate_wcss,
                        plot_elbow, calculate_silhouette_scores,
                        train_random_forest_regression_with_metrics, visualize_random_forest_regression)
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#------------------------------------------------Loading the data------------------------------------------------

data = read_csv('data/Walmart.csv')

#------------------------------------------------Sample plus row select-------------------------------------------

# Display a sample of the data
st.title('Walmart Data Visualization')
st.write('**This is a sample of the data:**')
st.write(data.sample(n=5))

# Check the length of the DataFrame
num_rows = len(data)
num_cols = data.shape[1]
# Write it to the screen
st.write(f'**The DataFrame has {num_rows} rows and {num_cols} columns.**')

# Create a number input widget to select the row number
selected_row_number = st.number_input('Enter the row number:', min_value=0, max_value=len(data), value=1, step=1)

# Check if the user has submitted the form
if st.button('Submit'):
    if selected_row_number >= 0 and selected_row_number <= len(data):
        selected_row = data.iloc[selected_row_number].to_frame().transpose()
        st.write('Selected row:')
        st.write(selected_row)
    else:
        st.write(f"Row number must be between 1 and {len(data)}.")

#------------------------------------------------Cleaning the data------------------------------------------------

data = clean_data(data)

st.write('**The data has been cleaned.**')

st.write('**Missing values has been dropped, the Date column has been converted to datetime and outliers has been removed.**')

# ------------------------------------------------Describe------------------------------------------------

# Display the descriptive statistics
data_descriptive = data.describe()
st.write('**Descriptive statistics:**')
st.write(data_descriptive)

# ------------------------------------------------Normality test------------------------------------------------

# Perform the normality test
st.write('**Normality test using the Anderson-Darling test:**')
normal_test_result = normal_test(data)

# Display the result
for result in normal_test_result:
    st.write(result)

st.write('**The normality test suggests that non of the columns is normally distributed.**')
st.write('**We personally do not trust this so we will make a visual representation of the normal distribution.**')
st.write('**We will visualize the normal distribution of the columns to confirm the result.**')

# ---------------------------Visualizing the normal distribution interactive-----------------------------------

# Optional column selection 
st.title("Normal Distribution Visualization")

# Selectbox for choosing the column
selected_column = st.selectbox("Select a column:", data.columns)

# Plot the normal distribution if a column is selected
if selected_column:
    st.write(f"### {selected_column}")
    plot_normal_distribution(data, selected_column)

st.markdown('---')
st.markdown('## Box Plot of columns without outliers')
st.markdown('**The box plot below shows the distribution of the colums without outliers.**')
box_plot_fig = create_box_plot(data)
st.pyplot(box_plot_fig)
st.markdown('---')

# ------------------------------------------Creating a heatmap  -------------------------------------------

st.markdown('## Correlation Heatmap With All Features')
correlation_heatmap_fig = create_correlation_heatmap(data)
st.pyplot(correlation_heatmap_fig)

#------------------------------------------------Cleaning out features------------------------------------

features_to_remove = ['Unemployment', 'Holiday_Flag']

cleaned_data = clean_out_features(data, features_to_remove)

#------------------------------------------------Creating a heatmap with cleaned out features-------------

st.markdown('## Correlation Heatmap With Cleaned Out Features')
st.write('**The heatmap below shows the correlation between the features in the cleaned data.**')
st.write('**The correlation coefficient ranges from -1 to 1.**')
st.write('**If the value is close to 1, it means that the features have a strong positive correlation.**')  
st.write('**If the value is close to -1, it means that the features have a strong negative correlation.**')
st.write('**If the value is close to 0, it means that the features have no correlation.**')
st.write('**The heatmap helps us understand the relationship between the features.**')
st.write('**This can help us identify which features are important for predicting Weekly Sales.**')
st.write('**We removed the features which has the least impact on Weekly Sales and decided not to inspect them further for now.**')
st.write('**As a result, the data is now represented by the following sample with 6 rows.**')
correlation_heatmap_fig_cleaned = create_correlation_heatmap(cleaned_data)
st.pyplot(correlation_heatmap_fig_cleaned)

#------------------------------------------------Sample plus row select-----------------------------------

# Explain the data preprocessing steps


st.markdown('## Sample of the Cleaned Data')

# Display the sample data frame
st.write(cleaned_data.sample(n=6))

# Indicate the number of columns remaining
st.write("The data frame now has 6 columns left.")

#---------------------------------------------Elbow method-----------------------------------------------

st.write('**The elbow method is a approximation method used in clustering analysis to determine the optimal number of clusters by identifying the point where the rate of decrease in within-cluster variance slows, resembling an "elbow" shape in the plot.**')

# Calculate WCSS for different numbers of clusters
wcss = calculate_wcss(data, max_clusters=10)

# Display elbow plot
fig_elbow = plot_elbow(wcss, max_clusters=10)
st.pyplot(fig_elbow)

# ------------------------------------------------Silhouette Score------------------------------------------

st.write('**The silhouette score is a metric used to evaluate the quality of clustering by measuring the cohesion and separation of clusters, with higher scores indicating better-defined clusters.**')

# Display silhouette scores in Streamlit app
def display_silhouette_scores():
    st.write("Calculating silhouette scores...")
    silhouette_scores = calculate_silhouette_scores(data, max_clusters=10)
    st.write("Silhouette Scores:")
    for n_clusters, score in enumerate(silhouette_scores, start=2):
        st.write(f"Number of clusters: {n_clusters}, Silhouette score: {score:.4f}")

# Center-align the button
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    if st.button("Calculate Silhouette Scores range 2-10 clusters"):
        display_silhouette_scores()

# ------------------------------------------------Clustering------------------------------------------

st.write('**The numeric features are scaled using StandardScaler to ensure that each feature contributes equally to the clustering process. Dimensionality reduction is performed using Principal Component Analysis (PCA) with n_components=2, transforming the scaled data into a 2-dimensional space. K-means clustering is applied to the reduced data with the specified number of clusters (num_clusters), utilizing the KMeans algorithm. The clusters are predicted for each data point. Finally, a scatter plot is created to visualize the clusters in the reduced 2-dimensional space, where each point represents a data point and is colored according to its assigned cluster.**')

# Apply K-Means clustering and visualize clusters
fig = apply_kmeans_clustering(data, sample_size=1000, num_clusters=2, random_state=42)
st.pyplot(fig)



# ---------------------------Random forest regression and visualize actual vs predicted-----------------------------
    
st.write("**This scatter plot compares what the model predicted with what actually happened. Each point on the plot represents one week of sales. The horizontal axis shows the actual sales that occurred, while the vertical axis shows the sales predicted by the model.**")

st.write("**Model Evaluation: It helps us evaluate the performance of the predictive model. If the points on the plot are close to the diagonal line, it indicates that the model's predictions are accurate. If they are scattered far from the line, it suggests that the model needs improvement.**")

# Train the Random Forest Regression model
fig = visualize_random_forest_regression(data, target_column='Weekly_Sales')

# Display the figure in Streamlit
st.pyplot(fig)

# ------------------------------------------------Random forest regression----------------------------------------

evaluation_metrics = train_random_forest_regression_with_metrics(data, target_column='Weekly_Sales')
st.write("Evaluation Metrics:")
for metric, value in evaluation_metrics.items():
    if metric == "Mean Absolute Error":
        st.write(f"{metric}: ${value:.4f}")
        st.write("**The mean absolute error (MAE) is the average absolute difference between the predicted and actual values. It measures the average magnitude of errors in a set of predictions, without considering their direction.**")
    elif metric == "Mean Squared Error":
        st.write(f"{metric}: ${value:.4f}")
        st.write("**The mean squared error (MSE) is the average of the squares of the errors between the predicted and actual values. It measures the average magnitude of the squared errors.**")
    elif metric == "Root Mean Squared Error":
        st.write(f"{metric}: ${value:.4f}")
        st.write("**The root mean squared error (RMSE) is the square root of the mean squared error. It measures the average magnitude of the errors in a set of predictions, considering their direction.**")
    elif metric == "R-squared":
        st.write(f"{metric}: {value:.4f}")
        st.write("**The R-squared value (R^2) is a measure of how well the model explains the variance in the target variable. It ranges from 0 to 1, with higher values indicating a better fit.**")

st.write("**The evaluation metrics help us understand how well the predictive model performs. If the MAE, MSE, and RMSE are close to 0, it indicates that the model's predictions are accurate. If R-squared is close to 1, it suggests that the model explains the variance in the target variable well.**")
st.write("**Understanding we are dealing with significant figures for Wallmart Store sales for whole weeks this is an excellent result. Especially considering that the R-squared is so close to 1, making it a near perfect prediction.**")
# ------------------------------------------------End of the app----------------------------------------
