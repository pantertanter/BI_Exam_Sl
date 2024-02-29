import streamlit as st
from data_loader import (read_csv, normal_test, clean_data,
                         create_and_save_box_plot, create_correlation_heatmap,
                         create_sales_by_stores_plot, create_box_plot_no_outliers,
                         clean_out_features, plot_normal_distribution)
import pandas as pd

# Read the CSV file
data = read_csv('data/Walmart.csv')

data = clean_data(data)

# Display a sample of the data
st.write('Sample of the data:')

st.title('Walmart Data Visualization')
st.write('This is a sample of the data:')
st.write(data.sample(n=5))

# Check the length of the DataFrame
num_rows = len(data)
num_cols = data.shape[1]
# Write it to the screen
st.write(f'The DataFrame has {num_rows} rows and {num_cols} columns.')

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

# Display the descriptive statistics
data_descriptive = data.describe()
st.write('Descriptive statistics:')
st.write(data_descriptive)

# Perform the normality test
st.write('Normality test:')
normal_test_result = normal_test(data)

# Display the result
for result in normal_test_result:
    st.write(result)

st.write('The normality test suggests that non of the columns is normally distributed.')
st.write('We personally do not trust this so we will make a visual representation of the normal distribution.')
st.write('We will visualize the normal distribution of the columns to confirm the result.')

# Optional column selection 
st.title("Normal Distribution Visualization")

# Selectbox for choosing the column
selected_column = st.selectbox("Select a column:", data.columns)

# Plot the normal distribution if a column is selected
if selected_column:
    st.write(f"### {selected_column}")
    plot_normal_distribution(data, selected_column)

st.markdown('## Weekly Sales by Stores')
sales_by_stores_plot_fig = create_sales_by_stores_plot(data)
st.pyplot(sales_by_stores_plot_fig)

st.markdown('---')
st.markdown('## Box Plot of Weekly Sales With Outliers')
st.markdown('The box plot below shows the distribution of the Weekly Sales with outliers and we can se that all the outliers are beond the max.')
box_plot_fig = create_and_save_box_plot(data)
st.pyplot(box_plot_fig)
st.markdown('---')

st.markdown('---')
st.markdown('## Box Plot of Weekly Sales Without Outliers')
box_plot_fig = create_box_plot_no_outliers(data)
st.pyplot(box_plot_fig)
st.markdown('---')

st.markdown('## Correlation Heatmap')
correlation_heatmap_fig = create_correlation_heatmap(data)
st.pyplot(correlation_heatmap_fig)

features_to_remove = ['Unemployment', 'Holiday_Flag']

cleaned_data = clean_out_features(data, features_to_remove)

st.markdown("## Data Preprocessing Summary")

# Explain the data preprocessing steps
st.write("We removed the features which has the least impact on Weekly Sales and decided not to inspect them further for now.")
st.write("As a result, the data is now represented by the following sample with 6 rows.")

# Display the sample data frame
st.write(cleaned_data.sample(n=6))

# Indicate the number of columns remaining
st.write("The data frame now has 6 columns left.")
