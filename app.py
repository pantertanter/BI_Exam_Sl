import streamlit as st
from data_loader import (read_csv, normal_test,
                         create_and_save_box_plot, create_correlation_heatmap,
                         create_sales_by_stores_plot, create_box_plot_no_outliers)
import pandas as pd

# Read the CSV file
data = read_csv('data/Walmart.csv')

data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

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

st.markdown('## Correlation Heatmap')
correlation_heatmap_fig = create_correlation_heatmap(data)
st.pyplot(correlation_heatmap_fig)

st.markdown('## Weekly Sales by Stores')
sales_by_stores_plot_fig = create_sales_by_stores_plot(data)
st.pyplot(sales_by_stores_plot_fig)

st.markdown('---')
st.markdown('## Box Plot of Weekly Sales With Outliers')
box_plot_fig = create_and_save_box_plot(data)
st.pyplot(box_plot_fig)
st.markdown('---')

st.markdown('---')
st.markdown('## Box Plot of Weekly Sales Without Outliers')
box_plot_fig = create_box_plot_no_outliers(data)
st.pyplot(box_plot_fig)
st.markdown('---')

# Header
st.header('This is a header')

# Subheader
st.subheader('This is a subheader')

# Text
st.write('This is some text.')

# Displaying Data
st.write('Here is some data:')
st.write(pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35]
}))

# Interactive Widgets
number = st.number_input('Enter a number')
st.write('The square of the number is:', number ** 2)

st.write(data.sample(n=5))
