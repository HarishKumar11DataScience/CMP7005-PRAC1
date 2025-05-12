import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
import joblib

# Define a function to display the first page (Introduction)
def introduction():
    st.title("Introduction")
    st.write("Welcome to the Data Analysis Assignment.")
    st.write("**Name**: Harish Kumar Gurjar")
    st.write("**Student ID**: st20320671")
    st.write("**Module Code**: CMP7005")
    st.write("**Module Title**: Programming for Data Analysis")
    st.write("**Academic Year**: 2024-2025, Semester 2")
    st.write("**Module Leader**: aprasad@cardiffmet.ac.uk")
    st.write("This Streamlit app presents a comprehensive analysis of air quality data, including data preprocessing, exploratory data analysis (EDA), and model building.")

# Function to display Data Overview
def data_overview():
    st.title("Data Overview")
    st.write("This section displays basic information about the dataset used for this analysis.")

    dataset_info = {
        'No': [1, 2, 3, 4, 5],
        'Year': [2013, 2013, 2013, 2013, 2013],
        'Month': [3, 3, 3, 3, 3],
        'Day': [1, 1, 1, 1, 1],
        'Hour': [0, 1, 2, 3, 4],
        'PM2.5': [4.0, 8.0, 7.0, 6.0, 3.0],
        'PM10': [4.0, 8.0, 7.0, 6.0, 3.0],
        'SO2': [4.0, 4.0, 5.0, 11.0, 12.0],
        'NO2': [7.0, 7.0, 10.0, 11.0, 12.0],
        'CO': [300.0, 300.0, 300.0, 300.0, 300.0],
        'O3': [77.0, 77.0, 73.0, 72.0, 72.0],
        'TEMP': [-0.7, -1.1, -1.1, -1.4, -2.0],
        'PRES': [1023.0, 1023.2, 1023.5, 1024.5, 1025.2],
        'DEWP': [-18.8, -18.2, -18.2, -19.4, -19.5],
        'RAIN': [0.0, 0.0, 0.0, 0.0, 0.0],
        'wd': ['NNW', 'N', 'NNW', 'NW', 'N'],
        'WSPM': [4.4, 4.7, 5.6, 3.1, 2.0],
        'station': ['Aotizhongxin', 'Aotizhongxin', 'Aotizhongxin', 'Aotizhongxin', 'Aotizhongxin']
    }
    
    # Create a DataFrame from the dataset info
    df = pd.DataFrame(dataset_info)

    # Display the first few rows of the dataset
    st.subheader("First Few Rows of the Dataset:")
    st.dataframe(df)

    # Display Data Types of each column
    st.subheader("Data Types of Each Column:")
    st.write(df.dtypes)

    # Basic Statistical Summary of Numerical Columns
    st.subheader("Statistical Summary of Numerical Columns:")
    st.write(df.describe())


# Define a function for the third page (Exploratory Data Analysis)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis (EDA)")
    st.write("This section provides a summary of the dataset and explores key features visually and statistically.")

    # Display the fundamental data understanding 
    st.subheader("Fundamental Data Understanding")
    st.write("The dataset contains 420,768 rows and 18 columns.")
    st.write("It includes features like year, month, hour, various pollutants (PM2.5, PM10, etc.), and meteorological data.")
    st.write("Missing data and unique values per column are shown in the previous section.")
    
    # Display images from image18 to image35 
    st.subheader("Statistical Plots and Visualizations")
    for i in range(1, 18):
        image_path = f"output_results/image{i}.png"  
        if os.path.exists(image_path):
            st.image(image_path, caption=f"Image {i}")


# Define a function for the fourth page (Statistics/Computation-Based Analysis and Visualization)
def statistics_analysis():
    st.title("Statistics/Computation-Based Analysis and Visualization")
    st.write("This section includes the application of statistical analysis and visualizations on the dataset.")

    # Display images for statistical visualizations
    st.subheader("Statistical Visualizations")
    for i in range(18, 36):  # Displaying images from image18 to image35
        image_path = f"output_results/image{i}.png" 
        if os.path.exists(image_path):
            st.image(image_path, caption=f"Image {i}")
    
    # Explanation of statistical techniques
    st.subheader("Statistical Techniques Used")
    st.write(""" 
    - **Descriptive Statistics**: The `describe()` function was used to generate a summary of statistics for numerical columns.
    - **Missing Value Analysis**: Missing values in the dataset were identified using `isnull().sum()`, and imputation strategies were applied.
    - **Correlation Analysis**: We examined correlations between variables using correlation matrices and visualized them with heatmaps.
    - **Visualizations**: Histograms, box plots, and scatter plots were used to visualize the distribution of pollutants and relationships between variables.
    - **Outlier Detection**: Outliers in certain columns (like PM2.5 and PM10) were identified and removed based on domain knowledge.
    """)


# Define a function for the Model Building page
def model_building():
    st.title("Model Building")
    st.write("In this section, we explain the model used for predicting PM2.5 levels and display the results.")

    # Explanation of Linear Regression Model
    st.subheader("Linear Regression Model")
    st.write("""
    **Linear Regression** is a statistical method used for modeling the relationship between a dependent variable (PM2.5) and one or more independent variables. 
    It assumes a linear relationship between the input variables (features) and the output (target). 
    We used **Linear Regression** to predict the PM2.5 values, which is a continuous variable.
    """)
    
    # Display images 36 and 37
    st.subheader("Model Visualizations")
    for i in range(36, 38):  # Displaying images image36.png and image37.png
        image_path = f"output_results/image{i}.png"
        if os.path.exists(image_path):
            st.image(image_path, caption=f"Image {i}")

    # Display the evaluation metrics
    st.subheader("Evaluation Metrics")
    st.write("""
    - **R-squared (R²)**: A measure of how well the model explains the variance in the target variable. A higher R² indicates a better model.
    - **Mean Squared Error (MSE)**: The average of the squared differences between the actual and predicted values. A lower MSE indicates a better fit.
    - **Mean Absolute Error (MAE)**: The average of the absolute differences between actual and predicted values. A lower MAE means the model is more accurate.
    """)

    # Display model performance metrics 
    st.write(f"**R-squared**: 0.8409")
    st.write(f"**Mean Squared Error (MSE)**: 0.0010")
    st.write(f"**Mean Absolute Error (MAE)**: 0.0206")

    # Display model saving message
    st.write("The model has been saved and can be used for future predictions.")
    st.write("Model saved as: /content/lr_model.pkl")

# Main function to control page navigation
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:", ["Introduction", "Data Overview", "Exploratory Data Analysis", "Statistics Analysis", "Model Building"])
    
    if page == "Introduction":
        introduction()
    elif page == "Data Overview":
        data_overview()
    elif page == "Exploratory Data Analysis":
        exploratory_data_analysis()
    elif page == "Statistics Analysis":
        statistics_analysis()
    elif page == "Model Building":
        model_building()

# Run the app
if __name__ == "__main__":
    main()
