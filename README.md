# Rossmann Pharmaceuticals

## Project Overview


## Business Need

As a Machine Learning Engineer at Rossmann Pharmaceuticals, the finance team has identified a critical need to forecast sales across all stores in various cities six weeks in advance. Currently, store managers rely on their experience and personal judgment for sales predictions, which can lead to inconsistencies and inaccuracies.

To enhance forecasting accuracy, the data team has pinpointed several key factors influencing sales, including promotions, competition, school and state holidays, seasonality, and locality. The goal is to develop a robust machine learning model that can forecast store sales more effectively by leveraging these factors.

In the initial phase of the project, exploratory data analysis (EDA) is essential to understand customer purchasing behavior across different stores. By analyzing factors such as the impact of promotions, holidays, and store openings, we can uncover valuable insights that drive sales. This task involves cleaning the data, handling outliers and missing values, and visualizing the relationships between key features, such as sales, promotions, and customer behaviors.

Key questions to explore include:

- How do promotions affect sales, and are they attracting more customers?

- What are the seasonal trends in purchasing behavior, particularly around holidays like Christmas and Easter?

- How does store competition, including proximity to other stores and the opening or closing of new competitors, influence sales?

Following this, we will preprocess the data for machine learning, including feature extraction, handling missing data, and scaling the data. Various models will be evaluated to predict daily sales, such as tree-based models (e.g., Random Forest Regressor) and deep learning techniques like Long Short-Term Memory (LSTM) networks, which are particularly useful for time-series forecasting.

The project will culminate in building an API to serve these models for real-time sales predictions, which will help Rossmann Pharmaceuticals plan ahead and optimize their resources effectively.

## Objective

The primary goal of this project is to develop and deploy an end-to-end machine learning product that accurately predicts sales for Rossmann’s stores. This product will provide valuable insights to the finance team, enabling more informed decision-making and strategic planning.

## Key Features

Sales Forecasting Model: Develop a robust model that integrates identified factors for accurate sales predictions.


Data Processing Pipeline: Implement a data pipeline to clean, preprocess, and transform data for model training and prediction.


Deployment: Serve the prediction model through a user-friendly interface accessible to finance analysts.


Performance Monitoring: Establish mechanisms to monitor model performance and update predictions as new data becomes available.


By delivering this solution, the project aims to empower the finance team with reliable sales forecasts, ultimately leading to better inventory management and optimized sales strategies across all Rossmann stores.



Getting Started

Follow the instructions below to set up and run the project on your local machine.

Prerequisites Ensure you have the following installed on your system:

Python 3.x pip virtualenv

1. Clone the repository

    Clone the project repository to your local machine using the following command:

    git clone https://github.com/brook1167/Rossmann-Pharmaceuticals

2. Install dependencies

   Navigate to the project directory and create a virtual environment using virtualenv:

   cd Rossmann-Pharmaceuticals
   
   virtualenv venv


3. Activate the virtual environment
    
    on Windows
        .\venv\Scripts\activate
        
    on Mac/Linus
        source venv/bin/activate
        
4. Install dependencies

    With the virtual environment activated, install all the required packages from the         
    requirements.txt file:
    
    pip install -r requirements.txt


5. Run the application

After installing the dependencies, you are all set! Run the application or script as needed.
