import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# DataProcessor class holds all non-plotting-related functions
class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    # Method to get the first 10 rows of the DataFrame
    def get_head(self):
        return self.df.head(10)
    
    # Method to get the last 10 rows of the DataFrame
    def get_tail(self):
        return self.df.tail(10)
    
    # Method to get descriptive statistics of the DataFrame
    def get_describe(self):
        return self.df.describe()
    
    # Method to get the shape (dimensions) of the DataFrame (rows, columns)
    def get_shape(self):
        return self.df.shape
    
    # Method to get the list of column names
    def get_columns(self):
        return self.df.columns.tolist()

    # Method to calculate missing values and their percentages for the DataFrame
    def get_missing_values(self):
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        missing_values = pd.DataFrame({
            'num_missing': missing_data,
            'percent_missing (%)': missing_percentage
        })
        return missing_values[missing_values['num_missing'] > 0].sort_values(by='percent_missing (%)', ascending=False)

    # Function to get missing values and their percentages in a given column
    def get_column_missing_values(self, column: str):
        if column not in self.df.columns:
            return f"Column '{column}' not found in DataFrame."
        
        num_missing = self.df[column].isnull().sum()
        percent_missing = (num_missing / len(self.df)) * 100
        
        return {
            'column': column,
            'num_missing': num_missing,
            'percent_missing (%)': percent_missing
        }

    # Method to fill missing values in specified columns with 0
    def fill_missing_values_in_columns(self):
        columns_to_fill = [
            'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval',
            'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'
        ]
        
        # Fill missing values in the specified columns with 0
        for column in columns_to_fill:
            if column in self.df.columns:
                self.df[column] = self.df[column].fillna(0)
        
        return self.df

    # Method to add Year, Month, Day, and WeekOfYear columns based on the 'Date' column
    def add_month_year(self):
        '''
        Converts the 'Date' column to datetime and adds Year, Month, Day, WeekOfYear columns.
        '''
        logging.info('Calculating Year, Month, Day, WeekOfYear for Dataframe')
        
        # Convert 'Date' column to datetime if it isn't already
        if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Day'] = self.df['Date'].dt.day
        self.df['WeekOfYear'] = self.df['Date'].dt.isocalendar().week  # Using isocalendar() for the week number
        
        return self.df

    # Method to get assortment description
    def get_assortment(self, value):
        logging.info(f'get assortment for value of {value}')
        assort = {'a': 'Basic', 'b': 'Extra', 'c': 'Extended'}
        return assort.get(value, None)
