import pandas as pd
import matplotlib.pyplot as plt

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

    # Method to plot the distribution of the 'Promo' column in both train and test DataFrames
    def plot_promo_distribution(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        # Ensure that the 'Promo' column exists in both DataFrames
        if 'Promo' not in df_train.columns or 'Promo' not in df_test.columns:
            return "The 'Promo' column is not present in both train and test DataFrames."
        
        # Get value counts for the 'Promo' column in both datasets
        train_distribution = df_train['Promo'].value_counts()
        test_distribution = df_test['Promo'].value_counts()

        # Create a figure with two subplots
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot training distribution
        ax[0].bar(train_distribution.index.astype(str), train_distribution)
        ax[0].set_title("Promo Distribution on Training Set")
        ax[0].set_xlabel("Promo")
        ax[0].set_ylabel("Count")

        # Plot testing distribution
        ax[1].bar(test_distribution.index.astype(str), test_distribution)
        ax[1].set_title("Promo Distribution on Testing Set")
        ax[1].set_xlabel("Promo")
        ax[1].set_ylabel("Count")

        # Show the plots
        plt.tight_layout()
        plt.show()

    # Method to plot average sales around Christmas and Easter
    def plot_sales_around_holidays(self, train_store: pd.DataFrame):
        # Ensure necessary columns are present
        required_columns = ['Open', 'Year', 'Month', 'Day', 'Sales']
        if not all(col in train_store.columns for col in required_columns):
            return "The required columns (Open, Year, Month, Day, Sales) are not present in the DataFrame."

        # Filter the data for open stores in 2014
        open_store = train_store[(train_store.Open == 1) & (train_store.Year == 2014)]

        # Define months for Christmas and Easter
        christmas_month = 12
        easter_month = 4

        # Filter data for December (Christmas) and April (Easter)
        christmas_data = open_store[open_store.Month == christmas_month]
        easter_data = open_store[open_store.Month == easter_month]

        # Filter around Christmas (Days 21 to 29)
        around_christmas = christmas_data[(christmas_data['Day'] > 20) & (christmas_data['Day'] < 30)]
        around_christmas = around_christmas[['Day', 'Sales']].groupby('Day').mean()

        # Filter around Easter (Days 16 to 24)
        around_easter = easter_data[(easter_data['Day'] > 15) & (easter_data['Day'] < 25)]
        around_easter = around_easter[['Day', 'Sales']].groupby('Day').mean()

        # Create subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # Plot for Christmas sales
        axs[0].bar(around_christmas.index, around_christmas['Sales'], color='green', alpha=0.7)
        axs[0].set_title('Sales during Christmas (Dec 25)')
        axs[0].set_xlabel('Day')
        axs[0].set_ylabel('Average Sales')

        # Plot for Easter sales
        axs[1].bar(around_easter.index, around_easter['Sales'], color='blue', alpha=0.7)
        axs[1].set_title('Sales during Easter (April 20)')
        axs[1].set_xlabel('Day')
        axs[1].set_ylabel('Average Sales')

        # Adjust layout
        plt.tight_layout()
        plt.show()
