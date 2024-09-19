import pandas as pd

class DataProcessor:
    def __init__(self, df):
        self.df = df

    # Function to return the dataframe head
    def get_head(self, n=10):
        return self.df.head(n)

    # Function to return the dataframe tail
    def get_tail(self, n=10):
        return self.df.tail(n)

    # Function to return dataframe description
    def get_describe(self):
        return self.df.describe()

    # Function to return dataframe info
    def get_info(self):
        return self.df.info

    # Function to return dataframe shape
    def get_shape(self):
        return self.df.shape

    # Function to return dataframe columns
    def get_columns(self):
        return self.df.columns

    # Function to return missing values and percentages
    def get_missing_values(self):
        num_missing = self.df.isnull().sum()
        num_rows = self.df.shape[0]

        data = {
            'num_missing': num_missing, 
            'percent_missing (%)': [round(x, 2) for x in num_missing / num_rows * 100]
        }

        missing_stats = pd.DataFrame(data)

        # Filter columns with missing values
        return missing_stats[missing_stats['num_missing'] != 0]

