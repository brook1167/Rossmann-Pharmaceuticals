import unittest
import pandas as pd
from datetime import datetime
from scripts.data_processing import *

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        # Create a simple DataFrame for testing
        data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
            'Promo2SinceWeek': [None, 12, None, 13],
            'Promo2SinceYear': [None, 2024, None, 2024],
            'PromoInterval': [None, 'Feb', None, 'Mar'],
            'CompetitionOpenSinceYear': [None, 2010, None, 2015],
            'CompetitionOpenSinceMonth': [None, 5, None, 6]
        }
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        self.processor = DataProcessor(df)

    def test_get_head(self):
        head = self.processor.get_head()
        self.assertEqual(len(head), 4)  # Since there are 4 rows
        self.assertEqual(head.iloc[0]['Date'], datetime(2024, 1, 1))

    def test_get_tail(self):
        tail = self.processor.get_tail()
        self.assertEqual(len(tail), 4)  # Same size as input data
        self.assertEqual(tail.iloc[-1]['Date'], datetime(2024, 1, 4))

    def test_get_describe(self):
        describe = self.processor.get_describe()
        self.assertTrue('Promo2SinceWeek' in describe.columns)

    def test_get_shape(self):
        shape = self.processor.get_shape()
        self.assertEqual(shape, (4, 6))  # 4 rows, 6 columns

    def test_get_columns(self):
        columns = self.processor.get_columns()
        self.assertEqual(columns, ['Date', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'])

    def test_get_missing_values(self):
        missing = self.processor.get_missing_values()
        self.assertEqual(len(missing), 5)  # 5 columns with missing values

    def test_get_column_missing_values(self):
        missing = self.processor.get_column_missing_values('Promo2SinceWeek')
        self.assertEqual(missing['num_missing'], 2)

    def test_fill_missing_values_in_columns(self):
        filled_df = self.processor.fill_missing_values_in_columns()
        self.assertEqual(filled_df['Promo2SinceWeek'].iloc[0], 0)
        self.assertEqual(filled_df['Promo2SinceYear'].iloc[0], 0)

    def test_add_month_year(self):
        df_with_date = self.processor.add_month_year()
        self.assertTrue('Year' in df_with_date.columns)
        self.assertEqual(df_with_date['Year'].iloc[0], 2024)
        self.assertEqual(df_with_date['WeekOfYear'].iloc[0], 1)

    def test_get_assortment(self):
        assortment = self.processor.get_assortment('a')
        self.assertEqual(assortment, 'Basic')
        assortment = self.processor.get_assortment('d')
        self.assertIsNone(assortment)

if __name__ == '__main__':
    unittest.main()
