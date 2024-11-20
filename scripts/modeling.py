import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

class TimeSeriesModel:
    def __init__(self, df):
        self.df = df
        self.preprocessor = self.create_preprocessing_pipeline()

    def extract_date_features(self):
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Day'] = self.df['Date'].dt.day
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        self.df['IsWeekend'] = self.df['DayOfWeek'].isin([5, 6]).astype(int)
        self.df['DayOfMonth'] = self.df['Date'].dt.day
        self.df['WeekOfYear'] = self.df['Date'].dt.isocalendar().week
        self.df['Season'] = pd.cut(self.df['Month'], 
                                   bins=[0, 3, 6, 9, 12], 
                                   labels=['Winter', 'Spring', 'Summer', 'Fall'],
                                   include_lowest=True)
        return self.df

    def check_stationarity(self, timeseries):
        """Check whether your time Series Data is Stationary."""
        result = adfuller(timeseries, autolag='AIC')
        print(f'ADF Statistic: {result[0]:.10f}')
        print(f'p-value: {result[1]:.10f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.10f}')
        
        # If p-value > 0.05, the time series is non-stationary
        if result[1] > 0.05:
            print("The time series is non-stationary")
        else:
            print("The time series is stationary")

    def create_supervised_data(self, data, n_step=1):
        """Transform the time series data into supervised learning data"""
        X, y = [], []
        for i in range(len(data) - n_step - 1):
            X.append(data[i:(i + n_step), 0])
            y.append(data[i + n_step, 0])

        return np.array(X), np.array(y)

    def create_preprocessing_pipeline(self):
        numeric_features = ['Store', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'Year', 'Month', 
                            'CompetitionDistance', 'CompetitionOpenSinceMonth', 
                            'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']
        categorical_features = ['StoreType', 'Assortment', 'StateHoliday', 'SchoolHoliday', 
                                'Season', 'Promo', 'Promo2']

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        return preprocessor

    def build_model(self):
        # Manually tuned parameters due to resource
        rf = RandomForestRegressor(
            n_estimators=200, 
            max_depth=64, 
            criterion='squared_error',
            min_samples_split=10, 
            min_samples_leaf=2, 
            n_jobs=-1, 
            random_state=42)
        return rf

    def train_model(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        msle = mean_squared_log_error(y_test, y_pred)
        return mse, mae, rmse, r2, msle

    def get_feature_importance(self, model):
        # Extract the names of numeric and one-hot encoded categorical features
        numeric_features = self.preprocessor.transformers_[0][2]
        categorical_transformer = self.preprocessor.transformers_[1][1]  # The pipeline that has OneHotEncoder
        categorical_features = self.preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(self.preprocessor.transformers_[1][2])
        
        # Combine both numeric and categorical feature names
        feature_names = list(numeric_features) + list(categorical_features)
        
        # Get feature importance from the model
        feature_importance = model.feature_importances_
        
        # Create a DataFrame for feature importance
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        return importance_df.sort_values('importance', ascending=False)

    def calculate_confidence_interval(self, y_pred, confidence=0.95):
        n = len(y_pred)
        m = np.mean(y_pred)
        se = np.std(y_pred, ddof=1) / np.sqrt(n)
        h = se * np.abs(np.random.standard_t(df=n-1, size=1))
        return m - h, m + h

    def serialize_model(self, model, path):
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")
        filename = f'model_{timestamp}.pkl'
        full_path = os.path.join(path, filename)
        joblib.dump(model, full_path)
        return filename

    def plot_acf_pacf(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        plot_acf(self.df['y'], ax=axes[0])
        plot_pacf(self.df['y'], ax=axes[1])
        plt.show()