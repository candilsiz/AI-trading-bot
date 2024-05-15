from yfinanca_crypto import get_df
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import mplfinance as mpf
from finta import TA
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
# from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import xgboost as xgb
import datetime
import requests
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

cols = ["Open", "High", "Low", "Close", "Adj" "Close", "Volume"]
ASSET = 'BTC-USD'
PERIOD1 = '2020-01-01' 
PERIOD2 = '2022-12-31' 
INTERVAL = 'daily'

# INITIALIZE   
def get_data():
    df = get_df(ASSET, PERIOD1, PERIOD2, INTERVAL)
    df['Date'] = pd.to_datetime(df['Date'])
    df.index = df['Date']
    return df

def fetch_fear_and_greed_index(asset_df, start_date, end_date):
    fear_greed_cols = ['Fear_and_Greed_Index', 'Classification_Extreme Fear', 
                       'Classification_Extreme Greed', 'Classification_Fear',
                       'Classification_Greed', 'Classification_Neutral']
    def get_data_from_api():
        base_url = "https://api.alternative.me/fng/"
        params = {'limit': '0',
                  'format': 'json'}
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.json()['data']
        else:
            print("Failed to fetch data: ", response.status_code)
            return None
    def convert_data(index_data):
        df = pd.DataFrame(index_data)
        df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df[(df['Date'] >= pd.to_datetime(start_date, format='%Y-%m-%d')) &
                (df['Date'] <= pd.to_datetime(end_date, format='%Y-%m-%d'))]
        df.set_index('Date', inplace=True)
        df.drop(columns=['time_until_update', 'timestamp'], inplace=True)
        df.rename(columns={'value': 'Fear_and_Greed_Index', 'value_classification': 'Classification'}, inplace=True)
        df.sort_index(ascending=True, inplace=True)
        return df
    def prepare_and_merge_data(df):
        result_df = pd.merge(asset_df, df[['Fear_and_Greed_Index', 'Classification']], left_index=True, right_index=True, how='left')
        result_df = pd.get_dummies(result_df, columns=['Classification'])
        result_df[fear_greed_cols] = result_df[fear_greed_cols].astype(int)
        return result_df
    index_data = get_data_from_api()
    if index_data is not None:
        df = convert_data(index_data)
        asset_df = prepare_and_merge_data(df)
    return asset_df

# FEATURE ENGINEERING
def get_timefeatures(df):
    """
    Adds Time derived features to the data frame.
    """
    df['year'] = df['Date'].dt.year
    df["quarter"] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
    df["dayofmonth"] = df['Date'].dt.day
    df['week'] = df['Date'].dt.isocalendar().week
    df['dayofweek'] = df['Date'].dt.day_of_week
    df['day'] = df['Date'].dt.day
    return df

def get_indicatorfeatures(df):
    df['RSI'] = TA.RSI(df, 16) # rsi
    df['SMA'] = TA. SMA(df, 20) # simple moving avarage
    df['SMA_L'] = TA.SMA(df, 41) #  simple moving avarage long
    df['OBV' ] = TA.OBV(df) # on balance volume
    df['VWAP'] = TA.VWAP(df) # volume weighted avarage price
    df['EMA'] = TA. EMA (df) # exponential moving avarage
    df['ATR'] = TA.ATR(df) # avarage true ranger
    return df

def get_statfeatures(df):
    features = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    windows = [3, 7, 30]
    for feature in features:
        for window in windows:
            df[f'{feature}_mean_lag{window}'] = df[feature].rolling(window=window, min_periods=1).mean()
            df[f'{feature}_std_lag{window}']  = df[feature].rolling(window=window, min_periods=1).std()
    return df

def train_test_split_bydate(df, y):
    # time series train_test_split
    mask = ~df.columns.isin([y])
    X, y = df.loc[:,mask], df[y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    return X_train, X_test, y_train, y_test

def preprocess(df): # add cols to keep
    """
    Combine all the feature engineering here.
    """
    PERIOD1 = '2020-01-01' 
    PERIOD2 = '2022-12-31' 
    df = df.drop(["Date"], axis=1)
    df = get_timefeatures(df)
    df = get_statfeatures(df)
    df = get_indicatorfeatures(df)
    df = fetch_fear_and_greed_index(df, PERIOD1, PERIOD2)
    return df

# PREDICTION
def forecast(df):
    # model_hyperparameters = dict()
    train_df = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split_bydate(train_df, "Close")
    evaluation_df = y_test.copy()
    model = xgb.XGBRegressor(
        colsample_bytree = 0.87,
        learning_rate = 0.01, 
        gamma = 3, 
        max_depth = 8, 
        min_child_weight = 7, 
        n_estimators = 900, 
        reg_alpha = 79, 
        reg_lambda = 0.0003
        )
    model.fit(X_train, y_train,
        eval_set = [(X_train,y_train), (X_test, y_test)],
        eval_metric = "rmse",
        verbose = 10)
    predictions = model.predict(X_test)
    evaluation_df = pd.DataFrame({'Actual_Close': y_test.values, 
                                  'Predicted_Close': predictions,
                                  'Date' : y_test.index.values})
    return train_df, evaluation_df

def evaluate(y_actual, y_pred):
    mape = mean_absolute_percentage_error(y_actual, y_pred) # MAPE
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred)) # RMSE
    return f"MAPE ERROR: {mape:.3f}%,\nRMSE ERROR: {rmse:.3f}"

def candle_stick_it(df):
    df.set_index('Date', inplace=True)
    one_month_df = df.iloc[-30:] # one month of dataframe
    mpf.plot(one_month_df, type='candle', volume=True, mav=(3, 6, 9), style='yahoo')

def vizualize(evaluation_df, plot_type='line'):
    plt.figure(figsize=(12, 6))
    plt.plot(evaluation_df['Date'], evaluation_df['Actual_Close'], label='Actual Close', color='blue')
    plt.plot(evaluation_df['Date'], evaluation_df['Predicted_Close'], label='Predicted Close', color='red')
    plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Close Prices')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # today = datetime.date.today()
    df = get_data()
    # candle_stick_it(df)
    train_df, evaluation_df = forecast(df)
    print(evaluate(evaluation_df['Actual_Close'], evaluation_df['Predicted_Close']))
    vizualize(evaluation_df)
    #print(df[['Fear and Greed Index', 'Classification']])
    #df = preprocess(df)
    fear_greed_cols = ['Fear_and_Greed_Index', 'Classification_Extreme Fear', 
                       'Classification_Extreme Greed', 'Classification_Fear',
                       'Classification_Greed', 'Classification_Neutral']
    # print(train_df[fear_greed_cols])
    # print(train_df)
    # print(train_df.columns)
    # print(train_df.info()) # do something about nulls exists in the df columns

