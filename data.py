import yfinance as yf
import pandas as pd

def download_data(ticker, start_date):
    return yf.download(ticker, start=start_date)


def compute_RSI(data, window):
    diff = data.diff()
    loss = diff.where(diff < 0, 0)
    gain = -diff.where(diff > 0, 0)
    
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def preprocess_data(df, horizon):
    rolling_avg = df.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    df[ratio_column] = df["Close"] / rolling_avg["Close"]

    df["Target"] = (df["Close"].shift(-horizon) > df["Close"]).astype(int)
    trend_column = f"Trend_{horizon}"
    df[trend_column] = df.shift(1).rolling(horizon).sum()["Target"]
    
    df.dropna(inplace=True)

    return df, ratio_column, trend_column


def create_dataset(X, y, look_back=1):
    dataX, dataY = [], []
    for i in range(len(X)-look_back-1):
        a = X[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(y[i + look_back])
    return np.array(dataX), np.array(dataY)