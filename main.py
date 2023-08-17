from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np

from data import download_data, compute_RSI, preprocess_data, create_dataset
from model import build_model
from utils import scale_data



def main():
    ticker = "AAPL"
    aapl = download_data(ticker, start_date="2000-01-01")

    aapl['RSI'] = compute_RSI(aapl['Close'], 14)
    aapl, ratio_column, trend_column = preprocess_data(aapl, horizon=5)

    X = aapl[[ratio_column, trend_column, 'RSI']].values
    y = aapl["Target"].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    look_back = 3
    X, y = create_dataset(X, y, look_back)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=2, shuffle=False)

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.55).astype(int)
    print(classification_report(y_test, y_pred))

    last_features = aapl[[ratio_column, trend_column, 'RSI']].values[-look_back:]
    last_features_scaled = scaler.transform(last_features)
    last_features_reshaped = np.reshape(last_features_scaled, (1, look_back, 3))

    next_day_prediction_probs = model.predict(last_features_reshaped)
    next_day_prediction = (next_day_prediction_probs > 0.55).astype(int)

    if next_day_prediction == 1:
        print(f"Prediction for the next trading day after {aapl.index[-1].strftime('%Y-%m-%d')}: UP")
    else:
        print(f"Prediction for the next trading day after {aapl.index[-1].strftime('%Y-%m-%d')}: DOWN")


if __name__ == "__main__":
    main()
