from sklearn.preprocessing import MinMaxScaler

def scale_data(X):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(X), scaler