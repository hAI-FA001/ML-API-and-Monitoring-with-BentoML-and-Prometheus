import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from california_housing_api.data_loading import load_and_prepare_data


def train_model():
    """Train RandomForestRegressor and save it."""
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    joblib.dump(model, "models/housing_model.joblib")


if __name__ == "__main__":
    train_model()
