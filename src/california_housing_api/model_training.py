import joblib
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from california_housing_api.data_loading import load_and_prepare_data


def objective(trial):
    """Objective function for Optuna"""
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 5, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def train_model():
    """Train and save a RandomForestRegressor model, with hyperparameter tuning"""
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)

    best_params = study.best_params
    print("Best Parameters:", best_params)

    X_train, X_test, y_train, y_test = load_and_prepare_data()
    best_model = RandomForestRegressor(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    joblib.dump(best_model, "models/housing_model.joblib")


if __name__ == "__main__":
    train_model()
