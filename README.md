# California Housing Price Prediction API

A REST API for predicting California housing prices using Random Forest Regressor.

## Setup

- Install dependencies using Poetry
  ```bash
  poetry install
  ```
- Install the project in editable mode
  ```bash
  poetry run pip install -e src/
  ```
- Train the model

  ```bash
  poetry run python src/california_housing_api/model_training.py
  ```

  - Trains and saves a `RandomForestRegressor` in `models/housing_model.joblib`.

- Run BentoML service

  ```bash
  bentoml serve src/california_housing_api/service.py:HousingPredictor
  ```

## Project Structure

<pre>
california-housing-api/
├── .venv/
├── src/
│   └── california_housing_api/
│       ├── __init__.py
│       ├── data_loading.py
│       ├── model_training.py
│       └── service.py
│       └── verify.py
├── models/
│   └── housing_model.joblib
├── pyproject.toml
├── poetry.lock
├── README.md
├── Dockerfile
└── .gitignore
</pre>
