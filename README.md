# California Housing Price Prediction API

A dockerized REST API for predicting California housing prices using Random Forest Regressor.<br>
Scikit-learn and Optuna are used for training and hyperparameter tuning, the endpoints are served using BentoML, and metrics are logged with Prometheus. <br>
The service is accessible on port 3000.

## Project Structure

<pre>
california-housing-api/
├── .github/
│   └── workflows/
│       ├── cd_workflow.yml
│       └── ci_workflow.yml
├── src/
│   └── california_housing_api/
│       ├── __init__.py
│       ├── data_loading.py
│       ├── model_training.py
│       ├── service.py
│       └── verify.py
├── models/
│   └── housing_model.joblib
├── tests/
│   └── test_api.py
├── pyproject.toml
├── poetry.lock
├── README.md
├── Dockerfile
└── .gitignore
</pre>

## Setup

- Install dependencies using Poetry
  ```bash
  poetry install
  ```
- Create the model

  ```bash
  poetry run python src/california_housing_api/model_training.py
  ```

  - Trains and saves a `RandomForestRegressor` in `models/housing_model.joblib`
  - Hyperparameters are tuned with Optuna

- Run BentoML service

  ```bash
  bentoml serve src/california_housing_api/service.py:HousingPredictor
  ```

### Docker

- Build the image

```bash
docker build -t california-housing-api .
```

- Run the container

```bash
docker run -p 3000:3000 california-housing-api
```

## Usage

Endpoints:

- `POST /predict`: Expects a JSON payload with housing features
- `POST /health`: Check if it's running
- `GET /metrics`: Shows Prometheus metrics
