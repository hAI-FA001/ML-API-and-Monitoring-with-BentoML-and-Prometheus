# California Housing Price Prediction API

An ML REST API containerized with Docker, monitored with Prometheus and Grafana, secured with API key authentication and managed with BentoML's Model Registry.

Tech:

- Scikit-Learn (RandomForestRegressor)
- Optuna for hyperparameter tuning
- BentoML to serve the API endpoints
- BentoML for model registry
- Prometheus for logging metrics and request latency
- Grafana for visualization through custom dashboards
- Dockerized BentoML, Prometheus and Grafana services
- CI/CD with GitHub Actions

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
│       ├── api_config.py
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
├── docker-compose.yml
├── prometheus.yml
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

  - Trains and saves a `RandomForestRegressor` in BentoML's model registry and under `models/` directory.

- Run the services

  ```bash
  docker-compose up --build
  ```

## Usage

Endpoints:

- `POST /predict`: Expects a JSON payload with housing features
- `POST /health`: Check if it's running
- `GET /metrics`: Shows Prometheus metrics

Each endpoint expects an API key, which you can get by inspecting `api_keys.txt`
