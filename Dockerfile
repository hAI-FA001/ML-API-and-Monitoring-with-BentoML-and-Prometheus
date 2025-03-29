FROM python:3.12-slim

WORKDIR /app

COPY poetry.lock pyproject.toml ./
COPY README.md ./
COPY src ./src
RUN mkdir ./models

RUN pip install poetry && poetry install && poetry env info --path > /tmp/poetry_path.txt && BENTO_PATH=$(cat /tmp/poetry_path.txt)/bin/bentoml && echo "Bento Path: $BENTO_PATH"
RUN poetry run python -m california_housing_api.model_training

EXPOSE 3000

CMD ["/bin/sh", "-c", "$(cat /tmp/poetry_path.txt)/bin/bentoml serve ./src/california_housing_api/service.py:HousingPredictor --host 0.0.0.0 --port 3000"]
