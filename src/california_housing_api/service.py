import bentoml
import joblib
import numpy as np
from pydantic import BaseModel

from prometheus_client import Counter, generate_latest, Histogram
import time

from bentoml.exceptions import InvalidArgument
from california_housing_api.api_config import API_KEYS


def authenticate(hashed):
    if hashed not in API_KEYS:
        raise InvalidArgument(error_code=401, message="Unauthorized")


class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


model = joblib.load("models/housing_model.joblib")

# prometheus metrics
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request Latency in Seconds")
REQUEST_COUNT = Counter("request_count", "Total number of Requests", ["endpoint"])
ERROR_COUNT = Counter("error_count", "Total number of Errors", ["endpoint"])


@bentoml.service(name="housing_predictor", runners=[])
class HousingPredictor:
    @bentoml.api
    def predict(self, input_data: HousingFeatures, api_key: str) -> np.ndarray:
        authenticate(api_key)

        start = time.time()
        REQUEST_COUNT.labels(endpoint="predict").inc()

        try:
            input_array = np.array(list(input_data.dict().values())).reshape(1, -1)
            prediction = model.predict(input_array)

            return {"prediction": prediction.tolist()}
        except Exception as e:
            ERROR_COUNT.labels(endpoint="predict").inc()
            return {"error": str(e)}
        finally:
            REQUEST_LATENCY.observe(time.time() - start)

    @bentoml.api(route="/health")
    def health(self, api_key: str) -> str:
        authenticate(api_key)

        REQUEST_COUNT.labels(endpoint="health").inc()
        return "OK"

    @bentoml.api(route="/metrics")
    def metrics(self, api_key: str) -> str:
        authenticate(api_key)

        REQUEST_COUNT.labels(endpoint="metrics").inc()
        return generate_latest().decode()
