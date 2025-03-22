import bentoml
import joblib
import numpy as np
from pydantic import BaseModel

from prometheus_client import Summary, Counter, generate_latest
import time


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
PREDICTION_TIME = Summary("prediction_time", "Time spent on Predictions")
PREDICTION_COUNTER = Counter("prediction_counter", "No. of Predictions")


@bentoml.service(name="housing_predictor", runners=[])
class HousingPredictor:
    @bentoml.api
    def predict(self, input_data: HousingFeatures) -> np.ndarray:
        start = time.time()

        input_array = np.array(list(input_data.dict().values())).reshape(1, -1)
        prediction = model.predict(input_array)

        end = time.time()

        PREDICTION_TIME.observe(end - start)
        PREDICTION_COUNTER.inc()

        return prediction

    @bentoml.api(route="/health")
    def health(self) -> str:
        return "OK"

    @bentoml.api(route="/metrics")
    def metrics(self) -> str:
        return generate_latest().decode()
