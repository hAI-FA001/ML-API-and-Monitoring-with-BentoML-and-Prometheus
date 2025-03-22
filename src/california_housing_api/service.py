import bentoml
import joblib
import numpy as np
from pydantic import BaseModel


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


@bentoml.service(name="housing_predictor", runners=[])
class HousingPredictor:
    @bentoml.api
    def predict(self, input_data: HousingFeatures) -> np.ndarray:
        input_array = np.array(list(input_data.dict().values())).reshape(1, -1)
        prediction = model.predict(input_array)
        return prediction
