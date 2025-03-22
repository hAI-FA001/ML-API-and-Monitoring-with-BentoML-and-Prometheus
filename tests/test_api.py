import requests
import json


def test_predict_endpoint():
    url = "http://localhost:3001/predict"
    payload = {
        "input_data": {
            "MedInc": 8.3252,
            "HouseAge": 41.0,
            "AveRooms": 6.984127,
            "AveBedrms": 1.023810,
            "Population": 322.0,
            "AveOccup": 2.555556,
            "Latitude": 37.88,
            "Longitude": -122.23,
        },
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    assert response.status_code == 200
    assert "prediction" in response.json() or "error" in response.json()


def test_health_endpoint():
    url = "http://localhost:3001/health"
    response = requests.post(url)

    assert response.status_code == 200
    assert response.text == "OK"


def test_metrics_endpoint():
    url = "http://localhost:3001/metrics"
    response = requests.get(url)

    assert response.status_code == 200
