import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from Web_App.app import app, process_prediction_request, calculate_engineered_features

def test_engineered_features():
    data = {
        'sleep_duration': 5.5,
        'stress_level': 'high',
        'sleep_quality': 'poor',
        'physical_activity_level': 'sedentary',
        'smoking_alcohol': 'yes',
        'bmi': 26.5
    }
    fe = calculate_engineered_features(data)
    assert fe['lifestyle_risk_index'] == 6
    assert fe['sleep_distance_from_8'] == 2.5
    assert fe['stress_num'] == 3

def test_prediction_process():
    data = {
        'age': 22,
        'gender': 'male',
        'sleep_duration': 8.0,
        'exercise_duration': 45,
        'stress_level': 'low',
        'physical_activity_level': 'active',
        'sleep_quality': 'good',
        'smoking_alcohol': 'no',
        'diet_type': 'balanced',
        'step_count': 10000,
        'calorie_expenditure': 2500,
        'heart_rate': 68,
        'bmi': 21.5,
        'water_intake': 3.0
    }
    res = process_prediction_request(data)
    assert res['status'] == 'success'
    assert 'prediction' in res
    assert res['prediction'] in ['fit', 'at-risk', 'unhealthy']
    assert 'probabilities' in res
    assert 'latency_ms' in res
    assert res['latency_ms'] >= 0.0

def test_fastapi_endpoints():
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # 1. Health check
        res_health = client.get("/api/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "online"
        
        # 2. Liveness check
        res_live = client.get("/health/live")
        assert res_live.status_code == 200
        assert res_live.json()["status"] == "alive"
        
        # 3. Prediction test
        payload = {
            "age": 21,
            "gender": "female",
            "sleep_duration": 7.5,
            "exercise_duration": 30,
            "stress_level": "moderate",
            "physical_activity_level": "moderate",
            "sleep_quality": "good",
            "smoking_alcohol": "no",
            "diet_type": "balanced",
            "step_count": 8000,
            "calorie_expenditure": 2100,
            "heart_rate": 72,
            "bmi": 22.0,
            "water_intake": 2.5
        }
        res_pred = client.post("/api/predict", json=payload)
        assert res_pred.status_code == 200
        data = res_pred.json()
        assert data["status"] == "success"
        assert data["prediction"] in ["fit", "at-risk", "unhealthy"]
    except ImportError:
        print("FastAPI / TestClient not installed, skipping TestClient verification.")

if __name__ == '__main__':
    print("Running Unit & Integration API Tests...")
    test_engineered_features()
    test_prediction_process()
    test_fastapi_endpoints()
    print("SUCCESS: All Unit & Integration API Tests PASSED!")
