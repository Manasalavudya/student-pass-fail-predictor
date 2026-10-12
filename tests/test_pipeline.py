import os
import unittest
from src.predict_pipeline import PredictPipeline
from app import app

class TestMLOpsPipeline(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_artifacts_exist(self):
        self.assertTrue(os.path.exists("artifacts/scaler.pkl"), "scaler.pkl artifact missing")
        self.assertTrue(os.path.exists("artifacts/metrics.json"), "metrics.json artifact missing")
        self.assertTrue(os.path.exists("artifacts/weights.npz"), "weights.npz artifact missing")

    def test_prediction_pass_case(self):
        pipeline = PredictPipeline()
        res = pipeline.predict(study_hours=8.0, attendance=90.0, previous_score=85.0)
        self.assertEqual(res["prediction"], "Pass")
        self.assertGreaterEqual(res["pass_probability"], 50.0)

    def test_prediction_fail_case(self):
        pipeline = PredictPipeline()
        res = pipeline.predict(study_hours=1.0, attendance=45.0, previous_score=30.0)
        self.assertEqual(res["prediction"], "Fail")
        self.assertLess(res["pass_probability"], 50.0)

    def test_flask_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "healthy")

    def test_flask_json_predict_endpoint(self):
        response = self.client.post("/predict", json={
            "study_hours": 7.0,
            "attendance": 85.0,
            "previous_score": 75.0
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "success")
        self.assertIn("result", response.json)
        self.assertEqual(response.json["result"]["prediction"], "Pass")

    def test_flask_monitoring_endpoint(self):
        response = self.client.get("/monitoring")
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
