import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime

class ModelMonitor:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.log_file = os.path.join(self.artifacts_dir, "prediction_logs.csv")
        self.baseline_file = os.path.join(self.artifacts_dir, "train.csv")
        self._init_log_file()

    def _init_log_file(self):
        if not os.path.exists(self.log_file):
            columns = [
                "timestamp", "study_hours", "attendance", "previous_score",
                "prediction", "pass_probability", "latency_ms"
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.log_file, index=False)

    def log_prediction(self, study_hours, attendance, previous_score, prediction, probability, latency_ms):
        """Logs each live inference request for telemetry and drift detection."""
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "study_hours": round(float(study_hours), 2),
            "attendance": round(float(attendance), 2),
            "previous_score": round(float(previous_score), 2),
            "prediction": prediction,
            "pass_probability": round(float(probability), 2),
            "latency_ms": round(float(latency_ms), 2)
        }
        df = pd.DataFrame([record])
        df.to_csv(self.log_file, mode='a', header=not os.path.exists(self.log_file), index=False)

    def get_monitoring_summary(self):
        """Calculates live telemetry, model metrics, and data drift status."""
        if not os.path.exists(self.log_file):
            return self._empty_summary()

        try:
            logs = pd.read_csv(self.log_file)
            if logs.empty:
                return self._empty_summary()
        except Exception:
            return self._empty_summary()

        total_requests = len(logs)
        pass_count = int((logs['prediction'] == 'Pass').sum())
        fail_count = int((logs['prediction'] == 'Fail').sum())
        pass_rate = round((pass_count / total_requests) * 100, 2) if total_requests > 0 else 0.0
        avg_latency = round(logs['latency_ms'].mean(), 2) if total_requests > 0 else 0.0
        avg_prob = round(logs['pass_probability'].mean(), 2) if total_requests > 0 else 0.0

        # Data Drift Detection (Comparing live request averages to training baseline)
        drift_status = self._detect_drift(logs)

        # Recent 10 prediction logs
        recent_logs = logs.tail(10).to_dict(orient="records")
        recent_logs.reverse() # latest first

        return {
            "total_requests": total_requests,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_rate": pass_rate,
            "avg_latency_ms": avg_latency,
            "avg_probability": avg_prob,
            "data_drift": drift_status,
            "recent_logs": recent_logs,
            "system_status": "HEALTHY" if not drift_status["drift_detected"] else "DRIFT_WARNING"
        }

    def _detect_drift(self, logs):
        """Checks if live feature distributions have drifted > 25% from training data."""
        baseline_means = {"StudyHours": 4.5, "Attendance": 70.0, "PreviousScore": 58.0}
        
        if os.path.exists(self.baseline_file):
            try:
                base_df = pd.read_csv(self.baseline_file)
                baseline_means = {
                    "StudyHours": float(base_df["StudyHours"].mean()),
                    "Attendance": float(base_df["Attendance"].mean()),
                    "PreviousScore": float(base_df["PreviousScore"].mean())
                }
            except Exception:
                pass

        live_means = {
            "StudyHours": float(logs["study_hours"].mean()),
            "Attendance": float(logs["attendance"].mean()),
            "PreviousScore": float(logs["previous_score"].mean())
        }

        drift_flags = {}
        drift_detected = False
        for feature, base_val in baseline_means.items():
            live_val = live_means.get(feature, base_val)
            pct_diff = abs(live_val - base_val) / (base_val if base_val != 0 else 1) * 100
            is_drifted = pct_diff > 25.0
            if is_drifted:
                drift_detected = True
            drift_flags[feature] = {
                "baseline_mean": round(base_val, 2),
                "live_mean": round(live_val, 2),
                "difference_pct": round(pct_diff, 2),
                "drifted": is_drifted
            }

        return {
            "drift_detected": drift_detected,
            "features": drift_flags
        }

    def _empty_summary(self):
        return {
            "total_requests": 0,
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate": 0.0,
            "avg_latency_ms": 0.0,
            "avg_probability": 0.0,
            "data_drift": {"drift_detected": False, "features": {}},
            "recent_logs": [],
            "system_status": "IDLE"
        }

if __name__ == "__main__":
    monitor = ModelMonitor()
    monitor.log_prediction(6.0, 80.0, 70.0, "Pass", 95.5, 4.2)
    print(monitor.get_monitoring_summary())
