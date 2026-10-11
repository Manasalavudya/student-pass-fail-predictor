import os
import joblib
import numpy as np

class PredictPipeline:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir
        self.scaler_path = os.path.join(self.artifacts_dir, "scaler.pkl")
        self.model_path = os.path.join(self.artifacts_dir, "model.keras")
        self.weights_path = os.path.join(self.artifacts_dir, "weights.npz")
        
        self.scaler = None
        self.keras_model = None
        self.weights = None
        self._load_artifacts()

    def _load_artifacts(self):
        # 1. Load Scaler
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
        else:
            raise FileNotFoundError(f"Scaler artifact not found at {self.scaler_path}. Run pipeline first!")

        # 2. Try loading Keras Model; if unavailable or memory-constrained, load weights
        try:
            from tensorflow import keras
            if os.path.exists(self.model_path):
                self.keras_model = keras.models.load_model(self.model_path)
        except Exception as e:
            # Fallback to weights
            self.keras_model = None

        if os.path.exists(self.weights_path):
            self.weights = np.load(self.weights_path)

    def predict(self, study_hours: float, attendance: float, previous_score: float):
        """
        Runs inference on given student features.
        Returns:
            dict containing pass_probability (0-100), prediction ('Pass' or 'Fail'),
            and engine ('TensorFlow/Keras' or 'Lightweight Neural Weights')
        """
        raw_features = np.array([[float(study_hours), float(attendance), float(previous_score)]])
        scaled_features = self.scaler.transform(raw_features)

        if self.keras_model is not None:
            raw_prob = float(self.keras_model.predict(scaled_features, verbose=0)[0][0])
            engine = "TensorFlow / Keras"
        elif self.weights is not None:
            # Exact Neural Network Forward Propagation:
            # Layer 1: ReLU(X * W1 + b1)
            w1, b1 = self.weights['w1'], self.weights['b1']
            z1 = np.dot(scaled_features, w1) + b1
            a1 = np.maximum(0, z1) # ReLU

            # Layer 2: Sigmoid(a1 * W2 + b2)
            w2, b2 = self.weights['w2'], self.weights['b2']
            z2 = np.dot(a1, w2) + b2
            raw_prob = float(1.0 / (1.0 + np.exp(-z2))[0][0]) # Sigmoid
            engine = "Lightweight Neural Weights (Render-optimized)"
        else:
            raise RuntimeError("No model or weights artifact available for prediction.")

        prob_percentage = round(raw_prob * 100, 2)
        result = "Pass" if raw_prob >= 0.5 else "Fail"

        return {
            "prediction": result,
            "pass_probability": prob_percentage,
            "confidence": prob_percentage if result == "Pass" else round(100 - prob_percentage, 2),
            "engine": engine
        }

class CustomData:
    def __init__(self, study_hours: float, attendance: float, previous_score: float):
        self.study_hours = study_hours
        self.attendance = attendance
        self.previous_score = previous_score

    def get_features(self):
        return [self.study_hours, self.attendance, self.previous_score]

if __name__ == "__main__":
    pipeline = PredictPipeline()
    sample = pipeline.predict(study_hours=6.0, attendance=85.0, previous_score=75.0)
    print("Inference Test:", sample)
