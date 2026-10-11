import os
import json
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelTrainer:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.model_path = os.path.join(self.artifacts_dir, "model.keras")
        self.weights_path = os.path.join(self.artifacts_dir, "weights.npz")
        self.metrics_path = os.path.join(self.artifacts_dir, "metrics.json")

    def build_neural_network(self):
        tf.random.set_seed(42)
        np.random.seed(42)

        model = keras.Sequential([
            keras.Input(shape=(3,), name="Input_Layer"),
            keras.layers.Dense(8, activation='relu', name="Hidden_Layer"),
            keras.layers.Dense(1, activation='sigmoid', name="Output_Layer")
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        print("\n--- [MLOps Step 3: Model Training & Evaluation] ---")
        model = self.build_neural_network()
        print("Neural Network Architecture Built:")
        model.summary(print_fn=lambda x: print(f"  {x}"))

        # Train model
        print("Training model (100 Epochs, batch size=4, Adam optimizer)...")
        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=4,
            verbose=0,
            validation_data=(X_test, y_test)
        )
        print("Training Complete!")

        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

        metrics = {
            "test_loss": float(round(test_loss, 4)),
            "test_accuracy": float(round(test_acc * 100, 2)),
            "precision": float(round(precision_score(y_test, y_pred, zero_division=0) * 100, 2)),
            "recall": float(round(recall_score(y_test, y_pred, zero_division=0) * 100, 2)),
            "f1_score": float(round(f1_score(y_test, y_pred, zero_division=0) * 100, 2)),
            "epochs": 100,
            "architecture": "Input(3) -> Dense(8, ReLU) -> Dense(1, Sigmoid)"
        }

        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"  • {k}: {v}")

        # Save model artifacts
        model.save(self.model_path)
        print(f"\nTrained Model saved -> {self.model_path}")

        # Save weights for ultra-lightweight server deployment fallback
        weights = model.get_weights()
        np.savez(
            self.weights_path,
            w1=weights[0], b1=weights[1],
            w2=weights[2], b2=weights[3]
        )
        print(f"Model Weights saved -> {self.weights_path}")

        # Save metrics artifact
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Evaluation Metrics saved -> {self.metrics_path}")

        return metrics

if __name__ == "__main__":
    from src.data_ingestion import DataIngestion
    from src.data_transformation import DataTransformation

    ingestion = DataIngestion()
    train_p, test_p = ingestion.initiate_data_ingestion()
    transformation = DataTransformation()
    X_train, y_train, X_test, y_test, _ = transformation.initiate_data_transformation(train_p, test_p)
    trainer = ModelTrainer()
    trainer.initiate_model_training(X_train, y_train, X_test, y_test)
