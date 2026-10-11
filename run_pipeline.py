"""
MLOps Pipeline Orchestrator
Executes: Data Ingestion -> Data Transformation -> Model Training -> Evaluation
"""
import time
from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.model_trainer import ModelTrainer
from src.predict_pipeline import PredictPipeline

def run_mlops_pipeline():
    start_time = time.time()
    print("===============================================================")
    print("        STUDENT PASS/FAIL PREDICTION - MLOps PIPELINE         ")
    print("===============================================================")

    # 1. Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 2. Transformation
    transformation = DataTransformation()
    X_train_scaled, y_train, X_test_scaled, y_test, scaler_path = transformation.initiate_data_transformation(
        train_path, test_path
    )

    # 3. Model Training & Evaluation
    trainer = ModelTrainer()
    metrics = trainer.initiate_model_training(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    # 4. Pipeline Sanity Test
    print("\n--- [MLOps Step 4: Pipeline Sanity Test] ---")
    predictor = PredictPipeline()
    sample_prediction = predictor.predict(study_hours=7.0, attendance=90.0, previous_score=80.0)
    print("Sanity Test Prediction (Hardworking Student):")
    print(f"  • Prediction:       {sample_prediction['prediction']}")
    print(f"  • Pass Probability: {sample_prediction['pass_probability']}%")
    print(f"  • Serving Engine:   {sample_prediction['engine']}")

    elapsed = round(time.time() - start_time, 2)
    print("\n===============================================================")
    print(f" MLOps Pipeline executed successfully in {elapsed} seconds!")
    print("===============================================================")
    return metrics

if __name__ == "__main__":
    run_mlops_pipeline()
