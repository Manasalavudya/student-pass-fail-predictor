import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataTransformation:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir
        self.scaler_path = os.path.join(self.artifacts_dir, "scaler.pkl")

    def initiate_data_transformation(self, train_path, test_path):
        print("\n--- [MLOps Step 2: Data Transformation & Feature Scaling] ---")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        feature_cols = ['StudyHours', 'Attendance', 'PreviousScore']
        target_col = 'Result'

        # Separate features and target
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].map({'Fail': 0, 'Pass': 1}).values

        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].map({'Fail': 0, 'Pass': 1}).values

        # Apply StandardScaler normalization (mean=0, std=1)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save transformer artifact
        joblib.dump(scaler, self.scaler_path)
        print(f"StandardScaler fitted and saved -> {self.scaler_path}")
        print(f"Features transformed: {feature_cols}")

        return X_train_scaled, y_train, X_test_scaled, y_test, self.scaler_path

if __name__ == "__main__":
    transformer = DataTransformation()
    transformer.initiate_data_transformation("artifacts/train.csv", "artifacts/test.csv")
